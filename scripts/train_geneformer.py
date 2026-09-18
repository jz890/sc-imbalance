"""Fine-tune Geneformer with a selected long-tail loss."""

import argparse
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import wandb
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup

from common import (
    base_env_info,
    compute_metrics,
    compute_rare_classes_for_run,
    load_adatas,
    load_config,
    save_embeddings,
    save_per_class_and_confusion,
    write_test_results_json,
)
from losses import LOSS_NAMES, build_loss

BACKBONE = "geneformer"
MODEL_INPUT_SIZE = 2048

SPECIAL_TOKEN = False
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.001
MAX_GRAD_NORM = 1.0


def tokenize_dataset_cached(adata, label_col: str, cache_dir: Path, nproc: int = 8):
    """Tokenize AnnData once and cache the resulting Arrow dataset."""
    from datasets import load_from_disk
    from geneformer import TranscriptomeTokenizer

    if cache_dir.exists():
        return load_from_disk(str(cache_dir))

    print(f"  tokenizing (not cached yet) -> {cache_dir} ...")
    tmp_dir = cache_dir.parent / f"_tokenize_tmp_{cache_dir.name}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    tmp_adata = adata.copy()
    tmp_adata.obs["label"] = tmp_adata.obs[label_col].astype(int)
    tmp_adata.write_h5ad(tmp_dir / "input.h5ad")

    tk = TranscriptomeTokenizer(
        custom_attr_name_dict={"label": "label"},
        nproc=nproc,
        model_version="V1",
    )
    tk.tokenize_data(
        data_directory=str(tmp_dir),
        output_directory=str(cache_dir.parent),
        output_prefix=cache_dir.stem,
        file_format="h5ad",
    )
    shutil.rmtree(tmp_dir)

    produced = cache_dir.parent / f"{cache_dir.stem}.dataset"
    if produced != cache_dir:
        produced.rename(cache_dir)
    return load_from_disk(str(cache_dir))


def forward_with_embedding(model, input_ids, attention_mask):
    """Return logits and the pooled CLS representation from the Geneformer model."""
    bert_out = model.bert(input_ids=input_ids, attention_mask=attention_mask)
    pooled = bert_out.pooler_output
    logits = model.classifier(model.dropout(pooled))
    return logits, pooled


def run_train_epoch(
    model, loader, optimizer, scheduler, criterion, device, amp, log_every=20
):
    model.train()
    total_loss, total_n = 0.0, 0
    num_batches = len(loader)
    epoch_start = time.time()
    for i, batch in enumerate(loader, start=1):
        labels = batch.pop("labels").to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.autocast(device_type="cuda", enabled=amp, dtype=torch.bfloat16):
            logits, _ = forward_with_embedding(model, input_ids, attention_mask)
            loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * len(labels)
        total_n += len(labels)
        if i % log_every == 0 or i == num_batches:
            elapsed = time.time() - epoch_start
            print(
                f"  batch {i:4d}/{num_batches} | loss {loss.item():.4f} | "
                f"{elapsed / i:.2f}s/batch | elapsed {elapsed:.0f}s",
                flush=True,
            )
    return total_loss / total_n


@torch.no_grad()
def evaluate(
    model,
    loader,
    id2type,
    rare_classes,
    device,
    amp,
    log_every=20,
    return_embeddings=False,
):
    model.eval()
    preds, trues, embs, logits_list = [], [], [], []
    num_batches = len(loader)
    start = time.time()
    for i, batch in enumerate(loader, start=1):
        labels = batch.pop("labels").to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.autocast(device_type="cuda", enabled=amp, dtype=torch.bfloat16):
            logits, pooled = forward_with_embedding(model, input_ids, attention_mask)
        preds.append(logits.argmax(1).cpu().numpy())
        trues.append(labels.cpu().numpy())
        if return_embeddings:
            embs.append(pooled.float().cpu().numpy())
            logits_list.append(logits.float().cpu().numpy())
        if num_batches > log_every and (i % log_every == 0 or i == num_batches):
            print(
                f"    eval batch {i:4d}/{num_batches} | elapsed {time.time() - start:.0f}s",
                flush=True,
            )
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    metrics = compute_metrics(trues, preds, id2type, rare_classes)
    if return_embeddings:
        return (
            metrics,
            preds,
            trues,
            np.concatenate(embs, axis=0),
            np.concatenate(logits_list, axis=0),
        )
    return metrics, preds, trues


def main(
    config_path: str,
    loss: str | None = None,
    loss_kwargs: dict | None = None,
    seed: int | None = None,
):
    import pickle
    from geneformer import DataCollatorForCellClassification, TOKEN_DICTIONARY_FILE_30M

    cfg = load_config(config_path)
    if loss is not None:
        cfg["loss"] = loss
    if loss_kwargs is not None:
        cfg["loss_kwargs"] = loss_kwargs
    if seed is not None:
        cfg["seed"] = seed

    if cfg["dataset_name"] not in (
        "ms",
        "zheng68k",
        "hpancreas",
        "ms_oversampled",
        "zheng68k_oversampled",
    ):
        raise NotImplementedError(
            f"dataset_name={cfg['dataset_name']!r}; use 'ms', 'zheng68k', 'hpancreas', "
            "'ms_oversampled', or 'zheng68k_oversampled'"
        )
    if cfg["loss"] not in LOSS_NAMES:
        raise NotImplementedError(f"loss={cfg['loss']!r}; use one of {LOSS_NAMES}")

    run_tag = f"{BACKBONE}_{cfg['dataset_name']}_{cfg['loss']}_seed{cfg['seed']}"
    cfg["output"]["save_dir"] = str(Path(cfg["output"]["save_dir"]) / run_tag)
    if not cfg["wandb"].get("run_name"):
        cfg["wandb"]["run_name"] = run_tag

    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    env_info = base_env_info(BACKBONE, cfg) | {
        "grad_clip_max_norm": MAX_GRAD_NORM,
        "model_input_size": MODEL_INPUT_SIZE,
        "special_token": SPECIAL_TOKEN,
    }

    load_dotenv()
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
    else:
        print(
            "  WARNING: WANDB_API_KEY not set, running with wandb disabled "
            "(wandb.init(mode='disabled')). This does not affect results: every "
            "run's metrics, training history, and environment info are written "
            "to test_results.json regardless of wandb status."
        )

    print("loading data...")
    adata, adata_test, id2type = load_adatas(cfg["data"])
    num_types = len(id2type)
    print(
        f"  train: {adata.shape}, test: {adata_test.shape}, {num_types} cell types total"
    )

    rare_classes = compute_rare_classes_for_run(cfg["data"], cfg["rare_class"], adata)
    print(
        f"rare classes (<{cfg['rare_class']['relative_threshold']:.0%} of training set, {len(rare_classes)} total):"
    )
    for c in rare_classes:
        print(f"  - {c}")

    data_dir = Path(cfg["data"]["train_h5ad"]).parent
    train_pool_ds = tokenize_dataset_cached(
        adata,
        "celltype_id",
        data_dir / f"{cfg['dataset_name']}_train_tokenized.dataset",
    )
    test_ds = tokenize_dataset_cached(
        adata_test,
        "celltype_id",
        data_dir / f"{cfg['dataset_name']}_test_tokenized.dataset",
    )

    val_fraction = cfg["data"]["val_fraction"]
    all_labels = np.array(train_pool_ds["label"])
    try:
        train_idx, val_idx = train_test_split(
            np.arange(len(train_pool_ds)),
            test_size=val_fraction,
            random_state=cfg["seed"],
            stratify=all_labels,
        )
    except ValueError as e:
        print(f"  stratified split failed ({e}); falling back to a plain random split")
        train_idx, val_idx = train_test_split(
            np.arange(len(train_pool_ds)),
            test_size=val_fraction,
            random_state=cfg["seed"],
        )

    train_ds = train_pool_ds.select(train_idx)
    val_ds = train_pool_ds.select(val_idx)

    train_labels = np.array(train_ds["label"])
    cls_num_list = [max(int((train_labels == i).sum()), 1) for i in range(num_types)]
    print(f"  train class counts: {cls_num_list}")

    with open(TOKEN_DICTIONARY_FILE_30M, "rb") as f:
        token_dictionary = pickle.load(f)
    collator = DataCollatorForCellClassification(token_dictionary=token_dictionary)
    batch_size = cfg["train"]["batch_size"]
    eval_batch_size = cfg["train"]["eval_batch_size"]
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator
    )
    val_loader = DataLoader(
        val_ds, batch_size=eval_batch_size, shuffle=False, collate_fn=collator
    )
    test_loader = DataLoader(
        test_ds, batch_size=eval_batch_size, shuffle=False, collate_fn=collator
    )

    print("building model...")

    model = BertForSequenceClassification.from_pretrained(
        cfg["pretrained_model_dir"],
        num_labels=num_types,
        classifier_dropout=cfg["model"]["dropout"],
        output_attentions=False,
        output_hidden_states=False,
    ).to(device)

    criterion, resolved_loss_kwargs = build_loss(
        cfg["loss"], cls_num_list, device, **cfg.get("loss_kwargs", {})
    )
    print(
        f"  loss: {cfg['loss']} ({type(criterion).__name__}), resolved_loss_kwargs: {resolved_loss_kwargs}"
    )
    env_info["resolved_loss_kwargs"] = resolved_loss_kwargs

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=WEIGHT_DECAY
    )
    total_steps = len(train_loader) * cfg["train"]["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps
    )
    amp = cfg["train"].get("amp", False) and device.type == "cuda"
    print(f"  amp: {amp} (bf16 -- requires Ampere/Ada or newer)")

    run = wandb.init(
        project=cfg["wandb"]["project"],
        entity=cfg["wandb"].get("entity"),
        name=cfg["wandb"]["run_name"],
        config=cfg | {"rare_classes": rare_classes},
        mode=None if wandb_api_key else "disabled",
    )

    save_dir = Path(cfg["output"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    emissions_tracker = None
    try:
        from codecarbon import EmissionsTracker

        emissions_tracker = EmissionsTracker(
            project_name=run_tag,
            output_dir=str(save_dir),
            log_level="error",
            save_to_file=True,
        )
        emissions_tracker.start()
    except Exception as e:
        print(
            f"  WARNING: codecarbon tracker unavailable, skipping emissions tracking ({e})"
        )
        emissions_tracker = None

    best_val_macro_f1, best_epoch = -1.0, -1
    history = []
    train_start = time.time()
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        epoch_start = time.time()
        train_loss = run_train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, amp
        )
        val_metrics, _, _ = evaluate(
            model, val_loader, id2type, rare_classes, device, amp
        )
        epoch_time = time.time() - epoch_start

        print(
            f"epoch {epoch:3d} | train_loss {train_loss:.4f} | val_macro_f1 {val_metrics['macro_f1']:.4f} | "
            f"val_rare_recall {val_metrics['rare_class_recall']:.4f} | val_balanced_acc {val_metrics['balanced_accuracy']:.4f}"
        )
        wandb.log(
            {"epoch": epoch, "train/loss": train_loss}
            | {f"val/{k}": v for k, v in val_metrics.items()}
        )
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "epoch_seconds": epoch_time}
            | {f"val_{k}": v for k, v in val_metrics.items()}
        )

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1, best_epoch = val_metrics["macro_f1"], epoch
            torch.save(model.state_dict(), save_dir / "best_model.pt")
            print(f"  new best val macro_f1={best_val_macro_f1:.4f}, checkpoint saved")
    total_train_seconds = time.time() - train_start

    emissions_kg = None
    if emissions_tracker is not None:
        try:
            emissions_kg = emissions_tracker.stop()
        except Exception as e:
            print(f"  WARNING: codecarbon stop() failed ({e})")

    peak_memory_mb = (
        {
            "allocated": torch.cuda.max_memory_allocated(device) / 1e6,
            "reserved": torch.cuda.max_memory_reserved(device) / 1e6,
        }
        if device.type == "cuda"
        else None
    )

    print("loading best checkpoint for final test evaluation...")
    model.load_state_dict(torch.load(save_dir / "best_model.pt", map_location=device))
    test_metrics, test_preds, test_trues, test_embeddings, test_logits = evaluate(
        model, test_loader, id2type, rare_classes, device, amp, return_embeddings=True
    )

    print("\n=== final test metrics ===")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
    wandb.log({f"test/{k}": v for k, v in test_metrics.items()})

    save_embeddings(
        test_embeddings,
        test_preds,
        test_trues,
        test_logits,
        id2type,
        cfg,
        run.id,
        BACKBONE,
    )
    per_class_report = save_per_class_and_confusion(
        test_trues, test_preds, id2type, rare_classes, save_dir
    )
    write_test_results_json(
        save_dir,
        test_metrics,
        rare_classes,
        id2type,
        per_class_report,
        cls_num_list,
        len(train_idx),
        len(val_idx),
        int(adata_test.n_obs),
        best_epoch,
        best_val_macro_f1,
        total_train_seconds,
        peak_memory_mb,
        emissions_kg,
        history,
        env_info,
        cfg,
    )
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--loss", default=None, choices=LOSS_NAMES)
    parser.add_argument("--loss-kwargs", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    loss_kwargs = json.loads(args.loss_kwargs) if args.loss_kwargs else None
    main(args.config, loss=args.loss, loss_kwargs=loss_kwargs, seed=args.seed)
