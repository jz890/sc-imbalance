"""Fine-tune scGPT with a selected long-tail loss."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import wandb
from dotenv import load_dotenv
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

import scgpt as scg
from scgpt.model import TransformerModel
from scgpt.preprocess import Preprocessor
from scgpt.tokenizer import tokenize_and_pad_batch
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import set_seed

from common import (
    base_env_info,
    compute_metrics,
    compute_rare_classes_for_run,
    get_gene_names,
    get_git_commit_hash,
    load_adatas,
    load_config,
    save_embeddings,
    save_per_class_and_confusion,
    write_test_results_json,
)
from losses import LOSS_NAMES, build_loss

PAD_TOKEN = "<pad>"
SPECIAL_TOKENS = [PAD_TOKEN, "<cls>", "<eoc>"]
PAD_VALUE = -2
GRAD_CLIP_MAX_NORM = 1.0
BACKBONE = "scgpt"


class CellDataset(Dataset):
    def __init__(self, data: dict):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


def build_vocab(pretrained_dir: str) -> GeneVocab:
    vocab = GeneVocab.from_file(Path(pretrained_dir) / "vocab.json")
    for tok in SPECIAL_TOKENS:
        if tok not in vocab:
            vocab.append_token(tok)
    vocab.set_default_index(vocab[PAD_TOKEN])
    return vocab


def filter_to_vocab(adata, vocab: GeneVocab, gene_name_col: str | None):
    mask = np.array([g in vocab for g in get_gene_names(adata, gene_name_col)])
    kept = int(mask.sum())
    print(f"  {kept}/{len(mask)} genes found in pretrained vocab")
    return adata[:, mask].copy()


def preprocess(adata, n_bins: int):
    prep = Preprocessor(
        use_key="X",
        filter_gene_by_counts=False,
        filter_cell_by_counts=False,
        normalize_total=1e4,
        result_normed_key="X_normed",
        log1p=False,
        result_log1p_key="X_log1p",
        subset_hvg=False,
        hvg_flavor="cell_ranger",
        binning=n_bins,
        result_binned_key="X_binned",
    )
    prep(adata, batch_key=None)
    return adata


def tokenize_split(
    adata, vocab: GeneVocab, gene_ids: np.ndarray, max_seq_len: int
) -> dict:
    binned = adata.layers["X_binned"]
    counts = binned.toarray() if issparse(binned) else np.asarray(binned)
    tokenized = tokenize_and_pad_batch(
        counts,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=PAD_TOKEN,
        pad_value=PAD_VALUE,
        append_cls=True,
        include_zero_gene=False,
    )
    return {
        "gene_ids": tokenized["genes"],
        "values": tokenized["values"],
        "celltype_labels": torch.from_numpy(adata.obs["celltype_id"].to_numpy()).long(),
    }


def build_model(
    vocab: GeneVocab, model_cfg: dict, pretrained_dir: str, num_types: int, device
):
    backend = model_cfg["fast_transformer_backend"]
    if backend not in ("linear", "flash", "none"):
        raise ValueError(
            f"fast_transformer_backend must be 'linear', 'flash', or 'none', got {backend!r}"
        )
    if backend == "flash":
        try:
            import flash_attn
        except ImportError as e:
            raise RuntimeError(
                "fast_transformer_backend='flash' requires the flash-attn package to be installed "
                "(pip install flash-attn --no-build-isolation). Use 'linear' or 'none' if you don't "
                "have it installed."
            ) from e
    use_fast_transformer = model_cfg["fast_transformer"] and backend != "none"

    with open(Path(pretrained_dir) / "args.json", encoding="utf-8") as f:
        margs = json.load(f)

    model = TransformerModel(
        len(vocab),
        margs["embsize"],
        margs["nheads"],
        margs["d_hid"],
        margs["nlayers"],
        nlayers_cls=3,
        n_cls=num_types,
        vocab=vocab,
        dropout=model_cfg["dropout"],
        pad_token=PAD_TOKEN,
        pad_value=PAD_VALUE,
        do_mvc=False,
        do_dab=False,
        use_batch_labels=False,
        num_batch_labels=1,
        domain_spec_batchnorm=False,
        input_emb_style="continuous",
        n_input_bins=model_cfg["n_bins"],
        cell_emb_style="cls",
        mvc_decoder_style="inner product",
        ecs_threshold=0.0,
        explicit_zero_prob=False,
        use_fast_transformer=use_fast_transformer,
        fast_transformer_backend=backend if use_fast_transformer else "linear",
        pre_norm=False,
    )

    state_dict = torch.load(Path(pretrained_dir) / "best_model.pt", map_location=device)
    model_dict = model.state_dict()
    matched = {
        k: v
        for k, v in state_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    print(
        f"  loaded {len(matched)}/{len(model_dict)} matching tensors from pretrained checkpoint"
    )
    model_dict.update(matched)
    model.load_state_dict(model_dict)
    return model.to(device)


def forward_pass(model, gene_ids, values, vocab):
    padding_mask = gene_ids.eq(vocab[PAD_TOKEN])
    return model(
        gene_ids,
        values,
        src_key_padding_mask=padding_mask,
        batch_labels=None,
        CLS=True,
        CCE=False,
        MVC=False,
        ECS=False,
        do_sample=False,
    )


def run_train_epoch(
    model, loader, optimizer, criterion, vocab, device, amp, log_every=20
):
    model.train()
    total_loss, total_n = 0.0, 0
    num_batches = len(loader)
    epoch_start = time.time()
    for i, batch in enumerate(loader, start=1):
        gene_ids = batch["gene_ids"].to(device)
        values = batch["values"].to(device)
        labels = batch["celltype_labels"].to(device)

        with torch.autocast(device_type="cuda", enabled=amp, dtype=torch.bfloat16):
            out = forward_pass(model, gene_ids, values, vocab)
            loss = criterion(out["cls_output"], labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
        optimizer.step()

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
    vocab,
    device,
    amp,
    id2type,
    rare_classes,
    log_every=20,
    return_embeddings=False,
):
    model.eval()
    preds, trues, embs, logits_list = [], [], [], []
    num_batches = len(loader)
    start = time.time()
    for i, batch in enumerate(loader, start=1):
        gene_ids = batch["gene_ids"].to(device)
        values = batch["values"].to(device)
        labels = batch["celltype_labels"].to(device)
        with torch.autocast(device_type="cuda", enabled=amp, dtype=torch.bfloat16):
            out = forward_pass(model, gene_ids, values, vocab)
        logits = out["cls_output"]
        preds.append(logits.argmax(1).cpu().numpy())
        trues.append(labels.cpu().numpy())
        if return_embeddings:

            embs.append(out["cell_emb"].float().cpu().numpy())

            logits_list.append(logits.float().cpu().numpy())
        if num_batches > log_every and (i % log_every == 0 or i == num_batches):
            print(
                f"    eval batch {i:4d}/{num_batches} | elapsed {time.time() - start:.0f}s",
                flush=True,
            )
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    if return_embeddings:
        embeddings = np.concatenate(embs, axis=0)
        all_logits = np.concatenate(logits_list, axis=0)

    metrics = compute_metrics(trues, preds, id2type, rare_classes)
    if return_embeddings:
        return metrics, preds, trues, embeddings, all_logits
    return metrics, preds, trues


def main(
    config_path: str,
    loss: str | None = None,
    loss_kwargs: dict | None = None,
    seed: int | None = None,
):
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
            f"dataset_name={cfg['dataset_name']!r} is not implemented yet; "
            "use 'ms', 'zheng68k', 'hpancreas', 'ms_oversampled', or 'zheng68k_oversampled'"
        )
    if cfg["loss"] not in LOSS_NAMES:
        raise NotImplementedError(
            f"loss={cfg['loss']!r} is not implemented; use one of {LOSS_NAMES}"
        )

    run_tag = f"{cfg['dataset_name']}_{cfg['loss']}_seed{cfg['seed']}"
    cfg["output"]["save_dir"] = str(Path(cfg["output"]["save_dir"]) / run_tag)
    if not cfg["wandb"].get("run_name"):
        cfg["wandb"]["run_name"] = run_tag

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    with open(Path(cfg["pretrained_model_dir"]) / "args.json", encoding="utf-8") as f:
        pretrained_model_args = json.load(f)

    env_info = base_env_info(BACKBONE, cfg) | {
        "scgpt_version": getattr(scg, "__version__", None),
        "requested_backend": cfg["model"]["fast_transformer_backend"],
        "flash_attn_backend": None,
        "grad_clip_max_norm": GRAD_CLIP_MAX_NORM,
        "pretrained_model_args": pretrained_model_args,
    }
    try:
        from scgpt.model.flash_attn_compat import flash_attn_backend as _fa_backend

        env_info["flash_attn_backend"] = _fa_backend
    except ImportError:
        pass

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
    gene_name_col = cfg["data"].get("gene_name_col")
    print(
        f"  train: {adata.shape}, test: {adata_test.shape}, {num_types} cell types total"
    )

    rare_classes = compute_rare_classes_for_run(cfg["data"], cfg["rare_class"], adata)
    print(
        f"rare classes (<{cfg['rare_class']['relative_threshold']:.0%} of training set, {len(rare_classes)} total):"
    )
    for c in rare_classes:
        print(f"  - {c}")

    print("building gene vocab from pretrained checkpoint...")
    vocab = build_vocab(cfg["pretrained_model_dir"])
    adata = filter_to_vocab(adata, vocab, gene_name_col)
    adata_test = filter_to_vocab(adata_test, vocab, gene_name_col)

    print("preprocessing (normalize -> bin)...")
    n_bins = cfg["model"]["n_bins"]
    preprocess(adata, n_bins)
    preprocess(adata_test, n_bins)

    gene_ids = np.array(vocab(get_gene_names(adata, gene_name_col)), dtype=int)
    gene_ids_test = np.array(
        vocab(get_gene_names(adata_test, gene_name_col)), dtype=int
    )

    val_fraction = cfg["data"]["val_fraction"]
    labels = adata.obs["celltype_id"].to_numpy()
    try:
        train_idx, val_idx = train_test_split(
            np.arange(adata.n_obs),
            test_size=val_fraction,
            random_state=cfg["seed"],
            stratify=labels,
        )
    except ValueError as e:
        print(f"  stratified split failed ({e}); falling back to a plain random split")
        train_idx, val_idx = train_test_split(
            np.arange(adata.n_obs), test_size=val_fraction, random_state=cfg["seed"]
        )

    train_labels = labels[train_idx]
    cls_num_list = [max(int((train_labels == i).sum()), 1) for i in range(num_types)]
    print(f"  train class counts: {cls_num_list}")

    max_seq_len = cfg["model"]["max_seq_len"]
    print(
        f"tokenizing splits (train {len(train_idx)}, val {len(val_idx)}, test {adata_test.n_obs}; max_seq_len={max_seq_len})..."
    )
    train_data = tokenize_split(adata[train_idx], vocab, gene_ids, max_seq_len)
    val_data = tokenize_split(adata[val_idx], vocab, gene_ids, max_seq_len)
    test_data = tokenize_split(adata_test, vocab, gene_ids_test, max_seq_len)
    print(
        f"  tokenized shapes -- train: {tuple(train_data['gene_ids'].shape)}, "
        f"val: {tuple(val_data['gene_ids'].shape)}, test: {tuple(test_data['gene_ids'].shape)} "
        f"(actual length is capped by each cell's nonzero gene count, may be well under max_seq_len)"
    )

    batch_size = cfg["train"]["batch_size"]
    eval_batch_size = cfg["train"]["eval_batch_size"]
    train_loader = DataLoader(
        CellDataset(train_data), batch_size=batch_size, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        CellDataset(val_data),
        batch_size=eval_batch_size,
        shuffle=False,
        pin_memory=True,
    )
    test_loader = DataLoader(
        CellDataset(test_data),
        batch_size=eval_batch_size,
        shuffle=False,
        pin_memory=True,
    )

    print("building model...")
    model = build_model(
        vocab, cfg["model"], cfg["pretrained_model_dir"], num_types, device
    )

    criterion, resolved_loss_kwargs = build_loss(
        cfg["loss"], cls_num_list, device, **cfg.get("loss_kwargs", {})
    )
    print(
        f"  loss: {cfg['loss']} ({type(criterion).__name__}), resolved_loss_kwargs: {resolved_loss_kwargs}"
    )
    env_info["resolved_loss_kwargs"] = resolved_loss_kwargs
    amp = cfg["train"]["amp"] and device.type == "cuda"

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"], eps=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 1, gamma=cfg["train"]["schedule_ratio"]
    )

    run = wandb.init(
        project=cfg["wandb"]["project"],
        entity=cfg["wandb"].get("entity"),
        name=cfg["wandb"].get("run_name"),
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

    best_val_macro_f1 = -1.0
    best_epoch = -1
    history = []
    train_start = time.time()
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        epoch_start = time.time()
        train_loss = run_train_epoch(
            model, train_loader, optimizer, criterion, vocab, device, amp
        )
        val_metrics, _, _ = evaluate(
            model, val_loader, vocab, device, amp, id2type, rare_classes
        )
        epoch_time = time.time() - epoch_start

        print(
            f"epoch {epoch:3d} | train_loss {train_loss:.4f} | "
            f"val_macro_f1 {val_metrics['macro_f1']:.4f} | "
            f"val_rare_recall {val_metrics['rare_class_recall']:.4f} | "
            f"val_balanced_acc {val_metrics['balanced_accuracy']:.4f}"
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
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            torch.save(model.state_dict(), save_dir / "best_model.pt")
            print(f"  new best val macro_f1={best_val_macro_f1:.4f}, checkpoint saved")

        scheduler.step()
    total_train_seconds = time.time() - train_start

    emissions_kg = None
    if emissions_tracker is not None:
        try:
            emissions_kg = emissions_tracker.stop()
        except Exception as e:
            print(
                f"  WARNING: codecarbon stop() failed, no emissions figure for this run ({e})"
            )

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
        model,
        test_loader,
        vocab,
        device,
        amp,
        id2type,
        rare_classes,
        return_embeddings=True,
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
    parser.add_argument("--config", default="configs/scgpt_ms.yaml")
    parser.add_argument(
        "--loss", default=None, choices=LOSS_NAMES, help="override config's loss field"
    )
    parser.add_argument(
        "--loss-kwargs",
        default=None,
        help="JSON string overriding config's loss_kwargs",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="override config's seed field"
    )
    args = parser.parse_args()
    loss_kwargs = json.loads(args.loss_kwargs) if args.loss_kwargs else None
    main(args.config, loss=args.loss, loss_kwargs=loss_kwargs, seed=args.seed)
