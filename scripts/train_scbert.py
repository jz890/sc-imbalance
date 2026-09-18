"""Fine-tune scBERT with a selected long-tail loss."""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import wandb
from dotenv import load_dotenv
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset

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

SCBERT_REPO_PATH = os.environ.get(
    "SCBERT_REPO_PATH", str(Path(__file__).resolve().parents[1] / "scBERT")
)
if SCBERT_REPO_PATH not in sys.path:
    sys.path.insert(0, SCBERT_REPO_PATH)

try:
    from performer_pytorch import PerformerLM
except ImportError as e:
    raise ImportError(
        f"could not import performer_pytorch from {SCBERT_REPO_PATH!r}. Clone the official repo "
        f"first: git clone https://github.com/TencentAILabHealthcare/scBERT.git {SCBERT_REPO_PATH} "
        f"-- or set SCBERT_REPO_PATH to wherever you cloned it."
    ) from e

BACKBONE = "scbert"
GRAD_CLIP_MAX_NORM = 1e6
BIN_VOCAB_SIZE = 7

DEFAULT_GRADIENT_ACCUMULATION = 60


PATIENCE = 10


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """Linear warmup followed by cosine decay, matching scBERT fine-tuning."""

    def __init__(
        self,
        optimizer,
        first_cycle_steps,
        cycle_mult=1.0,
        max_lr=0.1,
        min_lr=0.001,
        warmup_steps=0,
        gamma=1.0,
        last_epoch=-1,
    ):
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch
        super().__init__(optimizer, last_epoch)
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        if self.step_in_cycle < self.warmup_steps:
            return [
                (self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps
                + base_lr
                for base_lr in self.base_lrs
            ]
        progress = (self.step_in_cycle - self.warmup_steps) / (
            self.cur_cycle_steps - self.warmup_steps
        )
        return [
            base_lr + (self.max_lr - base_lr) * (1 + math.cos(math.pi * progress)) / 2
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = (
                    int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult)
                    + self.warmup_steps
                )
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(
                        math.log(
                            (
                                epoch / self.first_cycle_steps * (self.cycle_mult - 1)
                                + 1
                            ),
                            self.cycle_mult,
                        )
                    )
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps
                        * (self.cycle_mult**n - 1)
                        / (self.cycle_mult - 1)
                    )
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult**n
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


class ClassificationHead(nn.Module):
    """Classification head used for scBERT fine-tuning."""

    def __init__(self, dropout: float, seq_len: int, h_dim: int, out_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, (1, 200))
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(in_features=seq_len, out_features=512, bias=True)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(in_features=512, out_features=h_dim, bias=True)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(in_features=h_dim, out_features=out_dim, bias=True)

    def forward(self, x):
        x = x[:, None, :, :]
        x = self.conv1(x)
        x = self.act(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        h = self.fc2(x)
        x = self.act2(h)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x, h


class SCDataset(Dataset):
    """Convert binned expression into fixed-length token sequences."""

    def __init__(self, X, labels: np.ndarray):
        self.X = X
        self.labels = labels

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        row = self.X[idx]
        full_seq = row.toarray()[0] if issparse(row) else np.asarray(row).ravel()
        full_seq = np.clip(full_seq, None, BIN_VOCAB_SIZE - 2)
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0])))
        return full_seq, int(self.labels[idx])


class Gene2VecPositionalEmbedding(nn.Module):
    """Gene2Vec positional embedding aligned to the active gene panel."""

    def __init__(self, dim: int, max_seq_len: int, gene2vec_path: str):
        super().__init__()
        gene2vec_weight = np.load(gene2vec_path)
        assert gene2vec_weight.shape == (max_seq_len - 1, dim), (
            f"{gene2vec_path} shape {gene2vec_weight.shape} != expected "
            f"({max_seq_len - 1}, {dim}) -- this must be a per-dataset subset extracted by "
            f"extract_gene2vec_subset.py from THIS dataset's own h5ad, matching its n_genes."
        )
        gene2vec_weight = np.concatenate(
            (gene2vec_weight, np.zeros((1, gene2vec_weight.shape[1]))), axis=0
        )
        gene2vec_weight = torch.from_numpy(gene2vec_weight).float()
        self.emb = nn.Embedding.from_pretrained(gene2vec_weight)

    def forward(self, x):
        positions = torch.arange(x.shape[1], device=x.device)
        return self.emb(positions)


def build_model(
    pretrained_ckpt_path: str,
    seq_len: int,
    num_types: int,
    dropout: float,
    device,
    gene2vec_path: str,
):
    model = PerformerLM(
        num_tokens=BIN_VOCAB_SIZE,
        dim=200,
        depth=6,
        max_seq_len=seq_len,
        heads=10,
        local_attn_heads=0,
        g2v_position_emb=False,
    )
    ckpt = torch.load(pretrained_ckpt_path, map_location=device)

    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(
        f"  checkpoint load: {len(missing)} missing keys, {len(unexpected)} unexpected keys"
    )
    if missing:
        print(f"    missing (first 5): {missing[:5]}")
    if unexpected:
        print(f"    unexpected (first 5): {unexpected[:5]}")

    model.pos_emb = Gene2VecPositionalEmbedding(
        dim=200, max_seq_len=seq_len, gene2vec_path=gene2vec_path
    )

    for param in model.parameters():
        param.requires_grad = False
    for param in model.norm.parameters():
        param.requires_grad = True
    for param in model.performer.net.layers[-2].parameters():
        param.requires_grad = True

    model.to_out = ClassificationHead(
        dropout=dropout, seq_len=seq_len, h_dim=128, out_dim=num_types
    )
    return model.to(device)


def run_train_epoch(
    model, loader, optimizer, criterion, device, amp, accum_steps, log_every=20
):
    """Apply optimizer updates every ``accum_steps`` batches."""
    model.train()
    total_loss, total_n = 0.0, 0
    num_batches = len(loader)
    epoch_start = time.time()
    optimizer.zero_grad()
    for i, (data, labels) in enumerate(loader, start=1):
        data, labels = data.to(device), labels.to(device)
        with torch.autocast(device_type="cuda", enabled=amp, dtype=torch.bfloat16):
            logits, _ = model(data)
            loss = criterion(logits, labels)
        loss.backward()
        if i % accum_steps == 0 or i == num_batches:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
            optimizer.step()
            optimizer.zero_grad()

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
    for i, (data, labels) in enumerate(loader, start=1):
        data, labels = data.to(device), labels.to(device)
        with torch.autocast(device_type="cuda", enabled=amp, dtype=torch.bfloat16):
            logits, h = model(data)
        preds.append(logits.argmax(1).cpu().numpy())
        trues.append(labels.cpu().numpy())
        if return_embeddings:
            embs.append(h.float().cpu().numpy())
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

    gene2vec_path = cfg["model"]["gene2vec_path"]

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
        "grad_clip_max_norm": GRAD_CLIP_MAX_NORM,
        "grad_accumulation": cfg["train"].get(
            "grad_accumulation", DEFAULT_GRADIENT_ACCUMULATION
        ),
        "early_stopping_patience": PATIENCE,
        "gene2vec_path": gene2vec_path,
        "scbert_repo_path": SCBERT_REPO_PATH,
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

    seq_len = adata.n_vars + 1

    labels = adata.obs["celltype_id"].to_numpy()
    val_fraction = cfg["data"]["val_fraction"]
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

    test_labels = adata_test.obs["celltype_id"].to_numpy()
    train_dataset = SCDataset(adata.X[train_idx], train_labels)
    val_dataset = SCDataset(adata.X[val_idx], labels[val_idx])
    test_dataset = SCDataset(adata_test.X, test_labels)

    batch_size = cfg["train"]["batch_size"]
    eval_batch_size = cfg["train"]["eval_batch_size"]
    grad_accumulation = cfg["train"].get(
        "grad_accumulation", DEFAULT_GRADIENT_ACCUMULATION
    )
    print(
        f"  batch_size={batch_size}, grad_accumulation={grad_accumulation} "
        f"(effective batch={batch_size * grad_accumulation})"
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=eval_batch_size, shuffle=False, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=eval_batch_size, shuffle=False, pin_memory=True
    )

    print("building model...")
    model = build_model(
        cfg["pretrained_model_dir"],
        seq_len,
        num_types,
        cfg["model"]["dropout"],
        device,
        gene2vec_path,
    )

    criterion, resolved_loss_kwargs = build_loss(
        cfg["loss"], cls_num_list, device, **cfg.get("loss_kwargs", {})
    )
    print(
        f"  loss: {cfg['loss']} ({type(criterion).__name__}), resolved_loss_kwargs: {resolved_loss_kwargs}"
    )
    env_info["resolved_loss_kwargs"] = resolved_loss_kwargs
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=cfg["train"]["lr"])
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=15,
        cycle_mult=2,
        max_lr=cfg["train"]["lr"],
        min_lr=1e-6,
        warmup_steps=5,
        gamma=0.9,
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
    trigger_times = 0
    train_start = time.time()
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        epoch_start = time.time()
        train_loss = run_train_epoch(
            model, train_loader, optimizer, criterion, device, amp, grad_accumulation
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
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times > PATIENCE:
                print(
                    f"  early stopping triggered at epoch {epoch} (patience={PATIENCE})"
                )
                break

        scheduler.step()
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
