"""Shared data loading, evaluation, and result utilities."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import yaml
from scipy.special import softmax


from utils.metrics import compute_metrics, save_per_class_and_confusion


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_git_commit_hash() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def load_adatas(data_cfg: dict):
    """Load both splits and harmonize cell-type categories and identifiers."""
    if data_cfg.get("celltype_col") is None:
        raise ValueError("data.celltype_col must be set in the config")

    adata = sc.read_h5ad(data_cfg["train_h5ad"])
    adata_test = sc.read_h5ad(data_cfg["test_h5ad"])

    col = data_cfg["celltype_col"]
    adata.obs["celltype"] = adata.obs[col].astype("category")
    adata_test.obs["celltype"] = adata_test.obs[col].astype("category")

    all_types = pd.Categorical(
        pd.concat([adata.obs["celltype"], adata_test.obs["celltype"]])
    ).categories
    adata.obs["celltype"] = adata.obs["celltype"].cat.set_categories(all_types)
    adata_test.obs["celltype"] = adata_test.obs["celltype"].cat.set_categories(
        all_types
    )
    adata.obs["celltype_id"] = adata.obs["celltype"].cat.codes.values
    adata_test.obs["celltype_id"] = adata_test.obs["celltype"].cat.codes.values
    id2type = dict(enumerate(all_types))

    return adata, adata_test, id2type


def get_gene_names(adata, gene_name_col: str | None) -> list:
    """Return dataset-specific gene symbols."""
    if gene_name_col:
        return adata.var[gene_name_col].tolist()
    return adata.var.index.tolist()


def compute_rare_classes(celltype_series: pd.Series, threshold: float) -> list:
    freq = celltype_series.value_counts(normalize=True)
    return sorted(freq[freq < threshold].index.tolist())


def compute_rare_classes_for_run(data_cfg: dict, rare_class_cfg: dict, adata) -> list:
    """Determine rare classes from the unmodified training distribution."""
    reference_h5ad = rare_class_cfg.get("reference_h5ad")
    if reference_h5ad:
        ref_series = sc.read_h5ad(reference_h5ad).obs[data_cfg["celltype_col"]]
    else:
        ref_series = adata.obs["celltype"]
    return compute_rare_classes(ref_series, rare_class_cfg["relative_threshold"])


def save_embeddings(
    embeddings: np.ndarray,
    preds: np.ndarray,
    trues: np.ndarray,
    logits: np.ndarray,
    id2type: dict,
    cfg: dict,
    run_id: str,
    backbone: str,
) -> Path:
    """Save test embeddings, predictions, labels, logits, and probabilities."""
    out_dir = Path("embeddings")
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = (
        f"{backbone}_{cfg['dataset_name']}_{cfg['loss']}_seed{cfg['seed']}_{run_id}.npz"
    )
    path = out_dir / fname
    probabilities = softmax(logits, axis=1)
    np.savez(
        path,
        embeddings=embeddings,
        labels=trues,
        preds=preds,
        probabilities=probabilities,
        trues=trues,
        logits=logits,
        id2type=np.array([id2type[i] for i in range(len(id2type))], dtype=object),
    )
    print(
        f"  saved test-set cell embeddings+preds+logits+probabilities to {path} (shape {embeddings.shape})"
    )
    return path


def base_env_info(backbone: str, cfg: dict) -> dict:
    """Return common environment metadata for saved run results."""
    import torch

    return {
        "backbone": backbone,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "git_commit_hash": get_git_commit_hash(),
    }


def write_test_results_json(
    save_dir: Path,
    test_metrics: dict,
    rare_classes: list,
    id2type: dict,
    per_class_report: dict,
    cls_num_list: list,
    n_train: int,
    n_val: int,
    n_test: int,
    best_epoch: int,
    best_val_macro_f1: float,
    total_train_seconds: float,
    peak_gpu_memory_mb: dict | None,
    emissions_kg_co2eq: float | None,
    training_history: list,
    env_info: dict,
    resolved_config: dict,
):
    with open(save_dir / "test_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "test_metrics": test_metrics,
                "rare_classes": rare_classes,
                "id2type": {int(k): v for k, v in id2type.items()},
                "per_class_report": per_class_report,
                "cls_num_list": cls_num_list,
                "n_train": n_train,
                "n_val": n_val,
                "n_test": n_test,
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_val_macro_f1,
                "total_train_seconds": total_train_seconds,
                "peak_gpu_memory_mb": peak_gpu_memory_mb,
                "emissions_kg_co2eq": emissions_kg_co2eq,
                "training_history": training_history,
                "env_info": env_info,
                "resolved_config": resolved_config,
            },
            f,
            indent=2,
        )
