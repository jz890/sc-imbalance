"""Shared evaluation metrics and report writers."""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(
    trues: np.ndarray, preds: np.ndarray, id2type: dict, rare_classes: list
) -> dict:
    """The paper's four reported metrics (accuracy/precision/recall/macro_f1) plus this
    project's imbalance-focused additions (balanced_accuracy, rare_class_recall). Identical
    across all backbones -- do not fork this per-model."""
    rare_ids = [i for i, name in id2type.items() if name in rare_classes]
    rare_recall = (
        recall_score(trues, preds, labels=rare_ids, average="macro", zero_division=0)
        if rare_ids
        else float("nan")
    )
    return {
        "accuracy": accuracy_score(trues, preds),
        "precision_macro": precision_score(
            trues, preds, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(trues, preds, average="macro", zero_division=0),
        "macro_f1": f1_score(trues, preds, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(trues, preds),
        "rare_class_recall": rare_recall,
    }


def save_per_class_and_confusion(
    test_trues: np.ndarray,
    test_preds: np.ndarray,
    id2type: dict,
    rare_classes: list,
    save_dir: Path,
):
    """Writes per_class_report.csv and confusion_matrix.csv into save_dir; returns the
    per_class_report dict (also embedded in test_results.json)."""
    class_ids = list(range(len(id2type)))
    target_names = [id2type[i] for i in class_ids]

    per_class_report = classification_report(
        test_trues,
        test_preds,
        labels=class_ids,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    per_class_df = pd.DataFrame(per_class_report).transpose()
    per_class_df["is_rare"] = [name in rare_classes for name in per_class_df.index]
    per_class_df.to_csv(save_dir / "per_class_report.csv")
    print(f"  saved per-class report to {save_dir / 'per_class_report.csv'}")

    cm = confusion_matrix(test_trues, test_preds, labels=class_ids)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    cm_df.to_csv(save_dir / "confusion_matrix.csv")
    print(f"  saved confusion matrix to {save_dir / 'confusion_matrix.csv'}")

    return per_class_report
