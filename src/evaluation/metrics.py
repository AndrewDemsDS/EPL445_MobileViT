"""Metric computation for classification evaluation.

Covers the metrics required by the EPL445 course brief:
    Accuracy, Sensitivity (Recall), Specificity, F-Score, ROC/AUC.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    class_names: list[str] | None = None,
) -> dict:
    """Compute a comprehensive set of classification metrics.

    Parameters
    ----------
    y_true : array of int
        Ground-truth label indices, shape ``(N,)``.
    y_pred : array of int
        Predicted label indices, shape ``(N,)``.
    y_prob : array of float, optional
        Softmax probabilities, shape ``(N, C)``.  Needed for ROC / AUC.
    class_names : list[str], optional
        Ordered class names for the report.

    Returns
    -------
    dict with keys:
        accuracy, macro_f1, macro_precision, macro_recall,
        per_class (dict), confusion_matrix, classification_report (str),
        roc (dict, if y_prob provided), auc_macro.
    """
    num_classes = len(class_names) if class_names else len(set(y_true))

    result: dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    # Per-class metrics (sensitivity = recall, specificity computed from CM)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    result["confusion_matrix"] = cm.tolist()

    per_class: dict[str, dict[str, float]] = {}
    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        cls_name = class_names[i] if class_names else str(i)
        per_class[cls_name] = {
            "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
            "recall_sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            "f1": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0,
            "support": int(tp + fn),
        }
    result["per_class"] = per_class

    # Text classification report
    result["classification_report"] = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0,
    )

    # ROC / AUC (one-vs-rest)
    if y_prob is not None and num_classes > 1:
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
        # Handle binary edge case
        if num_classes == 2:
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

        roc_data: dict[str, dict] = {}
        auc_scores = []
        for i in range(num_classes):
            fpr, tpr, thresholds = roc_curve(y_true_bin[:, i], y_prob[:, i])
            cls_name = class_names[i] if class_names else str(i)
            try:
                auc_val = float(roc_auc_score(y_true_bin[:, i], y_prob[:, i]))
            except ValueError:
                auc_val = 0.0
            roc_data[cls_name] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": auc_val,
            }
            auc_scores.append(auc_val)

        result["roc"] = roc_data
        result["auc_macro"] = float(np.mean(auc_scores))

    return result
