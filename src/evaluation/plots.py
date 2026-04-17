"""Generate evaluation plots for the presentation.

Produces:
  - Confusion matrix heatmap
  - ROC curves (per class + macro)
  - Training loss / accuracy curves
  - Per-class F1 bar chart
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Use a clean, publication-ready style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("talk")


def plot_confusion_matrix(
    cm: list[list[int]] | np.ndarray,
    class_names: list[str],
    save_path: str | Path,
    title: str = "Confusion Matrix",
) -> None:
    """Save a confusion-matrix heatmap."""
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_roc_curves(
    roc_data: dict[str, dict],
    save_path: str | Path,
    title: str = "ROC Curves (One-vs-Rest)",
) -> None:
    """Save ROC curves for each class + macro average."""
    fig, ax = plt.subplots(figsize=(8, 7))
    aucs = []
    for cls_name, data in roc_data.items():
        auc = data["auc"]
        aucs.append(auc)
        ax.plot(data["fpr"], data["tpr"], label=f"{cls_name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    macro_auc = np.mean(aucs)
    ax.set_title(f"{title}\nMacro AUC = {macro_auc:.3f}")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_training_curves(
    history_csv: str | Path,
    save_path: str | Path,
) -> None:
    """Plot loss and accuracy curves from a training history CSV."""
    df = pd.read_csv(history_csv)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(df["epoch"], df["train_loss"], "o-", label="Train loss")
    ax1.plot(df["epoch"], df["val_loss"], "s-", label="Val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()

    # Accuracy
    ax2.plot(df["epoch"], df["train_accuracy"], "o-", label="Train acc")
    ax2.plot(df["epoch"], df["val_accuracy"], "s-", label="Val acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Curves")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_per_class_f1(
    per_class: dict[str, dict[str, float]],
    save_path: str | Path,
    title: str = "Per-Class F1 Score",
) -> None:
    """Save a horizontal bar chart of per-class F1 scores."""
    names = list(per_class.keys())
    f1_scores = [per_class[n]["f1"] for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("viridis", len(names))
    bars = ax.barh(names, f1_scores, color=colors)
    ax.set_xlabel("F1 Score")
    ax.set_title(title)
    ax.set_xlim(0, 1.05)

    # Add value labels
    for bar, val in zip(bars, f1_scores):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
