#!/usr/bin/env python3
"""Evaluate a trained model on the test set and generate metrics + plots.

Usage
-----
    python -m src.evaluation.evaluate --config configs/eval.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.traffic_dataset_builder import build_dataloaders
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.plots import (
    plot_confusion_matrix,
    plot_per_class_f1,
    plot_roc_curves,
    plot_training_curves,
)
from src.models.model_factory import build_model
from src.utils.device import get_device
from src.utils.io import ensure_dir, load_checkpoint, merge_configs, save_json
from src.utils.logging import setup_logger
from src.utils.seed import set_seed


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the model on *loader* and collect all predictions.

    Returns
    -------
    y_true, y_pred, y_prob : arrays of shape (N,), (N,), (N, C)
    """
    model.eval()
    all_true, all_pred, all_prob = [], [], []

    for images, labels in tqdm(loader, desc="Predicting", leave=False):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)

        all_true.append(labels.numpy())
        all_pred.append(logits.argmax(dim=1).cpu().numpy())
        all_prob.append(probs.cpu().numpy())

    return (
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_prob),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate traffic classifier")
    parser.add_argument(
        "--config", type=str, default="configs/eval.yaml",
        help="Path to eval config (merged with configs/default.yaml)",
    )
    args = parser.parse_args()

    # ── Config ───────────────────────────────────────────────────
    cfg = merge_configs("configs/default.yaml", args.config)
    logger = setup_logger()
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "auto"))

    # ── Data ─────────────────────────────────────────────────────
    loaders = build_dataloaders(cfg)
    if "test" not in loaders:
        logger.error("Test split CSV missing. Run scripts/prepare_dataset.py first.")
        return
    logger.info("Test batches: %d", len(loaders["test"]))

    # ── Model ────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    ckpt_path = cfg.get("checkpoint_path", "outputs/models/best_model.pth")
    load_checkpoint(ckpt_path, model, device=device)
    logger.info("Loaded checkpoint from %s", ckpt_path)

    # ── Predict ──────────────────────────────────────────────────
    y_true, y_pred, y_prob = collect_predictions(model, loaders["test"], device)
    class_names = cfg.get("class_names", [])

    # ── Metrics ──────────────────────────────────────────────────
    metrics = compute_all_metrics(y_true, y_pred, y_prob, class_names)
    logger.info("\n%s", metrics["classification_report"])

    # Save metrics JSON
    metrics_dir = ensure_dir(cfg.get("metrics_dir", "outputs/metrics"))
    # Remove non-serializable items for JSON
    json_metrics = {k: v for k, v in metrics.items() if k != "classification_report"}
    save_json(json_metrics, metrics_dir / "test_metrics.json")
    logger.info("Metrics saved to %s", metrics_dir / "test_metrics.json")

    # ── Plots ────────────────────────────────────────────────────
    figures_dir = ensure_dir(cfg.get("figures_dir", "outputs/figures"))

    if cfg.get("generate_confusion_matrix", True):
        plot_confusion_matrix(
            metrics["confusion_matrix"], class_names,
            figures_dir / "confusion_matrix.png",
        )
        logger.info("Confusion matrix saved")

    if cfg.get("generate_roc_curves", True) and "roc" in metrics:
        plot_roc_curves(metrics["roc"], figures_dir / "roc_curves.png")
        logger.info("ROC curves saved")

    if cfg.get("generate_f1_chart", True):
        plot_per_class_f1(metrics["per_class"], figures_dir / "per_class_f1.png")
        logger.info("Per-class F1 chart saved")

    # Training curves (if history file exists)
    history_path = Path(cfg.get("log_dir", cfg.get("metrics_dir", "outputs/metrics"))) / "training_history.csv"
    if history_path.exists():
        plot_training_curves(history_path, figures_dir / "training_curves.png")
        logger.info("Training curves saved")

    logger.info("Evaluation complete!")
    logger.info("  Accuracy:  %.4f", metrics["accuracy"])
    logger.info("  Macro F1:  %.4f", metrics["macro_f1"])
    if "auc_macro" in metrics:
        logger.info("  Macro AUC: %.4f", metrics["auc_macro"])


if __name__ == "__main__":
    main()
