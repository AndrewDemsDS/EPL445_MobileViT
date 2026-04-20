#!/usr/bin/env python3
"""Full training script for the traffic MobileViT classifier.

Usage
-----
    python -m src.training.train --config configs/train.yaml

The script:
1. Loads and merges default + training config.
2. Sets seed, builds model / dataloaders / optimizer / scheduler.
3. Runs the train+val loop with staged fine-tuning.
4. Saves the best checkpoint and per-epoch metrics CSV.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from src.datasets.traffic_dataset_builder import build_dataloaders
from src.models.model_factory import build_model
from src.training.engine import train_one_epoch, validate
from src.training.losses import build_criterion
from src.utils.device import get_device
from src.utils.io import ensure_dir, merge_configs, save_checkpoint
from src.utils.logging import setup_logger
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MobileViT traffic classifier")
    parser.add_argument(
        "--config", type=str, default="configs/train.yaml",
        help="Path to training config (merged with configs/default.yaml)",
    )
    args = parser.parse_args()

    # ── Config ───────────────────────────────────────────────────
    cfg = merge_configs("configs/default.yaml", args.config)
    logger = setup_logger(log_file=str(Path(cfg.get("log_dir", "outputs/metrics")) / "train.log"))
    logger.info("Config: %s", cfg)

    # ── Reproducibility ──────────────────────────────────────────
    set_seed(cfg.get("seed", 42))

    # ── Device ───────────────────────────────────────────────────
    device = get_device(cfg.get("device", "auto"))
    logger.info("Device: %s", device)

    # ── Data ─────────────────────────────────────────────────────
    loaders = build_dataloaders(cfg)
    if "train" not in loaders or "val" not in loaders:
        logger.error("Train or val split CSV missing. Run scripts/prepare_dataset.py first.")
        return
    logger.info("Train batches: %d, Val batches: %d", len(loaders["train"]), len(loaders["val"]))

    # ── Model ────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    freeze_epochs = cfg.get("freeze_backbone_epochs", 3)
    if freeze_epochs > 0:
        model.freeze_backbone()
        logger.info("Backbone frozen for first %d epoch(s)", freeze_epochs)

    # ── Optimizer & Scheduler ────────────────────────────────────
    lr = cfg.get("learning_rate", 3e-4)
    wd = cfg.get("weight_decay", 1e-4)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd
    )

    epochs = cfg.get("epochs", 15)
    scheduler_name = cfg.get("scheduler", "cosine")
    scheduler = None
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Loss ─────────────────────────────────────────────────────
    criterion = build_criterion(cfg, loaders["train"], device)

    # ── Training loop ────────────────────────────────────────────
    ckpt_dir = ensure_dir(cfg.get("checkpoint_dir", "outputs/models"))
    log_dir = ensure_dir(cfg.get("log_dir", "outputs/metrics"))
    log_every = cfg.get("log_every_n_steps", 20)
    patience = cfg.get("early_stopping_patience", 5)

    monitor = cfg.get("monitor_metric", "accuracy")  # 'accuracy' or 'macro_f1'
    best_val_score = 0.0
    epochs_no_improve = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        # Staged fine-tuning: unfreeze backbone after N epochs
        if epoch == freeze_epochs + 1 and freeze_epochs > 0:
            model.unfreeze_backbone()
            # Reset optimizer for full param set
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr * 0.1, weight_decay=wd)
            if scheduler_name == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs - epoch + 1
                )
            logger.info("Backbone unfrozen at epoch %d with reduced LR", epoch)

        train_metrics = train_one_epoch(
            model, loaders["train"], optimizer, criterion, device,
            epoch=epoch, log_every=log_every,
        )
        val_metrics = validate(model, loaders["val"], criterion, device)

        if scheduler:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch %d/%d — train_loss=%.4f train_acc=%.4f  "
            "val_loss=%.4f val_acc=%.4f val_f1=%.4f  lr=%.6f",
            epoch, epochs,
            train_metrics["loss"], train_metrics["accuracy"],
            val_metrics["loss"], val_metrics["accuracy"], val_metrics["macro_f1"],
            current_lr,
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "lr": current_lr,
        })

        # Checkpointing — select by configured monitor metric
        current_score = val_metrics[monitor]
        if current_score > best_val_score:
            best_val_score = current_score
            epochs_no_improve = 0
            save_checkpoint(
                model, optimizer, epoch,
                {**val_metrics, "train_loss": train_metrics["loss"]},
                ckpt_dir / "best_model.pth",
            )
            logger.info("  ✓ New best val_%s=%.4f — checkpoint saved", monitor, best_val_score)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info("  Early stopping triggered at epoch %d", epoch)
                break

    # ── Save training history ────────────────────────────────────
    history_path = log_dir / "training_history.csv"
    with open(history_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    logger.info("Training history saved to %s", history_path)
    logger.info("Best validation %s: %.4f", monitor, best_val_score)


if __name__ == "__main__":
    main()
