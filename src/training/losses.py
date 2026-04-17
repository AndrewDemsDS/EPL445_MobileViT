"""Loss functions for training."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_class_weights(
    dataloader: DataLoader,
    num_classes: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Compute inverse-frequency class weights from a DataLoader.

    Returns a ``(num_classes,)`` float tensor suitable for ``CrossEntropyLoss(weight=…)``.
    """
    counts = torch.zeros(num_classes)
    for _, labels in dataloader:
        for label in labels:
            counts[label.item()] += 1

    # Inverse frequency, normalised so weights sum to num_classes
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    return weights.to(device)


def build_criterion(
    cfg: dict[str, Any],
    dataloader: DataLoader | None = None,
    device: torch.device | str = "cpu",
) -> nn.Module:
    """Build a loss function from config.

    If ``cfg`` has ``use_class_weights: true`` and a *dataloader* is provided,
    the loss will use inverse-frequency weights.
    """
    num_classes = cfg.get("num_classes", 5)
    use_weights = cfg.get("use_class_weights", False)

    weight = None
    if use_weights and dataloader is not None:
        weight = compute_class_weights(dataloader, num_classes, device)

    return nn.CrossEntropyLoss(weight=weight)
