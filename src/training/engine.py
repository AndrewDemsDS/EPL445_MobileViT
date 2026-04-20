"""Training and validation engine — epoch-level functions."""

from __future__ import annotations

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int = 0,
    log_every: int = 20,
) -> dict[str, float]:
    """Run one training epoch.

    Returns
    -------
    dict with ``loss`` (average) and ``accuracy``.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Train epoch {epoch}", leave=False)
    for step, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        if (step + 1) % log_every == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / total:.4f}")

    return {
        "loss": running_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Run validation / test evaluation.

    Returns
    -------
    dict with ``loss`` and ``accuracy``.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for images, labels in tqdm(loader, desc="Validating", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    return {
        "loss": running_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "macro_f1": macro_f1,
    }
