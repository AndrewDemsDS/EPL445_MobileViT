"""I/O helpers: YAML loading, directory creation, checkpoint save/load."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml


# ── Config loading ───────────────────────────────────────────────


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return a dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_configs(default_path: str | Path, override_path: str | Path) -> dict[str, Any]:
    """Merge a default config with an override config (override wins)."""
    base = load_yaml(default_path)
    override = load_yaml(override_path)
    base.update(override)
    return base


# ── Directory helpers ────────────────────────────────────────────


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it does not exist, then return it."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── Checkpoint helpers ───────────────────────────────────────────


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    metrics: dict[str, float],
    path: str | Path,
) -> None:
    """Save a training checkpoint."""
    ensure_dir(Path(path).parent)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint into *model* (and optionally *optimizer*).

    Returns the full checkpoint dict so callers can inspect epoch/metrics.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


# ── JSON helpers ─────────────────────────────────────────────────


def save_json(data: Any, path: str | Path) -> None:
    """Write *data* as formatted JSON."""
    ensure_dir(Path(path).parent)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str | Path) -> Any:
    """Read a JSON file."""
    with open(path, "r") as f:
        return json.load(f)
