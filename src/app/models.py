"""Process-wide singletons for the MobileViT classifier and YOLO detector.

The FastAPI worker calls these once per job, so we cache the loaded
model on the first call. Saves ~3 s for the MobileViT load and ~12 s
for the YOLO compile on every subsequent job.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import numpy as np
import torch

from src.models.model_factory import build_model
from src.utils.device import get_device
from src.utils.io import load_checkpoint, merge_configs


@lru_cache(maxsize=1)
def get_classifier() -> Tuple[torch.nn.Module, torch.device]:
    """Return the fine-tuned MobileViT classifier and its device.

    Reads the same configs/default.yaml + configs/demo.yaml the job
    config does, so checkpoint_path and image_size match what the
    worker would have produced.
    """
    cfg = merge_configs("configs/default.yaml", "configs/demo.yaml")
    device = get_device(cfg.get("device", "auto"))
    model = build_model(cfg).to(device)
    ckpt_path = cfg.get("checkpoint_path", "outputs/models/best_model.pth")
    load_checkpoint(ckpt_path, model, device=device)
    model.eval()
    return model, device


@lru_cache(maxsize=1)
def get_yolo(weights: str = "yolov8n.pt"):
    """Return a warmed YOLOv8 model on the same device as the classifier.

    Runs one dummy forward pass to trigger the lazy weight download
    (if needed) and CUDA/ROCm kernel compilation, so the first real
    job doesn't pay the ~12 s spike.
    """
    from ultralytics import YOLO  # local import keeps cold start fast for sliding-only jobs

    _, device = get_classifier()
    model = YOLO(weights)
    model.to(str(device))
    dummy = np.zeros((360, 640, 3), dtype=np.uint8)
    model(dummy, verbose=False)
    return model


def warm_all() -> None:
    """Load and warm both models. Called from the FastAPI lifespan handler."""
    get_classifier()
    get_yolo()
