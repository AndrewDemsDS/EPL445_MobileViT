"""Model factory — single entry point for building models from config."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from src.models.mobilevit_classifier import MobileViTClassifier


def build_model(cfg: dict[str, Any]) -> nn.Module:
    """Construct and return a model based on the configuration dict.

    Parameters
    ----------
    cfg : dict
        Must contain at least ``model_name``, ``num_classes``, and ``pretrained``.
    """
    model_name = cfg.get("model_name", "mobilevit_s")
    num_classes = cfg.get("num_classes", 5)
    pretrained = cfg.get("pretrained", True)

    model = MobileViTClassifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
    )
    return model
