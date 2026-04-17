"""MobileViT classifier wrapper using the ``timm`` library.

Loads a pre-trained MobileViT backbone and replaces the classification head
with a new linear layer for the target number of classes.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn


class MobileViTClassifier(nn.Module):
    """Thin wrapper around a ``timm`` MobileViT model.

    Parameters
    ----------
    model_name : str
        ``timm`` model identifier, e.g. ``"mobilevit_s"``, ``"mobilevit_xs"``.
    num_classes : int
        Number of output classes.
    pretrained : bool
        Whether to load ImageNet-pretrained weights.
    drop_rate : float
        Dropout rate before the classifier head.
    """

    def __init__(
        self,
        model_name: str = "mobilevit_s",
        num_classes: int = 5,
        pretrained: bool = True,
        drop_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,          # remove original head
            drop_rate=drop_rate,
        )
        # Determine feature dimension from the backbone
        with torch.no_grad():
            dummy = torch.randn(1, 3, 256, 256)
            feat_dim = self.backbone(dummy).shape[-1]

        self.classifier = nn.Linear(feat_dim, num_classes)
        self._num_classes = num_classes

    # ── Forward pass ─────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits of shape ``(B, num_classes)``."""
        features = self.backbone(x)           # (B, feat_dim)
        return self.classifier(features)      # (B, num_classes)

    # ── Fine-tuning helpers ──────────────────────────────────────

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (only train the classifier head)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    @property
    def num_classes(self) -> int:
        return self._num_classes
