"""Smoke tests for the MobileViT model."""

import torch

from src.models.mobilevit_classifier import MobileViTClassifier


def test_model_forward_shape():
    """Model produces logits with correct shape (B, num_classes)."""
    num_classes = 5
    model = MobileViTClassifier(
        model_name="mobilevit_s",
        num_classes=num_classes,
        pretrained=False,  # skip download in CI
    )
    model.eval()

    batch = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        logits = model(batch)

    assert logits.shape == (2, num_classes), f"Expected (2, {num_classes}), got {logits.shape}"


def test_model_freeze_unfreeze():
    """Freeze/unfreeze backbone toggles requires_grad correctly."""
    model = MobileViTClassifier(
        model_name="mobilevit_s",
        num_classes=5,
        pretrained=False,
    )

    model.freeze_backbone()
    for p in model.backbone.parameters():
        assert not p.requires_grad, "Backbone param should be frozen"
    # Classifier head should still be trainable
    for p in model.classifier.parameters():
        assert p.requires_grad, "Classifier param should be trainable"

    model.unfreeze_backbone()
    for p in model.backbone.parameters():
        assert p.requires_grad, "Backbone param should be unfrozen"


def test_model_num_classes_property():
    """num_classes property returns the configured value."""
    model = MobileViTClassifier(num_classes=7, pretrained=False)
    assert model.num_classes == 7
