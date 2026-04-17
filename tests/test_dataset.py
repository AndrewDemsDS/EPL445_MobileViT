"""Tests for image transforms."""

import torch
from PIL import Image

from src.datasets.transforms import get_eval_transforms, get_train_transforms


def _make_dummy_image(width: int = 300, height: int = 200) -> Image.Image:
    """Create a random RGB PIL image."""
    import numpy as np
    arr = np.random.randint(0, 256, (height, width, 3), dtype="uint8")
    return Image.fromarray(arr)


def test_train_transform_output_shape():
    """Train transforms produce a (3, H, W) float tensor."""
    tfm = get_train_transforms(image_size=256)
    img = _make_dummy_image()
    tensor = tfm(img)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 256, 256), f"Expected (3, 256, 256), got {tensor.shape}"
    assert tensor.dtype == torch.float32


def test_eval_transform_output_shape():
    """Eval transforms produce a (3, H, W) float tensor."""
    tfm = get_eval_transforms(image_size=224)
    img = _make_dummy_image()
    tensor = tfm(img)

    assert tensor.shape == (3, 224, 224), f"Expected (3, 224, 224), got {tensor.shape}"


def test_transforms_are_deterministic_for_eval():
    """Eval transforms should give the same result for the same input."""
    tfm = get_eval_transforms(256)
    img = _make_dummy_image()
    t1 = tfm(img)
    t2 = tfm(img)
    assert torch.allclose(t1, t2), "Eval transforms should be deterministic"
