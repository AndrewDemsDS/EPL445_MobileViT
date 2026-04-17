"""Single-image prediction helper."""

from __future__ import annotations

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from src.datasets.transforms import get_eval_transforms


class ImagePredictor:
    """Classify a single PIL image.

    Parameters
    ----------
    model : nn.Module
        Trained classifier.
    device : torch.device
        Device to run inference on.
    class_names : list[str]
        Ordered class names.
    image_size : int
        Expected input size for the model.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        class_names: list[str],
        image_size: int = 256,
    ) -> None:
        self.model = model.eval()
        self.device = device
        self.class_names = class_names
        self.transform = get_eval_transforms(image_size)

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        """Classify a single PIL Image.

        Returns
        -------
        dict with ``class_name``, ``class_idx``, ``confidence``, ``probabilities``.
        """
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        idx = probs.argmax().item()

        return {
            "class_name": self.class_names[idx],
            "class_idx": int(idx),
            "confidence": float(probs[idx]),
            "probabilities": {
                name: float(probs[i]) for i, name in enumerate(self.class_names)
            },
        }
