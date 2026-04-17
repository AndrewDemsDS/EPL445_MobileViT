"""PyTorch Dataset for frame-level traffic classification.

Reads a CSV with columns ``image_path,label`` and returns
``(image_tensor, label_index)`` pairs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class TrafficFrameDataset(Dataset):
    """Dataset that loads images listed in a CSV file.

    Parameters
    ----------
    csv_path : str | Path
        Path to a CSV file with at least ``image_path`` and ``label`` columns.
    transform : transforms.Compose, optional
        Torchvision transforms to apply to each image.
    class_names : list[str], optional
        Ordered class names.  If *None*, labels are inferred from the CSV.
    data_root : str | Path, optional
        If image paths in the CSV are relative, they are resolved against this root.
    """

    def __init__(
        self,
        csv_path: str | Path,
        transform: transforms.Compose | None = None,
        class_names: list[str] | None = None,
        data_root: str | Path | None = None,
    ) -> None:
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.data_root = Path(data_root) if data_root else None

        # Build label ↔ index mapping
        if class_names is not None:
            self.class_names = list(class_names)
        else:
            self.class_names = sorted(self.df["label"].unique().tolist())
        self.label_to_idx = {name: i for i, name in enumerate(self.class_names)}

    # ── Dataset interface ────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = Path(row["image_path"])
        if self.data_root and not img_path.is_absolute():
            img_path = self.data_root / img_path

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = self.label_to_idx[row["label"]]
        return image, label

    # ── Helpers ──────────────────────────────────────────────────

    @property
    def num_classes(self) -> int:
        return len(self.class_names)
