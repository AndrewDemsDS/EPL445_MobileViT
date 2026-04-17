"""Build train / val / test DataLoaders from config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from src.datasets.transforms import get_eval_transforms, get_train_transforms
from src.datasets.video_frames_dataset import TrafficFrameDataset


def build_dataloaders(
    cfg: dict[str, Any],
) -> dict[str, DataLoader]:
    """Return a dict with ``"train"``, ``"val"``, and ``"test"`` DataLoaders.

    Parameters
    ----------
    cfg : dict
        Merged configuration dictionary (default + train/eval).
    """
    data_root = Path(cfg["data_root"])
    splits_dir = data_root / "splits"
    image_size = cfg.get("image_size", 256)
    batch_size = cfg.get("batch_size", 32)
    num_workers = cfg.get("num_workers", 4)
    class_names = cfg.get("class_names")

    train_transform = get_train_transforms(image_size)
    eval_transform = get_eval_transforms(image_size)

    loaders: dict[str, DataLoader] = {}

    for split, tfm, shuffle in [
        ("train", train_transform, True),
        ("val", eval_transform, False),
        ("test", eval_transform, False),
    ]:
        csv_path = splits_dir / f"{split}.csv"
        if not csv_path.exists():
            continue
        ds = TrafficFrameDataset(
            csv_path=csv_path,
            transform=tfm,
            class_names=class_names,
            # data_root not needed: CSV paths are project-root-relative
        )
        import torch
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split == "train"),
        )

    return loaders
