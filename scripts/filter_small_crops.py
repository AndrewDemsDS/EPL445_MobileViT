"""Filter crops smaller than MIN_SIDE pixels (shortest side) from the splits.

Very small crops (e.g. <64 px) are upscaled ~4x when resized to 256x256 for training.
The model can then learn interpolation artefacts rather than real vehicle features —
and test-set accuracy becomes inflated because the test distribution matches.
"""
from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm

MIN_SIDE = 64
SPLITS_DIR = Path("data/splits")


def min_side(path: str) -> int:
    try:
        with Image.open(path) as im:
            return min(im.size)
    except Exception:
        return 0


def main():
    for split in ("train", "val", "test"):
        csv = SPLITS_DIR / f"{split}.csv"
        df = pd.read_csv(csv)
        sizes = [min_side(p) for p in tqdm(df["image_path"], desc=f"Sizing {split}")]
        df["_min_side"] = sizes

        kept = df[df["_min_side"] >= MIN_SIDE].drop(columns="_min_side")
        dropped = len(df) - len(kept)
        pct = 100 * dropped / len(df)
        print(f"{split}: {len(df)} → {len(kept)}  (dropped {dropped}, {pct:.1f}%)")
        print(f"  Per-class kept:\n{kept['label'].value_counts().to_string()}\n")

        kept.to_csv(csv, index=False)


if __name__ == "__main__":
    main()
