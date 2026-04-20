"""Re-generate train/val/test splits at the sequence level to eliminate data leakage.

Patches from the same video sequence (MVI_XXXXX) are kept together in one split,
so the test set contains sequences the model has never seen during training.
"""
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SPLITS_DIR = Path("data/splits")
SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# test gets the remainder


def extract_sequence(path: str) -> str:
    m = re.search(r"(MVI_\d+)", path)
    return m.group(1) if m else "background"


def main():
    # Load all patches from the three existing CSVs
    dfs = [pd.read_csv(SPLITS_DIR / f"{s}.csv") for s in ("train", "val", "test")]
    all_df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset="image_path")
    all_df["sequence"] = all_df["image_path"].apply(extract_sequence)

    print(f"Total patches : {len(all_df)}")
    print(f"Unique seqs   : {all_df['sequence'].nunique()}")
    print(f"Class dist    :\n{all_df['label'].value_counts()}\n")

    # ── Sequence-level split ─────────────────────────────────────
    sequences = all_df["sequence"].unique()

    # Stratify by majority class per sequence so class balance is roughly preserved
    seq_label = (
        all_df.groupby("sequence")["label"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
    )
    seq_label.columns = ["sequence", "dominant_label"]

    train_seqs, temp_seqs = train_test_split(
        seq_label["sequence"],
        test_size=1 - TRAIN_RATIO,
        stratify=seq_label["dominant_label"],
        random_state=SEED,
    )
    relative_val = VAL_RATIO / (1 - TRAIN_RATIO)
    temp_label = seq_label[seq_label["sequence"].isin(temp_seqs)]["dominant_label"]
    val_seqs, test_seqs = train_test_split(
        temp_seqs,
        test_size=1 - relative_val,
        stratify=temp_label,
        random_state=SEED,
    )

    train_df = all_df[all_df["sequence"].isin(train_seqs)].drop(columns="sequence")
    val_df   = all_df[all_df["sequence"].isin(val_seqs)].drop(columns="sequence")
    test_df  = all_df[all_df["sequence"].isin(test_seqs)].drop(columns="sequence")

    # Verify zero overlap
    assert len(set(train_seqs) & set(test_seqs)) == 0, "Train/test overlap!"
    assert len(set(train_seqs) & set(val_seqs))  == 0, "Train/val overlap!"

    print("Sequence-level split (no leakage):")
    print(f"  Train: {len(train_seqs)} seqs, {len(train_df)} patches")
    print(f"  Val  : {len(val_seqs)} seqs,  {len(val_df)} patches")
    print(f"  Test : {len(test_seqs)} seqs,  {len(test_df)} patches")
    print(f"\nTest class distribution:\n{test_df['label'].value_counts()}")

    train_df.to_csv(SPLITS_DIR / "train.csv", index=False)
    val_df.to_csv(SPLITS_DIR  / "val.csv",   index=False)
    test_df.to_csv(SPLITS_DIR / "test.csv",  index=False)
    print("\nSplits saved to data/splits/")


if __name__ == "__main__":
    main()
