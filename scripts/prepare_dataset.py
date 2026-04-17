#!/usr/bin/env python3
"""Prepare the UA-DETRAC dataset for frame-level traffic classification.

Steps
-----
1. Download UA-DETRAC images and annotations (if not already present).
2. Parse XML annotations to extract vehicle bounding-box crops with class labels.
3. Save crops under ``data/processed/{class_name}/``.
4. Generate stratified train/val/test CSV splits under ``data/splits/``.

Usage
-----
    python scripts/prepare_dataset.py [--data-root data] [--max-crops-per-class 5000]

UA-DETRAC provides bounding box annotations with vehicle types:
  car, bus, van, others
We map these to our target labels:
  car, bus, truck (mapped from van+others), motorcycle (if present), background.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Label mapping ────────────────────────────────────────────────

# UA-DETRAC vehicle_type → our class names
DETRAC_LABEL_MAP = {
    "car": "car",
    "bus": "bus",
    "van": "truck",       # treat vans as trucks (heavy-ish vehicle)
    "others": "truck",    # misc heavy vehicles
}

TARGET_CLASSES = ["car", "motorcycle", "bus", "truck", "background"]

# ── Kaggle dataset ───────────────────────────────────────────────
# The official UA-DETRAC server blocks direct downloads (403).
# We use kagglehub to fetch the Kaggle mirror:
#   https://www.kaggle.com/datasets/bratjay/ua-detrac-orig
KAGGLEHUB_DATASET = "bratjay/ua-detrac-orig"


def download_via_kagglehub() -> Path:
    """Download UA-DETRAC using kagglehub and return the local cache path.

    Requires:
        pip install kagglehub
        KAGGLE_API_TOKEN env var  (or ~/.kaggle/kaggle.json)
    """
    try:
        import kagglehub
    except ImportError:
        print("  ✗ kagglehub not installed. Run: pip install kagglehub")
        raise SystemExit(1)

    print(f"  ↓ Downloading {KAGGLEHUB_DATASET} via kagglehub …")
    path = kagglehub.dataset_download(KAGGLEHUB_DATASET)
    print(f"  ✓ Dataset cached at: {path}")
    return Path(path)


def download_file(url: str, dest: Path) -> Path:
    """Download *url* to *dest* if it does not already exist (direct HTTP fallback)."""
    if dest.exists():
        print(f"  ✓ Already exists: {dest}")
        return dest
    print(f"  ↓ Downloading {url} …")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        import urllib.request
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        print("    Please download manually and place at:", dest)
        raise
    return dest


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract a zip file if destination does not look populated."""
    if any(extract_to.iterdir()) if extract_to.exists() else False:
        print(f"  ✓ Already extracted: {extract_to}")
        return
    print(f"  ⤓ Extracting {zip_path} → {extract_to} …")
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)


# ── Annotation parsing ──────────────────────────────────────────


def parse_detrac_xml(xml_path: Path) -> list[dict]:
    """Parse a single UA-DETRAC annotation XML file.

    Returns a list of dicts with keys:
        sequence, frame_num, bbox (x,y,w,h), vehicle_type
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    seq_name = root.attrib.get("name", xml_path.stem)
    records = []

    for frame_el in root.iter("frame"):
        frame_num = int(frame_el.attrib["num"])
        target_list = frame_el.find("target_list")
        if target_list is None:
            continue
        for target in target_list.findall("target"):
            box_el = target.find("box")
            attr_el = target.find("attribute")
            if box_el is None:
                continue
            x = float(box_el.attrib["left"])
            y = float(box_el.attrib["top"])
            w = float(box_el.attrib["width"])
            h = float(box_el.attrib["height"])
            vtype = attr_el.attrib.get("vehicle_type", "others") if attr_el is not None else "others"
            records.append({
                "sequence": seq_name,
                "frame_num": frame_num,
                "x": x, "y": y, "w": w, "h": h,
                "vehicle_type": vtype,
            })
    return records


def crop_and_save(
    records: list[dict],
    images_dir: Path,
    output_dir: Path,
    min_crop_size: int = 32,
) -> list[dict]:
    """Crop vehicles from frames and save as individual images.

    Returns a list of dicts with keys: image_path, label.
    """
    saved = []
    for rec in tqdm(records, desc="Cropping", leave=False):
        seq = rec["sequence"]
        frame_num = rec["frame_num"]
        # UA-DETRAC frame naming: img{frame_num:05d}.jpg
        frame_file = images_dir / seq / f"img{frame_num:05d}.jpg"
        if not frame_file.exists():
            continue

        img = cv2.imread(str(frame_file))
        if img is None:
            continue

        h_img, w_img = img.shape[:2]
        x = max(0, int(rec["x"]))
        y = max(0, int(rec["y"]))
        w = int(rec["w"])
        h = int(rec["h"])
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)

        if (x2 - x) < min_crop_size or (y2 - y) < min_crop_size:
            continue

        crop = img[y:y2, x:x2]
        label = DETRAC_LABEL_MAP.get(rec["vehicle_type"], "truck")

        class_dir = output_dir / label
        class_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{seq}_f{frame_num:05d}_{x}_{y}.jpg"
        out_path = class_dir / fname
        cv2.imwrite(str(out_path), crop)

        saved.append({"image_path": str(out_path), "label": label})

    return saved


def generate_background_crops(
    images_dir: Path,
    output_dir: Path,
    num_crops: int = 2000,
    crop_size: int = 128,
) -> list[dict]:
    """Generate random background crops from frames (no vehicles)."""
    bg_dir = output_dir / "background"
    bg_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    # Collect some frame paths
    frame_paths = []
    for seq_dir in sorted(images_dir.iterdir()):
        if not seq_dir.is_dir():
            continue
        frames = sorted(seq_dir.glob("*.jpg"))
        # Take every 20th frame to get variety
        frame_paths.extend(frames[::20])

    random.shuffle(frame_paths)
    frame_paths = frame_paths[:num_crops * 2]  # oversample

    count = 0
    for fp in frame_paths:
        if count >= num_crops:
            break
        img = cv2.imread(str(fp))
        if img is None:
            continue
        h, w = img.shape[:2]
        if h < crop_size or w < crop_size:
            continue
        # Random crop from corner / edge (less likely to contain vehicles)
        x = random.choice([0, w - crop_size])
        y = random.choice([0, h - crop_size])
        crop = img[y:y + crop_size, x:x + crop_size]
        fname = f"bg_{fp.parent.name}_{fp.stem}_{x}_{y}.jpg"
        out_path = bg_dir / fname
        cv2.imwrite(str(out_path), crop)
        saved.append({"image_path": str(out_path), "label": "background"})
        count += 1

    return saved


def create_splits(
    records: list[dict],
    splits_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    max_per_class: int | None = None,
    seed: int = 42,
) -> None:
    """Create stratified train/val/test CSV splits."""
    df = pd.DataFrame(records)
    splits_dir.mkdir(parents=True, exist_ok=True)

    # Optional: cap per-class count
    if max_per_class:
        dfs = []
        for label in df["label"].unique():
            subset = df[df["label"] == label]
            if len(subset) > max_per_class:
                subset = subset.sample(n=max_per_class, random_state=seed)
            dfs.append(subset)
        df = pd.concat(dfs, ignore_index=True)

    # Stratified split
    from sklearn.model_selection import train_test_split

    train_df, temp_df = train_test_split(
        df, test_size=(1 - train_ratio), stratify=df["label"], random_state=seed
    )
    relative_val = val_ratio / (1 - train_ratio)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - relative_val), stratify=temp_df["label"], random_state=seed
    )

    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)

    print(f"\n  Split sizes — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
    print(f"  Class distribution (train):")
    print(train_df["label"].value_counts().to_string(header=False))


# ── Main ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Prepare UA-DETRAC for classification")
    parser.add_argument("--data-root", type=str, default="data", help="Root data directory")
    parser.add_argument("--max-crops-per-class", type=int, default=5000,
                        help="Max crops per class (balance dataset)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download step (use existing data in data/raw/)")
    parser.add_argument("--dataset-path", type=str, default=None,
                        help="Path to already-downloaded UA-DETRAC dataset (skips download)")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    processed_dir = data_root / "processed"
    splits_dir = data_root / "splits"

    # ── Step 1: Download via kagglehub ───────────────────────────
    if args.dataset_path:
        raw_dir = Path(args.dataset_path)
        print(f"Step 1: Using provided dataset path: {raw_dir}")
    elif not args.skip_download:
        print("Step 1: Downloading UA-DETRAC via kagglehub …")
        raw_dir = download_via_kagglehub()
    else:
        raw_dir = data_root / "raw"
        print(f"Step 1: Skipping download, using {raw_dir}")

    # ── Step 2: Discover annotations ─────────────────────────────
    print("\nStep 2: Discovering dataset structure …")
    # Search for train annotations (prefer train, fall back to all XML)
    train_ann_dirs = list(raw_dir.rglob("DETRAC-Train-Annotations-XML"))
    if train_ann_dirs:
        xml_files = []
        for d in train_ann_dirs:
            xml_files.extend(d.rglob("*.xml"))
    else:
        xml_files = list(raw_dir.rglob("*.xml"))

    if not xml_files:
        print("  ✗ No XML annotation files found under", raw_dir)
        print("    Expected UA-DETRAC XML annotation files.")
        return

    print(f"  Found {len(xml_files)} annotation files")

    # ── Step 3: Parse annotations ────────────────────────────────
    print("\nStep 3: Parsing annotations …")
    all_records = []
    for xml_path in tqdm(xml_files, desc="Parsing XMLs"):
        all_records.extend(parse_detrac_xml(xml_path))
    print(f"  Total bounding boxes: {len(all_records)}")

    if not all_records:
        print("  ✗ No bounding boxes parsed. Check XML format.")
        return

    # ── Step 4: Discover images directory ────────────────────────
    # Look for sequence directories containing img*.jpg files
    images_dir = None
    for candidate in [raw_dir, raw_dir / "train_images", raw_dir / "DETRAC-train-data"]:
        if candidate.exists():
            # Check if this dir has sequence subdirs with images
            for sub in candidate.iterdir():
                if sub.is_dir() and list(sub.glob("img*.jpg")):
                    images_dir = candidate
                    break
        if images_dir:
            break

    # Fallback: search recursively for any dir containing img*.jpg
    if images_dir is None:
        for d in raw_dir.rglob("img00001.jpg"):
            images_dir = d.parent.parent  # go up from sequence dir
            break

    if images_dir is None:
        print("  ✗ Could not find image directories under", raw_dir)
        print("    Expected directories like MVI_XXXXX/ containing img00001.jpg files")
        return

    print(f"  Images directory: {images_dir}")
    seq_dirs = [d for d in images_dir.iterdir() if d.is_dir() and list(d.glob("img*.jpg"))]
    print(f"  Found {len(seq_dirs)} video sequences")

    # ── Step 5: Crop vehicles ────────────────────────────────────
    print("\nStep 4: Cropping vehicle patches …")
    vehicle_records = crop_and_save(all_records, images_dir, processed_dir)
    print(f"  Saved {len(vehicle_records)} vehicle crops")

    # ── Step 6: Background crops ─────────────────────────────────
    print("\nStep 5: Generating background crops …")
    bg_records = generate_background_crops(images_dir, processed_dir, num_crops=2000)
    print(f"  Saved {len(bg_records)} background crops")

    all_saved = vehicle_records + bg_records

    if not all_saved:
        print("  ✗ No crops were generated. Check that images match annotations.")
        return

    # ── Step 7: Create splits ────────────────────────────────────
    print("\nStep 6: Creating train/val/test splits …")
    create_splits(all_saved, splits_dir, max_per_class=args.max_crops_per_class)

    print("\n✓ Dataset preparation complete!")
    print(f"  Processed images: {processed_dir}")
    print(f"  Split CSVs:       {splits_dir}")


if __name__ == "__main__":
    main()

