"""Aggregate per-frame predictions into class counts and density estimates."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.io import save_json


def aggregate_predictions(
    csv_path: str | Path,
    output_path: str | Path | None = None,
) -> dict:
    """Read frame predictions CSV and produce aggregate statistics.

    Parameters
    ----------
    csv_path : path
        CSV with columns ``frame_idx, class_name, confidence`` (and optional
        bounding-box columns ``x, y, w, h``).
    output_path : path, optional
        If provided, save the aggregate as JSON.

    Returns
    -------
    dict with ``total_frames``, ``total_detections``, ``class_counts``,
    ``class_proportions``, ``dominant_class``, ``density_estimate``,
    and ``per_frame_vehicle_count``.
    """
    df = pd.read_csv(csv_path)

    # Total unique frames processed
    total_frames = int(df["frame_idx"].nunique())

    # Filter out background entries (w==0 rows are "no detection" markers)
    has_bbox = "w" in df.columns
    if has_bbox:
        vehicles = df[(df["class_name"] != "background") & (df["w"] > 0)]
    else:
        vehicles = df[df["class_name"] != "background"]

    total_detections = len(vehicles)

    # Class counts (vehicle classes only)
    counts = vehicles["class_name"].value_counts().to_dict() if len(vehicles) > 0 else {}

    # Proportions of detections
    proportions = {k: round(v / max(total_detections, 1), 4) for k, v in counts.items()}

    # Average vehicles per frame
    if has_bbox and total_frames > 0:
        veh_per_frame = vehicles.groupby("frame_idx").size()
        avg_veh_per_frame = round(float(veh_per_frame.mean()), 2) if len(veh_per_frame) > 0 else 0.0
    else:
        avg_veh_per_frame = round(total_detections / max(total_frames, 1), 2)

    # Frames with at least one vehicle detection
    frames_with_vehicles = int(vehicles["frame_idx"].nunique()) if len(vehicles) > 0 else 0
    density = round(frames_with_vehicles / max(total_frames, 1), 4)

    result = {
        "total_frames": total_frames,
        "total_detections": total_detections,
        "class_counts": counts,
        "class_proportions": proportions,
        "dominant_class": max(counts, key=counts.get) if counts else "background",
        "avg_vehicles_per_frame": avg_veh_per_frame,
        "frames_with_vehicles": frames_with_vehicles,
        "density_estimate": density,
    }

    if output_path:
        save_json(result, output_path)

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate frame predictions")
    parser.add_argument("--csv", required=True, help="Path to frame_predictions.csv")
    parser.add_argument("--output", default="outputs/predictions/class_counts.json")
    args = parser.parse_args()

    result = aggregate_predictions(args.csv, args.output)
    print(result)
