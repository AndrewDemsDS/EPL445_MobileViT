"""Analytics helpers: density timeline and ROI-filtered counts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def density_timeline(csv_path: str | Path) -> list[dict]:
    """Return per-frame vehicle counts as a list of {frame, count} dicts."""
    df = pd.read_csv(csv_path)
    vehicles = df[(df["class_name"] != "background") & (df.get("w", 1) > 0)]
    timeline = (
        vehicles.groupby("frame_idx")
        .size()
        .reset_index(name="count")
        .rename(columns={"frame_idx": "frame"})
        .sort_values("frame")
    )
    return timeline.to_dict(orient="records")


def roi_counts(
    csv_path: str | Path,
    x: int,
    y: int,
    w: int,
    h: int,
) -> dict:
    """Recompute class counts for detections whose bbox centre falls inside the ROI.

    ROI is given in the coordinate space of the annotated output video
    (i.e. original video resolution).
    """
    df = pd.read_csv(csv_path)
    vehicles = df[(df["class_name"] != "background") & (df["w"] > 0)].copy()

    if vehicles.empty:
        return {"class_counts": {}, "total_detections": 0, "roi": {"x": x, "y": y, "w": w, "h": h}}

    cx = vehicles["x"] + vehicles["w"] / 2
    cy = vehicles["y"] + vehicles["h"] / 2
    inside = (cx >= x) & (cx <= x + w) & (cy >= y) & (cy <= y + h)
    filtered = vehicles[inside]

    counts = filtered["class_name"].value_counts().to_dict()
    return {
        "class_counts": counts,
        "total_detections": int(len(filtered)),
        "roi": {"x": x, "y": y, "w": w, "h": h},
    }
