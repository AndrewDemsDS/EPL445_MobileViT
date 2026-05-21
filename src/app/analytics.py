"""Analytics helpers: density timeline, rectangle ROI, polygon lane counts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.inference.roi import ROILaneCounter


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


def lane_counts(csv_path: str | Path, lanes: dict[str, list[list[int]]]) -> dict:
    """Per-lane per-class counts using the polygon ROI counter.

    Parameters
    ----------
    csv_path : path to frame_predictions.csv
    lanes : ``{"lane_1": [[x,y], [x,y], ...], "lane_2": [...]}``

    Returns
    -------
    {
        "per_lane": {lane_name: {class: count}},
        "per_lane_unique": {lane_name: {class: unique_track_count}},  # when tracked
        "totals": {lane_name: total_detections}
    }
    """
    df = pd.read_csv(csv_path)
    vehicles = df[(df["class_name"] != "background") & (df["w"] > 0)].copy()

    counter = ROILaneCounter(lanes)
    tracked = "track_id" in df.columns

    per_lane: dict[str, dict[str, int]] = {name: {} for name in lanes}
    per_lane_unique: dict[str, dict[str, set]] = {name: {} for name in lanes}

    for _, row in vehicles.iterrows():
        x, y, w, h = int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"])
        cls = row["class_name"]
        cx, cy = x + w // 2, y + h // 2
        for lane_name in lanes:
            if counter.point_in_lane(cx, cy, lane_name):
                per_lane[lane_name][cls] = per_lane[lane_name].get(cls, 0) + 1
                if tracked:
                    tid = row.get("track_id")
                    if tid not in (None, "") and not pd.isna(tid):
                        per_lane_unique[lane_name].setdefault(cls, set()).add(tid)

    result = {
        "per_lane": per_lane,
        "totals": {name: sum(counts.values()) for name, counts in per_lane.items()},
        "lanes": lanes,
    }
    if tracked:
        result["per_lane_unique"] = {
            lane: {cls: len(ids) for cls, ids in classes.items()}
            for lane, classes in per_lane_unique.items()
        }
    return result
