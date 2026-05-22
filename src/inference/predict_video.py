#!/usr/bin/env python3
"""Video inference: scan each frame with a sliding window, classify patches,
and produce an annotated output video with bounding-box–style overlays.

The trained MobileViT model was fine-tuned on *cropped vehicle patches*, so
we tile each frame into overlapping windows, classify each window, and keep
the non-background detections above a confidence threshold.

Usage
-----
    python -m src.inference.predict_video --config configs/demo.yaml
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.datasets.transforms import get_eval_transforms
from src.models.model_factory import build_model
from src.utils.device import get_device
from src.utils.io import ensure_dir, load_checkpoint, merge_configs
from src.utils.logging import setup_logger


# ── Colour palette for bounding-box overlays ─────────────────────
CLASS_COLORS = {
    "car":        (0, 200, 0),     # green
    "bus":        (255, 140, 0),   # orange
    "truck":      (0, 120, 255),   # blue
    "background": (128, 128, 128), # grey (not drawn)
}


def _sliding_windows(
    frame_h: int,
    frame_w: int,
    win_sizes: list[int],
    stride_ratio: float = 0.5,
) -> list[tuple[int, int, int, int]]:
    """Generate (x, y, w, h) sliding-window coordinates at multiple scales."""
    windows: list[tuple[int, int, int, int]] = []
    for ws in win_sizes:
        stride = max(int(ws * stride_ratio), 1)
        for y in range(0, frame_h - ws + 1, stride):
            for x in range(0, frame_w - ws + 1, stride):
                windows.append((x, y, ws, ws))
    return windows


def _nms_boxes(
    boxes: list[dict],
    iou_threshold: float = 0.3,
) -> list[dict]:
    """Simple per-class greedy NMS on detected boxes."""
    if not boxes:
        return []

    # Group by class
    by_class: dict[str, list[dict]] = {}
    for b in boxes:
        by_class.setdefault(b["class_name"], []).append(b)

    kept: list[dict] = []
    for cls_name, cls_boxes in by_class.items():
        # Sort by confidence descending
        cls_boxes.sort(key=lambda b: b["confidence"], reverse=True)
        selected: list[dict] = []
        for box in cls_boxes:
            overlap = False
            for sel in selected:
                if _iou(box, sel) > iou_threshold:
                    overlap = True
                    break
            if not overlap:
                selected.append(box)
        kept.extend(selected)
    return kept


def _iou(a: dict, b: dict) -> float:
    """Compute IoU between two boxes (each has x, y, w, h keys)."""
    ax1, ay1 = a["x"], a["y"]
    ax2, ay2 = ax1 + a["w"], ay1 + a["h"]
    bx1, by1 = b["x"], b["y"]
    bx2, by2 = bx1 + b["w"], by1 + b["h"]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


# COCO classes from YOLOv8 that correspond to our vehicle taxonomy.
# We keep the YOLO box but re-classify the crop with MobileViT.
_YOLO_VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck


def _yolo_proposals(
    frame_rgb: np.ndarray,
    yolo_model,
    conf_threshold: float = 0.25,
) -> list[tuple[int, int, int, int]]:
    """Return (x, y, w, h) candidate boxes from YOLOv8 filtered to vehicles."""
    res = yolo_model(frame_rgb, conf=conf_threshold, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return []
    boxes_xyxy = res.boxes.xyxy.cpu().numpy()
    classes = res.boxes.cls.cpu().numpy().astype(int)
    out: list[tuple[int, int, int, int]] = []
    h_img, w_img = frame_rgb.shape[:2]
    for (x1, y1, x2, y2), cls in zip(boxes_xyxy, classes):
        if cls not in _YOLO_VEHICLE_CLASSES:
            continue
        x = max(0, int(x1))
        y = max(0, int(y1))
        w = min(w_img - x, int(x2 - x1))
        h = min(h_img - y, int(y2 - y1))
        if w < 16 or h < 16:
            continue
        out.append((x, y, w, h))
    return out


@torch.no_grad()
def _classify_patches_batch(
    frame_rgb: np.ndarray,
    windows: list[tuple[int, int, int, int]],
    model: torch.nn.Module,
    transform,
    device: torch.device,
    class_names: list[str],
    conf_threshold: float,
    batch_size: int = 128,
) -> list[dict]:
    """Classify all window patches and return non-background detections."""
    detections: list[dict] = []

    for i in range(0, len(windows), batch_size):
        batch_windows = windows[i:i + batch_size]
        tensors = []
        for (x, y, w, h) in batch_windows:
            crop = frame_rgb[y:y + h, x:x + w]
            pil_crop = Image.fromarray(crop)
            tensors.append(transform(pil_crop))

        batch_tensor = torch.stack(tensors).to(device)
        logits = model(batch_tensor)
        probs = torch.softmax(logits, dim=1)
        confs, idxs = probs.max(dim=1)

        for j, (x, y, w, h) in enumerate(batch_windows):
            cls_idx = idxs[j].item()
            conf = confs[j].item()
            cls_name = class_names[cls_idx]

            if cls_name != "background" and conf >= conf_threshold:
                detections.append({
                    "x": x, "y": y, "w": w, "h": h,
                    "class_name": cls_name,
                    "confidence": conf,
                })

    return detections


def run_video_inference(cfg: dict, preloaded: dict | None = None) -> None:
    """Run vehicle detection on a video and save results.

    `preloaded` can carry already-loaded models so we skip ~15 s of
    cold-start per job when invoked from the FastAPI worker:

        preloaded = {"model": ..., "device": ..., "yolo": ...}

    Each key is optional; anything missing falls back to building from
    `cfg`, matching the standalone CLI behaviour.
    """
    logger = setup_logger()
    preloaded = preloaded or {}

    # ── Model ────────────────────────────────────────────────────
    if "model" in preloaded and "device" in preloaded:
        model = preloaded["model"]
        device = preloaded["device"]
        logger.info("Using preloaded MobileViT classifier (device=%s)", device)
    else:
        device = get_device(cfg.get("device", "auto"))
        model = build_model(cfg).to(device)
        ckpt_path = cfg.get("checkpoint_path", "outputs/models/best_model.pth")
        load_checkpoint(ckpt_path, model, device=device)
        model.eval()
        logger.info("Loaded model from %s", ckpt_path)

    class_names = cfg.get("class_names", [])
    image_size = cfg.get("image_size", 256)
    transform = get_eval_transforms(image_size)

    # ── Video I/O ────────────────────────────────────────────────
    input_path = cfg.get("input_video", "data/raw/sample_traffic.mp4")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error("Cannot open video: %s", input_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_skip = cfg.get("frame_skip", 2)

    # Resize for faster processing if frame is very large
    max_dim = cfg.get("max_inference_dim", 960)
    scale = min(max_dim / max(orig_w, orig_h), 1.0)
    proc_w = int(orig_w * scale)
    proc_h = int(orig_h * scale)
    logger.info("Input: %dx%d @ %.1f fps, %d frames, processing at %dx%d",
                orig_w, orig_h, fps, total, proc_w, proc_h)

    # Sliding window sizes (relative to processed resolution)
    win_sizes = cfg.get("window_sizes", [128, 192, 256])
    stride_ratio = cfg.get("stride_ratio", 0.5)
    conf_threshold = cfg.get("confidence_threshold", 0.5)
    iou_threshold = cfg.get("nms_iou_threshold", 0.3)

    # Detection backend: "sliding" (multi-scale window) or "yolo" (YOLOv8n)
    detector = cfg.get("detector", "sliding").lower()
    yolo_model = None
    if detector == "yolo":
        if "yolo" in preloaded:
            yolo_model = preloaded["yolo"]
            logger.info("Using preloaded YOLO detector")
        else:
            from ultralytics import YOLO
            yolo_weights = cfg.get("yolo_weights", "yolov8n.pt")
            yolo_model = YOLO(yolo_weights)
            yolo_model.to(str(device))
            logger.info("YOLO detector enabled (weights=%s)", yolo_weights)
    else:
        logger.info("Sliding-window detector enabled (sizes=%s)", win_sizes)

    # Output video (original resolution with annotations)
    output_video = cfg.get("output_video", "outputs/predictions/annotated_output.mp4")
    ensure_dir(Path(output_video).parent)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_fps = fps / frame_skip
    writer = cv2.VideoWriter(output_video, fourcc, out_fps, (orig_w, orig_h))

    # Output CSV
    output_csv = cfg.get("output_csv", "outputs/predictions/frame_predictions.csv")
    ensure_dir(Path(output_csv).parent)
    # Line-buffered so the dashboard's progress poller can read rows as
    # they're written, not only after the file closes at end-of-job.
    csv_file = open(output_csv, "w", newline="", buffering=1)
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=["frame_idx", "track_id", "class_name", "confidence", "x", "y", "w", "h"],
    )
    csv_writer.writeheader()

    # ── Optional SORT tracker ────────────────────────────────────
    tracker = None
    if cfg.get("enable_tracking", False):
        from src.inference.tracker import SORTTracker
        tracker = SORTTracker(
            max_age=cfg.get("track_max_age", 5),
            min_hits=cfg.get("track_min_hits", 2),
            iou_threshold=cfg.get("track_iou_threshold", 0.3),
        )
        logger.info("SORT tracker enabled (max_age=%d, min_hits=%d)",
                    tracker.max_age, tracker.min_hits)

    # ── Process frames ───────────────────────────────────────────
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = cfg.get("font_scale", 0.7)
    thickness = 2

    frame_idx = 0
    total_detections = 0
    pbar = tqdm(total=total, desc="Video inference")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            # Resize for classification
            if scale < 1.0:
                proc_frame = cv2.resize(frame, (proc_w, proc_h))
            else:
                proc_frame = frame.copy()

            proc_rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)

            # Candidate windows: YOLO proposals or multi-scale sliding window
            if detector == "yolo":
                windows = _yolo_proposals(proc_rgb, yolo_model,
                                          conf_threshold=cfg.get("yolo_conf", 0.25))
            else:
                windows = _sliding_windows(proc_h, proc_w, win_sizes, stride_ratio)

            # Classify all patches with MobileViT
            detections = _classify_patches_batch(
                proc_rgb, windows, model, transform, device,
                class_names, conf_threshold,
            )

            # NMS only matters for sliding window (YOLO already applied its own).
            if detector != "yolo":
                detections = _nms_boxes(detections, iou_threshold)
            total_detections += len(detections)

            # Project detections to original resolution and (optionally) track
            inv_scale = 1.0 / scale if scale < 1.0 else 1.0
            projected: list[tuple[int, int, int, int, str, float]] = []
            for det in detections:
                x1 = int(det["x"] * inv_scale)
                y1 = int(det["y"] * inv_scale)
                x2 = int((det["x"] + det["w"]) * inv_scale)
                y2 = int((det["y"] + det["h"]) * inv_scale)
                projected.append((x1, y1, x2, y2, det["class_name"], det["confidence"]))

            # SORT tracker (optional). It needs to be called every analysed frame
            # so trackers can age out even on frames with no detections.
            if tracker is not None:
                track_in = [[x1, y1, x2, y2, cls] for (x1, y1, x2, y2, cls, _) in projected]
                tracked = tracker.update(track_in)
                # Match tracked outputs back to projected detections by IoU to recover confidence
                draw_items: list[tuple[int, int, int, int, str, float, int | None]] = []
                for (tx1, ty1, tx2, ty2, tid, tlabel) in tracked:
                    best_conf = 0.0
                    for (px1, py1, px2, py2, pcls, pconf) in projected:
                        if pcls != tlabel:
                            continue
                        ix1, iy1 = max(tx1, px1), max(ty1, py1)
                        ix2, iy2 = min(tx2, px2), min(ty2, py2)
                        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                        if inter > 0 and pconf > best_conf:
                            best_conf = pconf
                    draw_items.append((tx1, ty1, tx2, ty2, tlabel, best_conf or 1.0, tid))
            else:
                draw_items = [(x1, y1, x2, y2, cls, conf, None) for (x1, y1, x2, y2, cls, conf) in projected]

            # Draw + log
            for (x1, y1, x2, y2, cls, conf, tid) in draw_items:
                color = CLASS_COLORS.get(cls, (0, 255, 0))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                label = f"#{tid} {cls} {conf:.2f}" if tid is not None else f"{cls} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
                cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 4),
                            font, font_scale, (255, 255, 255), thickness)

                csv_writer.writerow({
                    "frame_idx": frame_idx,
                    "track_id": tid if tid is not None else "",
                    "class_name": cls,
                    "confidence": f"{conf:.4f}",
                    "x": x1, "y": y1,
                    "w": x2 - x1,
                    "h": y2 - y1,
                })

            # If no detection drawn, still note this frame
            if not draw_items:
                csv_writer.writerow({
                    "frame_idx": frame_idx,
                    "track_id": "",
                    "class_name": "background",
                    "confidence": "1.0000",
                    "x": 0, "y": 0, "w": 0, "h": 0,
                })

            # HUD overlay: frame count and detection summary
            counts: dict[str, int] = {}
            for (_x1, _y1, _x2, _y2, cls, _conf, _tid) in draw_items:
                counts[cls] = counts.get(cls, 0) + 1
            kind = "Tracked" if tracker is not None else "Detections"
            hud = f"Frame {frame_idx} | {kind}: {len(draw_items)}"
            if counts:
                hud += " | " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
            cv2.putText(frame, hud, (10, orig_h - 15),
                        font, font_scale * 0.8, (255, 255, 255), thickness)

            writer.write(frame)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()
    csv_file.close()

    logger.info("Annotated video saved to: %s", output_video)
    logger.info("Frame predictions saved to: %s", output_csv)
    logger.info("Total detections across all frames: %d", total_detections)


def main() -> None:
    parser = argparse.ArgumentParser(description="Video inference with MobileViT classifier")
    parser.add_argument(
        "--config", type=str, default="configs/demo.yaml",
        help="Path to demo config (merged with configs/default.yaml)",
    )
    args = parser.parse_args()
    cfg = merge_configs("configs/default.yaml", args.config)
    run_video_inference(cfg)


if __name__ == "__main__":
    main()
