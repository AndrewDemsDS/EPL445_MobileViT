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


@torch.no_grad()
def _classify_patches_batch(
    frame_rgb: np.ndarray,
    windows: list[tuple[int, int, int, int]],
    model: torch.nn.Module,
    transform,
    device: torch.device,
    class_names: list[str],
    conf_threshold: float,
    batch_size: int = 64,
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


def run_video_inference(cfg: dict) -> None:
    """Run sliding-window vehicle detection on a video and save results."""
    logger = setup_logger()
    device = get_device(cfg.get("device", "auto"))

    # ── Model ────────────────────────────────────────────────────
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

    # Output video (original resolution with annotations)
    output_video = cfg.get("output_video", "outputs/predictions/annotated_output.mp4")
    ensure_dir(Path(output_video).parent)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_fps = fps / frame_skip
    writer = cv2.VideoWriter(output_video, fourcc, out_fps, (orig_w, orig_h))

    # Output CSV
    output_csv = cfg.get("output_csv", "outputs/predictions/frame_predictions.csv")
    ensure_dir(Path(output_csv).parent)
    csv_file = open(output_csv, "w", newline="")
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=["frame_idx", "class_name", "confidence", "x", "y", "w", "h"],
    )
    csv_writer.writeheader()

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

            # Generate sliding windows
            windows = _sliding_windows(proc_h, proc_w, win_sizes, stride_ratio)

            # Classify all patches
            detections = _classify_patches_batch(
                proc_rgb, windows, model, transform, device,
                class_names, conf_threshold,
            )

            # Non-max suppression
            detections = _nms_boxes(detections, iou_threshold)
            total_detections += len(detections)

            # Draw detections on original-resolution frame
            inv_scale = 1.0 / scale if scale < 1.0 else 1.0
            for det in detections:
                x1 = int(det["x"] * inv_scale)
                y1 = int(det["y"] * inv_scale)
                x2 = int((det["x"] + det["w"]) * inv_scale)
                y2 = int((det["y"] + det["h"]) * inv_scale)
                color = CLASS_COLORS.get(det["class_name"], (0, 255, 0))

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                label = f"{det['class_name']} {det['confidence']:.2f}"
                (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
                cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 4),
                            font, font_scale, (255, 255, 255), thickness)

                csv_writer.writerow({
                    "frame_idx": frame_idx,
                    "class_name": det["class_name"],
                    "confidence": f"{det['confidence']:.4f}",
                    "x": x1, "y": y1,
                    "w": int(det["w"] * inv_scale),
                    "h": int(det["h"] * inv_scale),
                })

            # If no vehicle detected, still note it
            if not detections:
                csv_writer.writerow({
                    "frame_idx": frame_idx,
                    "class_name": "background",
                    "confidence": "1.0000",
                    "x": 0, "y": 0, "w": 0, "h": 0,
                })

            # HUD overlay: frame count and detection summary
            counts = {}
            for d in detections:
                counts[d["class_name"]] = counts.get(d["class_name"], 0) + 1
            hud = f"Frame {frame_idx} | Detections: {len(detections)}"
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
