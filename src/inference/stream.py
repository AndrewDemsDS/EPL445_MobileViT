"""Live (webcam, RTSP, file) traffic inference.

Two modes share the same input loop and JPEG-yielding generator:

- ``detector="yolo"`` (default) runs YOLOv8-nano per frame for tight
  vehicle proposals and re-classifies each crop with MobileViT in a
  single batched forward pass. ~5-10 fps live.
- ``detector="sliding"`` keeps the original Interim 2 sliding-window
  loop for reference. ~0.1 fps live.

The standalone CLI (``python -m src.inference.stream``) still uses
``cv2.imshow``; the FastAPI dashboard imports ``stream_frames`` and
serves the JPEG bytes as multipart/x-mixed-replace.
"""

from __future__ import annotations

from typing import Iterator, Tuple

import cv2
import numpy as np
import timm
import torch
from torchvision import transforms

from src.datasets.transforms import get_eval_transforms
from src.inference.predict_video import _classify_patches_batch, _yolo_proposals


CLASSES = ["car", "bus", "truck", "background"]

COLORS = {
    "car":        (0, 200, 0),     # green
    "bus":        (255, 140, 0),   # orange (BGR)
    "truck":      (0, 120, 255),   # blue (BGR)
    "background": (128, 128, 128),
}


def load_model(checkpoint_path: str, device: torch.device):
    """Standalone MobileViT loader for the CLI entry point.

    The FastAPI worker doesn't use this — it pulls from
    src.app.models.get_classifier() to share weights across jobs.
    """
    model = timm.create_model("mobilevit_s", pretrained=False, num_classes=4)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model.to(device)


def get_transform(img_size: int = 256):
    """Kept for backward compatibility; same output as get_eval_transforms."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


# ── Generator (used by the dashboard) ────────────────────────────


def stream_frames(
    source,
    checkpoint_path: str = "outputs/models/best_model.pth",
    img_size: int = 256,
    window_size: int = 64,
    stride: int = 32,
    max_frames: int = 0,
    detector: str = "yolo",
    yolo_conf: float = 0.25,
    classifier_conf: float = 0.5,
) -> Iterator[Tuple[bytes, dict]]:
    """Yield (jpeg_bytes, counts_dict) per annotated frame.

    Parameters
    ----------
    source : webcam index, file path, or rtsp://... URL (cv2.VideoCapture handles all three)
    detector : "yolo" (default, ~5-10 fps) or "sliding" (~0.1 fps)
    max_frames : 0 means until the source ends or the client disconnects
    """
    # STREAM_DEVICE=cpu pins the live stream to CPU so it doesn't fight the
    # offline job over the iGPU's shared VRAM. The offline jobs still use
    # GPU. Default is cpu because the live demo prizes stability over fps.
    import os
    force_cpu = os.environ.get("STREAM_DEVICE", "cpu").lower() == "cpu"

    if force_cpu:
        device = torch.device("cpu")
        model = load_model(checkpoint_path, device)
        yolo_model = None
        if detector == "yolo":
            from ultralytics import YOLO
            yolo_model = YOLO("yolov8n.pt")
            # ultralytics auto-selects cpu when device is unavailable; force it
            yolo_model.to("cpu")
    else:
        # Reuse the FastAPI-wide singletons (live on whichever device they're on).
        try:
            from src.app.models import get_classifier, get_yolo
            model, device = get_classifier()
            yolo_model = get_yolo() if detector == "yolo" else None
        except Exception:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = load_model(checkpoint_path, device)
            yolo_model = None
            if detector == "yolo":
                from ultralytics import YOLO
                yolo_model = YOLO("yolov8n.pt")
                yolo_model.to(str(device))

    transform = get_eval_transforms(img_size)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    frame_id = 0
    while max_frames == 0 or frame_id < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if detector == "yolo":
            # ~one tight box per vehicle from YOLO, batched MobileViT classify
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            windows = _yolo_proposals(frame_rgb, yolo_model, conf_threshold=yolo_conf)
            detections = _classify_patches_batch(
                frame_rgb, windows, model, transform, device,
                CLASSES, classifier_conf,
            )
            counts = {cls: 0 for cls in CLASSES if cls != "background"}
            for det in detections:
                cls = det["class_name"]
                counts[cls] = counts.get(cls, 0) + 1
                x, y, w, h = det["x"], det["y"], det["w"], det["h"]
                color = COLORS.get(cls, (0, 255, 0))
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                label = f"{cls} {det['confidence']:.2f}"
                cv2.putText(frame, label, (x, max(y - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            # Sliding-window fallback (slow, kept for the comparison demo)
            h_img, w_img = frame.shape[:2]
            counts = {cls: 0 for cls in CLASSES if cls != "background"}
            for y in range(0, h_img - window_size, stride):
                for x in range(0, w_img - window_size, stride):
                    patch = cv2.cvtColor(frame[y:y + window_size, x:x + window_size], cv2.COLOR_BGR2RGB)
                    from PIL import Image
                    tensor = transform(Image.fromarray(patch)).unsqueeze(0).to(device)
                    with torch.no_grad():
                        pred = torch.argmax(model(tensor), dim=1).item()
                    label = CLASSES[pred]
                    if label != "background":
                        counts[label] = counts.get(label, 0) + 1
                        cv2.rectangle(frame, (x, y), (x + window_size, y + window_size), COLORS[label], 1)

        # HUD
        y_off = 24
        for cls, n in counts.items():
            cv2.putText(frame, f"{cls}: {n}", (10, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[cls], 2)
            y_off += 22
        cv2.putText(frame, f"Frame {frame_id}  |  detector={detector}",
                    (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if ok:
            yield buf.tobytes(), counts
        frame_id += 1
    cap.release()


# ── Standalone CLI (unchanged behaviour) ─────────────────────────


def run_stream(source, checkpoint_path: str, img_size: int = 256,
               window_size: int = 64, stride: int = 32,
               detector: str = "yolo"):
    """Open a cv2.imshow window. Press 'q' to quit. Uses the new YOLO path."""
    print(f"Source: {source}, detector: {detector}")
    for jpeg, counts in stream_frames(
        source, checkpoint_path,
        img_size=img_size, window_size=window_size, stride=stride,
        detector=detector,
    ):
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        cv2.imshow("Traffic Stream", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="0=webcam, path to video, or rtsp://...")
    parser.add_argument("--checkpoint", default="outputs/models/best_model.pth")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--detector", default="yolo", choices=["yolo", "sliding"])
    args = parser.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    run_stream(source, args.checkpoint, args.img_size, detector=args.detector)
