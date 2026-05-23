"""RTSP/webcam streaming wrappers for the dashboard.

The standalone CLI in src/inference/stream.py uses cv2.imshow. For the
web dashboard we re-use the same model-loading and inference code via
stream_frames(), which yields JPEG bytes per annotated frame, and serve
it as a multipart/x-mixed-replace MJPEG stream.
"""

from __future__ import annotations

from src.inference.stream import stream_frames


# A boundary string the browser uses to delimit MJPEG frames.
_BOUNDARY = b"--frame"


def _parse_source(raw: str) -> str | int:
    """Convert "0" -> 0 (webcam), keep rtsp://... and file paths as-is."""
    return int(raw) if raw.isdigit() else raw


def mjpeg_stream(source: str, checkpoint_path: str = "outputs/models/best_model.pth",
                 max_frames: int = 0, detector: str = "yolo"):
    """Generator that yields MJPEG-framed bytes for FastAPI StreamingResponse.

    max_frames=0 streams until the source ends (file) or forever (webcam/RTSP).
    detector: "yolo" (default, ~5-10 fps) or "sliding" (~0.1 fps).
    """
    parsed = _parse_source(source)
    for jpeg, _counts in stream_frames(parsed, checkpoint_path,
                                       max_frames=max_frames, detector=detector):
        yield _BOUNDARY + b"\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"


def list_streams() -> list[dict]:
    """No persistent stream registry yet; UI uses a single feed at a time."""
    return []


def start_stream(url: str) -> dict:
    return {"status": "started", "url": url, "feed": f"/stream/feed?source={url}"}
