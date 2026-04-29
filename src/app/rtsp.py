"""RTSP stream support — stub for Phase 3 stretch goal.

To implement:
1. Accept an RTSP URL (e.g. rtsp://camera-ip:554/stream)
2. Open with cv2.VideoCapture(url)
3. Run sliding-window inference frame-by-frame
4. Stream annotated frames as MJPEG via a FastAPI StreamingResponse
5. Integrate ByteTrack/SORT for persistent vehicle IDs
"""

from __future__ import annotations


_active_streams: dict[str, dict] = {}


def list_streams() -> list[dict]:
    return list(_active_streams.values())


def start_stream(url: str) -> dict:
    # TODO: validate URL, spin up inference thread, return stream_id
    return {"error": "RTSP support not yet implemented", "url": url}
