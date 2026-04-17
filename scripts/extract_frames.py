#!/usr/bin/env python3
"""Extract frames from a video file at a configurable FPS.

Usage
-----
    python scripts/extract_frames.py --input video.mp4 --output data/interim/frames --fps 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    target_fps: float = 5.0,
) -> int:
    """Extract frames from *video_path* at approximately *target_fps*.

    Returns the number of frames saved.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, round(src_fps / target_fps))

    print(f"Source FPS: {src_fps:.1f}, extracting every {frame_interval} frame(s)")
    print(f"Total source frames: {total_frames}")

    saved = 0
    frame_idx = 0

    with tqdm(total=total_frames, desc="Extracting") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                fname = f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(output_dir / fname), frame)
                saved += 1
            frame_idx += 1
            pbar.update(1)

    cap.release()
    print(f"Saved {saved} frames to {output_dir}")
    return saved


def main():
    parser = argparse.ArgumentParser(description="Extract frames from video")
    parser.add_argument("--input", "-i", required=True, help="Input video path")
    parser.add_argument("--output", "-o", default="data/interim/frames", help="Output directory")
    parser.add_argument("--fps", type=float, default=5.0, help="Target FPS for extraction")
    args = parser.parse_args()

    extract_frames(args.input, args.output, args.fps)


if __name__ == "__main__":
    main()
