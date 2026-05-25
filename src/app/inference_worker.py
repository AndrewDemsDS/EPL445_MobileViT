"""Background inference worker: runs predict_video inside a job context."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

from src.app.jobs import Job, JobStatus, store
from src.inference.aggregate_counts import aggregate_predictions
from src.utils.io import load_yaml, merge_configs


def _make_cfg(job: Job) -> dict:
    """Build an inference config for this job, overriding paths to job dir."""
    cfg = merge_configs("configs/default.yaml", "configs/demo.yaml")
    cfg["input_video"] = str(job.input_video)
    cfg["output_video"] = str(job.output_video)
    cfg["output_csv"] = str(job.predictions_csv)
    cfg["output_counts"] = str(job.counts_json)
    # Per-job detector override from the upload form (defaults to demo.yaml).
    detector = getattr(job, "detector", None)
    if detector in ("yolo", "sliding"):
        cfg["detector"] = detector
    return cfg


def _progress_callback(job: Job, processed: int, total: int) -> None:
    job.processed_frames = processed
    job.total_frames = total
    job.progress = int(processed / max(total, 1) * 100)


def _encode_for_web(mp4_path: Path) -> None:
    """Produce a sibling H.264 MP4 (web_<name>.mp4) browsers can play.

    OpenCV's default mp4v codec is not in Chrome's allowed list, so the
    dashboard's <video> tag fails with a DEMUXER_ERROR_NO_SUPPORTED_STREAMS.
    """
    import shutil
    import subprocess

    if not mp4_path.exists() or not shutil.which("ffmpeg"):
        return
    web_path = mp4_path.with_name("web_" + mp4_path.name)
    if web_path.exists():
        return
    # `-preset ultrafast` cuts the encode time ~3x at the cost of a slightly
    # larger output, which is fine for the dashboard preview (browser plays
    # H.264 of any quality). `-tune zerolatency` helps when the source is a
    # live MP4 with mp4v that ffmpeg has to parse on the fly.
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y",
         "-i", str(mp4_path),
         "-vcodec", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
         "-pix_fmt", "yuv420p",
         "-movflags", "+faststart",
         str(web_path)],
        check=False,
    )


def run_inference_job(job_id: str) -> None:
    """Entry point for BackgroundTasks — runs full inference pipeline for job."""
    job = store.get(job_id)
    if job is None:
        return

    job.status = JobStatus.RUNNING
    store.save(job)

    try:
        # Import here to avoid loading torch at module import time
        from src.inference.predict_video import run_video_inference

        cfg = _make_cfg(job)

        # Patch run_video_inference with a progress callback if supported.
        # We wrap it so we can track frame progress even without modifying the original.
        _run_with_progress(job, cfg)

        # Aggregate counts from CSV
        aggregate_predictions(job.predictions_csv, job.counts_json)

        # Pre-encode to H.264 so the browser <video> plays without a 30s wait
        _encode_for_web(job.output_video)

        job.status = JobStatus.DONE
        job.progress = 100

    except Exception as exc:
        job.status = JobStatus.ERROR
        job.error = str(exc)

    finally:
        job.finished_at = datetime.utcnow().isoformat()
        store.save(job)


def _run_with_progress(job: Job, cfg: dict) -> None:
    """Run inference while updating job.progress via frame count in the CSV.

    This avoids modifying predict_video.py: we poll the CSV row count
    periodically to approximate progress during long runs.

    Phase 2 (streaming): refactor predict_video.run_video_inference to accept
    an optional progress_callback(processed, total) and call it per frame.
    """
    import threading
    import time

    import cv2

    # Pre-read total frame count so we can show progress
    cap = cv2.VideoCapture(str(cfg["input_video"]))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    cap.release()
    job.total_frames = total
    frame_skip = cfg.get("frame_skip", 3)
    expected_output_frames = max(total // frame_skip, 1)

    # Run inference in a sub-thread so we can poll progress from the main thread
    from src.inference.predict_video import run_video_inference
    from src.app.models import get_classifier, get_yolo

    # Use the FastAPI-wide singletons so we don't reload models per job
    # (saves ~3 s for MobileViT + ~12 s for YOLO compile).
    model, device = get_classifier()
    preloaded = {"model": model, "device": device}
    if cfg.get("detector", "yolo") == "yolo":
        preloaded["yolo"] = get_yolo(cfg.get("yolo_weights", "yolov8n.pt"))

    exc_holder: list[Exception] = []

    def _worker():
        try:
            run_video_inference(cfg, preloaded=preloaded)
        except Exception as e:
            exc_holder.append(e)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    csv_path = Path(cfg["output_csv"])
    while t.is_alive():
        if csv_path.exists():
            try:
                # The CSV has one row per detection (many per frame), so we
                # count unique frame_idx values to estimate progress. We also
                # track the max frame_idx seen, which is a tighter bound when
                # some frames have no detections at all.
                seen_frames: set[str] = set()
                max_frame_idx = -1
                with open(csv_path) as f:
                    next(f, None)  # skip header
                    for line in f:
                        idx = line.split(",", 1)[0].strip()
                        if not idx:
                            continue
                        seen_frames.add(idx)
                        try:
                            max_frame_idx = max(max_frame_idx, int(idx))
                        except ValueError:
                            pass
                # frame_idx is the original-video frame number, so we divide
                # by total (not expected_output_frames) to track real progress.
                processed = max(len(seen_frames), (max_frame_idx + 1) // max(frame_skip, 1))
                job.processed_frames = min(processed, expected_output_frames)
                job.progress = int(job.processed_frames / expected_output_frames * 100)
            except OSError:
                pass
        time.sleep(2)

    t.join()
    if exc_holder:
        raise exc_holder[0]

    # GPU-hang safety net: if the worker thread vanished without raising
    # *and* without producing the per-detection CSV + annotated MP4 that
    # run_video_inference always writes on a clean run, the inference did
    # not actually complete. gfx1103 SIGABRT can take down the worker
    # thread mid-loop without surfacing a Python exception. Surface this
    # as an error so the dashboard does not advertise a ghost RUNNING job.
    # (counts.json is written later by aggregate_predictions in the caller,
    # so we deliberately don't check it here.)
    if not Path(cfg["output_csv"]).exists() or not Path(cfg["output_video"]).exists():
        raise RuntimeError(
            "inference worker exited without producing output CSV/MP4 "
            "(likely a GPU hang or hard crash)"
        )
