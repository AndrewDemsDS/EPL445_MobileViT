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
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y",
         "-i", str(mp4_path),
         "-vcodec", "libx264", "-pix_fmt", "yuv420p",
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

    exc_holder: list[Exception] = []

    def _worker():
        try:
            run_video_inference(cfg)
        except Exception as e:
            exc_holder.append(e)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    csv_path = Path(cfg["output_csv"])
    while t.is_alive():
        if csv_path.exists():
            try:
                # Count non-header lines written so far as a proxy for progress
                with open(csv_path) as f:
                    rows = sum(1 for _ in f) - 1  # subtract header
                job.processed_frames = min(rows, expected_output_frames)
                job.progress = int(job.processed_frames / expected_output_frames * 100)
            except OSError:
                pass
        time.sleep(2)

    t.join()
    if exc_holder:
        raise exc_holder[0]
