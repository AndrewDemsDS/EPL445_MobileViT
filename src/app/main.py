"""FastAPI web dashboard for the MobileViT traffic classifier.

Run with:
    uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8000

Endpoints
---------
GET  /                          Serve index.html
POST /jobs                      Upload video, enqueue inference job
GET  /jobs                      List all jobs
GET  /jobs/{job_id}             Get job status + progress
GET  /jobs/{job_id}/counts      Per-class detection counts (JSON)
GET  /jobs/{job_id}/timeline    Per-frame vehicle count timeline (JSON)
POST /jobs/{job_id}/roi         Recompute counts inside a drawn ROI
GET  /jobs/{job_id}/video       Stream the annotated output video
GET  /jobs/{job_id}/frame       First frame of input video (for ROI canvas)
GET  /rtsp/streams              List active RTSP streams (stub)
POST /rtsp/streams              Start an RTSP stream job (stub)
"""

from __future__ import annotations

import json
import mimetypes
from pathlib import Path

import cv2
from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.app.analytics import density_timeline, roi_counts
from src.app.inference_worker import run_inference_job
from src.app.jobs import store
from src.app import rtsp as rtsp_module

app = FastAPI(title="MobileViT Traffic Dashboard", version="1.0")

# ── Static files ─────────────────────────────────────────────────
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


# ── Job endpoints ─────────────────────────────────────────────────

@app.post("/jobs", status_code=202)
async def create_job(file: UploadFile, background_tasks: BackgroundTasks):
    """Upload a video file and start inference in the background."""
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(400, "Only video files are accepted (.mp4 .avi .mov .mkv)")

    job = store.create()
    # Save uploaded video
    content = await file.read()
    job.input_video.write_bytes(content)

    background_tasks.add_task(run_inference_job, job.job_id)
    return {"job_id": job.job_id, "status": job.status.value}


@app.get("/jobs")
def list_jobs():
    return store.all()


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job.to_dict()


@app.get("/jobs/{job_id}/counts")
def get_counts(job_id: str):
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if not job.counts_json.exists():
        raise HTTPException(404, "Results not ready yet")
    return json.loads(job.counts_json.read_text())


@app.get("/jobs/{job_id}/timeline")
def get_timeline(job_id: str):
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if not job.predictions_csv.exists():
        raise HTTPException(404, "Results not ready yet")
    return density_timeline(job.predictions_csv)


@app.post("/jobs/{job_id}/roi")
def apply_roi(job_id: str, x: int, y: int, w: int, h: int):
    """Return counts filtered to the given ROI rectangle (no re-inference)."""
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if not job.predictions_csv.exists():
        raise HTTPException(404, "Results not ready yet")
    return roi_counts(job.predictions_csv, x, y, w, h)


@app.get("/jobs/{job_id}/video")
def get_video(job_id: str):
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if not job.output_video.exists():
        raise HTTPException(404, "Video not ready yet")
    return FileResponse(
        str(job.output_video),
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes"},
    )


@app.get("/jobs/{job_id}/frame")
def get_first_frame(job_id: str):
    """Return the first frame of the INPUT video as JPEG (used for ROI canvas)."""
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if not job.input_video.exists():
        raise HTTPException(404, "Input video not found")

    cap = cv2.VideoCapture(str(job.input_video))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise HTTPException(500, "Could not read frame from video")

    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise HTTPException(500, "Could not encode frame")

    return StreamingResponse(iter([buf.tobytes()]), media_type="image/jpeg")


# ── Dev / test helpers ───────────────────────────────────────────

@app.post("/dev/seed", include_in_schema=False)
def dev_seed():
    """Register a pre-computed job from existing outputs (dev/test only)."""
    import shutil
    from src.app.jobs import JobStatus
    from datetime import datetime

    job = store.create()
    src_video   = Path("outputs/predictions/annotated_output.mp4")
    src_csv     = Path("outputs/predictions/frame_predictions.csv")
    src_counts  = Path("outputs/predictions/class_counts.json")
    src_input   = Path("data/raw/sample_traffic.mp4")

    if src_input.exists():
        shutil.copy(src_input, job.input_video)
    if src_video.exists():
        shutil.copy(src_video, job.output_video)
    if src_csv.exists():
        shutil.copy(src_csv, job.predictions_csv)
    if src_counts.exists():
        shutil.copy(src_counts, job.counts_json)
        # Augment with synthetic tracking counts so the unique-vehicles pill renders.
        # Real tracking runs populate these via aggregate_counts; this is dev-only.
        try:
            data = json.loads(job.counts_json.read_text())
            if "unique_vehicles_by_class" not in data and "class_counts" in data:
                # Estimate unique ~= one-third of detections (heuristic for demo).
                cc = data["class_counts"]
                data["unique_vehicles_by_class"] = {k: max(1, round(v / 3)) for k, v in cc.items()}
                data["unique_vehicles_total"] = sum(data["unique_vehicles_by_class"].values())
                job.counts_json.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    job.status = JobStatus.DONE
    job.progress = 100
    job.total_frames = 1800
    job.processed_frames = 600
    job.finished_at = datetime.utcnow().isoformat()
    return job.to_dict()


# ── RTSP endpoints (stub) ─────────────────────────────────────────

@app.get("/rtsp/streams")
def list_rtsp_streams():
    return rtsp_module.list_streams()


@app.post("/rtsp/streams", status_code=501)
def start_rtsp_stream(url: str):
    return rtsp_module.start_stream(url)
