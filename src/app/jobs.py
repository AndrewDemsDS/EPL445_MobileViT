"""Job state machine for video inference jobs, persisted to disk.

Each job transitions: queued → running → done | error
Job files live under outputs/jobs/{job_id}/, with a job.json sidecar so the
Past Jobs table survives a uvicorn restart and demos don't have to re-run
inference on existing clips.
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


JOBS_ROOT = Path("outputs/jobs")


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


@dataclass
class Job:
    job_id: str
    status: JobStatus = JobStatus.QUEUED
    progress: int = 0          # 0-100
    total_frames: int = 0
    processed_frames: int = 0
    error: str | None = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    finished_at: str | None = None
    detector: str = "yolo"     # "yolo" or "sliding", set by the upload endpoint

    @property
    def dir(self) -> Path:
        return JOBS_ROOT / self.job_id

    @property
    def input_video(self) -> Path:
        return self.dir / "input.mp4"

    @property
    def output_video(self) -> Path:
        return self.dir / "output.mp4"

    @property
    def predictions_csv(self) -> Path:
        return self.dir / "frame_predictions.csv"

    @property
    def counts_json(self) -> Path:
        return self.dir / "class_counts.json"

    @property
    def sidecar(self) -> Path:
        return self.dir / "job.json"

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "progress": self.progress,
            "total_frames": self.total_frames,
            "processed_frames": self.processed_frames,
            "error": self.error,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "detector": self.detector,
        }

    def save(self) -> None:
        """Persist current state to outputs/jobs/{id}/job.json (atomic write)."""
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.sidecar.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2))
        tmp.replace(self.sidecar)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Job":
        return cls(
            job_id=d["job_id"],
            status=JobStatus(d.get("status", "queued")),
            progress=int(d.get("progress", 0)),
            total_frames=int(d.get("total_frames", 0)),
            processed_frames=int(d.get("processed_frames", 0)),
            error=d.get("error"),
            created_at=d.get("created_at", datetime.utcnow().isoformat()),
            finished_at=d.get("finished_at"),
            detector=d.get("detector", "yolo"),
        )


class JobStore:
    """Thread-safe job registry, backed by per-job JSON sidecars on disk."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        """Rehydrate jobs from outputs/jobs/.

        Two sources, in order of preference:
          1. A job.json sidecar (written by save() on every state change).
          2. A bare job dir with output.mp4 + frame_predictions.csv from a
             pre-sidecar run — synthesised as DONE so demos don't lose history.

        Jobs that were RUNNING when the process died are marked ERROR so the
        UI doesn't advertise ghost jobs after a restart.
        """
        if not JOBS_ROOT.exists():
            return
        seen: set[str] = set()
        for sidecar in JOBS_ROOT.glob("*/job.json"):
            try:
                data = json.loads(sidecar.read_text())
                job = Job.from_dict(data)
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
            if job.status in (JobStatus.RUNNING, JobStatus.QUEUED):
                job.status = JobStatus.ERROR
                job.error = job.error or "interrupted by server restart"
                job.finished_at = job.finished_at or datetime.utcnow().isoformat()
                try:
                    job.save()
                except OSError:
                    pass
            self._jobs[job.job_id] = job
            seen.add(job.job_id)

        for job_dir in JOBS_ROOT.iterdir():
            if not job_dir.is_dir() or job_dir.name in seen:
                continue
            output_mp4 = job_dir / "output.mp4"
            csv_path = job_dir / "frame_predictions.csv"
            if not (output_mp4.exists() and csv_path.exists()):
                continue
            mtime = datetime.utcfromtimestamp(output_mp4.stat().st_mtime).isoformat()
            job = Job(
                job_id=job_dir.name,
                status=JobStatus.DONE,
                progress=100,
                created_at=mtime,
                finished_at=mtime,
            )
            try:
                job.save()
            except OSError:
                pass
            self._jobs[job.job_id] = job

    def create(self) -> Job:
        with self._lock:
            job_id = str(uuid.uuid4())[:8]
            job = Job(job_id=job_id)
            job.dir.mkdir(parents=True, exist_ok=True)
            self._jobs[job_id] = job
        job.save()
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def all(self) -> list[dict]:
        return [j.to_dict() for j in sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)]

    def save(self, job: Job) -> None:
        """Persist a job's current state. Safe to call from worker threads."""
        try:
            job.save()
        except OSError:
            pass


# Module-level singleton shared across the FastAPI app
store = JobStore()
