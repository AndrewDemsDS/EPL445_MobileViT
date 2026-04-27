"""In-memory job state machine for video inference jobs.

Each job transitions: queued → running → done | error
Job files live under outputs/jobs/{job_id}/
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


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

    @property
    def dir(self) -> Path:
        return Path("outputs/jobs") / self.job_id

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
        }


class JobStore:
    """Thread-safe in-memory job registry."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}

    def create(self) -> Job:
        job_id = str(uuid.uuid4())[:8]
        job = Job(job_id=job_id)
        job.dir.mkdir(parents=True, exist_ok=True)
        self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def all(self) -> list[dict]:
        return [j.to_dict() for j in sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)]


# Module-level singleton shared across the FastAPI app
store = JobStore()
