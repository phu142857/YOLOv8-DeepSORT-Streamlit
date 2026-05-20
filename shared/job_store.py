"""Job registry — in-memory with optional filesystem persistence across API restarts."""

from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shared.artifacts import ArtifactStore
from shared.schemas import JobCreate, JobResponse, JobStatus
from shared.settings import settings

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class JobStore:
    def __init__(self, store: ArtifactStore | None = None) -> None:
        self._artifact_store = store or ArtifactStore()
        self._jobs: dict[str, JobResponse] = {}
        self._lock = threading.Lock()
        self._cancel_flags: set[str] = set()
        if settings.persist_jobs:
            self._load_from_disk()

    def _job_record_path(self, job_id: str) -> Path:
        return self._artifact_store.job_dir(job_id) / "job.json"

    def _persist(self, job: JobResponse) -> None:
        if not settings.persist_jobs:
            return
        self._artifact_store.write_json(self._job_record_path(job.id), job.model_dump(mode="json"))

    def _load_from_disk(self) -> None:
        jobs_root = self._artifact_store.root / "jobs"
        if not jobs_root.is_dir():
            return
        loaded = 0
        for job_dir in jobs_root.iterdir():
            if not job_dir.is_dir():
                continue
            record = job_dir / "job.json"
            if not record.exists():
                continue
            try:
                with record.open(encoding="utf-8") as f:
                    data = json.load(f)
                data["created_at"] = _parse_dt(data["created_at"])
                data["updated_at"] = _parse_dt(data["updated_at"])
                job = JobResponse(**data)
                self._jobs[job.id] = job
                loaded += 1
            except Exception as exc:
                logger.warning("Skipping corrupt job record %s: %s", record, exc)
        if loaded:
            logger.info("Restored %d job(s) from disk", loaded)

    def request_cancel(self, job_id: str) -> bool:
        with self._lock:
            self._cancel_flags.add(job_id)
        job = self.get(job_id)
        if job and job.status in {JobStatus.PENDING, JobStatus.QUEUED, JobStatus.RUNNING}:
            self.update(job_id, status=JobStatus.CANCELLED, message="cancelled by user")
            return True
        return False

    def is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            if job_id in self._cancel_flags:
                return True
        job = self.get(job_id)
        return job is not None and job.status == JobStatus.CANCELLED

    def create(self, spec: JobCreate) -> JobResponse:
        now = _utc_now()
        job = JobResponse(
            id=str(uuid.uuid4()),
            status=JobStatus.PENDING,
            source_type=spec.source_type,
            model_name=spec.model_name,
            confidence=spec.confidence,
            mlair_dataset_id=spec.mlair_dataset_id,
            mlair_dataset_version_id=spec.mlair_dataset_version_id,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._jobs[job.id] = job
        self._artifact_store.job_layout(job.id)
        self._persist(job)
        return job

    def get(self, job_id: str) -> JobResponse | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                return job.model_copy()
        if settings.persist_jobs:
            record = self._job_record_path(job_id)
            if record.exists():
                try:
                    with record.open(encoding="utf-8") as f:
                        data = json.load(f)
                    data["created_at"] = _parse_dt(data["created_at"])
                    data["updated_at"] = _parse_dt(data["updated_at"])
                    job = JobResponse(**data)
                    with self._lock:
                        self._jobs[job_id] = job
                    return job.model_copy()
                except Exception as exc:
                    logger.warning("Failed to load job %s from disk: %s", job_id, exc)
        return None

    def list_jobs(self, limit: int = 50) -> list[JobResponse]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return [j.model_copy() for j in jobs[:limit]]

    def update(
        self,
        job_id: str,
        *,
        status: JobStatus | None = None,
        progress: float | None = None,
        message: str | None = None,
        current_step: str | None = None,
        source_filename: str | None = None,
        error: str | None = None,
        artifact_manifest: dict[str, Any] | None = None,
        counters_in: dict[str, int] | None = None,
        counters_out: dict[str, int] | None = None,
        mlair_dataset_id: str | None = None,
        mlair_dataset_version_id: str | None = None,
        mlair_readiness: dict[str, Any] | None = None,
        mlair_ingest: dict[str, Any] | None = None,
        mlair_training: dict[str, Any] | None = None,
        mlair_model_version: dict[str, Any] | None = None,
    ) -> JobResponse | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            data = job.model_dump()
            if status is not None:
                data["status"] = status
            if progress is not None:
                data["progress"] = max(0.0, min(1.0, progress))
            if message is not None:
                data["message"] = message
            if current_step is not None:
                data["current_step"] = current_step
            if source_filename is not None:
                data["source_filename"] = source_filename
            if error is not None:
                data["error"] = error
            if artifact_manifest is not None:
                data["artifact_manifest"] = artifact_manifest
            if counters_in is not None:
                data["counters_in"] = counters_in
            if counters_out is not None:
                data["counters_out"] = counters_out
            if mlair_dataset_id is not None:
                data["mlair_dataset_id"] = mlair_dataset_id
            if mlair_dataset_version_id is not None:
                data["mlair_dataset_version_id"] = mlair_dataset_version_id
            if mlair_readiness is not None:
                data["mlair_readiness"] = mlair_readiness
            if mlair_ingest is not None:
                data["mlair_ingest"] = mlair_ingest
            if mlair_training is not None:
                data["mlair_training"] = mlair_training
            if mlair_model_version is not None:
                data["mlair_model_version"] = mlair_model_version
            data["updated_at"] = _utc_now()
            updated = JobResponse(**data)
            self._jobs[job_id] = updated

        self._persist(updated)
        return updated.model_copy()


def init_job_store(artifact_store: ArtifactStore) -> "JobStore":
    global job_store  # noqa: PLW0603
    job_store = JobStore(artifact_store)
    return job_store


job_store = JobStore()
