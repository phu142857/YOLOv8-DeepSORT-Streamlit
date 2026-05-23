"""Job lifecycle endpoints."""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException

from backend.orchestrator import run_job_pipeline
from backend.services.mlair_staging import stage_mlair_version
from inference.validation import ValidationError, validate_model
from shared.artifacts import ArtifactStore
from shared.job_store import job_store
from shared.schemas import JobCreate, JobResponse, JobStatus, SourceType
from shared.settings import settings

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


def _pick_job_source_file(source_dir: Path) -> Path | None:
    """Prefer image/video in source/; ignore mlair_pull.json and other sidecar files."""
    media: list[Path] = []
    other: list[Path] = []
    for f in source_dir.iterdir():
        if not f.is_file() or f.name == "mlair_pull.json":
            continue
        suf = f.suffix.lower()
        if suf in settings.video_extensions or suf in settings.image_extensions:
            media.append(f)
        else:
            other.append(f)
    if media:
        return sorted(media)[0]
    return sorted(other)[0] if other else None


def _get_store() -> ArtifactStore:
    from backend.main import artifact_store

    return artifact_store


@router.post("", response_model=JobResponse)
def create_job(spec: JobCreate) -> JobResponse:
    try:
        validate_model(spec.model_name)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if spec.mlair_dataset_version_id and spec.upload_id:
        raise HTTPException(status_code=400, detail="use either upload_id or mlair_dataset_version_id")

    job = job_store.create(spec)
    store = _get_store()

    if spec.mlair_dataset_version_id:
        try:
            staged = stage_mlair_version(job.id, spec.mlair_dataset_version_id, store)
            updated = job_store.update(
                job.id,
                source_type=SourceType.MLAIR,
                message=f"staged from MLAir version {spec.mlair_dataset_version_id}",
                source_filename=staged.name,
                mlair_dataset_version_id=spec.mlair_dataset_version_id,
            )
            return updated or job
        except Exception as exc:
            job_store.update(job.id, status=JobStatus.FAILED, error=str(exc), message="mlair staging failed")
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    if spec.upload_id:
        upload_dir = store.uploads_dir / spec.upload_id
        if not upload_dir.exists():
            raise HTTPException(status_code=404, detail="upload not found")
        files = [f for f in upload_dir.iterdir() if f.is_file()]
        if not files:
            raise HTTPException(status_code=400, detail="upload empty")
        source = store.save_upload_to_job(job.id, files[0])
        updated = job_store.update(
            job.id,
            message=f"source staged: {source.name}",
            source_filename=source.name,
        )
        return updated or job

    refreshed = job_store.get(job.id)
    return refreshed or job


@router.get("", response_model=list[JobResponse])
def list_jobs(limit: int = 50) -> list[JobResponse]:
    return job_store.list_jobs(limit=limit)


@router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: str) -> JobResponse:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@router.post("/{job_id}/start", response_model=JobResponse)
def start_job(job_id: str, background_tasks: BackgroundTasks, force: bool = False) -> JobResponse:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    if job.status == JobStatus.RUNNING:
        raise HTTPException(status_code=409, detail="job already running")
    if job.status == JobStatus.COMPLETED and not force:
        raise HTTPException(status_code=409, detail="job already completed (use ?force=true to re-run)")
    if job.status == JobStatus.QUEUED:
        return job

    store = _get_store()
    layout = store.job_layout(job_id)

    if force and job.status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}:
        for key in ("output", "detections", "tracking", "frames", "aggregates", "steps", "logs"):
            sub = layout[key]
            if sub.exists():
                shutil.rmtree(sub)
            sub.mkdir(parents=True, exist_ok=True)

    source_path = _pick_job_source_file(layout["source"])
    if source_path is None:
        raise HTTPException(status_code=400, detail="no source file — upload first")
    try:
        validate_model(job.model_name)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job_store.update(
        job_id,
        status=JobStatus.QUEUED,
        message="queued",
        progress=0.0,
        error=None,
        current_step="",
    )

    def _progress(p: float, msg: str) -> None:
        job_store.update(job_id, progress=min(0.98, p), message=msg)

    background_tasks.add_task(
        run_job_pipeline,
        job_id,
        source_path,
        job.model_name,
        job.confidence,
        store,
        _progress,
    )
    result = job_store.get(job_id)
    return result or job


@router.post("/{job_id}/cancel", response_model=JobResponse)
def cancel_job(job_id: str) -> JobResponse:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}:
        return job
    job_store.request_cancel(job_id)
    result = job_store.get(job_id)
    return result or job
