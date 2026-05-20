"""Job lifecycle timeline API."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from shared.artifacts import ArtifactStore
from shared.job_store import job_store
from shared.lifecycle import build_job_lifecycle
from shared.schemas import JobLifecycleResponse

router = APIRouter(prefix="/api/v1/jobs", tags=["lifecycle"])


def _get_store() -> ArtifactStore:
    from backend.main import artifact_store

    return artifact_store


@router.get("/{job_id}/lifecycle", response_model=JobLifecycleResponse)
def get_job_lifecycle(job_id: str) -> JobLifecycleResponse:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return build_job_lifecycle(job, _get_store())
