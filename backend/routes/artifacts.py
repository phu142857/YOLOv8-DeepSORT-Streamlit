"""Artifact download and listing."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from shared.artifacts import ArtifactStore
from shared.schemas import JobArtifactsResponse

router = APIRouter(prefix="/api/v1/jobs", tags=["artifacts"])

ARTIFACT_TYPES = frozenset(
    {"processed", "detections", "tracking", "manifest", "export", "aggregates", "log"}
)


def _get_store() -> ArtifactStore:
    from backend.main import artifact_store

    return artifact_store


@router.get("/{job_id}/artifacts", response_model=JobArtifactsResponse)
def list_artifacts(job_id: str) -> JobArtifactsResponse:
    from shared.job_store import job_store

    if job_store.get(job_id) is None:
        raise HTTPException(status_code=404, detail="job not found")
    store = _get_store()
    return JobArtifactsResponse(job_id=job_id, artifacts=store.list_artifacts(job_id))


@router.get("/{job_id}/artifacts/{artifact_type}")
def download_artifact(job_id: str, artifact_type: str) -> FileResponse:
    from shared.job_store import job_store

    if job_store.get(job_id) is None:
        raise HTTPException(status_code=404, detail="job not found")
    if artifact_type not in ARTIFACT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"unknown artifact type; allowed: {sorted(ARTIFACT_TYPES)}",
        )

    store = _get_store()
    path = store.resolve_artifact(job_id, artifact_type)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="artifact not found")

    return FileResponse(path, filename=path.name)
