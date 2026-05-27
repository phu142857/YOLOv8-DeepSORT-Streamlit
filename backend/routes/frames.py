"""Serve extracted frames (for MLAir manifest image_uri fetch)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter(prefix="/api/v1/jobs", tags=["frames"])


def _get_store():
    from backend.main import artifact_store

    return artifact_store


@router.get("/{job_id}/frames/{frame_name}")
def get_frame(job_id: str, frame_name: str) -> FileResponse:
    from shared.job_store import job_store

    safe_name = frame_name.replace("..", "").lstrip("/")
    store = _get_store()
    path = store.job_layout(job_id)["frames"] / safe_name
    if path.is_file():
        return FileResponse(path, media_type="image/jpeg", filename=safe_name)
    if job_store.get(job_id) is None:
        raise HTTPException(status_code=404, detail="job not found")
    raise HTTPException(status_code=404, detail="frame not found")
