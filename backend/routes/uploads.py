"""Upload endpoints."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from shared.artifacts import ArtifactStore
from shared.schemas import UploadResponse
from shared.settings import settings

router = APIRouter(prefix="/api/v1/uploads", tags=["uploads"])


def _get_store() -> ArtifactStore:
    from backend.main import artifact_store

    return artifact_store


@router.post("", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="filename required")

    suffix = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    allowed = settings.video_extensions | settings.image_extensions
    if suffix and suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"unsupported extension: {suffix}")

    store = _get_store()
    upload_id = store.new_upload_id()
    dest = store.upload_path(upload_id, file.filename)
    size = 0
    max_bytes = settings.max_upload_mb * 1024 * 1024

    with dest.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                dest.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail=f"file exceeds {settings.max_upload_mb}MB")
            out.write(chunk)

    return UploadResponse(
        upload_id=upload_id,
        filename=file.filename,
        path=store.relative(dest),
        size_bytes=size,
    )
