"""Bulk dataset import (ZIP of images → MLAir dataset version)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from mlair_adapter.dataset_zip_import import import_zip_to_mlair_dataset
from shared.schemas import DatasetZipImportResponse
from shared.settings import settings

router = APIRouter(prefix="/api/v1/datasets", tags=["datasets"])


def _get_store():
    from backend.main import artifact_store

    return artifact_store


@router.post("/import-zip", response_model=DatasetZipImportResponse)
async def import_dataset_zip(
    file: UploadFile = File(...),
    dataset_name: str = Form(...),
) -> DatasetZipImportResponse:
    """
    Upload a ZIP of images; extract to job frames; create MLAir dataset version.

    Images are served from ``GET /api/v1/jobs/{import_job_id}/frames/...`` for train/prepare.
    """
    if not file.filename or not str(file.filename).lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="a .zip file is required")

    name = str(dataset_name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="dataset_name is required")

    max_bytes = settings.dataset_zip_max_mb * 1024 * 1024
    size = 0
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        zip_path = Path(tmp.name)
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                zip_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"ZIP exceeds {settings.dataset_zip_max_mb}MB (CV_DATASET_ZIP_MAX_MB)",
                )
            tmp.write(chunk)

    try:
        result = import_zip_to_mlair_dataset(
            zip_path,
            dataset_name=name,
            store=_get_store(),
            source_label=file.filename,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        zip_path.unlink(missing_ok=True)

    return DatasetZipImportResponse(
        ok=True,
        import_job_id=result["import_job_id"],
        dataset_name=result["dataset_name"],
        image_count=result["image_count"],
        dataset_id=result.get("dataset_id") or None,
        dataset_version_id=result.get("dataset_version_id") or None,
        hub_url=settings.mlair_hub_url.rstrip("/"),
    )
