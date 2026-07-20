"""Import a ZIP of images into MLAir as a dataset version (CV UI bulk upload)."""

from __future__ import annotations

import csv
import logging
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

from mlair_adapter.dataset_client import MANIFEST_COLUMNS, DatasetClient
from shared.artifacts import ArtifactStore
from shared.job_store import job_store
from shared.schemas import JobCreate, JobStatus, SourceType
from shared.settings import settings

logger = logging.getLogger(__name__)

_IMAGE_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif", ".heic", ".heif"}
)


def _safe_zip_member(name: str) -> bool:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        return False
    return path.suffix.lower() in _IMAGE_SUFFIXES


def _normalize_frame_name(index: int, suffix: str) -> str:
    ext = suffix.lower()
    if ext == ".jpeg":
        ext = ".jpg"
    return f"{index:06d}{ext}"


def extract_zip_images(zip_path: Path, frames_dir: Path) -> list[Path]:
    """Extract image members from ZIP into ``frames_dir``; return written paths."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    idx = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        infos = [i for i in zf.infolist() if (not i.is_dir()) and _safe_zip_member(i.filename)]
        if not infos:
            return []
        if len(infos) > settings.dataset_zip_max_images:
            raise ValueError(
                f"ZIP contains {len(infos)} images; max is {settings.dataset_zip_max_images} "
                "(CV_DATASET_ZIP_MAX_IMAGES)"
            )
        total_unzipped = sum(int(i.file_size or 0) for i in infos)
        limit_bytes = int(settings.dataset_zip_max_unzipped_mb) * 1024 * 1024
        if total_unzipped > limit_bytes:
            raise ValueError(
                f"ZIP expands to ~{total_unzipped / (1024 * 1024):.1f}MB; max is "
                f"{settings.dataset_zip_max_unzipped_mb}MB (CV_DATASET_ZIP_MAX_UNZIPPED_MB)"
            )
        for info in infos:
            suffix = Path(info.filename).suffix.lower()
            if suffix == ".jpeg":
                suffix = ".jpg"
            dest_name = _normalize_frame_name(idx, suffix)
            dest = frames_dir / dest_name
            with zf.open(info, "r") as src, dest.open("wb") as out:
                shutil.copyfileobj(src, out)
            written.append(dest)
            idx += 1

    return written


def write_manifest_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(MANIFEST_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def build_manifest_rows(import_job_id: str, frame_paths: list[Path], *, source_label: str) -> list[dict[str, str]]:
    base = settings.api_base_url.rstrip("/")
    rows: list[dict[str, str]] = []
    for path in frame_paths:
        rows.append(
            {
                "image_uri": f"{base}/api/v1/jobs/{import_job_id}/frames/{path.name}",
                "frame_index": path.stem,
                "job_id": import_job_id,
                "source_file": source_label,
                "artifact_path": path.name,
            }
        )
    return rows


def import_zip_to_mlair_dataset(
    zip_path: Path,
    *,
    dataset_name: str,
    store: ArtifactStore | None = None,
    source_label: str | None = None,
) -> dict:
    """
    Extract images, stage under ``artifacts/jobs/{id}/frames/``, upload manifest to MLAir.

    Returns summary with ``import_job_id``, ``dataset_id``, ``dataset_version_id``, ``image_count``.
    """
    name = re.sub(r"[^\w.-]+", "-", str(dataset_name or "").strip()).strip("-")
    if not name:
        raise ValueError("dataset_name is required")

    client = DatasetClient()
    if not client.enabled:
        raise RuntimeError("MLAir is not configured (set CV_MLAIR_API_URL and CV_MLAIR_TOKEN)")

    store = store or ArtifactStore()
    label = source_label or Path(zip_path).name

    job = job_store.create(
        JobCreate(
            source_type=SourceType.IMAGE,
            model_name=settings.default_model,
            confidence=0.5,
        )
    )
    import_job_id = job.id
    frames_dir = store.job_layout(import_job_id)["frames"]

    try:
        frame_paths = extract_zip_images(zip_path, frames_dir)
        if not frame_paths:
            raise ValueError("ZIP contains no supported images (.jpg, .png, .webp, …)")

        rows = build_manifest_rows(import_job_id, frame_paths, source_label=label)
        manifest_path = store.job_layout(import_job_id)["base"] / "dataset_import_manifest.csv"
        write_manifest_csv(rows, manifest_path)

        upload = client.upload_manifest_csv(manifest_path, name)
        ds_id = str(upload.get("dataset_id") or "")
        ver_id = str(upload.get("dataset_version_id") or upload.get("version_id") or "")

        job_store.update(
            import_job_id,
            status=JobStatus.COMPLETED,
            progress=1.0,
            message=f"dataset zip import: {len(frame_paths)} images → {name}",
            mlair_dataset_id=ds_id or None,
            mlair_dataset_version_id=ver_id or None,
            artifact_manifest={
                "frames_dir": store.relative(frames_dir),
                "frame_count": len(frame_paths),
                "frames_extracted": len(frame_paths),
            },
        )

        logger.info(
            "dataset zip import job=%s dataset=%s images=%s version=%s",
            import_job_id,
            name,
            len(frame_paths),
            ver_id,
        )
        return {
            "ok": True,
            "import_job_id": import_job_id,
            "dataset_name": name,
            "image_count": len(frame_paths),
            "dataset_id": ds_id,
            "dataset_version_id": ver_id,
            "manifest_path": str(manifest_path),
            "upload": upload,
        }
    except Exception:
        job_store.update(
            import_job_id,
            status=JobStatus.FAILED,
            message="dataset zip import failed",
            error="import failed",
        )
        raise


def import_zip_bytes_to_mlair(
    data: bytes,
    *,
    dataset_name: str,
    filename: str = "upload.zip",
    store: ArtifactStore | None = None,
) -> dict:
    with tempfile.TemporaryDirectory(prefix="cv-dataset-zip-") as tmp:
        zip_path = Path(tmp) / Path(filename).name
        zip_path.write_bytes(data)
        return import_zip_to_mlair_dataset(
            zip_path,
            dataset_name=dataset_name,
            store=store,
            source_label=filename,
        )
