"""Pre-flight checks before starting a job."""

from __future__ import annotations

from pathlib import Path

from inference.engine import resolve_model_path
from shared.settings import settings


class ValidationError(Exception):
    pass


def validate_model(model_name: str) -> Path:
    path = resolve_model_path(model_name)
    if not path.is_file():
        raise ValidationError(
            f"Model weights not found: {model_name} (looked in {settings.detection_model_dir})"
        )
    return path


def validate_source(path: Path) -> str:
    if not path.is_file():
        raise ValidationError(f"Source file not found: {path}")
    suffix = path.suffix.lower()
    if suffix in settings.video_extensions:
        return "video"
    if suffix in settings.image_extensions:
        return "image"
    raise ValidationError(f"Unsupported file type: {suffix}")
