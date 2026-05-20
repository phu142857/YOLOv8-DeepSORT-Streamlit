"""Resolve YOLO weights path — local filename or MLAir registry."""

from __future__ import annotations

import logging
from pathlib import Path

import config
from mlair_adapter.model_client import ModelClient
from shared.settings import settings

logger = logging.getLogger(__name__)

REGISTRY_PREFIX = "registry:"


def is_registry_model(model_name: str) -> bool:
    return model_name.startswith(REGISTRY_PREFIX)


def registry_model_id(model_name: str) -> str:
    return model_name[len(REGISTRY_PREFIX) :]


def cache_path_for_model(model_id: str, version: int | None = None) -> Path:
    cache_root = settings.mlair_weights_cache_dir
    if version is not None:
        return cache_root / model_id / f"v{version}.pt"
    return cache_root / model_id / "production.pt"


def ensure_registry_weights(model_id: str, *, stage: str = "production") -> Path:
    client = ModelClient()
    if not client.enabled:
        raise RuntimeError("MLAir not configured for registry models")

    row = client.find_version_by_stage(model_id, stage)
    if row is None:
        row = client.find_latest_version(model_id)
    if row is None or not row.get("artifact_uri"):
        raise FileNotFoundError(f"no {stage} artifact for model {model_id}")

    version = int(row.get("version") or 0)
    dest = cache_path_for_model(model_id, version)
    if dest.is_file():
        return dest

    client.download_artifact(str(row["artifact_uri"]), dest)
    logger.info("Cached registry weights model=%s v%s -> %s", model_id, version, dest)
    return dest


def resolve_model_path(model_name: str) -> Path:
    if is_registry_model(model_name):
        return ensure_registry_weights(registry_model_id(model_name))

    path = Path(model_name)
    if path.exists():
        return path
    candidate = settings.detection_model_dir / model_name
    if candidate.exists():
        return candidate
    legacy = Path(config.DETECTION_MODEL_DIR) / model_name
    if legacy.exists():
        return legacy
    return candidate
