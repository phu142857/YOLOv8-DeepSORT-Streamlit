"""Resolve YOLO weights path — local filename or MLAir registry."""

from __future__ import annotations

import logging
from pathlib import Path

import config
from mlair_adapter.model_client import ModelClient
from shared.settings import settings
from shared.weights_catalog import (
    PRETRAINED_VERSION,
    find_weights_in_dir,
    parse_model_spec,
    resolve_local_weights,
    version_dir,
)

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

    row = client.resolve_version_row(model_id, stage=stage)
    if row is None or not row.get("artifact_uri"):
        raise FileNotFoundError(f"no {stage} artifact for model {model_id}")

    version = int(row.get("version") or 0)
    dest = cache_path_for_model(model_id, version)
    if dest.is_file():
        return dest

    client.download_artifact(str(row["artifact_uri"]), dest)
    logger.info("Cached registry weights model=%s v%s -> %s", model_id, version, dest)
    return dest


def resolve_inference_model_path(model_name: str) -> Path:
    """
    Weights for upload/Execution (Vehicle Detection).

    ``{model}/base`` may be an MLAir fine-tune with poor/zero detections.
    Prefer ``{model}/pretrained/weights.pt`` (COCO), seeded before Hub sync overwrites base.
    """
    if is_registry_model(model_name):
        return ensure_registry_weights(registry_model_id(model_name))

    parsed = parse_model_spec(model_name)
    if parsed:
        model, version = parsed
        if version == "base":
            for root in (settings.detection_model_dir, Path(config.DETECTION_MODEL_DIR)):
                root = Path(root).resolve()
                try:
                    pre = find_weights_in_dir(version_dir(root, model, PRETRAINED_VERSION))
                except ValueError:
                    pre = None
                if pre is not None:
                    logger.debug("inference weights: %s (pretrained)", pre)
                    return pre

    return resolve_model_path(model_name)


def resolve_model_path(model_name: str) -> Path:
    if is_registry_model(model_name):
        return ensure_registry_weights(registry_model_id(model_name))

    roots = [settings.detection_model_dir, Path(config.DETECTION_MODEL_DIR)]
    seen: set[Path] = set()
    last_exc: FileNotFoundError | None = None
    for root in roots:
        root = Path(root).resolve()
        if root in seen:
            continue
        seen.add(root)
        try:
            return resolve_local_weights(root, model_name)
        except FileNotFoundError as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise FileNotFoundError(f"model not found: {model_name}")
