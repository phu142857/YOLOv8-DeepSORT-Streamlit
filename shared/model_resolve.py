"""Resolve YOLO weights path — local filename or MLAir registry."""

from __future__ import annotations

import logging
from pathlib import Path

import config
from mlair_adapter.model_client import ModelClient
from shared.detection_weights_bootstrap import ensure_weights_for_spec
from shared.settings import settings
from shared.weights_catalog import (
    PRETRAINED_VERSION,
    find_weights_in_dir,
    parse_model_spec,
    resolve_local_weights,
    version_dir,
    weights_files_equivalent,
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


def ensure_registry_weights(
    model_id: str,
    *,
    stage: str = "production",
    version: int | None = None,
) -> Path:
    client = ModelClient()
    if not client.enabled:
        raise RuntimeError("MLAir not configured for registry models")

    if version is not None:
        version = int(version)
        dest = cache_path_for_model(model_id, version)
        if dest.is_file():
            return dest
        row = client.get_version(model_id, version)
        artifact_uri = str(row.get("artifact_uri") or "")
        if not artifact_uri:
            raise FileNotFoundError(f"no artifact for model {model_id} version {version}")
        client.download_artifact(artifact_uri, dest)
        logger.info("Cached registry weights model=%s v%s -> %s", model_id, version, dest)
        return dest

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


def _resolve_base_spec_weights(model: str, root: Path) -> Path | None:
    """``{model}/base`` = MLAir production mirror on disk (``base/`` == ``production/``)."""
    try:
        base = find_weights_in_dir(version_dir(root, model, "base"))
        pre = find_weights_in_dir(version_dir(root, model, PRETRAINED_VERSION))
    except ValueError:
        return None

    if base is None:
        return pre

    if pre is not None and weights_files_equivalent(base, pre):
        logger.debug("inference weights: %s (base, same bytes as pretrained)", base)
    else:
        prod = find_weights_in_dir(version_dir(root, model, "production"))
        if prod is not None and weights_files_equivalent(base, prod):
            logger.info("inference weights: %s (MLAir production / base)", base)
        elif pre is not None:
            logger.info(
                "inference weights: %s (base; differs from pretrained %s)",
                base,
                pre,
            )
        else:
            logger.info("inference weights: %s (base)", base)
    return base


def resolve_inference_model_path(model_name: str) -> Path:
    """
    Weights for upload/Execution (Vehicle Detection).

    ``{model}/base`` follows MLAir production on ``base/weights.pt`` (see model sync).
    ``{model}/v2`` etc. resolve to explicit version folders when selected in the UI.
    """
    if is_registry_model(model_name):
        return ensure_registry_weights(registry_model_id(model_name))

    parsed = parse_model_spec(model_name)
    if parsed:
        model, version = parsed
        if version == "base":
            for root in (settings.detection_model_dir, Path(config.DETECTION_MODEL_DIR)):
                root = Path(root).resolve()
                path = _resolve_base_spec_weights(model, root)
                if path is not None:
                    return path

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
            try:
                return ensure_weights_for_spec(model_name, root)
            except FileNotFoundError:
                pass
    if last_exc is not None:
        raise last_exc
    raise FileNotFoundError(f"model not found: {model_name}")
