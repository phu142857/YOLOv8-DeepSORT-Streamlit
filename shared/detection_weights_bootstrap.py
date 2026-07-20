"""Ensure detection weights exist on disk before inference (S3 sync + Ultralytics COCO fallback)."""

from __future__ import annotations

import logging
from pathlib import Path

from shared.settings import settings
from shared.weights_catalog import (
    PRETRAINED_VERSION,
    find_weights_in_dir,
    parse_model_spec,
    resolve_local_weights,
    version_dir,
)

logger = logging.getLogger(__name__)


def list_bootstrap_model_names() -> list[str]:
    return _bootstrap_model_names()


def _bootstrap_model_names() -> list[str]:
    raw = settings.detection_bootstrap_models.strip()
    if not raw:
        return []
    return [m.strip() for m in raw.split(",") if m.strip()]


def seed_ultralytics_pretrained(model: str, root: Path | None = None) -> Path | None:
    """
    Download official COCO weights into ``{model}/pretrained`` and ``{model}/base``.

    Seeds ``pretrained/`` and ``base/`` with the same COCO bytes until Hub promote replaces ``base/``.
    """
    root = Path(root or settings.detection_model_dir)
    try:
        pre_dir = version_dir(root, model, PRETRAINED_VERSION)
        base_dir = version_dir(root, model, "base")
    except ValueError:
        return None

    pre_pt = pre_dir / "weights.pt"
    base_pt = base_dir / "weights.pt"
    if pre_pt.is_file() and base_pt.is_file():
        return pre_pt

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.warning("ultralytics not available; cannot seed %s", model)
        return None

    pre_dir.mkdir(parents=True, exist_ok=True)
    base_dir.mkdir(parents=True, exist_ok=True)
    logger.info("seeding Ultralytics COCO weights for %s", model)
    m = YOLO(f"{model}.pt")
    data = Path(m.ckpt_path).read_bytes()
    pre_pt.write_bytes(data)
    base_pt.write_bytes(data)
    logger.info("seeded %s -> %s (+ base)", model, pre_pt)
    return pre_pt


def ensure_model_weights(model: str, root: Path | None = None) -> Path | None:
    """
    Ensure ``weights/detection/{model}/base`` (or production) has a checkpoint.

    Order: existing local → S3 production manifest → Ultralytics pretrained seed.
    """
    root = Path(root or settings.detection_model_dir)
    for version in ("base", "production", "current"):
        try:
            weights = find_weights_in_dir(version_dir(root, model, version))
        except ValueError:
            weights = None
        if weights is not None:
            return weights

    if settings.s3_models_sync_on_startup:
        try:
            from shared.s3_model_store import s3_enabled, sync_model_from_s3

            if s3_enabled():
                synced = sync_model_from_s3(model, root)
                if synced is not None:
                    return synced
        except Exception:
            logger.exception("S3 sync failed for model %s", model)

    return seed_ultralytics_pretrained(model, root)


def ensure_weights_for_spec(spec: str, root: Path | None = None) -> Path:
    """Resolve spec or bootstrap missing canonical weights."""
    root = Path(root or settings.detection_model_dir)
    try:
        return resolve_local_weights(root, spec)
    except FileNotFoundError:
        pass

    parsed = parse_model_spec(spec)
    if parsed:
        model, version = parsed
        if version in ("base", "production", "current"):
            path = ensure_model_weights(model, root)
            if path is not None:
                return resolve_local_weights(root, spec)
        raise FileNotFoundError(f"no weights for spec {spec!r} under {root}")

    path = ensure_model_weights(spec.strip(), root)
    if path is None:
        raise FileNotFoundError(f"could not bootstrap weights for {spec!r}")
    return resolve_local_weights(root, spec)


def startup_ensure_detection_weights() -> dict[str, str]:
    """Called once at API startup — warm EFS / empty volumes."""
    root = Path(settings.detection_model_dir)
    root.mkdir(parents=True, exist_ok=True)
    results: dict[str, str] = {}
    for model in _bootstrap_model_names():
        try:
            path = ensure_model_weights(model, root)
            results[model] = str(path) if path else "missing"
        except Exception as exc:
            logger.exception("bootstrap failed for %s", model)
            results[model] = f"error:{exc}"
    return results
