"""Inspect local / registry checkpoints for CV inference debugging."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from shared.model_resolve import parse_model_spec, resolve_inference_model_path
from shared.settings import settings
from shared.weights_catalog import (
    PRETRAINED_VERSION,
    find_weights_in_dir,
    sha256_file,
    version_dir,
    weights_files_equivalent,
)

logger = logging.getLogger(__name__)


def _path_info(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    st = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": st.st_size,
        "sha256": sha256_file(path),
    }


def _yolo_metadata(weights_path: Path) -> dict[str, Any]:
    try:
        from ultralytics import YOLO

        model = YOLO(str(weights_path))
        names = getattr(model, "names", None) or {}
        if isinstance(names, dict):
            name_list = [names[i] for i in sorted(names)]
        elif isinstance(names, (list, tuple)):
            name_list = list(names)
        else:
            name_list = []
        return {"nc": len(name_list), "names": name_list[:20]}
    except Exception as exc:
        logger.debug("yolo metadata failed for %s: %s", weights_path, exc)
        return {"error": str(exc)}


def detection_model_diagnostics(spec: str) -> dict[str, Any]:
    """
    Compare base / production / pretrained / inference resolution for a model spec.

    Helps explain empty detections after Hub promote (wrong bytes on disk, collapsed fine-tune, etc.).
    """
    root = Path(settings.detection_model_dir)
    parsed = parse_model_spec(spec)
    model = parsed[0] if parsed else spec.split("/")[0]

    base = find_weights_in_dir(version_dir(root, model, "base"))
    production = find_weights_in_dir(version_dir(root, model, "production"))
    pretrained = find_weights_in_dir(version_dir(root, model, PRETRAINED_VERSION))

    version_paths: dict[str, Any] = {}
    model_dir = root / model
    if model_dir.is_dir():
        for child in sorted(model_dir.iterdir()):
            if child.is_dir() and child.name.startswith("v") and child.name[1:].isdigit():
                w = find_weights_in_dir(child)
                if w is not None:
                    version_paths[child.name] = _path_info(w)

    inference_path: Path | None = None
    inference_error: str | None = None
    try:
        inference_path = resolve_inference_model_path(spec)
    except Exception as exc:
        inference_error = str(exc)

    same_base_pretrained = (
        base is not None and pretrained is not None and weights_files_equivalent(base, pretrained)
    )
    same_base_production = (
        base is not None
        and production is not None
        and weights_files_equivalent(base, production)
    )

    out: dict[str, Any] = {
        "spec": spec,
        "model": model,
        "root": str(root.resolve()),
        "paths": {
            "base": _path_info(base),
            "production": _path_info(production),
            "pretrained": _path_info(pretrained),
            "version_folders": version_paths,
        },
        "inference_resolved": _path_info(inference_path) if inference_path else None,
        "inference_error": inference_error,
        "flags": {
            "base_equals_pretrained": same_base_pretrained,
            "base_equals_production": same_base_production,
            "inference_uses_pretrained_coco": bool(
                inference_path and pretrained and weights_files_equivalent(inference_path, pretrained)
            ),
            "inference_uses_production_base": bool(
                inference_path and base and weights_files_equivalent(inference_path, base)
            ),
        },
        "hints": [],
    }

    if inference_path and inference_path.is_file():
        out["yolo"] = _yolo_metadata(inference_path)

    if same_base_pretrained and not spec.endswith(f"/{PRETRAINED_VERSION}"):
        out["hints"].append(
            "base and pretrained are identical (COCO) — Hub fine-tune may not have been mirrored to disk."
        )
    if (
        base is not None
        and pretrained is not None
        and not same_base_pretrained
        and inference_path
        and weights_files_equivalent(inference_path, base)
    ):
        out["hints"].append(
            "Inference uses fine-tuned base/production (not COCO pretrained). "
            "Empty detections usually mean the promoted checkpoint is weak or trained without labels."
        )
    if inference_error:
        out["hints"].append(f"Inference resolution failed: {inference_error}")

    return out
