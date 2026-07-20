"""YOLO + DeepSORT inference (framework-agnostic)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

import config
from shared.model_resolve import resolve_inference_model_path as _resolve_model_path


def _model_cache_key(path: Path) -> str:
    """Invalidate in-process YOLO cache when checkpoint file is replaced (e.g. Hub promote v2)."""
    resolved = path.resolve()
    st = resolved.stat()
    return f"{resolved}:{st.st_mtime_ns}:{st.st_size}"


@lru_cache(maxsize=8)
def _load_model_cached(cache_key: str, resolved_path: str) -> YOLO:
    return YOLO(resolved_path)


def load_model(model_path: str) -> YOLO:
    path = Path(model_path)
    return _load_model_cached(_model_cache_key(path), str(path.resolve()))


def resolve_model_path(model_name: str) -> Path:
    return _resolve_model_path(model_name)


def predict_frame(
    model: YOLO, frame: np.ndarray, confidence: float
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, int], dict[str, int]]:
    """Run detection (+ tracking via DetectionPredictor) on one frame."""
    results = model.predict(frame, conf=confidence, verbose=False)
    result = results[0]
    plotted = result.plot()
    detections: list[dict[str, Any]] = []
    if result.boxes is not None and len(result.boxes):
        for box in result.boxes:
            cls_id = int(box.cls.item()) if box.cls is not None else -1
            name = result.names.get(cls_id, str(cls_id))
            detections.append(
                {
                    "class_id": cls_id,
                    "class_name": name,
                    "confidence": float(box.conf.item()) if box.conf is not None else None,
                    "xyxy": box.xyxy[0].tolist(),
                }
            )
    counters_in = dict(config.OBJECT_COUNTER1 or {})
    counters_out = dict(config.OBJECT_COUNTER or {})
    return plotted, detections, counters_in, counters_out
