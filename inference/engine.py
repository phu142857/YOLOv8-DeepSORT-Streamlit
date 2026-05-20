"""YOLO + DeepSORT inference (framework-agnostic)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

import config
from shared.settings import settings


@lru_cache(maxsize=4)
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def resolve_model_path(model_name: str) -> Path:
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
