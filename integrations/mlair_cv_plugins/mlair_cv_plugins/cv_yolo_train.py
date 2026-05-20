"""
cv_yolo_train — MLAir plugin contract (registry + validate only).

Execution theo framework MLAir:
- HTTP pipeline task → cv-api (mặc định, xem examples/mlair/pipelines/*.json)
- Hoặc external worker → scripts/mlair_cv_train_worker.py

Không implement train trong ml-air core / mlair_runner.
"""

from __future__ import annotations

from typing import Any


class CvYoloTrainPlugin:
    meta = {
        "name": "cv_yolo_train",
        "version": "0.1.0",
        "engine_version": "1.0.0",
        "inputs": {
            "dataset_version_id": "string",
            "model_id": "string",
            "dataset_id": "string",
            "artifact_uri": "string",
        },
        "outputs": {"checkpoint": "string", "metrics": "object"},
        "ui_schema": None,
        "lineage": {
            "inputs": ["dataset_version"],
            "outputs": ["model_checkpoint"],
        },
    }

    def validate(self, context: dict[str, Any]) -> bool:
        if not str(context.get("dataset_version_id") or "").strip():
            raise ValueError("dataset_version_id is required")
        if not str(context.get("model_id") or context.get("mlair_model_id") or "").strip():
            raise ValueError("model_id or mlair_model_id is required")
        return True
