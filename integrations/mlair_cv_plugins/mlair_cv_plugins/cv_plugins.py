"""MLAir plugin contracts for CV lifecycle DAG (validate-only; execution in cv worker)."""

from __future__ import annotations

from typing import Any


def _require(context: dict[str, Any], *keys: str) -> None:
    for key in keys:
        if key == "model_id":
            if not str(context.get("model_id") or context.get("mlair_model_id") or "").strip():
                raise ValueError("model_id or mlair_model_id is required")
            continue
        if not str(context.get(key) or "").strip():
            raise ValueError(f"{key} is required")


class CvYoloPreparePlugin:
    meta = {
        "name": "cv_yolo_prepare",
        "version": "0.2.0",
        "engine_version": "1.0.0",
        "inputs": {"dataset_version_id": "string", "model_id": "string"},
        "outputs": {"data_yaml": "string", "train_images": "number"},
        "ui_schema": None,
        "lineage": {"inputs": ["dataset_version"], "outputs": ["yolo_dataset"]},
    }

    def validate(self, context: dict[str, Any]) -> bool:
        _require(context, "dataset_version_id", "model_id")
        return True


class CvYoloDetectPlugin:
    meta = {
        "name": "cv_yolo_detect",
        "version": "0.2.0",
        "engine_version": "1.0.0",
        "inputs": {"dataset_version_id": "string", "model_id": "string"},
        "outputs": {
            "detected": "number",
            "already_labeled": "number",
            "skipped": "boolean",
        },
        "ui_schema": None,
        "lineage": {"inputs": ["dataset_version", "production_model"], "outputs": ["job_detections"]},
    }

    def validate(self, context: dict[str, Any]) -> bool:
        _require(context, "dataset_version_id", "model_id")
        return True


class CvYoloTrainPlugin:
    meta = {
        "name": "cv_yolo_train",
        "version": "0.2.0",
        "engine_version": "1.0.0",
        "inputs": {
            "dataset_version_id": "string",
            "model_id": "string",
            "artifact_uri": "string",
        },
        "outputs": {"checkpoint": "string", "metrics": "object"},
        "ui_schema": None,
        "lineage": {"inputs": ["dataset_version", "yolo_dataset"], "outputs": ["model_checkpoint"]},
    }

    def validate(self, context: dict[str, Any]) -> bool:
        _require(context, "dataset_version_id", "model_id")
        return True


class CvYoloEvalPlugin:
    meta = {
        "name": "cv_yolo_eval",
        "version": "0.2.0",
        "engine_version": "1.0.0",
        "inputs": {"dataset_version_id": "string", "model_id": "string"},
        "outputs": {"mAP50": "number", "metrics": "object"},
        "ui_schema": None,
        "lineage": {"inputs": ["model_checkpoint"], "outputs": ["eval_metrics"]},
    }

    def validate(self, context: dict[str, Any]) -> bool:
        _require(context, "dataset_version_id", "model_id")
        return True


class CvYoloGatePlugin:
    meta = {
        "name": "cv_yolo_gate",
        "version": "0.2.0",
        "engine_version": "1.0.0",
        "inputs": {"dataset_version_id": "string", "model_id": "string"},
        "outputs": {"passed": "boolean", "promoted": "boolean"},
        "ui_schema": None,
        "lineage": {"inputs": ["eval_metrics", "production_model"], "outputs": ["production_slot"]},
    }

    def validate(self, context: dict[str, Any]) -> bool:
        _require(context, "dataset_version_id", "model_id")
        return True


class CvHardExampleMinePlugin:
    meta = {
        "name": "cv_hard_example_mine",
        "version": "0.2.0",
        "engine_version": "1.0.0",
        "inputs": {"dataset_version_id": "string", "model_id": "string"},
        "outputs": {"appended": "number"},
        "ui_schema": None,
        "lineage": {"inputs": ["dataset_version"], "outputs": ["hard_example_buffer"]},
    }

    def validate(self, context: dict[str, Any]) -> bool:
        _require(context, "dataset_version_id", "model_id")
        return True
