"""MLAir pipeline version config for CV YOLO training (HTTP task → cv-api)."""

from __future__ import annotations

import json
from pathlib import Path

from shared.settings import settings

_PIPELINES_DIR = Path(__file__).resolve().parents[1] / "examples" / "mlair" / "pipelines"


def load_cv_yolo_pipeline_config(
    *,
    mode: str = "http",
    dataset_logical_name: str | None = None,
    cv_train_url: str | None = None,
) -> dict:
    """Load official JSON from examples/ or build inline."""
    if mode == "plugin":
        path = _PIPELINES_DIR / "cv-yolo-vehicle-train.plugin.config.json"
        config = json.loads(path.read_text(encoding="utf-8"))
    else:
        path = _PIPELINES_DIR / "cv-yolo-vehicle-train.http.config.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        if cv_train_url:
            config["tasks"][0]["http"]["url"] = cv_train_url.rstrip("/")
    if dataset_logical_name:
        for row in config.get("inputs") or []:
            if isinstance(row, dict) and "dataset" in row:
                row["dataset"] = dataset_logical_name
    return config


def cv_yolo_train_pipeline_config(
    *,
    dataset_logical_name: str | None = None,
    cv_train_url: str | None = None,
) -> dict:
    """
    Pipeline version ``config`` registered on MLAir.

    See ml-air docs: http-pipeline-tasks.md, configure-data-readiness-gating.md.
    """
    dataset = dataset_logical_name or settings.mlair_dataset_name
    train_url = (cv_train_url or "http://cv-api:8000/api/v1/mlair/train/execute").rstrip("/")
    return {
        "inputs": [{"dataset": dataset, "required_size": 1}],
        "tasks": [
            {
                "id": "yolo_train",
                "type": "http",
                "http": {
                    "method": "POST",
                    "url": train_url,
                    "headers": {"Content-Type": "application/json"},
                    "json_body": {
                        "run_id": "{{ run_id }}",
                        "task_id": "{{ task_id }}",
                        "tenant_id": "{{ tenant_id }}",
                        "project_id": "{{ project_id }}",
                        "trace_id": "{{ trace_id }}",
                        "model_id": "{{ context.model_id }}",
                        "mlair_model_id": "{{ context.mlair_model_id }}",
                        "dataset_id": "{{ context.dataset_id }}",
                        "dataset_version_id": "{{ context.dataset_version_id }}",
                        "artifact_uri": "{{ context.artifact_uri }}",
                        "base_weights_source": "{{ context.base_weights_source }}",
                    },
                    "secret_env": "CV_MLAIR_TRAIN_CALLBACK_TOKEN",
                    "timeout_seconds": 7200,
                },
            }
        ],
    }
