"""MLAir pipeline version configs for CV YOLO lifecycle (Phase B)."""

from __future__ import annotations

import json
from pathlib import Path

from shared.settings import settings

_PIPELINES_DIR = Path(__file__).resolve().parents[1] / "examples" / "mlair" / "pipelines"

_PIPELINE_FILES = {
    "cv-yolo-lifecycle-train": "cv-yolo-lifecycle-train.plugin.config.json",
    "cv-yolo-vehicle-train": "cv-yolo-vehicle-train.plugin.config.json",
    "cv-hard-example-mine": "cv-hard-example-mine.plugin.config.json",
}


def load_pipeline_config(
    pipeline_id: str | None = None,
    *,
    dataset_logical_name: str | None = None,
    cv_train_url: str | None = None,
    mode: str | None = None,
) -> dict:
    """Load pipeline JSON by id; falls back to inline legacy HTTP config."""
    pipeline_id = pipeline_id or settings.mlair_train_pipeline_id
    mode = (mode or settings.mlair_pipeline_mode or "plugin").strip().lower()

    if mode == "http" and pipeline_id == settings.mlair_legacy_train_pipeline_id:
        return _load_http_legacy(dataset_logical_name, cv_train_url)

    filename = _PIPELINE_FILES.get(pipeline_id)
    if filename:
        path = _PIPELINES_DIR / filename
        if path.is_file():
            config = json.loads(path.read_text(encoding="utf-8"))
            if dataset_logical_name:
                for row in config.get("inputs") or []:
                    if isinstance(row, dict) and "dataset" in row:
                        row["dataset"] = dataset_logical_name
            return config

    if pipeline_id == settings.mlair_hard_example_pipeline_id:
        return json.loads((_PIPELINES_DIR / "cv-hard-example-mine.plugin.config.json").read_text(encoding="utf-8"))

    return cv_yolo_train_pipeline_config(
        mode="plugin",
        dataset_logical_name=dataset_logical_name,
        cv_train_url=cv_train_url,
    )


def load_cv_yolo_pipeline_config(
    *,
    mode: str = "plugin",
    dataset_logical_name: str | None = None,
    cv_train_url: str | None = None,
    pipeline_id: str | None = None,
) -> dict:
    """Backward-compatible alias — defaults to lifecycle train pipeline."""
    return load_pipeline_config(
        pipeline_id or settings.mlair_train_pipeline_id,
        dataset_logical_name=dataset_logical_name,
        cv_train_url=cv_train_url,
        mode=mode,
    )


def _load_http_legacy(
    dataset_logical_name: str | None,
    cv_train_url: str | None,
) -> dict:
    path = _PIPELINES_DIR / "cv-yolo-vehicle-train.http.config.json"
    if path.is_file():
        config = json.loads(path.read_text(encoding="utf-8"))
        if cv_train_url:
            config["tasks"][0]["http"]["url"] = cv_train_url.rstrip("/")
    else:
        config = cv_yolo_train_pipeline_config(
            mode="http",
            dataset_logical_name=dataset_logical_name,
            cv_train_url=cv_train_url,
        )
    if dataset_logical_name:
        for row in config.get("inputs") or []:
            if isinstance(row, dict) and "dataset" in row:
                row["dataset"] = dataset_logical_name
    return config


def cv_yolo_train_pipeline_config(
    *,
    mode: str = "plugin",
    dataset_logical_name: str | None = None,
    cv_train_url: str | None = None,
) -> dict:
    """Inline single-task pipeline (legacy)."""
    dataset = dataset_logical_name or settings.mlair_dataset_name
    if mode == "plugin":
        return {
            "inputs": [{"dataset": dataset, "required_size": settings.mlair_pipeline_required_size}],
            "tasks": [
                {
                    "id": "yolo_train",
                    "plugin": "cv_yolo_train",
                    "plugin_version": ">=0.1.0,<3.0.0",
                }
            ],
        }
    train_url = (cv_train_url or "http://cv-api:8000/api/v1/mlair/train/execute").rstrip("/")
    return {
        "inputs": [{"dataset": dataset, "required_size": settings.mlair_pipeline_required_size}],
        "tasks": [
            {
                "id": "yolo_train",
                "type": "http",
                "http": {
                    "method": "POST",
                    "url": train_url,
                    "headers": {"Content-Type": "application/json"},
                    "json_body_jsonpath": "$",
                    "json_body": {
                        "run_id": "{{ run_id }}",
                        "task_id": "{{ task_id }}",
                        "tenant_id": "{{ tenant_id }}",
                        "project_id": "{{ project_id }}",
                        "trace_id": "{{ trace_id }}",
                    },
                    "secret_env": "CV_MLAIR_TRAIN_CALLBACK_TOKEN",
                    "timeout_seconds": 7200,
                },
            }
        ],
    }
