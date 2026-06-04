"""Build MLAir ``POST /tasks/{id}/complete`` payload (metrics + artifacts)."""

from __future__ import annotations

from typing import Any


def metrics_from_plugin_result(result: dict[str, Any]) -> dict[str, Any]:
    """Raw metric keys — MLAir adds ``{plugin}.`` prefix in ``_persist_run_plugin_tracking``."""
    metrics: dict[str, Any] = dict(result.get("metrics") or {})
    for key in ("mAP50", "production_map50", "train_images", "val_images", "imported_version"):
        if result.get(key) is not None:
            metrics[key] = result[key]
    gate = result.get("gate")
    if isinstance(gate, dict):
        for k in ("candidate_map50", "production_map50", "passed", "promoted"):
            if gate.get(k) is not None:
                metrics[k] = gate[k]
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            out[key] = 1.0 if value else 0.0
        elif isinstance(value, (int, float)):
            out[key] = float(value)
    return out


def artifacts_from_plugin_result(result: dict[str, Any], *, plugin: str) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    step = str(result.get("step") or plugin)

    checkpoint = str(result.get("checkpoint") or "").strip()
    if checkpoint:
        uri = checkpoint if checkpoint.startswith("file://") else f"file://{checkpoint}"
        items.append({"path": f"{step}/checkpoint", "uri": uri})

    data_yaml = str(result.get("data_yaml") or "").strip()
    if data_yaml:
        uri = data_yaml if data_yaml.startswith("file://") else f"file://{data_yaml}"
        items.append({"path": f"{step}/data.yaml", "uri": uri})

    workspace = str(result.get("workspace") or "").strip()
    if workspace and not data_yaml:
        items.append({"path": f"{step}/workspace", "uri": f"file://{workspace}"})

    return items


def _attach_usage_report(body: dict[str, Any], usage_report: dict[str, Any] | None) -> None:
    if not isinstance(usage_report, dict):
        return
    ru = usage_report.get("resource_usage")
    if isinstance(ru, dict) and ru:
        body["resource_usage"] = ru
    samples = usage_report.get("usage_samples")
    if isinstance(samples, list) and samples:
        body["usage_samples"] = samples


def build_complete_task_body(
    worker_id: str,
    result: dict[str, Any],
    *,
    plugin: str,
    usage_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "worker_id": worker_id,
        "metrics": metrics_from_plugin_result(result),
    }
    artifacts = artifacts_from_plugin_result(result, plugin=plugin)
    if artifacts:
        body["artifacts"] = artifacts
    lineage = result.get("lineage")
    if isinstance(lineage, dict) and (lineage.get("inputs") or lineage.get("outputs")):
        body["lineage"] = lineage
    _attach_usage_report(body, usage_report)
    return body


def build_fail_task_body(
    worker_id: str,
    error: str,
    *,
    usage_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {"worker_id": worker_id, "error": error}
    _attach_usage_report(body, usage_report)
    return body
