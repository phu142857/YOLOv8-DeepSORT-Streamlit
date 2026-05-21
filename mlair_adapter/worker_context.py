"""Build plugin execution context from MLAir external-worker lease payload."""

from __future__ import annotations

import logging
from typing import Any

from mlair_adapter.base_client import MLAirClient
from mlair_adapter.run_workspace import load_state

logger = logging.getLogger(__name__)

_CONTEXT_KEYS = (
    "model_id",
    "mlair_model_id",
    "dataset_id",
    "dataset_version_id",
    "artifact_uri",
    "base_weights_source",
    "base_version_id",
)


def _copy_keys(target: dict[str, Any], source: dict[str, Any] | None) -> None:
    if not isinstance(source, dict):
        return
    for key in _CONTEXT_KEYS:
        val = source.get(key)
        if val is not None and str(val).strip() and not target.get(key):
            target[key] = val


def plugin_context_from_lease_task(task: dict[str, Any]) -> dict[str, Any]:
    """
    MLAir ``POST /v1/tasks/lease`` returns::

        { "task_id", "run_id", "plugin", "payload": { "context", "override_config", ... } }
    """
    payload = task.get("payload") if isinstance(task.get("payload"), dict) else {}

    ctx: dict[str, Any] = {}
    _copy_keys(ctx, payload.get("context") if isinstance(payload.get("context"), dict) else None)
    _copy_keys(ctx, task.get("plugin_context") if isinstance(task.get("plugin_context"), dict) else None)
    _copy_keys(ctx, task.get("context") if isinstance(task.get("context"), dict) else None)
    _copy_keys(ctx, payload.get("override_config") if isinstance(payload.get("override_config"), dict) else None)

    params = ctx.get("params") if isinstance(ctx.get("params"), dict) else payload.get("params")
    if isinstance(params, dict):
        _copy_keys(ctx, params)

    for key in _CONTEXT_KEYS:
        if task.get(key) and not ctx.get(key):
            ctx[key] = task[key]

    ctx.setdefault("run_id", task.get("run_id"))
    ctx.setdefault("task_id", task.get("task_id"))
    ctx.setdefault("tenant_id", task.get("tenant_id"))
    ctx.setdefault("project_id", task.get("project_id"))
    ctx.setdefault("pipeline_id", task.get("pipeline_id"))

    run_id = str(ctx.get("run_id") or "")
    if run_id:
        _copy_keys(ctx, load_state(run_id))

    if not str(ctx.get("dataset_version_id") or "").strip():
        ctx = _enrich_from_run_api(ctx)

    return ctx


def _enrich_from_run_api(ctx: dict[str, Any]) -> dict[str, Any]:
    run_id = str(ctx.get("run_id") or "").strip()
    if not run_id:
        return ctx
    client = MLAirClient()
    if not client.enabled:
        return ctx
    try:
        run = client.get(f"{client._prefix()}/runs/{run_id}")
        if isinstance(run, dict):
            _copy_keys(ctx, run.get("plugin_context") if isinstance(run.get("plugin_context"), dict) else None)
            _copy_keys(ctx, run.get("override_config") if isinstance(run.get("override_config"), dict) else None)
    except Exception as exc:
        logger.warning("could not load run %s for plugin context: %s", run_id, exc)
    return ctx
