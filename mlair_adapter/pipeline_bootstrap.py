"""Register ``cv-yolo-vehicle-train`` pipeline version on MLAir (idempotent)."""

from __future__ import annotations

import logging
import threading

from mlair_adapter.base_client import MLAirClient, items_from_response
from mlair_adapter.model_client import ModelClient
from mlair_adapter.pipeline_config import load_pipeline_config
from shared.settings import settings

logger = logging.getLogger(__name__)


def _tasks_have_plugin(config: dict) -> bool:
    tasks = config.get("tasks") or []
    if not isinstance(tasks, list) or not tasks:
        return False
    for t in tasks:
        if not isinstance(t, dict) or not str(t.get("plugin") or "").strip():
            return False
    return True


def _tasks_have_http(config: dict) -> bool:
    tasks = config.get("tasks") or []
    if not isinstance(tasks, list) or not tasks:
        return False
    return any(isinstance(t, dict) and str(t.get("type") or "").strip().lower() == "http" for t in tasks)


def _latest_pipeline_config(client: MLAirClient, pfx: str, pipeline_id: str) -> dict | None:
    try:
        existing = client.get(f"{pfx}/pipelines/{pipeline_id}/versions", params={"limit": 1})
        items = items_from_response(existing)
        if not items:
            return None
        cfg = items[0].get("config")
        return cfg if isinstance(cfg, dict) else None
    except Exception:
        return None


def _pipeline_task_signature(config: dict) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    """Stable compare key for DAG tasks (id, plugin/type, depends_on)."""
    tasks = config.get("tasks") or []
    sig: list[tuple[str, str, tuple[str, ...]]] = []
    for t in tasks:
        if not isinstance(t, dict):
            continue
        task_id = str(t.get("id") or "").strip()
        kind = str(t.get("plugin") or t.get("type") or "").strip().lower()
        deps = tuple(str(d).strip() for d in (t.get("depends_on") or []) if str(d).strip())
        sig.append((task_id, kind, deps))
    return tuple(sig)


def _needs_new_pipeline_version(
    *,
    latest: dict | None,
    mode: str,
    pipeline_id: str,
    train_url: str | None = None,
) -> bool:
    if not latest:
        return True
    if mode == "plugin" and not _tasks_have_plugin(latest):
        return True
    if mode == "http" and not _tasks_have_http(latest):
        return True
    desired = load_pipeline_config(pipeline_id, mode=mode, cv_train_url=train_url)
    return _pipeline_task_signature(desired) != _pipeline_task_signature(latest)


def map_all_models_to_pipeline(
    *,
    pipeline_id: str | None = None,
    client: MLAirClient | None = None,
) -> dict:
    """PUT pipeline-mapping for every model in registry (required for Hub Train with model)."""
    client = client or MLAirClient()
    pipeline_id = pipeline_id or settings.mlair_train_pipeline_id
    pfx = client._prefix()
    mc = ModelClient()
    mapped: list[str] = []
    errors: list[dict] = []

    if not client.enabled or not mc.enabled:
        return {"ok": False, "reason": "mlair_not_configured", "mapped_models": mapped}

    for m in mc.list_models():
        mid = m.get("model_id")
        name = m.get("name")
        if not mid:
            continue
        try:
            client.put(f"{pfx}/models/{mid}/pipeline-mapping", json={"pipeline_id": pipeline_id})
            mapped.append(str(name or mid))
        except Exception as exc:
            logger.warning("pipeline-mapping failed for %s: %s", name, exc)
            errors.append({"model": name, "error": str(exc)})

    return {
        "ok": True,
        "pipeline_id": pipeline_id,
        "mapped_models": mapped,
        "errors": errors,
    }


def reload_cv_plugins(client: MLAirClient | None = None) -> dict:
    """Best-effort plugin registry reload (cv_yolo_train must be installed in ml-air-api)."""
    client = client or MLAirClient()
    if not client.enabled:
        return {"ok": False, "reason": "mlair_not_configured"}
    try:
        return client.post("/v1/plugins/reload", json={})
    except Exception as exc:
        logger.warning("plugin reload failed: %s", exc)
        return {"ok": False, "error": str(exc)}


def _publish_pipeline_version(
    client: MLAirClient,
    pfx: str,
    pipeline_id: str,
    *,
    mode: str,
    train_url: str | None,
    force_republish: bool,
) -> dict:
    latest_cfg = _latest_pipeline_config(client, pfx, pipeline_id)
    needs_publish = force_republish or _needs_new_pipeline_version(
        latest=latest_cfg,
        mode=mode,
        pipeline_id=pipeline_id,
        train_url=train_url,
    )
    version_id: str | None = None
    skipped = False
    republished = False
    if needs_publish:
        config = load_pipeline_config(pipeline_id, mode=mode, cv_train_url=train_url)
        client.post("/v1/pipelines/validate", json=config)
        ver = client.post(f"{pfx}/pipelines/{pipeline_id}/versions", json={"config": config})
        version_id = ver.get("version_id") if isinstance(ver, dict) else None
        republished = latest_cfg is not None
        logger.info("MLAir pipeline %s mode=%s version_id=%s", pipeline_id, mode, version_id)
    else:
        skipped = True
        existing = client.get(f"{pfx}/pipelines/{pipeline_id}/versions", params={"limit": 1})
        items = items_from_response(existing)
        if items:
            version_id = str(items[0].get("version_id") or "") or None
    return {
        "pipeline_id": pipeline_id,
        "version_id": version_id,
        "skipped": skipped,
        "republished": republished,
    }


def ensure_cv_yolo_pipeline(
    *,
    map_models: bool = True,
    required_size: int | None = None,
    train_url: str | None = None,
    mode: str | None = None,
    force_republish: bool = False,
    client: MLAirClient | None = None,
    register_hard_example: bool = True,
) -> dict:
    """
    Publish pipeline version so Hub lists ``cv-yolo-vehicle-train``.

    Default mode ``plugin`` (Hub Train requires ``tasks[].plugin``).
    Republishes when the latest version is HTTP-only (legacy bootstrap).
    """
    client = client or MLAirClient()
    if not client.enabled:
        return {"ok": False, "reason": "mlair_not_configured"}

    if required_size is None:
        required_size = settings.mlair_pipeline_required_size

    pipeline_id = settings.mlair_train_pipeline_id
    pfx = client._prefix()
    train_url = train_url or "http://cv-api:8000/api/v1/mlair/train/execute"
    mode = (mode or settings.mlair_pipeline_mode or "plugin").strip().lower()
    if mode not in ("plugin", "http"):
        mode = "plugin"

    version_id: str | None = None
    pipeline_skipped = False
    republished = False
    hard_mine: dict | None = None
    try:
        main = _publish_pipeline_version(
            client,
            pfx,
            pipeline_id,
            mode=mode,
            train_url=train_url,
            force_republish=force_republish,
        )
        version_id = main.get("version_id")
        pipeline_skipped = main.get("skipped", False)
        republished = main.get("republished", False)
        if register_hard_example:
            hard_mine = _publish_pipeline_version(
                client,
                pfx,
                settings.mlair_hard_example_pipeline_id,
                mode="plugin",
                train_url=None,
                force_republish=force_republish,
            )
    except Exception as exc:
        logger.exception("pipeline register failed")
        return {"ok": False, "reason": str(exc), "pipeline_id": pipeline_id, "mode": mode}

    plugin_reload = reload_cv_plugins(client)

    policy_result: dict | None = None
    try:
        policy_result = _ensure_training_policy(client, pfx, required_size=required_size)
    except Exception as exc:
        logger.warning("training policy not configured (dataset may not exist yet): %s", exc)

    mapping_out: dict = {"mapped_models": [], "errors": []}
    if map_models:
        mapping_out = map_all_models_to_pipeline(pipeline_id=pipeline_id, client=client)

    return {
        "ok": True,
        "skipped": pipeline_skipped,
        "republished": republished,
        "mode": mode,
        "pipeline_id": pipeline_id,
        "version_id": version_id,
        "plugin_reload": plugin_reload,
        "training_policy": policy_result,
        "mapped_models": mapping_out.get("mapped_models") or [],
        "mapping_errors": mapping_out.get("errors") or [],
        "hard_example_pipeline": hard_mine,
    }


def _ensure_training_policy(client: MLAirClient, pfx: str, *, required_size: int) -> dict:
    datasets = client.get(f"{pfx}/datasets", params={"limit": 50})
    ds_row = None
    for row in items_from_response(datasets):
        if row.get("name") == settings.mlair_dataset_name:
            ds_row = row
            break
    if not ds_row:
        return {"ok": False, "reason": f"dataset {settings.mlair_dataset_name!r} not found"}

    dataset_id = str(ds_row["dataset_id"])
    policy_body = {
        "trigger_mode": "manual",
        "required_size": required_size,
        "freshness_hours": 168,
        "validation_rules": [],
    }
    try:
        return client.post(f"{pfx}/datasets/{dataset_id}/training-policies", json=policy_body)
    except Exception:
        policies = client.get(f"{pfx}/datasets/{dataset_id}/training-policies")
        items = items_from_response(policies)
        if not items:
            raise
        pid = items[0].get("policy_id")
        return client.put(
            f"{pfx}/datasets/{dataset_id}/training-policies",
            json={**policy_body, "policy_id": pid},
        )


def start_pipeline_bootstrap_background() -> None:
    if not settings.mlair_bootstrap_pipeline_on_startup:
        return
    if not MLAirClient().enabled:
        logger.info("MLAir pipeline bootstrap disabled (API not configured)")
        return

    def _run() -> None:
        try:
            out = ensure_cv_yolo_pipeline(map_models=True)
            if out.get("ok"):
                logger.info(
                    "MLAir pipeline bootstrap: %s mode=%s republished=%s",
                    out.get("pipeline_id"),
                    out.get("mode"),
                    out.get("republished"),
                )
        except Exception:
            logger.exception("MLAir pipeline bootstrap failed")

    threading.Thread(target=_run, name="mlair-pipeline-bootstrap", daemon=True).start()
