"""Register ``cv-yolo-vehicle-train`` pipeline version on MLAir (idempotent)."""

from __future__ import annotations

import logging
import threading

from mlair_adapter.base_client import MLAirClient, items_from_response
from mlair_adapter.model_client import ModelClient
from mlair_adapter.pipeline_config import load_cv_yolo_pipeline_config
from shared.settings import settings

logger = logging.getLogger(__name__)


def ensure_cv_yolo_pipeline(
    *,
    map_models: bool = True,
    required_size: int = 100,
    train_url: str | None = None,
    client: MLAirClient | None = None,
) -> dict:
    """
    Publish pipeline version so Hub lists ``cv-yolo-vehicle-train``.
    Skips if a version already exists. Training policy requires dataset to exist.
    """
    client = client or MLAirClient()
    if not client.enabled:
        return {"ok": False, "reason": "mlair_not_configured"}

    pipeline_id = settings.mlair_train_pipeline_id
    pfx = client._prefix()
    train_url = train_url or "http://cv-api:8000/api/v1/mlair/train/execute"

    try:
        existing = client.get(f"{pfx}/pipelines/{pipeline_id}/versions", params={"limit": 5})
        if items_from_response(existing):
            return {
                "ok": True,
                "skipped": True,
                "reason": "already_registered",
                "pipeline_id": pipeline_id,
            }
    except Exception as exc:
        logger.debug("pipeline versions probe failed: %s", exc)

    config = load_cv_yolo_pipeline_config(mode="http", cv_train_url=train_url)
    client.post("/v1/pipelines/validate", json={"config": config})
    ver = client.post(f"{pfx}/pipelines/{pipeline_id}/versions", json={"config": config})
    version_id = ver.get("version_id") if isinstance(ver, dict) else None
    logger.info("MLAir pipeline registered: %s version_id=%s", pipeline_id, version_id)

    policy_result: dict | None = None
    try:
        policy_result = _ensure_training_policy(client, pfx, required_size=required_size)
    except Exception as exc:
        logger.warning("training policy not configured (dataset may not exist yet): %s", exc)

    mapped: list[str] = []
    if map_models:
        mc = ModelClient()
        if mc.enabled:
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

    return {
        "ok": True,
        "pipeline_id": pipeline_id,
        "version_id": version_id,
        "training_policy": policy_result,
        "mapped_models": mapped,
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
            if out.get("ok") and not out.get("skipped"):
                logger.info("MLAir pipeline bootstrap: %s", out.get("pipeline_id"))
        except Exception:
            logger.exception("MLAir pipeline bootstrap failed")

    threading.Thread(target=_run, name="mlair-pipeline-bootstrap", daemon=True).start()
