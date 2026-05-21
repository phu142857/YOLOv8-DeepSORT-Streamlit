"""MLAir registry core sync (Vet-AI ``_sync_mlair_project_registry_core`` analogue)."""

from __future__ import annotations

import logging
import time

from mlair_adapter.model_sync import ModelSyncService, _hub_models_missing_versions, _wait_for_mlair_api
from mlair_adapter.pipeline_bootstrap import ensure_cv_yolo_pipeline
from shared.settings import settings

logger = logging.getLogger(__name__)


def sync_mlair_registry_core(
    *,
    force: bool = False,
    map_pipeline: bool = True,
    bootstrap_pipeline: bool = True,
) -> dict:
    """
    CV workload registry sync:

    1. Push canonical checkpoints from ``weights/detection`` → MLAir ``versions/import``.
    2. Pull production → ``base`` + ``production`` local folders.
    3. Optionally register pipeline + map models (Hub Train).

    Inference always uses **local disk**; MLAir is catalog/governance only.
    """
    svc = ModelSyncService()
    if not svc.enabled:
        return {"ok": False, "reason": "sync_disabled"}

    out: dict = {"ok": True, "ts": time.time()}
    out["models"] = svc.sync_full(force=force)

    if _hub_models_missing_versions():
        logger.info("registry core: Hub missing versions after sync, forcing push")
        out["models_retry"] = svc.sync_full(force=True)

    if bootstrap_pipeline:
        try:
            out["pipeline"] = ensure_cv_yolo_pipeline(map_models=map_pipeline)
        except Exception as exc:
            logger.warning("pipeline bootstrap in registry core failed: %s", exc)
            out["pipeline"] = {"ok": False, "error": str(exc)}

    return out


def run_registry_sync_loop() -> None:
    interval = max(60.0, float(settings.mlair_registry_resync_seconds))
    while True:
        try:
            if ModelSyncService().enabled and _wait_for_mlair_api(attempts=3, delay_sec=2.0):
                summary = sync_mlair_registry_core(force=False, map_pipeline=False)
                pushed = int((summary.get("models") or {}).get("pushed") or 0)
                if pushed:
                    logger.info("registry resync: pushed=%s", pushed)
        except Exception:
            logger.exception("registry resync loop failed")
        time.sleep(interval)


def start_registry_sync_background() -> None:
    """Startup + periodic registry sync (disk → MLAir catalog)."""
    if not settings.mlair_auto_sync_models:
        return
    if (
        not settings.mlair_registry_sync_at_startup
        and not settings.mlair_sync_on_startup
        and settings.mlair_registry_resync_seconds <= 0
    ):
        return

    def _startup() -> None:
        try:
            if not _wait_for_mlair_api():
                logger.warning("MLAir API not ready; registry sync will retry on interval")
                return
            summary = sync_mlair_registry_core(
                force=False,
                map_pipeline=settings.mlair_bootstrap_pipeline_on_startup,
            )
            models = summary.get("models") or {}
            if _hub_models_missing_versions():
                sync_mlair_registry_core(force=True, map_pipeline=False)
            logger.info(
                "MLAir registry core done: pushed=%s pulled=%s",
                models.get("pushed"),
                models.get("pulled"),
            )
        except Exception:
            logger.exception("MLAir registry startup sync failed")

    import threading

    if settings.mlair_registry_sync_at_startup or settings.mlair_sync_on_startup:
        threading.Thread(target=_startup, name="mlair-registry-sync-startup", daemon=True).start()

    if settings.mlair_registry_resync_seconds > 0:
        threading.Thread(target=run_registry_sync_loop, name="mlair-registry-sync-loop", daemon=True).start()
