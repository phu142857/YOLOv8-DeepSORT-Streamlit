"""MLAir registry core sync (Vet-AI ``_sync_mlair_project_registry_core`` analogue)."""

from __future__ import annotations

import logging
import time

from mlair_adapter.model_sync import ModelSyncService, _hub_models_missing_versions, _wait_for_mlair_api
from mlair_adapter.pipeline_bootstrap import ensure_cv_yolo_pipeline, map_all_models_to_pipeline
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
    2. When ``CV_MLAIR_SYNC_PRODUCTION_FROM_HUB=1`` (default): if Hub ``production_version``
       differs from local ``base/``, copy that version → ``base`` + ``production`` + ``v{N}/``
       (older ``v1``, ``v2`` folders are kept as archives).

    Full registry pull on ``sync-full`` only when ``CV_MLAIR_MIRROR_REGISTRY_TO_LOCAL=1``.

    Inference always uses **local disk**; MLAir decides *which* version is production.
    """
    svc = ModelSyncService()
    if not svc.enabled:
        return {"ok": False, "reason": "sync_disabled"}

    out: dict = {"ok": True, "ts": time.time()}
    out["models"] = svc.sync_full(force=force)

    if settings.mlair_sync_production_from_hub:
        try:
            state = svc.load_sync_state()
            pulled = svc.sync_pull_registry_models(state)
            out["production_pull"] = pulled
            out["production_pulled"] = sum(1 for r in pulled if r.get("action") == "pull")
        except Exception as exc:
            logger.warning("registry core: Hub production pull failed: %s", exc)
            out["production_pull"] = {"ok": False, "error": str(exc)}

    if _hub_models_missing_versions():
        logger.info("registry core: Hub missing versions after sync, forcing push")
        out["models_retry"] = svc.sync_full(force=True)

    if bootstrap_pipeline:
        try:
            out["pipeline"] = ensure_cv_yolo_pipeline(map_models=map_pipeline)
        except Exception as exc:
            logger.warning("pipeline bootstrap in registry core failed: %s", exc)
            out["pipeline"] = {"ok": False, "error": str(exc)}
    elif map_pipeline:
        try:
            out["pipeline_mapping"] = map_all_models_to_pipeline()
        except Exception as exc:
            logger.warning("pipeline mapping failed: %s", exc)
            out["pipeline_mapping"] = {"ok": False, "error": str(exc)}

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
