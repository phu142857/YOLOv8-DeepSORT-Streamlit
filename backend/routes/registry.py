"""Model registry proxy for client model selection."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from mlair_adapter.model_client import ModelClient
from mlair_adapter.model_sync import ModelSyncService
from mlair_adapter.pipeline_bootstrap import ensure_cv_yolo_pipeline, map_all_models_to_pipeline
from mlair_adapter.registry_sync import sync_mlair_registry_core
from shared.model_resolve import REGISTRY_PREFIX, ensure_registry_weights
from shared.schemas import RegistryModelOption, RegistryModelsResponse
from shared.settings import settings

router = APIRouter(prefix="/api/v1/registry", tags=["registry"])


@router.get("/models", response_model=RegistryModelsResponse)
def list_registry_models() -> RegistryModelsResponse:
    client = ModelClient()
    if not client.enabled:
        return RegistryModelsResponse(configured=False, items=[])

    options: list[RegistryModelOption] = []
    for row in client.list_models():
        model_id = row.get("model_id") or ""
        if not model_id:
            continue
        resolved = client.resolve_version_row(model_id, stage="production")
        prod = client.find_version_by_stage(model_id, "production")
        version_num = int(resolved.get("version")) if resolved and resolved.get("version") is not None else None
        options.append(
            RegistryModelOption(
                model_id=model_id,
                name=row.get("name") or model_id,
                registry_value=f"{REGISTRY_PREFIX}{model_id}",
                production_version=int(prod.get("version")) if prod else None,
                latest_version=version_num,
                artifact_uri=str(resolved.get("artifact_uri")) if resolved and resolved.get("artifact_uri") else None,
            )
        )
    return RegistryModelsResponse(configured=True, items=options)


@router.post("/models/sync-all")
def sync_all_local_models(force: bool = False) -> dict:
    """Full bidirectional sync: local canonical ↔ MLAir production (base + production folders)."""
    svc = ModelSyncService()
    if not svc.enabled:
        raise HTTPException(status_code=503, detail="MLAir model sync not configured")
    return svc.sync_full(force=force)


@router.post("/models/sync-full")
def sync_full_models(force: bool = False) -> dict:
    """Alias for :func:`sync_all_local_models` — true two-way sync. Use ``?force=1`` after ``down -v``."""
    return sync_all_local_models(force=force)


@router.post("/sync-core")
def registry_sync_core(force: bool = False) -> dict:
    """Vet-AI-style registry sync: disk weights → MLAir + optional pipeline mapping."""
    out = sync_mlair_registry_core(force=force, map_pipeline=True)
    if not out.get("ok"):
        raise HTTPException(status_code=503, detail=out.get("reason", "sync_disabled"))
    return out


@router.post("/pipeline/bootstrap")
def bootstrap_mlair_pipeline(force: bool = False) -> dict:
    """Register ``cv-yolo-vehicle-train`` (plugin mode by default; republish if latest is HTTP-only)."""
    out = ensure_cv_yolo_pipeline(map_models=True, force_republish=force)
    if not out.get("ok"):
        raise HTTPException(status_code=503, detail=out.get("reason", "bootstrap_failed"))
    return out


@router.post("/pipeline/map-models")
def map_models_to_train_pipeline() -> dict:
    """Map all registry models → ``cv-yolo-vehicle-train`` (fixes Hub Train with model unresolved)."""
    out = map_all_models_to_pipeline()
    if not out.get("ok"):
        raise HTTPException(status_code=503, detail=out.get("reason", "mapping_failed"))
    return out


@router.get("/models/{model_id}/resolved-pipeline")
def get_resolved_pipeline(model_id: str) -> dict:
    """Proxy MLAir resolved-pipeline for debugging Train with model."""
    client = ModelClient()
    if not client.enabled:
        raise HTTPException(status_code=503, detail="MLAir not configured")
    return client.get(f"{client._prefix()}/models/{model_id}/resolved-pipeline")


@router.post("/models/{model_id}/sync-weights")
def sync_registry_weights(model_id: str, stage: str = "production") -> dict:
    client = ModelClient()
    if not client.enabled:
        raise HTTPException(status_code=503, detail="MLAir not configured")
    try:
        path = ensure_registry_weights(model_id, stage=stage)
        return {"model_id": model_id, "stage": stage, "local_path": str(path)}
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/config")
def registry_config() -> dict:
    return {
        "mlair_model_id": settings.mlair_model_id or None,
        "auto_train": settings.mlair_auto_train,
        "auto_promote": settings.mlair_auto_promote,
        "auto_sync_models": settings.mlair_auto_sync_models,
        "sync_interval_sec": settings.mlair_sync_interval_sec,
        "sync_unifies_base_and_production": True,
        "weights_cache_dir": str(settings.mlair_weights_cache_dir),
    }
