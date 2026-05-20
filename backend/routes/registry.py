"""Model registry proxy for client model selection."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from mlair_adapter.model_client import ModelClient
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
        prod = client.find_version_by_stage(model_id, "production")
        options.append(
            RegistryModelOption(
                model_id=model_id,
                name=row.get("name") or model_id,
                registry_value=f"{REGISTRY_PREFIX}{model_id}",
                production_version=int(prod.get("version")) if prod else None,
                artifact_uri=prod.get("artifact_uri") if prod else None,
            )
        )
    return RegistryModelsResponse(configured=True, items=options)


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
        "weights_cache_dir": str(settings.mlair_weights_cache_dir),
    }
