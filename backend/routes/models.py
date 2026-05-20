"""Local detection weights catalog (model/version layout)."""

from __future__ import annotations

from fastapi import APIRouter

from shared.schemas import LocalModelOption, LocalModelsResponse, UnifiedModelsResponse
from shared.settings import settings
from shared.unified_catalog import list_unified_detection_models
from shared.weights_catalog import scan_detection_models

router = APIRouter(prefix="/api/v1/models", tags=["models"])


@router.get("", response_model=UnifiedModelsResponse)
@router.get("/unified", response_model=UnifiedModelsResponse)
def list_unified_models() -> UnifiedModelsResponse:
    """One model per ``weights/detection/{name}``; MLAir registry row matched by the same name."""
    return list_unified_detection_models()


@router.get("/local", response_model=LocalModelsResponse)
def list_local_models() -> LocalModelsResponse:
    root = settings.detection_model_dir
    items = [
        LocalModelOption(
            model=e.model,
            version=e.version,
            spec=e.spec,
            label=e.label,
            weights_path=str(e.weights_path),
        )
        for e in scan_detection_models(root)
    ]
    return LocalModelsResponse(root=str(root), items=items)
