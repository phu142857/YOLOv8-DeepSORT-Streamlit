"""Local detection weights catalog (model/version layout)."""

from __future__ import annotations

from fastapi import APIRouter

from shared.schemas import LocalModelOption, LocalModelsResponse
from shared.settings import settings
from shared.weights_catalog import scan_detection_models

router = APIRouter(prefix="/api/v1/models", tags=["models"])


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
