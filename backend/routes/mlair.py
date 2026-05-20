"""MLAir proxy endpoints for UI (datasets, readiness, buffer)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from mlair_adapter.dataset_client import DatasetClient
from mlair_adapter.readiness_client import ReadinessClient, normalize_readiness
from shared.schemas import MLAirReadinessResponse

router = APIRouter(prefix="/api/v1/mlair", tags=["mlair"])


def _require_client() -> DatasetClient:
    client = DatasetClient()
    if not client.enabled:
        raise HTTPException(
            status_code=503,
            detail="MLAir not configured — set CV_MLAIR_API_URL and CV_MLAIR_TOKEN",
        )
    return client


@router.get("/status")
def mlair_status() -> dict:
    client = DatasetClient()
    return {
        "configured": client.enabled,
        "api_url": client.base_url if client.enabled else None,
        "tenant": client.tenant,
        "project": client.project,
    }


@router.get("/datasets")
def list_datasets(limit: int = 50) -> dict:
    client = _require_client()
    items = client.list_datasets(limit=limit)
    return {"items": items}


@router.get("/datasets/{dataset_id}/versions")
def list_versions(dataset_id: str) -> dict:
    client = _require_client()
    return {"items": client.list_versions(dataset_id)}


@router.get("/datasets/{dataset_id}/buffer")
def get_buffer(dataset_id: str) -> dict:
    client = _require_client()
    return client.get_buffer(dataset_id)


@router.post("/datasets/{dataset_id}/materialize")
def materialize(dataset_id: str) -> dict:
    client = _require_client()
    return client.materialize_buffer(dataset_id)


def _to_readiness_response(
    dataset_id: str,
    normalized: dict,
    *,
    version_hint: str | None = None,
) -> MLAirReadinessResponse:
    return MLAirReadinessResponse(
        dataset_id=dataset_id,
        dataset_version_id=version_hint or normalized.get("dataset_version_id"),
        status=normalized.get("status", ""),
        ready=bool(normalized.get("ready")),
        reasons=list(normalized.get("reasons") or []),
        raw=normalized.get("raw") or normalized,
    )


@router.get("/datasets/{dataset_id}/readiness", response_model=MLAirReadinessResponse)
def get_readiness(
    dataset_id: str,
    dataset_version_id: str | None = None,
) -> MLAirReadinessResponse:
    client = ReadinessClient()
    if not client.enabled:
        raise HTTPException(status_code=503, detail="MLAir not configured")
    normalized = client.get_readiness(dataset_id, dataset_version_id=dataset_version_id)
    return _to_readiness_response(dataset_id, normalized, version_hint=dataset_version_id)


@router.post("/datasets/{dataset_id}/readiness/evaluate", response_model=MLAirReadinessResponse)
def evaluate_readiness(
    dataset_id: str,
    dataset_version_id: str | None = None,
) -> MLAirReadinessResponse:
    client = ReadinessClient()
    if not client.enabled:
        raise HTTPException(status_code=503, detail="MLAir not configured")
    normalized = client.evaluate_readiness(
        dataset_id,
        dataset_version_id=dataset_version_id,
        source="cv_workload",
    )
    return _to_readiness_response(dataset_id, normalized, version_hint=dataset_version_id)
