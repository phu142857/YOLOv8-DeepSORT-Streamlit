"""MLAir proxy endpoints for UI (datasets, readiness, buffer)."""

from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException

from pydantic import BaseModel

from mlair_adapter.dataset_client import DatasetClient
from mlair_adapter.model_client import ModelClient
from mlair_adapter.model_sync import ModelSyncService
from mlair_adapter.readiness_client import ReadinessClient, normalize_readiness
from mlair_adapter.training_client import TrainingClient
from shared.schemas import MLAirReadinessResponse
from shared.settings import settings


class TriggerTrainingRequest(BaseModel):
    model_id: str
    dataset_id: str
    dataset_version_id: str | None = None


class PromoteWebhookBody(BaseModel):
    """MLAir ``MLAIR_MODEL_PROMOTE_WEBHOOK_*`` outbound JSON."""

    tenant_id: str = ""
    project_id: str = ""
    model_id: str
    version: int
    artifact_uri: str = ""
    idempotency_key: str | None = None


router = APIRouter(prefix="/api/v1/mlair", tags=["mlair"])


def _check_promote_webhook_token(authorization: str | None) -> None:
    expected = settings.mlair_promote_webhook_token.strip()
    if not expected:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    token = authorization[7:].strip()
    if token != expected:
        raise HTTPException(status_code=403, detail="invalid promote webhook token")


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


@router.post("/runs/trigger")
def trigger_training(body: TriggerTrainingRequest) -> dict:
    """Proxy for Hub/tooling — primary train path is worker `mlair_train` after readiness."""
    client = TrainingClient()
    if not client.enabled:
        raise HTTPException(status_code=503, detail="MLAir not configured")
    return client.trigger_run_by_model(
        body.model_id,
        body.dataset_id,
        dataset_version_id=body.dataset_version_id,
        context={"source": "cv_api_proxy"},
    )


@router.get("/runs/{run_id}")
def get_training_run(run_id: str) -> dict:
    client = TrainingClient()
    if not client.enabled:
        raise HTTPException(status_code=503, detail="MLAir not configured")
    return client.get_run(run_id)


@router.post("/promote-webhook")
def mlair_promote_webhook(
    body: PromoteWebhookBody,
    authorization: str | None = Header(default=None),
) -> dict:
    """
    MLAir Hub promote → update ``weights/detection/{model}/base`` for Vehicle Detection.
    Configure on ml-air-api: ``MLAIR_MODEL_PROMOTE_WEBHOOK_URL=http://cv-api:8000/api/v1/mlair/promote-webhook``.
    """
    _check_promote_webhook_token(authorization)
    if not settings.mlair_sync_on_hub_promote:
        return {"ok": True, "skipped": True, "reason": "sync_on_hub_promote_disabled"}

    svc = ModelSyncService()
    if not svc.enabled:
        raise HTTPException(status_code=503, detail="MLAir model sync not configured")
    try:
        return svc.apply_mlair_promotion_to_local(
            body.model_id,
            body.version,
            stage=settings.mlair_promote_stage,
            artifact_uri=body.artifact_uri or None,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/models/{model_id}/promote")
def promote_model(model_id: str, version: int, stage: str = "production") -> dict:
    client = ModelClient()
    if not client.enabled:
        raise HTTPException(status_code=503, detail="MLAir not configured")
    promoted = client.promote(model_id, version, stage=stage)
    if settings.mlair_sync_on_hub_promote:
        try:
            local = ModelSyncService().apply_mlair_promotion_to_local(
                model_id,
                version,
                stage=stage,
            )
            promoted = {**promoted, "local_sync": local}
        except Exception as exc:
            promoted = {**promoted, "local_sync": {"ok": False, "error": str(exc)}}
    return promoted


@router.get("/training/config")
def training_config() -> dict:
    return {
        "model_id": settings.mlair_model_id or None,
        "auto_train": settings.mlair_auto_train,
        "auto_promote": settings.mlair_auto_promote,
        "promote_stage": settings.mlair_promote_stage,
    }
