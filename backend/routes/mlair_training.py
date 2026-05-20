"""MLAir executor callback — real YOLO training on user dataset versions."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from mlair_adapter.yolo_train_pipeline import run_yolo_training
from shared.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/mlair/train", tags=["mlair-training"])


class TrainExecuteRequest(BaseModel):
    """Body from MLAir HTTP pipeline task (Jinja-rendered)."""

    run_id: str = ""
    task_id: str = ""
    tenant_id: str = ""
    project_id: str = ""
    trace_id: str = ""
    model_id: str = ""
    mlair_model_id: str = ""
    dataset_id: str = ""
    dataset_version_id: str = ""
    artifact_uri: str | None = None
    base_weights_source: str | None = None
    context: dict = Field(default_factory=dict)


def _check_callback_token(authorization: str | None) -> None:
    expected = settings.mlair_train_callback_token.strip()
    if not expected:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    token = authorization[7:].strip()
    if token != expected:
        raise HTTPException(status_code=403, detail="invalid train callback token")


@router.post("/execute")
def execute_training(
    body: TrainExecuteRequest,
    authorization: str | None = Header(default=None),
) -> dict:
    """
    Invoked by ml-air **executor** HTTP task (`cv-yolo-vehicle-train` pipeline).

    Downloads the pinned dataset version, builds YOLO labels from CV job detections,
    fine-tunes from registry base weights, imports checkpoint back to MLAir model registry.
    """
    _check_callback_token(authorization)

    ctx = dict(body.context or {})
    ctx.setdefault("run_id", body.run_id or ctx.get("run_id") or "unknown")
    for key in (
        "model_id",
        "mlair_model_id",
        "dataset_id",
        "dataset_version_id",
        "artifact_uri",
        "base_weights_source",
        "tenant_id",
        "project_id",
        "trace_id",
        "task_id",
    ):
        val = getattr(body, key, None) or ctx.get(key)
        if val:
            ctx[key] = val
    if body.mlair_model_id and not ctx.get("model_id"):
        ctx["model_id"] = body.mlair_model_id

    if not ctx.get("dataset_version_id"):
        raise HTTPException(status_code=422, detail="dataset_version_id required")
    if not (ctx.get("model_id") or ctx.get("mlair_model_id")):
        raise HTTPException(status_code=422, detail="model_id required")

    try:
        result = run_yolo_training(ctx)
        return {
            "ok": True,
            "params": {"source": "cv_yolo_train", "run_id": ctx["run_id"]},
            "metrics": result.get("metrics") or {},
            "artifacts": [
                {
                    "path": "weights/best.pt",
                    "uri": str(result.get("checkpoint", "")),
                }
            ],
            "lineage": {
                "inputs": [
                    {
                        "name": "dataset_version",
                        "version": str(ctx.get("dataset_version_id")),
                    }
                ],
                "outputs": [
                    {
                        "name": "model_checkpoint",
                        "version": str((result.get("imported_version") or {}).get("version", "")),
                    }
                ],
            },
            "result": result,
        }
    except Exception as exc:
        logger.exception("MLAir YOLO training failed run_id=%s", ctx.get("run_id"))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/health")
def train_health() -> dict:
    return {
        "service": "cv-yolo-train",
        "epochs": settings.mlair_train_epochs,
        "pipeline_id": settings.mlair_train_pipeline_id,
    }
