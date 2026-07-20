"""FastAPI control plane — inference orchestration and artifact management."""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from shared.artifacts import ArtifactStore
from shared.job_store import init_job_store
from shared.settings import settings

logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
logger = logging.getLogger(__name__)

artifact_store = ArtifactStore()
init_job_store(artifact_store)

# Import routes after init_job_store so all modules share the same JobStore instance.
from backend.routes import (  # noqa: E402
    artifacts,
    datasets,
    frames,
    jobs,
    lifecycle,
    mlair,
    mlair_training,
    models,
    registry,
    uploads,
)
from mlair_adapter.registry_sync import start_registry_sync_background  # noqa: E402
from shared.detection_weights_bootstrap import startup_ensure_detection_weights  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    summary = startup_ensure_detection_weights()
    if summary:
        logger.info("detection weights bootstrap: %s", summary)
    start_registry_sync_background()
    yield


app = FastAPI(
    title="CV Lifecycle Workload API",
    description="YOLOv8 + DeepSORT workload with artifact persistence (MLAir-ready)",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(uploads.router)
app.include_router(datasets.router)
app.include_router(jobs.router)
app.include_router(artifacts.router)
app.include_router(frames.router)
app.include_router(mlair.router)
app.include_router(mlair_training.router)
app.include_router(lifecycle.router)
app.include_router(registry.router)
app.include_router(models.router)


@app.middleware("http")
async def trace_middleware(request: Request, call_next):
    trace_id = request.headers.get(settings.trace_header) or str(uuid.uuid4())[:8]
    request.state.trace_id = trace_id
    response = await call_next(request)
    response.headers[settings.trace_header] = trace_id
    return response


@app.get("/health")
def health() -> dict:
    jobs_root = artifact_store.root / "jobs"
    job_count = len(list(jobs_root.iterdir())) if jobs_root.is_dir() else 0
    return {
        "status": "ok",
        "service": "cv-lifecycle-api",
        "version": "0.2.0",
        "jobs_on_disk": job_count,
        "persist_jobs": settings.persist_jobs,
    }


@app.get("/api/v1/runtime")
def runtime_config() -> dict:
    return {
        "mlair_configured": bool(settings.mlair_api_url and settings.mlair_token),
        "mlair_auto_train": settings.mlair_auto_train,
        "mlair_auto_sync_models": settings.mlair_auto_sync_models,
        "mlair_sync_interval_sec": settings.mlair_sync_interval_sec,
        "mlair_model_id": settings.mlair_model_id or None,
        "mlair_api_url": settings.mlair_api_url or None,
        "s3_models_bucket": settings.s3_models_bucket or None,
        "artifact_root": str(settings.artifact_root),
        "api_base_url": settings.api_base_url,
        "max_video_frames": settings.max_video_frames,
        "frame_extract_interval": settings.frame_extract_interval,
        "persist_jobs": settings.persist_jobs,
    }
