"""Chains workers — same interface MLAir executor will call later."""

from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from inference.validation import ValidationError, validate_model, validate_source
from shared.artifacts import ArtifactStore
from shared.job_store import job_store
from shared.schemas import JobStatus
from workers import (
    AggregationWorker,
    DetectionWorker,
    ExportWorker,
    MLAirIngestWorker,
    MLAirReadinessWorker,
    TrackingWorker,
)
from workers.base import WorkerContext

logger = logging.getLogger(__name__)

WORKER_CHAIN = [
    DetectionWorker(),
    TrackingWorker(),
    AggregationWorker(),
    ExportWorker(),
    MLAirIngestWorker(),
    MLAirReadinessWorker(),
]

# Progress budget per worker (sums to ~1.0)
_PROGRESS_WEIGHTS = {
    "detection": 0.55,
    "tracking": 0.08,
    "aggregation": 0.08,
    "export": 0.07,
    "mlair_ingest": 0.12,
    "mlair_readiness": 0.10,
}


def _log(store: ArtifactStore, job_id: str, msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{ts}] {msg}"
    store.append_log(job_id, line)
    logger.info("job=%s %s", job_id, msg)


def run_job_pipeline(
    job_id: str,
    source_path: Path,
    model_name: str,
    confidence: float,
    store: ArtifactStore | None = None,
    on_progress: Callable[[float, str], None] | None = None,
) -> None:
    artifact_store = store or ArtifactStore()

    def _set_progress(value: float, message: str, step: str = "") -> None:
        if on_progress:
            on_progress(value, message)
        job_store.update(
            job_id,
            progress=value,
            message=message,
            current_step=step or None,
        )

    try:
        validate_model(model_name)
        validate_source(source_path)
    except ValidationError as exc:
        _log(artifact_store, job_id, f"validation failed: {exc}")
        job_store.update(
            job_id,
            status=JobStatus.FAILED,
            progress=1.0,
            message="validation failed",
            error=str(exc),
        )
        return

    if job_store.is_cancelled(job_id):
        return

    job_store.update(job_id, status=JobStatus.RUNNING, progress=0.0, message="starting pipeline")
    _log(artifact_store, job_id, f"pipeline start source={source_path.name} model={model_name}")

    base_progress = 0.0

    def detection_progress(p: float, msg: str) -> None:
        weight = _PROGRESS_WEIGHTS["detection"]
        _set_progress(base_progress + p * weight, msg, "detection")

    job_meta = job_store.get(job_id)
    ctx = WorkerContext(
        job_id=job_id,
        source_path=source_path,
        model_name=model_name,
        confidence=confidence,
        store=artifact_store,
        metadata={
            "progress_callback": detection_progress,
            "cancel_check": lambda: job_store.is_cancelled(job_id),
            "skip_mlair_ingest": bool(job_meta and job_meta.mlair_dataset_version_id),
            "mlair_dataset_id": job_meta.mlair_dataset_id if job_meta else None,
        },
    )

    try:
        for worker in WORKER_CHAIN:
            if job_store.is_cancelled(job_id):
                _log(artifact_store, job_id, "cancelled before " + worker.name)
                return

            _log(artifact_store, job_id, f"worker start: {worker.name}")
            job_store.update(
                job_id,
                message=f"running {worker.name}",
                current_step=worker.name,
                progress=base_progress,
            )

            result = worker.run(ctx)
            artifact_store.save_step(
                job_id,
                worker.name,
                {"ok": result.ok, "message": result.message, "metadata_keys": list(result.metadata.keys())},
            )

            if not result.ok:
                _log(artifact_store, job_id, f"worker failed: {worker.name} — {result.message}")
                job_store.update(
                    job_id,
                    status=JobStatus.FAILED,
                    message=result.message,
                    error=result.message,
                    progress=1.0,
                    current_step=worker.name,
                )
                return

            ctx.metadata.update(result.metadata)
            base_progress += _PROGRESS_WEIGHTS.get(worker.name, 0.1)
            _log(artifact_store, job_id, f"worker done: {worker.name}")

        manifest = ctx.metadata.get("manifest", {})
        counters_in: dict[str, int] = {}
        counters_out: dict[str, int] = {}
        aggregates = artifact_store.job_layout(job_id)["aggregates"] / "counts.json"
        if aggregates.exists():
            with aggregates.open(encoding="utf-8") as f:
                data = json.load(f)
                counters_in = data.get("counters_in", {})
                counters_out = data.get("counters_out", {})

        _log(artifact_store, job_id, "pipeline complete")
        mlair_meta = ctx.metadata.get("mlair_ingest") or {}
        job_store.update(
            job_id,
            status=JobStatus.COMPLETED,
            progress=1.0,
            message="pipeline complete",
            current_step="done",
            artifact_manifest=manifest,
            counters_in=counters_in,
            counters_out=counters_out,
            mlair_dataset_id=mlair_meta.get("dataset_id") or (job_meta.mlair_dataset_id if job_meta else None),
            mlair_dataset_version_id=mlair_meta.get("dataset_version_id")
            or (job_meta.mlair_dataset_version_id if job_meta else None),
            mlair_readiness=ctx.metadata.get("mlair_readiness") or {},
        )
    except InterruptedError:
        _log(artifact_store, job_id, "pipeline interrupted (cancelled)")
        job_store.update(job_id, status=JobStatus.CANCELLED, message="cancelled during processing")
    except Exception as exc:
        tb = traceback.format_exc()
        _log(artifact_store, job_id, f"pipeline error: {exc}")
        logger.exception("Job %s failed", job_id)
        artifact_store.write_json(
            artifact_store.job_layout(job_id)["logs"] / "error.json",
            {"error": str(exc), "traceback": tb},
        )
        job_store.update(
            job_id,
            status=JobStatus.FAILED,
            message="pipeline error",
            error=str(exc),
            progress=1.0,
        )
