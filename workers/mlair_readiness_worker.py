"""Evaluate MLAir dataset readiness after ingest (Phase 2)."""

from __future__ import annotations

import logging

from mlair_adapter.events_client import emit_event
from mlair_adapter.readiness_client import ReadinessClient
from workers.base import Worker, WorkerContext, WorkerResult

logger = logging.getLogger(__name__)


class MLAirReadinessWorker:
    name = "mlair_readiness"

    def run(self, ctx: WorkerContext) -> WorkerResult:
        if ctx.metadata.get("skip_mlair_ingest"):
            return WorkerResult(ok=True, message="readiness skipped (mlair pull source)")

        ingest = ctx.metadata.get("mlair_ingest") or {}
        dataset_id = ingest.get("dataset_id") or ctx.metadata.get("mlair_dataset_id")
        version_id = ingest.get("dataset_version_id") or ctx.metadata.get("mlair_dataset_version_id")

        if not dataset_id:
            return WorkerResult(ok=True, message="no dataset_id for readiness")

        client = ReadinessClient()
        if not client.enabled:
            return WorkerResult(ok=True, message="mlair not configured")

        try:
            readiness = client.evaluate_readiness(
                dataset_id,
                dataset_version_id=version_id,
                source="cv_workload",
            )
            ctx.metadata["mlair_readiness"] = readiness

            emit_event(
                "dataset.readiness.updated",
                {
                    "dataset_id": dataset_id,
                    "dataset_version_id": version_id,
                    "status": readiness.get("status"),
                    "ready": readiness.get("ready"),
                    "reasons": readiness.get("reasons", []),
                },
                job_id=ctx.job_id,
                store=ctx.store,
            )

            status = readiness.get("status", "")
            ready = readiness.get("ready", False)
            return WorkerResult(
                ok=True,
                message=f"readiness: {status} ({'READY' if ready else 'NOT READY'})",
                metadata=ctx.metadata,
            )
        except Exception as exc:
            logger.exception("Readiness evaluation failed for job %s", ctx.job_id)
            return WorkerResult(ok=False, message=f"readiness failed: {exc}")
