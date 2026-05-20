"""Evaluate MLAir dataset readiness after ingest (Phase 2)."""

from __future__ import annotations

import logging

import httpx

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

        if not version_id:
            return WorkerResult(
                ok=True,
                message="readiness skipped (no dataset_version yet — run after accumulation creates a version)",
                metadata=ctx.metadata,
            )

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
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 422 and _is_dataset_version_required(exc.response):
                return WorkerResult(
                    ok=True,
                    message="readiness skipped (dataset_version_id required by MLAir — evaluate after version is materialized)",
                    metadata=ctx.metadata,
                )
            logger.exception("Readiness HTTP error for job %s", ctx.job_id)
            return WorkerResult(ok=True, message=f"readiness skipped (HTTP {exc.response.status_code})")
        except ValueError as exc:
            if "dataset_version_id_required" in str(exc):
                return WorkerResult(
                    ok=True,
                    message="readiness skipped (no pinned dataset_version_id)",
                    metadata=ctx.metadata,
                )
            return WorkerResult(ok=True, message=f"readiness skipped: {exc}")
        except Exception as exc:
            logger.exception("Readiness evaluation failed for job %s", ctx.job_id)
            return WorkerResult(ok=True, message=f"readiness skipped: {exc}")


def _is_dataset_version_required(response: httpx.Response) -> bool:
    try:
        body = response.json()
    except Exception:
        return False
    nested = body.get("detail") if isinstance(body, dict) else None
    if isinstance(nested, dict) and nested.get("reason") == "DATASET_VERSION_REQUIRED":
        return True
    return body.get("detail") == "dataset_version_id_required"
