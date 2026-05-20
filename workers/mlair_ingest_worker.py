"""Push job frame artifacts to MLAir dataset (Phase 2)."""

from __future__ import annotations

import logging
from pathlib import Path

from mlair_adapter.dataset_client import DatasetClient
from mlair_adapter.events_client import emit_event
from shared.settings import settings
from workers.base import WorkerContext, WorkerResult

logger = logging.getLogger(__name__)


class MLAirIngestWorker:
    name = "mlair_ingest"

    def run(self, ctx: WorkerContext) -> WorkerResult:
        if ctx.metadata.get("skip_mlair_ingest"):
            return WorkerResult(ok=True, message="mlair ingest skipped (pulled source)")
        if not settings.mlair_auto_ingest:
            return WorkerResult(ok=True, message="mlair auto ingest disabled")

        client = DatasetClient()
        if not client.enabled:
            return WorkerResult(ok=True, message="mlair not configured")

        layout = ctx.store.job_layout(ctx.job_id)
        frames_dir = layout["frames"]
        if not frames_dir.is_dir() or not any(frames_dir.glob("*.jpg")):
            return WorkerResult(ok=True, message="no frames to ingest")

        source_file = Path(ctx.source_path).name

        try:
            result = client.ingest_job_frames(
                ctx.job_id,
                frames_dir,
                dataset_name=settings.mlair_dataset_name,
                dataset_id=ctx.metadata.get("mlair_dataset_id"),
                source_file=source_file,
                auto_materialize=settings.mlair_auto_materialize,
            )
            ctx.metadata["mlair_ingest"] = result

            emit_event(
                "dataset.ingest.completed",
                {
                    "job_id": ctx.job_id,
                    "rows": result.get("rows"),
                    "dataset_id": result.get("dataset_id"),
                    "current_size": result.get("current_size") or result.get("mlair_buffer_current_size"),
                    "target_threshold": result.get("target_threshold"),
                    "accumulation_strategy": result.get("accumulation_strategy"),
                    "dataset_version_id": result.get("dataset_version_id"),
                },
                job_id=ctx.job_id,
                store=ctx.store,
            )

            if result.get("materialized"):
                emit_event(
                    "dataset.version.created",
                    result["materialized"],
                    job_id=ctx.job_id,
                    store=ctx.store,
                )

            current = result.get("current_size") or result.get("mlair_buffer_current_size") or "?"
            threshold = result.get("target_threshold", "?")
            if result.get("dataset_version_id"):
                msg = f"mlair: version created (buffer {current}/{threshold})"
            else:
                msg = f"mlair: appended {result.get('rows', 0)} rows (buffer {current}/{threshold})"
            return WorkerResult(ok=True, message=msg, metadata=ctx.metadata)
        except Exception as exc:
            logger.exception("MLAir ingest failed for job %s", ctx.job_id)
            return WorkerResult(ok=False, message=f"mlair ingest failed: {exc}")
