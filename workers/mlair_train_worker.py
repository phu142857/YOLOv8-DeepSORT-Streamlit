"""Trigger MLAir training when dataset readiness is READY (Phase 3)."""

from __future__ import annotations

import logging

from mlair_adapter.events_client import emit_event
from mlair_adapter.model_client import ModelClient
from mlair_adapter.training_client import TrainingClient
from shared.settings import settings
from workers.base import WorkerContext, WorkerResult

logger = logging.getLogger(__name__)


class MLAirTrainWorker:
    name = "mlair_train"

    def run(self, ctx: WorkerContext) -> WorkerResult:
        if not settings.mlair_auto_train:
            return WorkerResult(ok=True, message="auto train disabled")

        if ctx.metadata.get("skip_mlair_ingest"):
            return WorkerResult(ok=True, message="train skipped (mlair pull source)")

        model_id = settings.mlair_model_id
        if not model_id:
            return WorkerResult(ok=True, message="train skipped (CV_MLAIR_MODEL_ID unset)")

        ingest = ctx.metadata.get("mlair_ingest") or {}
        dataset_id = ingest.get("dataset_id") or ctx.metadata.get("mlair_dataset_id")
        version_id = ingest.get("dataset_version_id") or ctx.metadata.get("mlair_dataset_version_id")

        if not dataset_id:
            return WorkerResult(ok=True, message="train skipped (no dataset_id)")
        if not version_id:
            return WorkerResult(ok=True, message="train skipped (pin dataset_version_id first)")

        readiness = ctx.metadata.get("mlair_readiness") or {}
        if not readiness.get("ready"):
            reasons = readiness.get("reasons") or []
            detail = reasons[0] if reasons else readiness.get("status", "NOT READY")
            return WorkerResult(ok=True, message=f"train gated: {detail}")

        train_client = TrainingClient()
        if not train_client.enabled:
            return WorkerResult(ok=True, message="mlair not configured")

        try:
            trigger = train_client.trigger_run_by_model(
                model_id,
                dataset_id,
                dataset_version_id=version_id,
                idempotency_key=f"cv-job-{ctx.job_id}",
                context={"source": "cv_workload", "job_id": ctx.job_id},
            )
            ctx.metadata["mlair_training"] = {"trigger": trigger}

            if TrainingClient.is_blocked(trigger):
                emit_event(
                    "training.run.blocked",
                    {"model_id": model_id, "dataset_version_id": version_id, "trigger": trigger},
                    job_id=ctx.job_id,
                    store=ctx.store,
                )
                return WorkerResult(ok=True, message="training blocked by readiness gate")

            run_id = TrainingClient.run_id_from_trigger(trigger)
            if not run_id:
                return WorkerResult(ok=False, message="training trigger returned no run_id")

            polled = train_client.poll_run(
                run_id,
                timeout_sec=settings.mlair_train_poll_timeout_sec,
                interval_sec=settings.mlair_train_poll_interval_sec,
            )
            ctx.metadata["mlair_training"]["run_id"] = run_id
            ctx.metadata["mlair_training"]["run"] = polled

            success = bool(polled.get("_poll_success"))
            status = str(polled.get("status") or "unknown")

            if success:
                emit_event(
                    "training.run.completed",
                    {"run_id": run_id, "model_id": model_id, "status": status},
                    job_id=ctx.job_id,
                    store=ctx.store,
                )
                model_version = self._maybe_promote(model_id, run_id, ctx)
                if model_version:
                    ctx.metadata["mlair_model_version"] = model_version
            else:
                emit_event(
                    "training.run.failed",
                    {"run_id": run_id, "status": status, "run": polled},
                    job_id=ctx.job_id,
                    store=ctx.store,
                )

            if not success and settings.mlair_train_fail_pipeline:
                return WorkerResult(ok=False, message=f"training run {status}")

            return WorkerResult(
                ok=True,
                message=f"training run {run_id[:8]}… → {status}",
                metadata=ctx.metadata,
            )
        except Exception as exc:
            logger.exception("Training failed for job %s", ctx.job_id)
            if settings.mlair_train_fail_pipeline:
                return WorkerResult(ok=False, message=f"training failed: {exc}")
            return WorkerResult(ok=True, message=f"training error (non-fatal): {exc}")

    def _maybe_promote(self, model_id: str, run_id: str, ctx: WorkerContext) -> dict | None:
        if not settings.mlair_auto_promote:
            return None
        model_client = ModelClient()
        if not model_client.enabled:
            return None
        version_row = model_client.find_latest_version(model_id, run_id=run_id)
        if not version_row:
            version_row = model_client.find_latest_version(model_id)
        if not version_row:
            return None
        version_num = int(version_row.get("version") or 0)
        if version_num < 1:
            return None
        promoted = model_client.promote(
            model_id,
            version_num,
            stage=settings.mlair_promote_stage,
        )
        emit_event(
            "model.version.promoted",
            {"model_id": model_id, "version": version_num, "stage": settings.mlair_promote_stage},
            job_id=ctx.job_id,
            store=ctx.store,
        )
        return promoted
