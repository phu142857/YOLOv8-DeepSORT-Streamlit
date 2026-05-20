"""Detection + frame artifact extraction."""

from __future__ import annotations

from inference.pipeline import process_image_job, process_video_job
from shared.settings import settings
from workers.base import WorkerContext, WorkerResult


class DetectionWorker:
    name = "detection"

    def run(self, ctx: WorkerContext) -> WorkerResult:
        suffix = ctx.source_path.suffix.lower()
        progress = ctx.metadata.get("progress_callback")
        should_cancel = ctx.metadata.get("cancel_check")
        layout = ctx.store.job_layout(ctx.job_id)
        mlair_marker = layout["source"] / "mlair_pull.json"

        try:
            if mlair_marker.exists():
                from inference.pipeline import process_frames_dir_job

                frames_dir = layout["frames"]
                manifest = process_frames_dir_job(
                    ctx.job_id,
                    frames_dir,
                    ctx.model_name,
                    ctx.confidence,
                    ctx.store,
                    on_progress=progress,
                    should_cancel=should_cancel,
                )
                ctx.metadata["manifest"] = manifest.model_dump()
                return WorkerResult(ok=True, message="detection complete (mlair frames)", metadata=ctx.metadata)

            if suffix in settings.video_extensions:
                manifest = process_video_job(
                    ctx.job_id,
                    ctx.source_path,
                    ctx.model_name,
                    ctx.confidence,
                    ctx.store,
                    on_progress=progress,
                    should_cancel=should_cancel,
                )
            elif suffix in settings.image_extensions:
                manifest = process_image_job(
                    ctx.job_id,
                    ctx.source_path,
                    ctx.model_name,
                    ctx.confidence,
                    ctx.store,
                )
            else:
                return WorkerResult(ok=False, message=f"unsupported source type: {suffix}")

            ctx.metadata["manifest"] = manifest.model_dump()
            return WorkerResult(ok=True, message="detection complete", metadata=ctx.metadata)
        except InterruptedError:
            return WorkerResult(ok=False, message="cancelled")
        except Exception as exc:
            return WorkerResult(ok=False, message=str(exc))
