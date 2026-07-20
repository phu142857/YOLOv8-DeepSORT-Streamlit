"""Finalize export manifest for downstream MLAir dataset ingestion."""

from __future__ import annotations

from shared.schemas import ArtifactManifest
from workers.base import Worker, WorkerContext, WorkerResult


class ExportWorker:
    name = "export"

    def run(self, ctx: WorkerContext) -> WorkerResult:
        manifest_data = ctx.metadata.get("manifest")
        if manifest_data:
            manifest = ArtifactManifest(**manifest_data)
        else:
            manifest = ctx.store.load_manifest(ctx.job_id)
        if manifest is None:
            return WorkerResult(ok=False, message="manifest missing")

        export_path = ctx.store.job_layout(ctx.job_id)["base"] / "export.json"
        payload = {
            "job_id": ctx.job_id,
            "artifacts": manifest.model_dump(),
            "ready_for_mlair_ingest": bool(manifest.frames_dir or manifest.frame_count),
        }
        ctx.store.write_json(export_path, payload)
        ctx.metadata["export_path"] = str(export_path)
        return WorkerResult(ok=True, message="export manifest written", metadata=ctx.metadata)
