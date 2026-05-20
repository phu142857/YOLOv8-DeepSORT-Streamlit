"""Tracking is fused in DetectionPredictor; this worker validates track artifacts."""

from __future__ import annotations

from workers.base import Worker, WorkerContext, WorkerResult


class TrackingWorker:
    name = "tracking"

    def run(self, ctx: WorkerContext) -> WorkerResult:
        layout = ctx.store.job_layout(ctx.job_id)
        tracks = layout["tracking"] / "tracks.jsonl"
        if tracks.exists():
            return WorkerResult(ok=True, message="tracking artifacts present")
        detections = layout["detections"] / "detections.jsonl"
        if detections.exists():
            return WorkerResult(ok=True, message="tracking embedded in detection pass")
        single = layout["detections"] / "detections.json"
        if single.exists():
            return WorkerResult(ok=True, message="single-frame detection complete")
        return WorkerResult(ok=False, message="no tracking artifacts found")
