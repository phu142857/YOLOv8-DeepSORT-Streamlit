"""Aggregate counters and frame stats for readiness / governance."""

from __future__ import annotations

import json

from workers.base import Worker, WorkerContext, WorkerResult


class AggregationWorker:
    name = "aggregation"

    def run(self, ctx: WorkerContext) -> WorkerResult:
        layout = ctx.store.job_layout(ctx.job_id)
        agg_path = layout["aggregates"] / "counts.json"
        if not agg_path.exists():
            return WorkerResult(ok=False, message="aggregates missing — run detection first")

        with agg_path.open(encoding="utf-8") as f:
            aggregates = json.load(f)

        summary = {
            "frames_processed": aggregates.get("frames_processed", 1),
            "total_in": sum(aggregates.get("counters_in", {}).values()),
            "total_out": sum(aggregates.get("counters_out", {}).values()),
            "classes_in": list(aggregates.get("counters_in", {}).keys()),
            "classes_out": list(aggregates.get("counters_out", {}).keys()),
        }
        summary_path = layout["aggregates"] / "summary.json"
        ctx.store.write_json(summary_path, summary)
        ctx.metadata["aggregate_summary"] = summary
        return WorkerResult(ok=True, message="aggregation complete", metadata=ctx.metadata)
