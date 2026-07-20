"""Build lifecycle timeline from job record, worker steps, and pipeline log."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from shared.artifacts import ArtifactStore
from shared.schemas import JobLifecycleResponse, JobResponse, LifecycleStep

_EVENT_RE = re.compile(r"EVENT\s+(\S+)\s+(\{.*\})")


def _parse_log_events(log_path: Path) -> list[LifecycleStep]:
    if not log_path.is_file():
        return []
    steps: list[LifecycleStep] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if "EVENT " not in line:
            continue
        m = _EVENT_RE.search(line)
        if not m:
            continue
        event_type, payload_raw = m.group(1), m.group(2)
        try:
            payload = json.loads(payload_raw)
        except json.JSONDecodeError:
            payload = {"raw": payload_raw}
        steps.append(
            LifecycleStep(
                id=event_type,
                label=event_type.replace(".", " ").replace("_", " ").title(),
                status="done",
                detail=str(payload)[:200],
                payload=payload,
            )
        )
    return steps


def build_job_lifecycle(job: JobResponse, store: ArtifactStore | None = None) -> JobLifecycleResponse:
    artifact_store = store or ArtifactStore()
    layout = artifact_store.job_layout(job.id)
    steps: list[LifecycleStep] = []

    steps.append(
        LifecycleStep(
            id="job.created",
            label="Job created",
            status="done",
            detail=f"{job.source_type.value} · {job.model_name}",
        )
    )

    worker_order = (
        "detection",
        "tracking",
        "aggregation",
        "export",
        "mlair_ingest",
        "mlair_readiness",
        "mlair_train",
    )
    for name in worker_order:
        step_file = layout["steps"] / f"{name}.json"
        if not step_file.exists():
            continue
        with step_file.open(encoding="utf-8") as f:
            data = json.load(f)
        ok = data.get("ok", True)
        steps.append(
            LifecycleStep(
                id=f"worker.{name}",
                label=name.replace("_", " ").title(),
                status="done" if ok else "failed",
                detail=data.get("message", ""),
            )
        )

    steps.extend(_parse_log_events(layout["logs"] / "pipeline.log"))

    if job.mlair_dataset_id:
        ver = job.mlair_dataset_version_id or "—"
        steps.append(
            LifecycleStep(
                id="mlair.dataset",
                label="MLAir dataset linked",
                status="done",
                detail=f"dataset={job.mlair_dataset_id[:12]}… version={ver[:12] if ver != '—' else ver}",
            )
        )

    if job.mlair_readiness:
        r = job.mlair_readiness
        ready = r.get("ready", False)
        steps.append(
            LifecycleStep(
                id="dataset.readiness",
                label="Readiness evaluation",
                status="done" if ready else "blocked",
                detail=r.get("status", ""),
                payload=r,
            )
        )

    if job.mlair_training:
        tr = job.mlair_training
        run = tr.get("run") or {}
        status = str(run.get("status") or tr.get("trigger", {}).get("status") or "")
        success = bool(run.get("_poll_success"))
        steps.append(
            LifecycleStep(
                id="training.run",
                label="Training run",
                status="done" if success else ("failed" if run.get("_poll_terminal") else "pending"),
                detail=f"run={tr.get('run_id', '')[:12]}… {status}".strip(),
                payload=tr,
            )
        )

    if job.mlair_model_version:
        mv = job.mlair_model_version
        steps.append(
            LifecycleStep(
                id="model.promoted",
                label="Model version promoted",
                status="done",
                detail=f"v{mv.get('version', '?')} · {mv.get('stage', '')}",
                payload=mv if isinstance(mv, dict) else {},
            )
        )

    if job.status.value == "completed":
        steps.append(
            LifecycleStep(
                id="job.completed",
                label="Pipeline complete",
                status="done",
                detail=job.message,
            )
        )
    elif job.status.value == "failed":
        steps.append(
            LifecycleStep(
                id="job.failed",
                label="Pipeline failed",
                status="failed",
                detail=job.error or job.message,
            )
        )

    return JobLifecycleResponse(job_id=job.id, steps=steps, job_status=job.status.value)
