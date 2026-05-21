"""Persist MLAir pipeline run state between external-worker tasks (same run_id)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from shared.settings import settings


def workspace_dir(run_id: str) -> Path:
    rid = str(run_id or "unknown").strip() or "unknown"
    d = (settings.artifact_root / "mlair_pipeline_runs" / rid).resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d


def state_path(run_id: str) -> Path:
    return workspace_dir(run_id) / "state.json"


def load_state(run_id: str) -> dict[str, Any]:
    path = state_path(run_id)
    if not path.is_file():
        return {"run_id": run_id}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {"run_id": run_id}
    except json.JSONDecodeError:
        return {"run_id": run_id}


def save_state(run_id: str, patch: dict[str, Any]) -> dict[str, Any]:
    data = load_state(run_id)
    data.update(patch)
    data["run_id"] = run_id
    state_path(run_id).write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def require_keys(state: dict[str, Any], keys: tuple[str, ...], *, step: str) -> None:
    missing = [k for k in keys if not state.get(k)]
    if missing:
        raise ValueError(f"{step} requires prior step output: missing {missing}")
