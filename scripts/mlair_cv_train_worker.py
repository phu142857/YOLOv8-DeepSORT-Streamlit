#!/usr/bin/env python3
"""
MLAir external worker for plugin ``cv_yolo_train`` (integrate-external-executor.md).

Requires:
  ML_AIR_TASK_EXECUTION_MODE=external on ml-air api + scheduler
  Plugin ``cv_yolo_train`` installed in api (integrations/mlair_cv_plugins)
  Pipeline version using examples/mlair/pipelines/cv-yolo-vehicle-train.plugin.config.json
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlair_adapter.worker_context import plugin_context_from_lease_task
from mlair_adapter.yolo_train_pipeline import run_yolo_training


def _post_json(url: str, token: str, body: dict) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(req, timeout=7200) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def main() -> None:
    base = os.getenv("MLAIR_API_BASE_URL", "http://localhost:8080").rstrip("/")
    token = (os.getenv("MLAIR_WORKER_TOKEN") or os.getenv("ML_AIR_WORKER_TOKEN") or "").strip()
    if not token:
        raise SystemExit("set MLAIR_WORKER_TOKEN")
    worker_id = os.getenv("MLAIR_WORKER_ID", "cv-yolo-train-worker").strip()
    caps = os.getenv("MLAIR_CAPABILITIES", "cv_yolo_train")

    capabilities = [c.strip() for c in caps.split(",") if c.strip()]
    lease_url = f"{base}/v1/tasks/lease"
    print(f"cv_yolo_train worker started id={worker_id} caps={capabilities}", flush=True)

    while True:
        try:
            res = _post_json(
                lease_url,
                token,
                {"worker_id": worker_id, "capabilities": capabilities, "max_tasks": 1},
            )
        except Exception as exc:  # noqa: BLE001 — keep polling through transient controller errors
            print(f"lease_error {exc}", flush=True)
            time.sleep(5)
            continue

        if res.get("execution_mode") != "external":
            print("api_not_external_mode — set ML_AIR_TASK_EXECUTION_MODE=external", flush=True)
            time.sleep(10)
            continue

        tasks = res.get("tasks") or []
        if not tasks:
            time.sleep(2)
            continue

        for task in tasks:
            tid = task["task_id"]
            plugin = task.get("plugin") or ""
            print(f"leased task_id={tid} plugin={plugin}", flush=True)
            if plugin != "cv_yolo_train":
                _post_json(
                    f"{base}/v1/tasks/{urllib.parse.quote(tid, safe='')}/fail",
                    token,
                    {"worker_id": worker_id, "error": f"unsupported_plugin:{plugin}"},
                )
                continue

            try:
                ctx = plugin_context_from_lease_task(task)
                result = run_yolo_training(ctx)
                checkpoint = str(result.get("checkpoint") or "")
                metrics = result.get("metrics") or {}
                complete_body: dict[str, Any] = {
                    "worker_id": worker_id,
                    "metrics": metrics,
                }
                if checkpoint:
                    uri = f"file://{checkpoint}" if checkpoint.startswith("/") else checkpoint
                    complete_body["artifacts"] = [{"path": "train/checkpoint", "uri": uri}]
                _post_json(
                    f"{base}/v1/tasks/{urllib.parse.quote(tid, safe='')}/complete",
                    token,
                    complete_body,
                )
                print(f"complete task_id={tid} checkpoint={checkpoint or 'none'}", flush=True)
            except Exception as exc:
                print(f"fail task_id={tid} err={exc}", flush=True)
                try:
                    _post_json(
                        f"{base}/v1/tasks/{urllib.parse.quote(tid, safe='')}/fail",
                        token,
                        {"worker_id": worker_id, "error": str(exc)},
                    )
                except Exception:
                    pass


if __name__ == "__main__":
    raise SystemExit(main())
