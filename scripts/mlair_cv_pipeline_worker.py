#!/usr/bin/env python3
"""
MLAir external worker for CV lifecycle plugins (Phase B DAG).

Capabilities: cv_yolo_prepare, cv_yolo_train, cv_yolo_eval, cv_yolo_gate, cv_hard_example_mine
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlair_adapter.torch_compat import apply_torch_checkpoint_compat

apply_torch_checkpoint_compat()

from mlair_adapter.task_log_client import TaskLogSink
from mlair_adapter.worker_context import plugin_context_from_lease_task
from mlair_adapter.worker_log_capture import capture_task_logs
from mlair_adapter.yolo_lifecycle import (
    run_eval,
    run_gate,
    run_hard_example_mine,
    run_legacy_monolithic_train,
    run_prepare,
    run_train_step,
)

PLUGIN_HANDLERS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "cv_yolo_prepare": run_prepare,
    "cv_yolo_train": lambda ctx: _dispatch_train(ctx),
    "cv_yolo_eval": run_eval,
    "cv_yolo_gate": run_gate,
    "cv_hard_example_mine": run_hard_example_mine,
}

DEFAULT_CAPS = ",".join(PLUGIN_HANDLERS.keys())


def _dispatch_train(ctx: dict[str, Any]) -> dict[str, Any]:
    """Lifecycle DAG uses prepare→train; legacy single-task pipeline uses monolithic train."""
    run_id = str(ctx.get("run_id") or "")
    from mlair_adapter.run_workspace import load_state

    state = load_state(run_id)
    if state.get("prepare_ok"):
        return run_train_step(ctx)
    return run_legacy_monolithic_train(ctx)


def _post_json(url: str, token: str, body: dict, *, timeout: float = 7200) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def _task_url(base: str, task_id: str, suffix: str) -> str:
    tid = urllib.parse.quote(task_id, safe=":")
    return f"{base}/v1/tasks/{tid}/{suffix}"


def _heartbeat_loop(
    base: str,
    token: str,
    worker_id: str,
    task_id: str,
    stop: threading.Event,
) -> None:
    """Keep lease alive during long YOLO train (ML_AIR_TASK_LEASE_SECONDS default is 30s)."""
    interval = max(5, int(os.getenv("MLAIR_HEARTBEAT_INTERVAL_SEC", "15")))
    url = _task_url(base, task_id, "heartbeat")
    while not stop.wait(interval):
        try:
            _post_json(url, token, {"worker_id": worker_id}, timeout=30)
        except Exception as exc:
            print(f"heartbeat_error task_id={task_id} err={exc}", flush=True)


def _run_handler_with_heartbeat(
    base: str,
    token: str,
    worker_id: str,
    task_id: str,
    handler: Callable[[dict[str, Any]], dict[str, Any]],
    ctx: dict[str, Any],
) -> dict[str, Any]:
    stop = threading.Event()
    hb = threading.Thread(
        target=_heartbeat_loop,
        args=(base, token, worker_id, task_id, stop),
        name=f"heartbeat-{task_id}",
        daemon=True,
    )
    hb.start()
    try:
        return handler(ctx)
    finally:
        stop.set()
        hb.join(timeout=5)


def _metrics_from_result(result: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(result.get("metrics") or {})
    if result.get("mAP50") is not None:
        metrics["mAP50"] = result["mAP50"]
    if result.get("production_map50") is not None:
        metrics["production_mAP50"] = result["production_map50"]
    gate = result.get("gate")
    if isinstance(gate, dict):
        for k in ("candidate_map50", "production_map50", "passed"):
            if gate.get(k) is not None:
                metrics[f"gate_{k}"] = gate[k]
    return metrics


def main() -> None:
    base = os.getenv("MLAIR_API_BASE_URL", "http://localhost:8080").rstrip("/")
    token = (os.getenv("MLAIR_WORKER_TOKEN") or os.getenv("ML_AIR_WORKER_TOKEN") or "").strip()
    if not token:
        raise SystemExit("set MLAIR_WORKER_TOKEN")
    worker_id = os.getenv("MLAIR_WORKER_ID", "cv-lifecycle-worker").strip()
    caps = os.getenv("MLAIR_CAPABILITIES", DEFAULT_CAPS)
    capabilities = [c.strip() for c in caps.split(",") if c.strip()]
    lease_url = f"{base}/v1/tasks/lease"
    print(
        f"cv lifecycle worker id={worker_id} caps={capabilities} train_checkpoint_resolver=v2",
        flush=True,
    )

    while True:
        try:
            res = _post_json(
                lease_url,
                token,
                {"worker_id": worker_id, "capabilities": capabilities, "max_tasks": 1},
            )
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
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
            plugin = str(task.get("plugin") or "").strip()
            print(f"leased task_id={tid} plugin={plugin}", flush=True)
            handler = PLUGIN_HANDLERS.get(plugin)
            if not handler:
                _post_json(
                    _task_url(base, tid, "fail"),
                    token,
                    {"worker_id": worker_id, "error": f"unsupported_plugin:{plugin}"},
                )
                continue

            try:
                ctx = plugin_context_from_lease_task(task)
                log_sink = TaskLogSink(base, token, worker_id, tid)
                with capture_task_logs(log_sink):
                    result = _run_handler_with_heartbeat(base, token, worker_id, tid, handler, ctx)
                checkpoint = str(result.get("checkpoint") or "")
                metrics = _metrics_from_result(result)
                body: dict[str, Any] = {"worker_id": worker_id, "metrics": metrics}
                if checkpoint:
                    body["artifact_uri"] = (
                        f"file://{checkpoint}" if checkpoint.startswith("/") else checkpoint
                    )
                _post_json(_task_url(base, tid, "complete"), token, body)
                print(f"complete task_id={tid} plugin={plugin} step={result.get('step')}", flush=True)
            except Exception as exc:
                print(f"fail task_id={tid} plugin={plugin} err={exc}", flush=True)
                try:
                    _post_json(
                        _task_url(base, tid, "fail"),
                        token,
                        {"worker_id": worker_id, "error": str(exc)},
                    )
                except Exception:
                    pass


if __name__ == "__main__":
    raise SystemExit(main())
