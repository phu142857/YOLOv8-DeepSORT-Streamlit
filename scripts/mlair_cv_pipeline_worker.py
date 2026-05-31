#!/usr/bin/env python3
"""
MLAir external worker for CV lifecycle plugins (Phase B DAG).

Capabilities: cv_yolo_detect, cv_yolo_prepare, cv_yolo_train, cv_yolo_eval, cv_yolo_gate, cv_hard_example_mine
"""

from __future__ import annotations

import os
import sys
import time
import urllib.error
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlair_adapter.torch_compat import apply_torch_checkpoint_compat

apply_torch_checkpoint_compat()

from mlair_adapter.run_tracking_client import build_complete_task_body, build_fail_task_body
from mlair_adapter.task_log_client import TaskLogSink
from mlair_adapter.worker_context import plugin_context_from_lease_task
from mlair_adapter.worker_log_capture import capture_task_logs
from mlair_adapter.worker_task_runtime import post_json, run_handler_with_heartbeat, task_url
from mlair_adapter.yolo_lifecycle import (
    run_detect,
    run_eval,
    run_gate,
    run_hard_example_mine,
    run_legacy_monolithic_train,
    run_prepare,
    run_train_step,
)

PLUGIN_HANDLERS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "cv_yolo_detect": run_detect,
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


def main() -> None:
    base = os.getenv("MLAIR_API_BASE_URL", "http://localhost:8080").rstrip("/")
    token = (os.getenv("MLAIR_WORKER_TOKEN") or os.getenv("ML_AIR_WORKER_TOKEN") or "").strip()
    if not token:
        raise SystemExit("set MLAIR_WORKER_TOKEN")
    worker_id = (os.getenv("MLAIR_WORKER_ID") or "").strip()
    if not worker_id:
        worker_id = (os.getenv("HOSTNAME") or "").strip() or "cv-lifecycle-worker"
    caps = os.getenv("MLAIR_CAPABILITIES", DEFAULT_CAPS)
    capabilities = [c.strip() for c in caps.split(",") if c.strip()]
    lease_url = f"{base}/v1/tasks/lease"
    print(
        f"cv lifecycle worker id={worker_id} caps={capabilities} "
        f"usage_monitor=on train_checkpoint_resolver=v2",
        flush=True,
    )

    while True:
        try:
            res = post_json(
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
                post_json(
                    task_url(base, tid, "fail"),
                    token,
                    build_fail_task_body(worker_id, f"unsupported_plugin:{plugin}"),
                )
                continue

            usage_bundle: dict[str, Any] = {}
            try:
                ctx = plugin_context_from_lease_task(task)
                log_sink = TaskLogSink(base, token, worker_id, tid)

                def _run() -> dict[str, Any]:
                    return handler(ctx)

                with capture_task_logs(log_sink):
                    result, usage_bundle = run_handler_with_heartbeat(
                        base, token, worker_id, tid, _run
                    )
                body = build_complete_task_body(
                    worker_id, result, plugin=plugin, usage_report=usage_bundle
                )
                post_json(task_url(base, tid, "complete"), token, body)
                ru = usage_bundle.get("resource_usage") or {}
                n_samples = len(usage_bundle.get("usage_samples") or [])
                print(
                    f"complete task_id={tid} plugin={plugin} step={result.get('step')} "
                    f"usage_samples={n_samples} cpu_percent_peak={ru.get('cpu_percent_peak')} "
                    f"memory_mb_peak={ru.get('memory_mb_peak')} "
                    f"gpu_percent_peak={ru.get('gpu_percent_peak')} "
                    f"gpu_memory_mb_peak={ru.get('gpu_memory_mb_peak')}",
                    flush=True,
                )
            except Exception as exc:
                print(f"fail task_id={tid} plugin={plugin} err={exc}", flush=True)
                try:
                    post_json(
                        task_url(base, tid, "fail"),
                        token,
                        build_fail_task_body(worker_id, str(exc), usage_report=usage_bundle or None),
                    )
                except Exception:
                    pass


if __name__ == "__main__":
    raise SystemExit(main())
