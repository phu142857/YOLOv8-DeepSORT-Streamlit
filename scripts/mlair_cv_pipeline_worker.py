#!/usr/bin/env python3
"""
MLAir external worker for CV lifecycle plugins (Phase B DAG).

Capabilities: cv_yolo_split, cv_yolo_detect, cv_yolo_prepare, cv_yolo_train, cv_yolo_eval, cv_yolo_gate, cv_hard_example_mine
"""

from __future__ import annotations

import os
import socket
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

from mlair_adapter.base_client import MLAirClient
from mlair_adapter.lineage_client import ingest_lineage_blocks
from mlair_adapter.run_tracking_client import (
    artifacts_from_plugin_result,
    metrics_from_plugin_result,
    normalize_usage_bundle_for_complete,
)
from shared.settings import settings
from mlair_adapter.task_log_client import TaskLogSink
from mlair_adapter.worker_context import plugin_context_from_lease_task
from mlair_adapter.worker_log_capture import capture_task_logs
from mlair_adapter.worker_task_runtime import (
    TaskCancelled,
    post_json,
    run_handler_with_heartbeat,
)
from sdk.worker_client import post_task_complete_from_bundle, post_task_fail
from mlair_adapter.yolo_lifecycle import (
    run_detect,
    run_eval,
    run_gate,
    run_hard_example_mine,
    run_legacy_monolithic_train,
    run_prepare,
    run_split,
    run_train_step,
)

PLUGIN_HANDLERS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "cv_yolo_split": run_split,
    "cv_yolo_detect": run_detect,
    "cv_yolo_prepare": run_prepare,
    "cv_yolo_train": lambda ctx: _dispatch_train(ctx),
    "cv_yolo_eval": run_eval,
    "cv_yolo_gate": run_gate,
    "cv_hard_example_mine": run_hard_example_mine,
}

DEFAULT_CAPS = ",".join(PLUGIN_HANDLERS.keys())


def _lineage_blocks_for_post_ingest(
    complete_body: dict[str, Any],
    blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Avoid duplicating block[0] when Hub already ingests ``complete_body['lineage']``."""
    mode = settings.mlair_lineage_post_ingest
    if mode in ("0", "false", "no", "off"):
        return []
    if mode in ("all", "full", "legacy"):
        return blocks
    on_complete = complete_body.get("lineage")
    if not blocks:
        return []
    if not isinstance(on_complete, dict) or not (
        on_complete.get("inputs") or on_complete.get("outputs")
    ):
        return blocks
    return blocks[1:]


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
        worker_id = (os.getenv("HOSTNAME") or socket.gethostname() or "cv-lifecycle-worker").strip()
    caps = os.getenv("MLAIR_CAPABILITIES", DEFAULT_CAPS)
    capabilities = [c.strip() for c in caps.split(",") if c.strip()]
    lease_url = f"{base}/v1/tasks/lease"
    print(
        f"cv lifecycle worker id={worker_id} caps={capabilities} "
        f"usage_monitor=sdk train_checkpoint_resolver=v2",
        flush=True,
    )

    while True:
        try:
            res = post_json(
                lease_url,
                token,
                {"worker_id": worker_id, "capabilities": capabilities, "max_tasks": 1},
            )
        except Exception as exc:  # noqa: BLE001 — keep polling through 502/refused/reset on controller
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
                post_task_fail(
                    tid,
                    worker_id=worker_id,
                    error=f"unsupported_plugin:{plugin}",
                    token=token,
                    base_url=base,
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
                result_lineage = result.get("lineage")
                complete_lineage = (
                    result_lineage
                    if isinstance(result_lineage, dict)
                    and (result_lineage.get("inputs") or result_lineage.get("outputs"))
                    else None
                )
                artifacts = artifacts_from_plugin_result(result, plugin=plugin)
                post_task_complete_from_bundle(
                    tid,
                    worker_id=worker_id,
                    usage_bundle=normalize_usage_bundle_for_complete(usage_bundle),
                    metrics=metrics_from_plugin_result(result),
                    artifacts=artifacts or None,
                    lineage=complete_lineage,
                    token=token,
                    base_url=base,
                )

                run_id = str(task.get("run_id") or result.get("run_id") or "")
                lineage_blocks: list[dict[str, Any]] = []
                raw_ingests = result.get("lineage_ingests")
                if isinstance(raw_ingests, list):
                    lineage_blocks = [b for b in raw_ingests if isinstance(b, dict)]
                if not lineage_blocks:
                    legacy = result.get("lineage")
                    if isinstance(legacy, dict) and (legacy.get("inputs") or legacy.get("outputs")):
                        lineage_blocks = [legacy]
                post_blocks = _lineage_blocks_for_post_ingest(
                    {"lineage": complete_lineage}, lineage_blocks
                )
                if post_blocks and run_id:
                    hub = MLAirClient(base_url=base, token=token)
                    for ing_out in ingest_lineage_blocks(
                        hub, run_id=run_id, task_id=tid, blocks=post_blocks
                    ):
                        print(
                            f"lineage_ingest_post task_id={tid} run_id={run_id} "
                            f"edges={ing_out.get('edges')} ingested={ing_out.get('ingested')}",
                            flush=True,
                        )

                ru = usage_bundle.get("resource_usage") or {}
                n_samples = len(usage_bundle.get("usage_samples") or [])
                print(
                    f"complete task_id={tid} plugin={plugin} step={result.get('step')} "
                    f"log_lines={log_sink.posted_lines} usage_samples={n_samples} "
                    f"cpu_percent_peak={ru.get('cpu_percent_peak')} "
                    f"memory_mb_peak={ru.get('memory_mb_peak')} "
                    f"gpu_percent_peak={ru.get('gpu_percent_peak')} "
                    f"gpu_memory_mb_peak={ru.get('gpu_memory_mb_peak')}",
                    flush=True,
                )
            except TaskCancelled:
                # MLAir already marked the task/run CANCELLED; don't post complete/fail.
                print(f"cancelled task_id={tid} plugin={plugin} — aborted by cancel", flush=True)
                continue
            except Exception as exc:
                print(f"fail task_id={tid} plugin={plugin} err={exc}", flush=True)
                try:
                    post_task_fail(
                        tid,
                        worker_id=worker_id,
                        error=str(exc),
                        usage_bundle=normalize_usage_bundle_for_complete(usage_bundle) or None,
                        token=token,
                        base_url=base,
                    )
                except Exception:
                    pass


if __name__ == "__main__":
    raise SystemExit(main())
