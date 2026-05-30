"""Shared external-worker lease helpers: heartbeat + resource usage."""

from __future__ import annotations

import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable

from mlair_adapter.resource_monitor import ResourceMonitor
from mlair_adapter.task_resource_monitor import default_sample_interval_seconds


def _heartbeat_interval_sec() -> float:
    raw = (
        os.getenv("MLAIR_HEARTBEAT_INTERVAL_SEC")
        or os.getenv("ML_AIR_RESOURCE_FLUSH_INTERVAL")
        or "3"
    ).strip()
    try:
        return max(3.0, float(raw))
    except ValueError:
        return 3.0


def post_json(url: str, token: str, body: dict, *, timeout: float = 7200) -> dict:
    import json

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


def task_url(base: str, task_id: str, suffix: str) -> str:
    tid = urllib.parse.quote(task_id, safe=":")
    return f"{base}/v1/tasks/{tid}/{suffix}"


def _heartbeat_loop(
    base: str,
    token: str,
    worker_id: str,
    task_id: str,
    stop: threading.Event,
    monitor: ResourceMonitor | None,
) -> None:
    interval = _heartbeat_interval_sec()
    url = task_url(base, task_id, "heartbeat")
    while not stop.wait(interval):
        try:
            body: dict[str, Any] = {"worker_id": worker_id}
            if monitor is not None:
                usage = monitor.latest_heartbeat_usage()
                if usage:
                    body["usage"] = usage
            post_json(url, token, body, timeout=30)
        except Exception as exc:
            print(f"heartbeat_error task_id={task_id} err={exc}", flush=True)


def run_handler_with_heartbeat(
    base: str,
    token: str,
    worker_id: str,
    task_id: str,
    handler: Callable[[], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run task handler; return (plugin_result, complete_bundle for complete/fail)."""
    stop = threading.Event()
    sample_iv = default_sample_interval_seconds()
    monitor = ResourceMonitor(
        task_id=task_id,
        flush_interval_seconds=0,
        interval_seconds=sample_iv,
    )
    with monitor:
        hb = threading.Thread(
            target=_heartbeat_loop,
            args=(base, token, worker_id, task_id, stop, monitor),
            name=f"heartbeat-{task_id}",
            daemon=True,
        )
        hb.start()
        exc_to_raise: BaseException | None = None
        result: dict[str, Any] = {}
        try:
            result = handler()
        except BaseException as exc:
            exc_to_raise = exc
        finally:
            stop.set()
            hb.join(timeout=max(_heartbeat_interval_sec() + 2.0, 5.0))
    bundle = monitor.complete_bundle()
    if exc_to_raise is not None:
        raise exc_to_raise
    return result, bundle
