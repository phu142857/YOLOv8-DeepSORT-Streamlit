"""External-worker lease helpers: heartbeat, resource usage (SDK) and cancellation.

For **external execution mode** MLAir cannot measure the worker's CPU/RAM/GPU on its own
(the worker runs in a separate container/process). The worker task API only *stores* the
usage the worker posts (`heartbeat.usage`, `complete.usage_samples`/`resource_usage`).
So sampling runs inside the worker using MLAir's own SDK ``sdk.resource_monitor``.

Cancellation: when a run/task is cancelled (or the lease is lost), MLAir's
``POST /v1/tasks/{id}/heartbeat`` returns ``{"ok": false}`` (its UPDATE only matches a
still-``RUNNING`` task leased by this worker). The heartbeat loop watches for that and
sets a cancel ``Event``; running plugin code (training callback / detect loop) calls
``raise_if_cancelled()`` to abort promptly instead of running to completion.
"""

from __future__ import annotations

import contextvars
import os
import threading
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable

from sdk.resource_monitor import ResourceMonitor, default_sample_interval_seconds


class TaskCancelled(Exception):
    """Raised inside a plugin handler when its MLAir task was cancelled/lease lost."""


_cancel_event: contextvars.ContextVar[threading.Event | None] = contextvars.ContextVar(
    "_cancel_event",
    default=None,
)


def current_cancel_event() -> threading.Event | None:
    return _cancel_event.get()


def is_cancelled() -> bool:
    ev = _cancel_event.get()
    return ev is not None and ev.is_set()


def raise_if_cancelled() -> None:
    """Abort the in-flight task if MLAir signalled cancellation. Call in long loops."""
    if is_cancelled():
        raise TaskCancelled()


def _heartbeat_interval_sec() -> float:
    raw = (os.getenv("MLAIR_HEARTBEAT_INTERVAL_SEC") or "3").strip()
    try:
        return max(1.0, float(raw))
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
    auth_token: str,
    worker_id: str,
    task_id: str,
    stop: threading.Event,
    monitor: ResourceMonitor | None,
    cancel: threading.Event,
) -> None:
    interval = _heartbeat_interval_sec()
    url = task_url(base, task_id, "heartbeat")
    while True:
        try:
            body: dict[str, Any] = {"worker_id": worker_id}
            if monitor is not None:
                usage = monitor.latest_heartbeat_usage()
                if usage:
                    body["usage"] = usage
            resp = post_json(url, auth_token, body, timeout=30)
            # MLAir returns ok=false once the task is no longer RUNNING+leased by us
            # (cancelled, or lease lost/taken) → signal the handler to abort.
            if isinstance(resp, dict) and resp.get("ok") is False and not cancel.is_set():
                print(f"cancel_detected task_id={task_id} (heartbeat ok=false)", flush=True)
                cancel.set()
        except Exception as exc:
            print(f"heartbeat_error task_id={task_id} err={exc}", flush=True)
        if stop.wait(interval):
            break


def run_handler_with_heartbeat(
    base: str,
    token: str,
    worker_id: str,
    task_id: str,
    handler: Callable[[], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the task handler under the SDK resource monitor + lease heartbeat.

    Returns ``(plugin_result, complete_bundle)``. Raises :class:`TaskCancelled` if MLAir
    cancelled the task while it ran (handler code aborts via ``raise_if_cancelled`` /
    the training callback).
    """
    stop = threading.Event()
    cancel = threading.Event()
    monitor = ResourceMonitor(
        task_id=task_id,
        interval_seconds=default_sample_interval_seconds(),
        flush_interval_seconds=0,
    )
    token_ctx = _cancel_event.set(cancel)
    try:
        # ``with monitor`` starts the sampling thread (see __enter__); __exit__ stops it.
        with monitor:
            hb = threading.Thread(
                target=_heartbeat_loop,
                args=(base, token, worker_id, task_id, stop, monitor, cancel),
                name=f"heartbeat-{task_id}",
                daemon=True,
            )
            hb.start()
            try:
                result = handler()
            finally:
                stop.set()
                hb.join(timeout=max(_heartbeat_interval_sec() + 2.0, 5.0))
        if cancel.is_set():
            raise TaskCancelled()
        return result, monitor.complete_bundle()
    finally:
        _cancel_event.reset(token_ctx)
