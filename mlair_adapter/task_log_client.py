"""Push task log lines to MLAir run log stream (Hub Runner logs)."""

from __future__ import annotations

import json
import logging
import os
import threading
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

_MAX_LINES_PER_REQUEST = 100
_FLUSH_BATCH = int(os.getenv("MLAIR_LOG_FLUSH_BATCH", "40"))


def task_logs_url(base: str, task_id: str) -> str:
    """Task ids like ``{run_id}:train`` — keep colon unescaped (MLAir route)."""
    tid = urllib.parse.quote(task_id, safe=":")
    return f"{base.rstrip('/')}/v1/tasks/{tid}/logs"


def post_task_logs(
    base: str,
    token: str,
    *,
    worker_id: str,
    task_id: str,
    lines: list[dict[str, str]],
    timeout: float = 30,
) -> dict[str, Any]:
    if not lines:
        return {"ok": True, "appended": 0}
    url = task_logs_url(base, task_id)
    body = {"worker_id": worker_id, "lines": lines[:_MAX_LINES_PER_REQUEST]}
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


class TaskLogSink:
    """Buffered POST /v1/tasks/{task_id}/logs (max 100 lines per request)."""

    def __init__(self, base: str, token: str, worker_id: str, task_id: str) -> None:
        self._base = base
        self._token = token
        self._worker_id = worker_id
        self._task_id = task_id
        self._buf: list[dict[str, str]] = []
        self._lock = threading.Lock()
        self._dropped = 0
        self._max_buffer = int(os.getenv("MLAIR_LOG_MAX_BUFFER", "500"))

    def line(self, message: str, *, level: str = "INFO") -> None:
        msg = (message or "").strip()
        if not msg:
            return
        entry = {"level": (level or "INFO").upper()[:16], "message": msg[:8000]}
        with self._lock:
            if len(self._buf) >= self._max_buffer:
                self._dropped += 1
                return
            self._buf.append(entry)
            if len(self._buf) >= _FLUSH_BATCH:
                self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        while self._buf:
            batch = self._buf[:_MAX_LINES_PER_REQUEST]
            del self._buf[: len(batch)]
            try:
                post_task_logs(
                    self._base,
                    self._token,
                    worker_id=self._worker_id,
                    task_id=self._task_id,
                    lines=batch,
                )
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
                logger.debug("task_log_flush_failed task_id=%s err=%s", self._task_id, exc)
