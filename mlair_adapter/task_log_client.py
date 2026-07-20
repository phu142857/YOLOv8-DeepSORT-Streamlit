"""Push task log lines to MLAir run log stream (Hub task detail / Runner logs)."""

from __future__ import annotations

import json
import logging
import os
import threading
import urllib.error
from typing import Any

from sdk.worker_client import post_task_logs as sdk_post_task_logs

logger = logging.getLogger(__name__)

_MAX_LINES_PER_REQUEST = 100
_FLUSH_BATCH = int(os.getenv("MLAIR_LOG_FLUSH_BATCH", "5"))


def post_task_logs(
    base: str,
    token: str,
    *,
    worker_id: str,
    task_id: str,
    lines: list[dict[str, str]],
    timeout: float = 30,
) -> dict[str, Any]:
    """POST ``/v1/tasks/{task_id}/logs`` via the official MLAir SDK."""
    del timeout  # sdk uses its own default
    if not lines:
        return {"ok": True, "appended": 0}
    return sdk_post_task_logs(
        task_id,
        worker_id=worker_id,
        lines=lines[:_MAX_LINES_PER_REQUEST],
        token=token,
        base_url=base,
    )


def _format_log_http_error(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        detail = json.loads(body).get("detail", body) if body else ""
    except Exception:
        detail = ""
    return f"HTTP {exc.code} detail={detail!r}"


class TaskLogSink:
    """Buffered POST /v1/tasks/{task_id}/logs (max 100 lines per request).

    MLAir only accepts logs while the task is ``RUNNING`` and ``leased_by`` matches
    ``worker_id``. Flush frequently (default every 5 lines + periodic timer) so Hub
    task detail shows live logs and the final buffer is not lost on fast tasks.
    """

    def __init__(self, base: str, token: str, worker_id: str, task_id: str) -> None:
        self._base = base
        self._token = token
        self._worker_id = worker_id
        self._task_id = task_id
        self._buf: list[dict[str, str]] = []
        self._lock = threading.Lock()
        self._dropped = 0
        self._posted = 0
        self._max_buffer = int(os.getenv("MLAIR_LOG_MAX_BUFFER", "500"))
        self._ever_flushed = False

    @property
    def posted_lines(self) -> int:
        return self._posted

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
            if not self._ever_flushed or len(self._buf) >= _FLUSH_BATCH:
                self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        while self._buf:
            batch = self._buf[:_MAX_LINES_PER_REQUEST]
            del self._buf[: len(batch)]
            try:
                out = post_task_logs(
                    self._base,
                    self._token,
                    worker_id=self._worker_id,
                    task_id=self._task_id,
                    lines=batch,
                )
                self._ever_flushed = True
                self._posted += int(out.get("appended") or len(batch))
            except urllib.error.HTTPError as exc:
                detail = _format_log_http_error(exc)
                # 409 = task no longer leased (cancelled/completed) — drop remainder quietly.
                if exc.code == 409:
                    logger.debug("task_log_flush_skipped task_id=%s %s", self._task_id, detail)
                    return
                logger.warning("task_log_flush_failed task_id=%s %s", self._task_id, detail)
                print(f"task_log_flush_failed task_id={self._task_id} {detail}", flush=True)
                return
            except Exception as exc:
                logger.warning("task_log_flush_failed task_id=%s err=%s", self._task_id, exc)
                print(f"task_log_flush_failed task_id={self._task_id} err={exc}", flush=True)
                return
