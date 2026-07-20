"""Tee stdout/stderr and logging into MLAir task log sink."""

from __future__ import annotations

import logging
import os
import re
import sys
import threading
import time
from typing import TextIO

from mlair_adapter.task_log_client import TaskLogSink

_LOG_FLUSH_INTERVAL_SEC = float(os.getenv("MLAIR_LOG_FLUSH_INTERVAL_SEC", "2"))

# Ultralytics/tqdm write progress + warnings to stderr — not task failures.
_STDERR_ERROR_HINTS = re.compile(
    r"(traceback|exception:|^error[:\s]|failed|fatal|cannot open|file not found)",
    re.IGNORECASE,
)
_STDERR_WARN_HINTS = re.compile(
    r"(warning|futurewarning|userwarning|deprecated)",
    re.IGNORECASE,
)
_STDERR_PROGRESS_HINTS = re.compile(
    r"(%[\s#|]|it/s\]|\d+/\d+\s+\[|█|▌|▎|▏|Scanning |Downloading )",
)


def _classify_stderr_line(line: str) -> str:
    text = line.strip()
    if not text:
        return "INFO"
    if _STDERR_ERROR_HINTS.search(text):
        return "ERROR"
    if _STDERR_WARN_HINTS.search(text):
        return "WARN"
    if _STDERR_PROGRESS_HINTS.search(text):
        return "INFO"
    return "INFO"


class _StreamTee(TextIO):
    def __init__(
        self,
        original: TextIO,
        sink: TaskLogSink,
        *,
        level: str,
        classify_stderr: bool = False,
    ) -> None:
        self._original = original
        self._sink = sink
        self._level = level
        self._classify_stderr = classify_stderr
        self._pending = ""

    def _emit(self, line: str) -> None:
        if not line.strip():
            return
        lvl = _classify_stderr_line(line) if self._classify_stderr else self._level
        self._sink.line(line.rstrip(), level=lvl)

    def write(self, data: str) -> int:
        if not data:
            return 0
        self._original.write(data)
        # tqdm progress uses \r without \n — treat as a line boundary for live Hub logs.
        if "\r" in data and "\n" not in data:
            parts = data.split("\r")
            tail = parts[-1]
            if tail.strip():
                self._emit(tail)
            return len(data)
        self._pending += data
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            self._emit(line)
        return len(data)

    def flush(self) -> None:
        self._original.flush()
        if self._pending.strip():
            self._emit(self._pending.rstrip())
            self._pending = ""

    def isatty(self) -> bool:
        return getattr(self._original, "isatty", lambda: False)()


class _LoggingHandler(logging.Handler):
    def __init__(self, sink: TaskLogSink) -> None:
        super().__init__()
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            if msg:
                self._sink.line(msg, level=record.levelname)
        except Exception:
            pass


class capture_task_logs:
    def __init__(self, sink: TaskLogSink) -> None:
        self._sink = sink
        self._orig_out: TextIO | None = None
        self._orig_err: TextIO | None = None
        self._tee_out: _StreamTee | None = None
        self._tee_err: _StreamTee | None = None
        self._handler: _LoggingHandler | None = None
        self._flush_stop = threading.Event()
        self._flush_thread: threading.Thread | None = None

    def _periodic_flush(self) -> None:
        while not self._flush_stop.wait(_LOG_FLUSH_INTERVAL_SEC):
            if self._tee_err is not None:
                self._tee_err.flush()
            if self._tee_out is not None:
                self._tee_out.flush()
            self._sink.flush()

    def __enter__(self) -> TaskLogSink:
        self._sink.line("task execution started", level="INFO")
        self._orig_out = sys.stdout
        self._orig_err = sys.stderr
        self._tee_out = _StreamTee(self._orig_out, self._sink, level="INFO")
        self._tee_err = _StreamTee(self._orig_err, self._sink, level="INFO", classify_stderr=True)
        sys.stdout = self._tee_out  # type: ignore[assignment]
        sys.stderr = self._tee_err  # type: ignore[assignment]
        self._flush_stop.clear()
        self._flush_thread = threading.Thread(
            target=self._periodic_flush,
            name=f"task-log-flush-{self._sink._task_id}",
            daemon=True,
        )
        self._flush_thread.start()
        self._handler = _LoggingHandler(self._sink)
        self._handler.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger().addHandler(self._handler)
        for name in ("ultralytics", "mlair_adapter"):
            logging.getLogger(name).addHandler(self._handler)
        return self._sink

    def __exit__(self, *exc: object) -> None:
        self._flush_stop.set()
        if self._flush_thread is not None:
            self._flush_thread.join(timeout=_LOG_FLUSH_INTERVAL_SEC + 1.0)
        if self._tee_out is not None:
            self._tee_out.flush()
        if self._tee_err is not None:
            self._tee_err.flush()
        if self._orig_out is not None:
            sys.stdout = self._orig_out
        if self._orig_err is not None:
            sys.stderr = self._orig_err
        if self._handler is not None:
            logging.getLogger().removeHandler(self._handler)
            for name in ("ultralytics", "mlair_adapter"):
                try:
                    logging.getLogger(name).removeHandler(self._handler)
                except ValueError:
                    pass
        if exc[0] is not None:
            self._sink.line(f"task error: {exc[1]}", level="ERROR")
        self._sink.flush()
