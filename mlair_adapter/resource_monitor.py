"""Resource Usage Contract v1 — context manager for CV external workers."""

from __future__ import annotations

import os
from typing import Any

from mlair_adapter.task_resource_monitor import TaskResourceMonitor
from mlair_adapter.usage_contract import (
    contract_complete_resource_usage,
    contract_heartbeat_from_sample,
    contract_summary_from_report,
)


class ResourceMonitor:
    """Same surface as ml-air ``sdk.resource_monitor.ResourceMonitor``."""

    def __init__(
        self,
        *,
        task_id: str | None = None,
        pid: int | None = None,
        interval_seconds: float | None = None,
        flush_interval_seconds: float | None = None,
        autostart_pid: bool = True,
    ) -> None:
        self._pid = pid
        self._autostart_pid = autostart_pid
        self._inner = TaskResourceMonitor(
            task_id=task_id,
            interval_seconds=interval_seconds,
            flush_interval_seconds=flush_interval_seconds,
        )
        self._report: dict[str, Any] | None = None

    def __enter__(self) -> ResourceMonitor:
        if self._autostart_pid:
            self._inner.start(self._pid if self._pid is not None else os.getpid())
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        if self._inner.root_pid is not None or self._inner._thread is not None:
            self._report = self._inner.stop()
        return False

    def start(self, pid: int) -> None:
        self._inner.start(pid)

    def attach_pid(self, pid: int) -> None:
        self._inner.attach_pid(pid)

    def stop(self) -> dict[str, Any]:
        self._report = self._inner.stop()
        return self._report

    def _report_or_build(self) -> dict[str, Any]:
        if self._report is not None:
            return self._report
        return self._inner.build_report()

    def summary(self) -> dict[str, Any]:
        return contract_summary_from_report(self._report_or_build())

    def complete_bundle(self) -> dict[str, Any]:
        report = self._report_or_build()
        return {
            "resource_usage": contract_complete_resource_usage(report),
            "usage_samples": report.get("usage_samples") or [],
        }

    def latest_heartbeat_usage(self) -> dict[str, Any] | None:
        report = self._report_or_build()
        samples = report.get("usage_samples") or []
        if samples and isinstance(samples[-1], dict):
            return contract_heartbeat_from_sample(samples[-1])
        return contract_heartbeat_from_sample(self._inner.sample_once())
