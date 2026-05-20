"""Training run trigger and polling (MLAir Phase 3)."""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from mlair_adapter.base_client import MLAirClient

logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = frozenset(
    {"success", "succeeded", "completed", "failed", "cancelled", "blocked", "error"}
)
_SUCCESS_STATUSES = frozenset({"success", "succeeded", "completed"})


class TrainingClient(MLAirClient):
    def trigger_run_by_model(
        self,
        model_id: str,
        dataset_id: str,
        *,
        dataset_version_id: str | None = None,
        experiment_id: str | None = None,
        context: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
        priority: str = "normal",
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model_id": model_id,
            "dataset_id": dataset_id,
            "priority": priority,
        }
        if dataset_version_id:
            body["dataset_version_id"] = dataset_version_id
        if experiment_id:
            body["experiment_id"] = experiment_id
        if context:
            body["context"] = context
        if idempotency_key:
            body["idempotency_key"] = idempotency_key
        return self.post(f"{self._prefix()}/runs/trigger", json=body)

    def get_run(self, run_id: str) -> dict[str, Any]:
        return self.get(f"{self._prefix()}/runs/{run_id}")

    def poll_run(
        self,
        run_id: str,
        *,
        timeout_sec: float = 3600.0,
        interval_sec: float = 5.0,
    ) -> dict[str, Any]:
        deadline = time.time() + timeout_sec
        last: dict[str, Any] = {}
        while time.time() < deadline:
            last = self.get_run(run_id)
            status = str(last.get("status") or "").lower()
            if status in _TERMINAL_STATUSES:
                last["_poll_terminal"] = True
                last["_poll_success"] = status in _SUCCESS_STATUSES
                return last
            time.sleep(interval_sec)
        last["_poll_terminal"] = False
        last["_poll_success"] = False
        last["_poll_timeout"] = True
        return last

    @staticmethod
    def run_id_from_trigger(response: dict[str, Any]) -> str | None:
        return response.get("run_id") or response.get("id")

    @staticmethod
    def is_blocked(response: dict[str, Any]) -> bool:
        if response.get("blocked_by_gate"):
            return True
        status = str(response.get("status") or "").lower()
        return status == "blocked"
