"""Readiness and eligibility against MLAir policies."""

from __future__ import annotations

from typing import Any

import httpx

from mlair_adapter.base_client import MLAirClient


def normalize_readiness(raw: dict[str, Any]) -> dict[str, Any]:
    """Map MLAir readiness payload to a stable shape for UI and workers."""
    status = str(raw.get("status") or raw.get("readiness_status") or raw.get("state") or "")
    ready = raw.get("ready")
    if ready is None:
        ready = status.upper() in {"READY", "PASS", "PASSED", "OK"}
    reasons: list[str] = []
    for key in ("reasons", "blocking_reasons", "why_blocked"):
        val = raw.get(key)
        if isinstance(val, list):
            for item in val:
                if isinstance(item, str):
                    reasons.append(item)
                elif isinstance(item, dict):
                    reasons.append(str(item.get("message") or item.get("reason") or item))
        elif isinstance(val, dict):
            reasons.extend(f"{k}: {v}" for k, v in val.items())
    return {
        "dataset_id": raw.get("dataset_id"),
        "dataset_version_id": raw.get("dataset_version_id"),
        "status": status,
        "ready": bool(ready),
        "reasons": reasons,
        "raw": raw,
    }


class ReadinessClient(MLAirClient):
    def get_readiness(
        self,
        dataset_id: str,
        *,
        dataset_version_id: str | None = None,
        policy_id: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, str] = {}
        if dataset_version_id:
            params["dataset_version_id"] = dataset_version_id
        if policy_id:
            params["policy_id"] = policy_id
        raw = self.get(f"{self._prefix()}/datasets/{dataset_id}/readiness", params=params or None)
        return normalize_readiness(raw)

    def evaluate_readiness(
        self,
        dataset_id: str,
        *,
        dataset_version_id: str | None = None,
        policy_id: str | None = None,
        source: str = "cv_workload",
    ) -> dict[str, Any]:
        params: dict[str, str] = {"source": source}
        if dataset_version_id:
            params["dataset_version_id"] = dataset_version_id
        if policy_id:
            params["policy_id"] = policy_id
        url = f"{self.base_url}{self._prefix()}/datasets/{dataset_id}/readiness/evaluate"
        r = httpx.post(url, headers=self._headers(), params=params, timeout=self.timeout)
        if r.status_code == 422:
            body = r.json() if r.content else {}
            nested = body.get("detail") if isinstance(body, dict) else None
            reason = nested.get("reason") if isinstance(nested, dict) else None
            if reason == "DATASET_VERSION_REQUIRED" or body.get("detail") == "dataset_version_id_required":
                raise ValueError(
                    "dataset_version_id_required: pin dataset_version_id or wait until accumulation materializes a version"
                ) from None
        r.raise_for_status()
        raw = r.json() if r.content else {}
        return normalize_readiness(raw)

    def get_eligibility(
        self,
        dataset_id: str,
        *,
        dataset_version_id: str | None = None,
    ) -> dict[str, Any]:
        params = {}
        if dataset_version_id:
            params["dataset_version_id"] = dataset_version_id
        return self.get(f"{self._prefix()}/datasets/{dataset_id}/eligibility", params=params or None)
