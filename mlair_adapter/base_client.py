"""HTTP client for MLAir /v1 APIs."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from shared.settings import settings

logger = logging.getLogger(__name__)


def items_from_response(data: Any) -> list[dict[str, Any]]:
    """Parse MLAir list endpoints: ``{"items": [...]}`` or a bare list."""
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        items = data.get("items")
        if isinstance(items, list):
            return [row for row in items if isinstance(row, dict)]
    return []


class MLAirClient:
    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        tenant: str | None = None,
        project: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = (base_url or settings.mlair_api_url).rstrip("/")
        self.token = token or settings.mlair_token
        self.tenant = tenant or settings.mlair_tenant
        self.project = project or settings.mlair_project
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.token)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    def _prefix(self) -> str:
        return f"/v1/tenants/{self.tenant}/projects/{self.project}"

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        r = httpx.get(url, headers=self._headers(), params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json() if r.content else {}

    def post(self, path: str, json: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        r = httpx.post(url, headers={**self._headers(), "Content-Type": "application/json"}, json=json, timeout=self.timeout)
        r.raise_for_status()
        return r.json() if r.content else {}

    def patch(self, path: str, json: dict[str, Any]) -> Any:
        url = f"{self.base_url}{path}"
        r = httpx.patch(url, headers={**self._headers(), "Content-Type": "application/json"}, json=json, timeout=self.timeout)
        r.raise_for_status()
        return r.json() if r.content else {}

    def put(self, path: str, json: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        r = httpx.put(url, headers={**self._headers(), "Content-Type": "application/json"}, json=json, timeout=self.timeout)
        r.raise_for_status()
        return r.json() if r.content else {}

    def post_multipart(self, path: str, data: dict[str, str], files: dict[str, Any]) -> Any:
        url = f"{self.base_url}{path}"
        r = httpx.post(url, headers=self._headers(), data=data, files=files, timeout=self.timeout)
        r.raise_for_status()
        return r.json() if r.content else {}

    def get_bytes(self, path: str) -> bytes:
        url = f"{self.base_url}{path}"
        r = httpx.get(url, headers=self._headers(), timeout=self.timeout)
        r.raise_for_status()
        return r.content
