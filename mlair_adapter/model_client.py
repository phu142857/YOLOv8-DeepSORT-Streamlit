"""Model registry — versions, promote, artifact download (MLAir Phase 3)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from mlair_adapter.base_client import MLAirClient

logger = logging.getLogger(__name__)


class ModelClient(MLAirClient):
    def list_models(self, limit: int = 100) -> list[dict[str, Any]]:
        data = self.get(f"{self._prefix()}/models", params={"limit": limit})
        return list(data.get("items") or [])

    def get_model(self, model_id: str) -> dict[str, Any]:
        return self.get(f"{self._prefix()}/models/{model_id}")

    def list_versions(self, model_id: str) -> list[dict[str, Any]]:
        data = self.get(f"{self._prefix()}/models/{model_id}/versions")
        return list(data.get("items") or [])

    def get_version(self, model_id: str, version: int) -> dict[str, Any]:
        return self.get(f"{self._prefix()}/models/{model_id}/versions/{version}")

    def promote(
        self,
        model_id: str,
        version: int,
        *,
        stage: str = "production",
    ) -> dict[str, Any]:
        return self.post(
            f"{self._prefix()}/models/{model_id}/promote",
            json={"version": version, "stage": stage},
        )

    def find_version_by_stage(self, model_id: str, stage: str) -> dict[str, Any] | None:
        for row in self.list_versions(model_id):
            if str(row.get("stage") or "").lower() == stage.lower():
                return row
        return None

    def find_latest_version(self, model_id: str, *, run_id: str | None = None) -> dict[str, Any] | None:
        versions = self.list_versions(model_id)
        if run_id:
            matched = [v for v in versions if v.get("run_id") == run_id]
            if matched:
                versions = matched
        if not versions:
            return None
        return max(versions, key=lambda v: int(v.get("version") or 0))

    def download_artifact(self, artifact_uri: str, dest: Path) -> Path:
        """Fetch weights from file:// or http(s) artifact_uri."""
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)

        parsed = urlparse(artifact_uri)
        if parsed.scheme in {"", "file"}:
            src = Path(parsed.path if parsed.scheme == "file" else artifact_uri)
            if not src.is_file():
                raise FileNotFoundError(f"artifact not found: {artifact_uri}")
            dest.write_bytes(src.read_bytes())
            return dest

        if parsed.scheme in {"http", "https"}:
            r = httpx.get(artifact_uri, timeout=120.0)
            r.raise_for_status()
            dest.write_bytes(r.content)
            return dest

        raise ValueError(f"unsupported artifact_uri scheme: {artifact_uri}")
