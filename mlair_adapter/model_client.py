"""Model registry — versions, promote, artifact download (MLAir Phase 3)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from mlair_adapter.base_client import MLAirClient, items_from_response
from shared.settings import settings
from shared.weights_catalog import find_weights_in_dir

logger = logging.getLogger(__name__)

# Shared with ml-air-api when CV_MLAIR_MODEL_ARTIFACT_MOUNT is set (docker-compose).
MLAIR_ARTIFACT_ROOT = Path("/mlair/artifacts")


class ModelClient(MLAirClient):
    def list_models(self, limit: int = 100) -> list[dict[str, Any]]:
        data = self.get(f"{self._prefix()}/models", params={"limit": limit})
        return items_from_response(data)

    def find_model_by_name(self, name: str) -> dict[str, Any] | None:
        """Hub model whose ``name`` matches a folder under ``weights/detection/{name}/``."""
        target = str(name or "").strip()
        if not target:
            return None
        matches = [row for row in self.list_models() if str(row.get("name") or "") == target]
        if not matches:
            return None
        if len(matches) > 1:
            logger.warning(
                "multiple MLAir models named %r (%s rows); using first model_id=%s",
                target,
                len(matches),
                matches[0].get("model_id"),
            )
        return matches[0]

    def get_model(self, model_id: str) -> dict[str, Any]:
        return self.get(f"{self._prefix()}/models/{model_id}")

    def create_model(self, name: str, *, description: str = "") -> dict[str, Any]:
        return self.post(
            f"{self._prefix()}/models",
            json={"name": name, "description": description or ""},
        )

    def import_version(
        self,
        model_id: str,
        weights_path: Path,
        *,
        stage: str = "production",
    ) -> dict[str, Any]:
        """Upload a .pt checkpoint into MLAir model registry."""
        path = Path(weights_path)
        with path.open("rb") as f:
            files = {"model_file": (path.name, f, "application/octet-stream")}
            data = {"stage": stage}
            url = f"{self.base_url}{self._prefix()}/models/{model_id}/versions/import"
            r = httpx.post(
                url,
                headers=self._headers(),
                data=data,
                files=files,
                timeout=max(self.timeout, 300.0),
            )
            r.raise_for_status()
            return r.json() if r.content else {}

    def list_versions(self, model_id: str) -> list[dict[str, Any]]:
        data = self.get(f"{self._prefix()}/models/{model_id}/versions")
        return items_from_response(data)

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

    def resolve_version_row(
        self, model_id: str, *, stage: str = "production"
    ) -> dict[str, Any] | None:
        """Production version first, else highest version number."""
        row = self.find_version_by_stage(model_id, stage)
        if row and row.get("artifact_uri"):
            return row
        return self.find_latest_version(model_id)

    def find_latest_version(self, model_id: str, *, run_id: str | None = None) -> dict[str, Any] | None:
        versions = self.list_versions(model_id)
        if run_id:
            matched = [v for v in versions if v.get("run_id") == run_id]
            if matched:
                versions = matched
        if not versions:
            return None
        return max(versions, key=lambda v: int(v.get("version") or 0))

    def _resolve_file_artifact(self, artifact_uri: str) -> Path:
        parsed = urlparse(artifact_uri)
        if parsed.scheme not in {"", "file"}:
            raise ValueError(f"not a file uri: {artifact_uri}")

        candidates: list[Path] = []
        if parsed.scheme == "file":
            candidates.append(Path(parsed.path))
        candidates.append(Path(artifact_uri))

        mount = Path(settings.mlair_model_artifact_mount)
        for raw in candidates:
            if raw.is_file():
                return raw
            if raw.is_dir():
                weights = find_weights_in_dir(raw)
                if weights is not None:
                    return weights
            # file:///mlair/artifacts/models/... — same path when volume is mounted on cv-api
            if mount.is_dir() and str(raw).startswith("/mlair/"):
                try:
                    alt = mount / raw.relative_to(MLAIR_ARTIFACT_ROOT)
                except ValueError:
                    alt = raw
                if alt.is_file():
                    return alt
                if alt.is_dir():
                    weights = find_weights_in_dir(alt)
                    if weights is not None:
                        return weights

        raise FileNotFoundError(
            f"artifact not found: {artifact_uri}. "
            f"Mount MLAir model artifacts at {mount} (see docker-compose ml_air_model_artifacts)."
        )

    def download_artifact(self, artifact_uri: str, dest: Path) -> Path:
        """Fetch weights from file:// (shared volume) or http(s) artifact_uri."""
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)

        parsed = urlparse(artifact_uri)
        if parsed.scheme in {"", "file"}:
            src = self._resolve_file_artifact(artifact_uri)
            dest.write_bytes(src.read_bytes())
            return dest

        if parsed.scheme in {"http", "https"}:
            r = httpx.get(artifact_uri, timeout=120.0)
            r.raise_for_status()
            dest.write_bytes(r.content)
            return dest

        raise ValueError(f"unsupported artifact_uri scheme: {artifact_uri}")
