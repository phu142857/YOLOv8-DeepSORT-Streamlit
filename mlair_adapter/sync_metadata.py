"""Per-version disk ↔ MLAir mapping (Vet-AI ``mlair-sync.json`` pattern)."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from shared.settings import settings
from shared.weights_catalog import version_dir

logger = logging.getLogger(__name__)

SYNC_METADATA_FILENAME = "mlair-sync.json"
METADATA_JSON = "metadata.json"
METADATA_KEY = "cv_mlair_sync"


@dataclass
class VersionSyncMetadata:
    """Maps ``weights/detection/{model}/{disk_version}/`` to MLAir registry row."""

    cv_disk_version: str
    cv_model: str
    cv_spec: str
    sha256: str = ""
    mlair_model_id: str = ""
    mlair_version: int = 0
    mlair_version_id: str = ""
    mlair_artifact_uri: str = ""
    mlair_stage: str = "production"
    synced_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "cv_disk_version": self.cv_disk_version,
            "cv_model": self.cv_model,
            "cv_spec": self.cv_spec,
            "sha256": self.sha256,
            "mlair": {
                "model_id": self.mlair_model_id,
                "version": self.mlair_version,
                "version_id": self.mlair_version_id,
                "artifact_uri": self.mlair_artifact_uri,
                "stage": self.mlair_stage,
            },
            "synced_at": self.synced_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VersionSyncMetadata | None:
        if not isinstance(data, dict):
            return None
        mlair = data.get("mlair") if isinstance(data.get("mlair"), dict) else {}
        model = str(data.get("cv_model") or "")
        ver = str(data.get("cv_disk_version") or "")
        if not model or not ver:
            return None
        return cls(
            cv_disk_version=ver,
            cv_model=model,
            cv_spec=str(data.get("cv_spec") or f"{model}/{ver}"),
            sha256=str(data.get("sha256") or ""),
            mlair_model_id=str(mlair.get("model_id") or data.get("mlair_model_id") or ""),
            mlair_version=int(mlair.get("version") or data.get("mlair_version") or 0),
            mlair_version_id=str(mlair.get("version_id") or data.get("mlair_version_id") or ""),
            mlair_artifact_uri=str(mlair.get("artifact_uri") or data.get("mlair_artifact_uri") or ""),
            mlair_stage=str(mlair.get("stage") or data.get("mlair_stage") or "production"),
            synced_at=float(data.get("synced_at") or 0.0),
        )


def sync_metadata_path(version_dir_path: Path) -> Path:
    return Path(version_dir_path) / SYNC_METADATA_FILENAME


def read_version_sync_metadata(version_dir_path: Path) -> VersionSyncMetadata | None:
    path = sync_metadata_path(version_dir_path)
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            row = VersionSyncMetadata.from_dict(data)
            if row:
                return row
        except (json.JSONDecodeError, OSError, ValueError):
            logger.debug("invalid %s under %s", SYNC_METADATA_FILENAME, version_dir_path)

    meta_path = Path(version_dir_path) / METADATA_JSON
    if meta_path.is_file():
        try:
            root = json.loads(meta_path.read_text(encoding="utf-8"))
            nested = root.get(METADATA_KEY) if isinstance(root, dict) else None
            if isinstance(nested, dict):
                return VersionSyncMetadata.from_dict(nested)
        except (json.JSONDecodeError, OSError, ValueError):
            pass
    return None


def write_version_sync_metadata(
    version_dir_path: Path,
    *,
    meta: VersionSyncMetadata,
    also_embed_in_metadata_json: bool = True,
) -> Path:
    version_dir_path = Path(version_dir_path)
    path = sync_metadata_path(version_dir_path)
    payload = meta.to_dict()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if also_embed_in_metadata_json:
        meta_path = version_dir_path / METADATA_JSON
        root: dict[str, Any] = {}
        if meta_path.is_file():
            try:
                loaded = json.loads(meta_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    root = loaded
            except (json.JSONDecodeError, OSError):
                root = {}
        root[METADATA_KEY] = payload
        meta_path.write_text(json.dumps(root, indent=2), encoding="utf-8")

    return path


def is_disk_version_importable(disk_version: str) -> bool:
    """Like Vet-AI: only canonical folders go to MLAir unless import-all is enabled."""
    if settings.mlair_disk_import_all_versions:
        return True
    if disk_version in ("base", "production"):
        return True
    if disk_version.startswith("v") and disk_version[1:].isdigit():
        return True
    return False


def version_dir_for_entry(model: str, disk_version: str) -> Path:
    return version_dir(settings.detection_model_dir, model, disk_version)
