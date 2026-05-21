"""Bidirectional sync: ``weights/detection`` ↔ MLAir model registry (same bytes on base + production)."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from mlair_adapter.model_client import ModelClient
from mlair_adapter.sync_metadata import (
    VersionSyncMetadata,
    is_disk_version_importable,
    read_version_sync_metadata,
    write_version_sync_metadata,
)
from shared.model_resolve import ensure_registry_weights
from shared.settings import settings
from shared.weights_catalog import (
    CANONICAL_LOCAL_VERSIONS,
    LocalModelEntry,
    ensure_version_layout,
    list_detection_model_names,
    model_spec,
    pick_canonical_local_entry,
)

logger = logging.getLogger(__name__)

_STATE_LOCK = threading.Lock()
_SYNC_LOCK = threading.Lock()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class ModelSyncService:
    """True sync: push local canonical weights → MLAir, then pull production → local base + production."""

    def __init__(self, client: ModelClient | None = None) -> None:
        self._client = client or ModelClient()

    @property
    def enabled(self) -> bool:
        return bool(settings.mlair_auto_sync_models and self._client.enabled)

    @property
    def state_path(self) -> Path:
        return settings.mlair_sync_state_path

    def load_sync_state(self) -> dict[str, Any]:
        """Persisted sync map (spec → model_id, sha256, aligned)."""
        return self._load_state()

    def _hub_has_model(self, model_id: str) -> bool:
        if not model_id:
            return False
        try:
            row = self._client.get_model(model_id)
            return bool(row and row.get("model_id"))
        except Exception:
            return False

    def _reconcile_state_with_hub(self, state: dict[str, Any]) -> int:
        """
        Drop stale entries when Hub was reset (``docker compose down -v``) but
        ``artifacts/.mlair_model_sync_state.json`` on the host still exists.
        """
        try:
            hub_ids = {
                str(r.get("model_id"))
                for r in self._client.list_models()
                if r.get("model_id")
            }
        except Exception:
            return 0

        removed = 0
        for key in list(state.keys()):
            row = state.get(key)
            if not isinstance(row, dict):
                continue
            mid = str(row.get("model_id") or "")
            if mid and mid not in hub_ids:
                state.pop(key, None)
                removed += 1
                continue
            if key.startswith("registry:"):
                rid = key.split(":", 1)[-1]
                if rid and rid not in hub_ids:
                    state.pop(key, None)
                    removed += 1
        if removed:
            logger.info(
                "pruned %s stale model sync state entries (Hub empty or DB was recreated)",
                removed,
            )
        return removed

    def resolve_model_id_for_spec(self, model_spec_or_name: str) -> str | None:
        from shared.unified_catalog import resolve_mlair_model_id

        return resolve_mlair_model_id(
            model_spec_or_name,
            client=self._client,
            state=self._load_state(),
        )

    def _load_state(self) -> dict[str, Any]:
        path = self.state_path
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_state(self, state: dict[str, Any]) -> None:
        path = self.state_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with _STATE_LOCK:
            path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def _update_state_aligned(
        self,
        state: dict[str, Any],
        *,
        model_name: str,
        model_id: str,
        version: int,
        sha256: str,
        stage: str,
        source: str,
    ) -> None:
        for ver in CANONICAL_LOCAL_VERSIONS:
            state[model_spec(model_name, ver)] = {
                "sha256": sha256,
                "stage": stage,
                "model_id": model_id,
                "registry_version": version,
                "synced_at": time.time(),
                "source": source,
                "aligned": True,
            }
        state[f"registry:{model_id}"] = {
            "sha256": sha256,
            "model_name": model_name,
            "registry_version": version,
            "synced_at": time.time(),
            "source": source,
        }

    def mirror_production_to_canonical_local(
        self,
        model_id: str,
        *,
        model_name: str | None = None,
        version: int | None = None,
        stage: str | None = None,
    ) -> dict[str, Path]:
        """
        Pull MLAir production (or given version) into local ``{model}/production`` and ``{model}/base``.
        Both folders receive the same ``weights.pt`` so [local] base == [MLAir] production bytes.
        """
        stage = stage or settings.mlair_sync_stage
        if not model_name:
            model_row = self._client.get_model(model_id)
            model_name = str(model_row.get("name") or model_id)

        if version is None:
            row = self._client.resolve_version_row(model_id, stage=stage)
            if not row:
                raise FileNotFoundError(f"no {stage} version for model {model_id}")
            version = int(row.get("version") or 0)

        cache_path = ensure_registry_weights(model_id, stage=stage)
        if not cache_path.is_file():
            row = self._client.get_version(model_id, version)
            artifact_uri = str(row.get("artifact_uri") or "")
            self._client.download_artifact(artifact_uri, cache_path)

        digest = _sha256_file(cache_path)
        dest: dict[str, Path] = {}
        for ver in CANONICAL_LOCAL_VERSIONS:
            folder = ensure_version_layout(settings.detection_model_dir, model_name, ver)
            target = folder / "weights.pt"
            shutil.copy2(cache_path, target)
            dest[ver] = target

        return {"production": dest["production"], "base": dest["base"], "sha256": digest, "version": version}

    def sync_pull_registry_models(self, state: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """MLAir → local: every registry model production copied to base + production."""
        state = state if state is not None else self._load_state()
        results: list[dict[str, Any]] = []

        root = Path(settings.detection_model_dir)
        local_names = set(list_detection_model_names(root))

        for row in self._client.list_models():
            model_id = str(row.get("model_id") or "")
            model_name = str(row.get("name") or "")
            if not model_id or not model_name:
                continue
            if model_name not in local_names:
                logger.debug("skip pull for hub-only model %s (no weights/detection folder)", model_name)
                continue
            try:
                mirrored = self.mirror_production_to_canonical_local(model_id, model_name=model_name)
                self._update_state_aligned(
                    state,
                    model_name=model_name,
                    model_id=model_id,
                    version=int(mirrored["version"]),
                    sha256=str(mirrored["sha256"]),
                    stage=settings.mlair_sync_stage,
                    source="pull_registry",
                )
                results.append(
                    {
                        "model": model_name,
                        "model_id": model_id,
                        "ok": True,
                        "action": "pull",
                        "version": mirrored["version"],
                        "local_base": str(mirrored["base"]),
                    }
                )
            except Exception as exc:
                logger.exception("pull registry failed for %s", model_name)
                results.append({"model": model_name, "model_id": model_id, "ok": False, "error": str(exc)})

        self._save_state(state)
        return results

    def sync_local_entry(
        self,
        entry: LocalModelEntry,
        *,
        stage: str | None = None,
        state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Push one local checkpoint to MLAir when bytes differ from last synced state."""
        stage = stage or settings.mlair_sync_stage
        spec = entry.spec
        weights = Path(entry.weights_path)
        if not weights.is_file():
            return {"spec": spec, "skipped": True, "reason": "missing_weights"}

        if not is_disk_version_importable(entry.version):
            return {"spec": spec, "skipped": True, "reason": "non_canonical_disk_version"}

        digest = _sha256_file(weights)
        state = state if state is not None else self._load_state()
        version_dir_path = weights.parent
        hub_row = self._client.find_model_by_name(entry.model)

        if settings.mlair_disk_sync_mode == "metadata":
            disk_meta = read_version_sync_metadata(version_dir_path)
            if (
                disk_meta
                and disk_meta.sha256 == digest
                and hub_row
                and str(hub_row.get("model_id") or "") == disk_meta.mlair_model_id
                and self._hub_has_model(disk_meta.mlair_model_id)
            ):
                return {"spec": spec, "skipped": True, "reason": "already_aligned_metadata"}

        prev = state.get(spec) or {}
        if prev.get("sha256") == digest and hub_row:
            model_id_prev = str(prev.get("model_id") or "")
            hub_id = str(hub_row.get("model_id") or "")
            if model_id_prev == hub_id and self._hub_has_model(hub_id):
                reg = state.get(f"registry:{hub_id}") or {}
                if reg.get("sha256") == digest:
                    return {"spec": spec, "skipped": True, "reason": "already_aligned"}

        row = hub_row
        if row and row.get("model_id"):
            model_id = str(row["model_id"])
            action = "import_version"
            out = self._client.import_version(model_id, weights, stage=stage)
        else:
            action = "create_and_import"
            created = self._client.create_model(
                entry.model,
                description=f"CV workload {entry.spec}",
            )
            model_id = str(created.get("model_id") or "")
            if not model_id:
                return {"spec": spec, "ok": False, "error": "create_model_failed", "response": created}
            out = self._client.import_version(model_id, weights, stage=stage)

        version = int(out.get("version") or 0)
        version_row = self._client.get_version(model_id, version) if version else {}
        artifact_uri = str(version_row.get("artifact_uri") or "")
        write_version_sync_metadata(
            version_dir_path,
            meta=VersionSyncMetadata(
                cv_disk_version=entry.version,
                cv_model=entry.model,
                cv_spec=spec,
                sha256=digest,
                mlair_model_id=model_id,
                mlair_version=version,
                mlair_version_id=str(version_row.get("version_id") or ""),
                mlair_artifact_uri=artifact_uri,
                mlair_stage=stage,
                synced_at=time.time(),
            ),
        )
        self.mirror_production_to_canonical_local(model_id, model_name=entry.model, version=version, stage=stage)
        self._update_state_aligned(
            state,
            model_name=entry.model,
            model_id=model_id,
            version=version,
            sha256=digest,
            stage=stage,
            source="push_local",
        )
        self._save_state(state)

        ensure_registry_weights(model_id, stage=stage)
        return {
            "spec": spec,
            "ok": True,
            "action": action,
            "model_id": model_id,
            "version": version,
            "stage": stage,
            "aligned": True,
        }

    def sync_push_local_models(self, *, root: Path | None = None, state: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Local canonical checkpoint → MLAir (one push per model name)."""
        root = Path(root or settings.detection_model_dir)
        state = state if state is not None else self._load_state()
        results: list[dict[str, Any]] = []

        for model_name in list_detection_model_names(root):
            entry = pick_canonical_local_entry(root, model_name)
            if entry is None:
                continue
            try:
                results.append(self.sync_local_entry(entry, state=state))
            except Exception as exc:
                logger.exception("push failed for %s", model_name)
                results.append({"spec": entry.spec, "ok": False, "error": str(exc)})

        self._save_state(state)
        return results

    def sync_full(self, *, root: Path | None = None, force: bool = False) -> dict[str, Any]:
        """
        True bidirectional sync:
        1. Push local canonical weights that changed.
        2. Pull every MLAir model production → ``{name}/base`` and ``{name}/production`` (same file).
        """
        if not self.enabled:
            return {"ok": False, "reason": "sync_disabled"}

        root = Path(root or settings.detection_model_dir)
        state = self._load_state()
        pruned = self._reconcile_state_with_hub(state)
        if pruned:
            self._save_state(state)
        if force:
            for key in list(state.keys()):
                if not key.startswith("registry:"):
                    state.pop(key, None)
            self._save_state(state)

        with _SYNC_LOCK:
            push_results = self.sync_push_local_models(root=root, state=state)
            pull_results = self.sync_pull_registry_models(state)

            # Models only on disk, not yet in registry
            registry_names = {str(r.get("name") or "") for r in self._client.list_models()}
            for model_name in list_detection_model_names(root):
                if model_name in registry_names:
                    continue
                entry = pick_canonical_local_entry(root, model_name)
                if entry is None:
                    continue
                try:
                    push_results.append(self.sync_local_entry(entry, state=state))
                except Exception as exc:
                    push_results.append({"spec": entry.spec, "ok": False, "error": str(exc)})

        pushed = sum(1 for r in push_results if r.get("ok"))
        pulled = sum(1 for r in pull_results if r.get("ok"))
        return {
            "ok": True,
            "mode": "full",
            "force": force,
            "state_pruned": pruned,
            "root": str(root),
            "pushed": pushed,
            "pulled": pulled,
            "push_results": push_results,
            "pull_results": pull_results,
        }

    def sync_all_local_models(self, *, root: Path | None = None) -> dict[str, Any]:
        """Backward-compatible alias → :meth:`sync_full`."""
        return self.sync_full(root=root)

    def sync_after_training(self, model_id: str, *, run_id: str | None = None) -> dict[str, Any]:
        """After train: align local base/production with new production on Hub."""
        if not self.enabled:
            return {"ok": False, "reason": "sync_disabled"}

        stage = settings.mlair_sync_stage
        row = self._client.find_latest_version(model_id, run_id=run_id) if run_id else None
        if not row:
            row = self._client.resolve_version_row(model_id, stage=stage)
        if not row or not row.get("artifact_uri"):
            return {"ok": False, "reason": "no_version_row"}

        version = int(row.get("version") or 0)
        model_row = self._client.get_model(model_id)
        model_name = str(model_row.get("name") or model_id)

        mirrored = self.mirror_production_to_canonical_local(
            model_id, model_name=model_name, version=version, stage=stage
        )
        state = self._load_state()
        self._update_state_aligned(
            state,
            model_name=model_name,
            model_id=model_id,
            version=version,
            sha256=str(mirrored["sha256"]),
            stage=stage,
            source="mlair_train",
        )
        self._save_state(state)
        ensure_registry_weights(model_id, stage=stage)

        base_dir = mirrored["base"].parent
        write_version_sync_metadata(
            base_dir,
            meta=VersionSyncMetadata(
                cv_disk_version="base",
                cv_model=model_name,
                cv_spec=model_spec(model_name, "base"),
                sha256=str(mirrored["sha256"]),
                mlair_model_id=model_id,
                mlair_version=version,
                mlair_version_id=str(row.get("version_id") or ""),
                mlair_artifact_uri=str(row.get("artifact_uri") or ""),
                mlair_stage=stage,
                synced_at=time.time(),
            ),
        )

        return {
            "ok": True,
            "model_id": model_id,
            "model_name": model_name,
            "version": version,
            "aligned": True,
            "local_base": str(mirrored["base"]),
            "local_production": str(mirrored["production"]),
        }


def sync_training_checkpoint_to_mlair(
    model_id: str,
    checkpoint_path: Path,
    *,
    model_name: str | None = None,
    disk_version: str = "base",
    run_id: str | None = None,
) -> dict[str, Any]:
    """Vet-AI ``sync_training_directory_to_mlair`` — import + pull to canonical local + ``mlair-sync.json``."""
    svc = ModelSyncService()
    if not svc.enabled:
        return {"ok": False, "reason": "sync_disabled"}
    return svc.sync_after_training(model_id, run_id=run_id)

    def mirror_registry_version_to_local(
        self,
        model_id: str,
        version: int,
        *,
        model_name: str | None = None,
    ) -> Path:
        """Legacy: mirror version folder ``v{N}``; prefer :meth:`mirror_production_to_canonical_local`."""
        mirrored = self.mirror_production_to_canonical_local(
            model_id, model_name=model_name, version=version
        )
        return mirrored["base"]

    def run_background_sync_loop(self) -> None:
        interval = max(60.0, float(settings.mlair_sync_interval_sec))
        while True:
            try:
                if self.enabled:
                    summary = self.sync_full()
                    if summary.get("pushed") or summary.get("pulled"):
                        logger.info(
                            "mlair full sync: pushed=%s pulled=%s",
                            summary.get("pushed"),
                            summary.get("pulled"),
                        )
            except Exception:
                logger.exception("background model sync failed")
            time.sleep(interval)


def _hub_models_missing_versions(client: ModelClient | None = None) -> bool:
    """True when registry rows exist but no imported version (common after permission 500)."""
    client = client or ModelClient()
    if not client.enabled:
        return False
    try:
        for row in client.list_models():
            mid = str(row.get("model_id") or "")
            if not mid:
                continue
            if row.get("production_version") is not None:
                continue
            if not client.list_versions(mid):
                return True
    except Exception:
        return False
    return False


def _wait_for_mlair_api(*, attempts: int = 30, delay_sec: float = 2.0) -> bool:
    client = ModelClient()
    if not client.enabled:
        return False
    import time

    for i in range(attempts):
        try:
            client.get(f"{client._prefix()}/models", params={"limit": 1})
            return True
        except Exception as exc:
            if i == 0 or (i + 1) % 5 == 0:
                logger.debug("waiting for MLAir API (%s/%s): %s", i + 1, attempts, exc)
            time.sleep(delay_sec)
    return False


def start_model_sync_background() -> None:
    """Backward-compatible entry → Vet-AI-style :func:`registry_sync.start_registry_sync_background`."""
    from mlair_adapter.registry_sync import start_registry_sync_background

    start_registry_sync_background()
