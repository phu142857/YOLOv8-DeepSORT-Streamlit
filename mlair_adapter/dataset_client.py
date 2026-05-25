"""Dataset ingest (push) and materialization (pull) for CV frame manifests."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from mlair_adapter.base_client import MLAirClient, items_from_response
from mlair_adapter.model_client import MLAIR_ARTIFACT_ROOT
from shared.artifacts import ArtifactStore
from shared.settings import settings

logger = logging.getLogger(__name__)

# CSV contract shared with MLAir dataset versions for this workload
MANIFEST_COLUMNS = ("image_uri", "frame_index", "job_id", "source_file", "artifact_path")


def _items_from_response(data: Any) -> list[dict[str, Any]]:
    return items_from_response(data)


def normalize_buffer_append_result(
    out: dict[str, Any],
    *,
    job_id: str,
    row_count: int,
    dataset_name: str,
) -> dict[str, Any]:
    """Flatten MLAir ``buffer/append`` response for job metadata and UI."""
    out.setdefault("job_id", job_id)
    out.setdefault("rows", row_count)
    out.setdefault("dataset_name", dataset_name)

    ver = out.get("dataset_version_id") or out.get("version_id")
    version_obj = out.get("version")
    if not ver and isinstance(version_obj, dict):
        ver = version_obj.get("dataset_version_id") or version_obj.get("version_id") or version_obj.get("id")
    if ver:
        out["dataset_version_id"] = str(ver)

    ds = out.get("dataset_id")
    dataset_obj = out.get("dataset")
    if not ds and isinstance(dataset_obj, dict):
        ds = dataset_obj.get("dataset_id") or dataset_obj.get("id")
    if ds:
        out["dataset_id"] = str(ds)

    return out


def materialized_version_event_payload(result: dict[str, Any]) -> dict[str, Any]:
    """
    Build a dict for ``emit_event`` when MLAir returns ``materialized: true`` (bool)
    instead of a nested object.
    """
    mat = result.get("materialized")
    if isinstance(mat, dict):
        return mat
    return {
        "materialized": True,
        "dataset_version_id": result.get("dataset_version_id"),
        "dataset_id": result.get("dataset_id"),
        "current_size": result.get("current_size") or result.get("mlair_buffer_current_size"),
        "target_threshold": result.get("target_threshold"),
        "triggered_by": result.get("triggered_by", "buffer_threshold"),
    }


class DatasetClient(MLAirClient):
    def list_datasets(self, limit: int = 100) -> list[dict[str, Any]]:
        data = self.get(f"{self._prefix()}/datasets", params={"limit": limit})
        return _items_from_response(data)

    def get_dataset(self, dataset_id: str) -> dict[str, Any]:
        return self.get(f"{self._prefix()}/datasets/{dataset_id}")

    def list_versions(self, dataset_id: str) -> list[dict[str, Any]]:
        data = self.get(f"{self._prefix()}/datasets/{dataset_id}/versions")
        return _items_from_response(data)

    def get_version(self, version_id: str) -> dict[str, Any]:
        return self.get(f"{self._prefix()}/dataset-versions/{version_id}")

    def get_buffer(self, dataset_id: str) -> dict[str, Any]:
        return self.get(f"{self._prefix()}/datasets/{dataset_id}/buffer")

    def patch_buffer(
        self,
        dataset_id: str,
        *,
        target_threshold: int | None = None,
        accumulation_strategy: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if target_threshold is not None:
            body["target_threshold"] = target_threshold
        if accumulation_strategy:
            body["accumulation_strategy"] = accumulation_strategy
        return self.patch(f"{self._prefix()}/datasets/{dataset_id}/buffer", body)

    def materialize_buffer(self, dataset_id: str) -> dict[str, Any]:
        return self.post(f"{self._prefix()}/datasets/{dataset_id}/materialize")

    def materialize_when_threshold_met(self, dataset_id: str) -> dict[str, Any]:
        """
        Create a new dataset version only when accumulation threshold is satisfied.
        Does not materialize for manual-only strategies unless buffer is ready.
        """
        try:
            buf = self.get_buffer(dataset_id)
        except httpx.HTTPError as exc:
            logger.warning("Cannot read buffer for %s: %s", dataset_id, exc)
            return {"skipped": True, "reason": "buffer_unavailable", "error": str(exc)}

        current = int(buf.get("current_size") or buf.get("record_count") or 0)
        threshold = int(buf.get("target_threshold") or 0)
        strategy = str(buf.get("accumulation_strategy") or "")

        if threshold <= 0:
            return {
                "skipped": True,
                "reason": "no_threshold",
                "current_size": current,
                "target_threshold": threshold,
            }

        if current < threshold:
            return {
                "skipped": True,
                "reason": "below_threshold",
                "materialized": False,
                "current_size": current,
                "target_threshold": threshold,
            }

        if strategy == "rolling_accumulate":
            return {
                "skipped": True,
                "reason": "rolling_accumulate_requires_manual_materialize",
                "current_size": current,
                "target_threshold": threshold,
            }

        try:
            result = self.materialize_buffer(dataset_id)
            result["triggered_by"] = "cv_workload_threshold_met"
            result["materialized"] = True
            result["current_size"] = current
            result["target_threshold"] = threshold
            return result
        except httpx.HTTPError as exc:
            logger.warning("Materialize failed for %s: %s", dataset_id, exc)
            return {"skipped": True, "materialized": False, "error": str(exc)}

    def maybe_auto_materialize(self, dataset_id: str) -> dict[str, Any] | None:
        """Alias for threshold-gated materialize (legacy callers)."""
        return self.materialize_when_threshold_met(dataset_id)

    def _mlair_runtime_frames_dir(self, job_id: str) -> Path:
        """Shared MLAir dataset volume path (survives container recreate; visible to train worker)."""
        root = Path(settings.mlair_model_artifact_mount) / "datasets"
        return (root / settings.mlair_runtime_frames_subdir / job_id).resolve()

    def persist_frame_to_mlair_volume(self, job_id: str, frame_path: Path) -> str | None:
        """Copy frame into ``ml_air_dataset_artifacts``; return ``file://`` URI or None if mount missing."""
        datasets_root = Path(settings.mlair_model_artifact_mount) / "datasets"
        if not datasets_root.exists():
            logger.warning("MLAir datasets mount not found at %s", datasets_root)
            return None
        dest_dir = self._mlair_runtime_frames_dir(job_id)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / frame_path.name
        shutil.copy2(frame_path, dest)
        return f"file://{dest}"

    def build_frame_manifest_csv(
        self,
        job_id: str,
        frames_dir: Path,
        *,
        public_base_url: str | None = None,
        source_file: str = "",
        persist_to_mlair: bool | None = None,
    ) -> tuple[Path, int]:
        """Write manifest CSV; default URIs are ``file://`` on MLAir dataset volume (container storage)."""
        frames_dir = Path(frames_dir)
        rows: list[dict[str, str]] = []
        base = (public_base_url or settings.api_base_url).rstrip("/")
        use_mlair_store = (
            settings.mlair_persist_ingest_frames if persist_to_mlair is None else persist_to_mlair
        )

        for frame_path in sorted(frames_dir.glob("*.jpg")):
            frame_index = frame_path.stem
            image_uri: str | None = None
            if use_mlair_store:
                image_uri = self.persist_frame_to_mlair_volume(job_id, frame_path)
            if not image_uri:
                image_uri = f"{base}/api/v1/jobs/{job_id}/frames/{frame_path.name}"
            rows.append(
                {
                    "image_uri": image_uri,
                    "frame_index": frame_index,
                    "job_id": job_id,
                    "source_file": source_file,
                    "artifact_path": frame_path.name,
                }
            )

        out = frames_dir.parent / "mlair_manifest.csv"
        with out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        return out, len(rows)

    def upload_manifest_csv(
        self,
        csv_path: Path,
        dataset_name: str,
        *,
        required_cols: list[str] | None = None,
    ) -> dict[str, Any]:
        """POST .../datasets/upload — creates dataset + immutable version."""
        cols = required_cols or list(MANIFEST_COLUMNS)
        with csv_path.open("rb") as f:
            return self.post_multipart(
                f"{self._prefix()}/datasets/upload",
                data={
                    "dataset_name": dataset_name,
                    "required_cols": json.dumps(cols),
                },
                files={"file": (csv_path.name, f, "text/csv")},
            )

    def append_buffer_rows(
        self,
        dataset_id: str,
        *,
        rows: list[dict[str, Any]],
        source_type: str | None = None,
        execution_id: str | None = None,
    ) -> dict[str, Any]:
        """
        POST .../datasets/{dataset_id}/buffer/append
        Appends rows into MLAir accumulation buffer window and may auto-materialize.
        """
        body: dict[str, Any] = {"rows": rows}
        if source_type:
            body["source_type"] = source_type
        if execution_id:
            body["execution_id"] = execution_id
        return self.post(f"{self._prefix()}/datasets/{dataset_id}/buffer/append", json=body)

    def append_buffer_rows_by_name(
        self,
        dataset_name: str,
        *,
        rows: list[dict[str, Any]],
        source_type: str | None = None,
        execution_id: str | None = None,
    ) -> dict[str, Any]:
        """
        POST .../datasets/by-name/{dataset_name}/buffer/append
        Creates dataset if missing, then appends rows into its accumulation buffer.
        """
        safe = str(dataset_name or "").strip()
        if not safe:
            raise ValueError("dataset_name_required")
        body: dict[str, Any] = {"rows": rows}
        if source_type:
            body["source_type"] = source_type
        if execution_id:
            body["execution_id"] = execution_id
        return self.post(f"{self._prefix()}/datasets/by-name/{safe}/buffer/append", json=body)

    def ingest_job_frames(
        self,
        job_id: str,
        frames_dir: Path,
        *,
        dataset_name: str | None = None,
        dataset_id: str | None = None,
        source_file: str = "",
        public_base_url: str | None = None,
        auto_materialize: bool = False,
    ) -> dict[str, Any]:
        """
        Append extracted frame manifest rows into MLAir dataset buffer.
        Returns ingest summary with dataset_id / dataset_version_id when materialized.
        """
        if not self.enabled:
            return {"skipped": True, "reason": "mlair_not_configured"}

        manifest_path, row_count = self.build_frame_manifest_csv(
            job_id,
            frames_dir,
            public_base_url=public_base_url,
            source_file=source_file,
        )
        if row_count == 0:
            return {"skipped": True, "reason": "no_frames", "job_id": job_id}

        # NOTE: materialization is decided by MLAir buffer policy; cv-api only appends rows.
        _ = auto_materialize

        # Convert manifest CSV back into list-of-dicts rows for MLAir buffer/append.
        rows: list[dict[str, Any]] = []
        with Path(manifest_path).open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append({k: (r.get(k) or "") for k in MANIFEST_COLUMNS})

        name = dataset_name or settings.mlair_dataset_name
        if dataset_id:
            out = self.append_buffer_rows(
                dataset_id,
                rows=rows,
                source_type="runtime_manifest",
                execution_id=job_id,
            )
        else:
            out = self.append_buffer_rows_by_name(
                name,
                rows=rows,
                source_type="runtime_manifest",
                execution_id=job_id,
            )
        return normalize_buffer_append_result(
            out, job_id=job_id, row_count=row_count, dataset_name=name
        )

    def download_version_csv(self, version_id: str) -> bytes:
        return self.get_bytes(f"{self._prefix()}/dataset-versions/{version_id}/download")

    @staticmethod
    def parse_version_manifest(text: str) -> list[dict[str, Any]]:
        """
        MLAir ``.../dataset-versions/{id}/download`` returns NDJSON (one JSON row per line).
        Legacy uploads may be CSV with a header row.
        """
        text = (text or "").strip()
        if not text:
            return []

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return []

        if lines[0].startswith("{"):
            rows: list[dict[str, Any]] = []
            for ln in lines:
                try:
                    obj = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
            return rows

        reader = csv.DictReader(io.StringIO(text))
        if not reader.fieldnames:
            return []
        return [dict(r) for r in reader]

    @staticmethod
    def _normalize_image_fetch_uri(uri: str) -> str:
        """Rewrite host-only URLs saved as localhost so Docker workers reach cv-api."""
        base = settings.api_base_url.rstrip("/")
        for prefix in (
            "http://127.0.0.1:8000",
            "http://localhost:8000",
            "https://127.0.0.1:8000",
            "https://localhost:8000",
        ):
            if uri.startswith(prefix):
                return base + uri[len(prefix) :]
        return uri

    @staticmethod
    def _copy_frame_from_job_artifacts(row: dict[str, Any], dest: Path) -> bool:
        """Prefer local ``artifacts/jobs/{job_id}/frames/`` (same volume as cv-api/worker)."""
        job_id = str(row.get("job_id") or "").strip()
        if not job_id:
            return False
        layout = ArtifactStore().job_layout(job_id)
        frames_dir = layout["frames"]
        candidates: list[Path] = []
        artifact_path = str(row.get("artifact_path") or "").strip()
        frame_index = str(row.get("frame_index") or "").strip()
        if artifact_path:
            candidates.append(frames_dir / Path(artifact_path).name)
        if frame_index:
            candidates.append(frames_dir / f"{frame_index}.jpg")
            if frame_index.isdigit():
                candidates.append(frames_dir / f"{frame_index.zfill(6)}.jpg")
        for src in candidates:
            if src.is_file():
                dest.write_bytes(src.read_bytes())
                return True
        return False

    @staticmethod
    def _fetch_uri_to_file(uri: str, dest: Path) -> bool:
        uri = (uri or "").strip()
        if not uri:
            return False
        if uri.startswith("http://") or uri.startswith("https://"):
            fetch_uri = DatasetClient._normalize_image_fetch_uri(uri)
            try:
                r = httpx.get(fetch_uri, timeout=60.0)
                r.raise_for_status()
                dest.write_bytes(r.content)
                return True
            except httpx.HTTPError as exc:
                logger.warning("HTTP fetch failed %s: %s", fetch_uri, exc)
                return False
        parsed = urlparse(uri)
        if parsed.scheme == "file":
            candidates: list[Path] = []
            if parsed.path:
                candidates.append(Path(parsed.path))
            # file:///mlair/artifacts/datasets/... → mount at /mlair/artifacts/datasets/...
            mount = Path(settings.mlair_model_artifact_mount)
            if parsed.path.startswith("/mlair/"):
                candidates.append(Path(parsed.path))
                try:
                    candidates.append(mount / "datasets" / Path(parsed.path).relative_to("/mlair/artifacts/datasets"))
                except ValueError:
                    try:
                        rel = Path(parsed.path).relative_to(MLAIR_ARTIFACT_ROOT)
                        candidates.append(mount.parent / rel)
                    except ValueError:
                        pass
            for src in candidates:
                if src.is_file():
                    dest.write_bytes(src.read_bytes())
                    return True
        src = Path(uri)
        if src.is_file():
            dest.write_bytes(src.read_bytes())
            return True
        return False

    def fetch_version_frames(
        self,
        version_id: str,
        dest_dir: Path,
        *,
        max_frames: int = 0,
    ) -> tuple[Path, int]:
        """
        Pull: download dataset version CSV and fetch each image_uri into dest_dir.
        Returns (dest_dir, count_downloaded).
        """
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        raw = self.download_version_csv(version_id)
        text = raw.decode("utf-8")
        manifest_rows = self.parse_version_manifest(text)
        if not manifest_rows:
            raise ValueError(f"empty dataset version manifest for {version_id}")

        downloaded = 0

        for row in manifest_rows:
            if max_frames > 0 and downloaded >= max_frames:
                break
            uri = (row.get("image_uri") or row.get("uri") or "").strip()
            if not uri:
                continue

            frame_idx = str(row.get("frame_index") or f"{downloaded:06d}").strip()
            out_name = f"{Path(frame_idx).stem}.jpg"
            dest = dest_dir / out_name

            ok = self._copy_frame_from_job_artifacts(row, dest)
            if not ok:
                ok = self._fetch_uri_to_file(uri, dest)
            if not ok:
                logger.warning(
                    "skip frame job_id=%s uri=%s (local artifacts + HTTP fetch failed)",
                    row.get("job_id"),
                    uri[:120],
                )
                continue

            downloaded += 1

        if downloaded == 0:
            raise ValueError(
                f"no frames downloaded from version {version_id} "
                "(check job artifacts under artifacts/jobs/ or CV_API_BASE_URL for ingest)"
            )

        manifest_copy = dest_dir.parent / "pulled_manifest.csv"
        manifest_copy.write_bytes(raw)
        return dest_dir, downloaded

    @staticmethod
    def checksum_dir(frames_dir: Path) -> str:
        h = hashlib.sha256()
        for p in sorted(Path(frames_dir).glob("*.jpg")):
            h.update(p.read_bytes())
        return h.hexdigest()[:16]
