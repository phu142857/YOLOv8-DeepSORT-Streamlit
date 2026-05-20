"""Dataset ingest (push) and materialization (pull) for CV frame manifests."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
from pathlib import Path
from typing import Any

import httpx

from mlair_adapter.base_client import MLAirClient
from shared.settings import settings

logger = logging.getLogger(__name__)

# CSV contract shared with MLAir dataset versions for this workload
MANIFEST_COLUMNS = ("image_uri", "frame_index", "job_id", "source_file", "artifact_path")


def _items_from_response(data: Any) -> list[dict[str, Any]]:
    """Parse MLAir list endpoints: ``{"items": [...]}`` or a bare list."""
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        items = data.get("items")
        if isinstance(items, list):
            return [row for row in items if isinstance(row, dict)]
    return []


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

    def build_frame_manifest_csv(
        self,
        job_id: str,
        frames_dir: Path,
        *,
        public_base_url: str | None = None,
        source_file: str = "",
    ) -> tuple[Path, int]:
        """Write manifest CSV; image_uri is fetchable URL when public_base_url is set."""
        frames_dir = Path(frames_dir)
        rows: list[dict[str, str]] = []
        base = (public_base_url or settings.api_base_url).rstrip("/")

        for frame_path in sorted(frames_dir.glob("*.jpg")):
            frame_index = frame_path.stem
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
        # Normalize keys so the rest of the CV workload can rely on them.
        out.setdefault("job_id", job_id)
        out.setdefault("rows", row_count)
        out.setdefault("dataset_name", name)
        return out

    def download_version_csv(self, version_id: str) -> bytes:
        return self.get_bytes(f"{self._prefix()}/dataset-versions/{version_id}/download")

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
        reader = csv.DictReader(io.StringIO(text))
        if not reader.fieldnames:
            raise ValueError("empty dataset version CSV")

        uri_col = "image_uri" if "image_uri" in reader.fieldnames else reader.fieldnames[0]
        downloaded = 0

        for row in reader:
            if max_frames > 0 and downloaded >= max_frames:
                break
            uri = (row.get(uri_col) or "").strip()
            if not uri:
                continue

            frame_idx = row.get("frame_index", f"{downloaded:06d}")
            out_name = f"{frame_idx}.jpg"
            dest = dest_dir / out_name

            if uri.startswith("http://") or uri.startswith("https://"):
                r = httpx.get(uri, timeout=60.0)
                r.raise_for_status()
                dest.write_bytes(r.content)
            else:
                src = Path(uri)
                if src.is_file():
                    dest.write_bytes(src.read_bytes())
                else:
                    logger.warning("Skipping non-fetchable uri: %s", uri)
                    continue

            downloaded += 1

        if downloaded == 0:
            raise ValueError(f"no frames downloaded from version {version_id}")

        manifest_copy = dest_dir.parent / "pulled_manifest.csv"
        manifest_copy.write_bytes(raw)
        return dest_dir, downloaded

    @staticmethod
    def checksum_dir(frames_dir: Path) -> str:
        h = hashlib.sha256()
        for p in sorted(Path(frames_dir).glob("*.jpg")):
            h.update(p.read_bytes())
        return h.hexdigest()[:16]
