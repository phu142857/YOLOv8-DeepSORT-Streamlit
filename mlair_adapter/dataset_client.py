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


class DatasetClient(MLAirClient):
    def list_datasets(self, limit: int = 100) -> list[dict[str, Any]]:
        data = self.get(f"{self._prefix()}/datasets", params={"limit": limit})
        return list(data.get("items") or data if isinstance(data, list) else [])

    def get_dataset(self, dataset_id: str) -> dict[str, Any]:
        return self.get(f"{self._prefix()}/datasets/{dataset_id}")

    def list_versions(self, dataset_id: str) -> list[dict[str, Any]]:
        data = self.get(f"{self._prefix()}/datasets/{dataset_id}/versions")
        return list(data.get("items") or [])

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

    def maybe_auto_materialize(self, dataset_id: str) -> dict[str, Any] | None:
        """
        Materialize buffer when size >= threshold or when manual/schedule strategy allows.
        Returns materialize result dict or None if skipped.
        """
        try:
            buf = self.get_buffer(dataset_id)
        except httpx.HTTPError as exc:
            logger.warning("Cannot read buffer for %s: %s", dataset_id, exc)
            return None

        current = int(buf.get("current_size") or buf.get("record_count") or 0)
        threshold = int(buf.get("target_threshold") or 0)
        strategy = str(buf.get("accumulation_strategy") or "")

        should = False
        if threshold > 0 and current >= threshold:
            should = True
        if strategy in {"manual_materialize_only", "snapshot_on_schedule"} and current > 0:
            should = True

        if not should:
            return {"skipped": True, "current_size": current, "target_threshold": threshold}

        try:
            result = self.materialize_buffer(dataset_id)
            result["triggered_by"] = "cv_workload_auto_materialize"
            return result
        except httpx.HTTPError as exc:
            logger.warning("Materialize failed for %s: %s", dataset_id, exc)
            return {"skipped": True, "error": str(exc)}

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
        Push extracted frames to MLAir as a CSV dataset version.
        Returns ingest summary with dataset_id / dataset_version_id when present.
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

        name = dataset_name or settings.mlair_dataset_name
        result = self.upload_manifest_csv(manifest_path, name)

        out: dict[str, Any] = {
            "job_id": job_id,
            "rows": row_count,
            "manifest": str(manifest_path),
            "upload": result,
        }

        ds_id = result.get("dataset_id") or dataset_id
        ver_id = result.get("dataset_version_id") or result.get("version_id")
        if ds_id:
            out["dataset_id"] = ds_id
        if ver_id:
            out["dataset_version_id"] = ver_id

        if auto_materialize and ds_id:
            mat = self.maybe_auto_materialize(ds_id)
            if mat and not mat.get("skipped"):
                out["materialized"] = mat
                out["dataset_version_id"] = (
                    mat.get("dataset_version_id") or out.get("dataset_version_id")
                )

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
