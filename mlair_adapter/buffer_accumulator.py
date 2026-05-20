"""Local CSV staging — MLAir Hub decides threshold and accumulation strategy."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlair_adapter.dataset_client import DatasetClient, MANIFEST_COLUMNS
from shared.settings import settings

logger = logging.getLogger(__name__)

MLAIR_DEFAULT_BUFFER_THRESHOLD = 1000
AUTO_FLUSH_STRATEGY = "snapshot_on_threshold"


@dataclass(frozen=True)
class MLAirBufferPolicy:
    target_threshold: int
    accumulation_strategy: str
    current_size: int

    @property
    def auto_flush_on_threshold(self) -> bool:
        return self.accumulation_strategy == AUTO_FLUSH_STRATEGY


def _buffer_dir(dataset_name: str) -> Path:
    safe = dataset_name.replace("/", "_").strip() or "default"
    return settings.artifact_root / "mlair_buffer" / safe


def _meta_path(dataset_name: str) -> Path:
    return _buffer_dir(dataset_name) / "meta.json"


def _accumulation_path(dataset_name: str) -> Path:
    return _buffer_dir(dataset_name) / "accumulation.csv"


def load_meta(dataset_name: str) -> dict[str, Any]:
    path = _meta_path(dataset_name)
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_meta(dataset_name: str, meta: dict[str, Any]) -> None:
    path = _meta_path(dataset_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def count_accumulated_rows(dataset_name: str) -> int:
    path = _accumulation_path(dataset_name)
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return sum(1 for _ in reader)


def append_manifest_rows(dataset_name: str, manifest_path: Path) -> int:
    """Append job manifest rows into the local staging file (frame data until MLAir flush)."""
    accum = _accumulation_path(dataset_name)
    accum.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open(encoding="utf-8", newline="") as src:
        reader = csv.DictReader(src)
        rows = list(reader)

    write_header = not accum.exists() or accum.stat().st_size == 0
    with accum.open("a", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=list(MANIFEST_COLUMNS))
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in MANIFEST_COLUMNS})

    total = count_accumulated_rows(dataset_name)
    logger.info("Local staging %s: +%d rows, total=%d", dataset_name, len(rows), total)
    return total


def resolve_dataset_id(client: DatasetClient, dataset_name: str) -> str | None:
    meta = load_meta(dataset_name)
    ds_id = meta.get("dataset_id")
    if ds_id:
        return str(ds_id)

    for row in client.list_datasets():
        if (row.get("name") or "") == dataset_name:
            ds_id = str(row.get("dataset_id") or "")
            if ds_id:
                meta["dataset_id"] = ds_id
                save_meta(dataset_name, meta)
                return ds_id
    return None


def fetch_mlair_buffer_policy(client: DatasetClient, dataset_id: str | None) -> MLAirBufferPolicy:
    """Threshold and strategy come from MLAir Hub (GET .../datasets/{id}/buffer)."""
    if settings.mlair_buffer_threshold > 0:
        logger.warning(
            "CV_MLAIR_BUFFER_THRESHOLD is set but ignored — accumulation is controlled by MLAir Hub buffer config"
        )

    if not dataset_id:
        return MLAirBufferPolicy(
            target_threshold=MLAIR_DEFAULT_BUFFER_THRESHOLD,
            accumulation_strategy=AUTO_FLUSH_STRATEGY,
            current_size=0,
        )

    try:
        buf = client.get_buffer(dataset_id)
        threshold = int(buf.get("target_threshold") or 0)
        strategy = str(buf.get("accumulation_strategy") or AUTO_FLUSH_STRATEGY).strip() or AUTO_FLUSH_STRATEGY
        current = int(buf.get("current_size") or buf.get("record_count") or 0)
        return MLAirBufferPolicy(
            target_threshold=max(1, threshold) if threshold > 0 else MLAIR_DEFAULT_BUFFER_THRESHOLD,
            accumulation_strategy=strategy,
            current_size=max(0, current),
        )
    except Exception as exc:
        logger.warning("MLAir buffer unavailable for %s: %s — using defaults", dataset_id, exc)
        return MLAirBufferPolicy(
            target_threshold=MLAIR_DEFAULT_BUFFER_THRESHOLD,
            accumulation_strategy=AUTO_FLUSH_STRATEGY,
            current_size=0,
        )


def ingest_with_accumulation(
    client: DatasetClient,
    *,
    job_id: str,
    manifest_path: Path,
    row_count: int,
    dataset_name: str,
    dataset_id: str | None = None,
    auto_materialize: bool = True,
) -> dict[str, Any]:
    """
    Stage frames locally; flush to MLAir only when Hub policy allows (snapshot_on_threshold + size met).
    Threshold and accumulation_strategy are always read from MLAir buffer API.
    """
    _ = auto_materialize  # flush timing is governed by MLAir accumulation_strategy

    total = append_manifest_rows(dataset_name, manifest_path)
    ds_id = dataset_id or resolve_dataset_id(client, dataset_name)
    policy = fetch_mlair_buffer_policy(client, ds_id)

    out: dict[str, Any] = {
        "job_id": job_id,
        "rows": row_count,
        "manifest": str(manifest_path),
        "dataset_name": dataset_name,
        "buffer_mode": "mlair_policy",
        "pending_rows": total,
        "target_threshold": policy.target_threshold,
        "accumulation_strategy": policy.accumulation_strategy,
        "mlair_buffer_current_size": policy.current_size,
    }

    if ds_id:
        out["dataset_id"] = ds_id
        meta = load_meta(dataset_name)
        meta["dataset_id"] = ds_id
        save_meta(dataset_name, meta)

    if not policy.auto_flush_on_threshold:
        out["materialize_pending"] = True
        out["flush_gated_by"] = f"mlair_strategy:{policy.accumulation_strategy}"
        return out

    if total < policy.target_threshold:
        out["materialize_pending"] = True
        return out

    accum_path = _accumulation_path(dataset_name)
    upload = client.upload_manifest_csv(accum_path, dataset_name)
    out["upload"] = upload
    ds_id = upload.get("dataset_id") or ds_id
    if ds_id:
        out["dataset_id"] = ds_id
        meta = load_meta(dataset_name)
        meta["dataset_id"] = ds_id
        save_meta(dataset_name, meta)

    ver_id = upload.get("dataset_version_id") or upload.get("version_id")
    if ver_id:
        out["dataset_version_id"] = ver_id
        out["materialized"] = {"materialized": True, "source": "batch_upload", "dataset_version_id": ver_id}

    accum_path.unlink(missing_ok=True)
    meta = load_meta(dataset_name)
    if ds_id:
        meta["dataset_id"] = ds_id
    save_meta(dataset_name, meta)
    out["pending_rows"] = 0
    return out
