"""Split source manifest → detected / not-detected Hub datasets → merged train-ready version."""

from __future__ import annotations

import csv
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Keep in sync with mlair_adapter.dataset_client.MANIFEST_COLUMNS (avoid heavy import in tests).
MANIFEST_COLUMNS = ("image_uri", "frame_index", "job_id", "source_file", "artifact_path")

if TYPE_CHECKING:
    from mlair_adapter.dataset_client import DatasetClient
from mlair_adapter.job_detections import detections_usable, load_job_detections, normalize_frame_key
from shared.settings import settings

logger = logging.getLogger(__name__)


def filter_rows_with_job_id(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("job_id") or "").strip():
            out.append(row)
    return out


def normalize_manifest_row(row: dict[str, Any]) -> dict[str, str]:
    return {col: str(row.get(col) or "") for col in MANIFEST_COLUMNS}


def write_manifest_csv(rows: list[dict[str, Any]], dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    normalized = [normalize_manifest_row(r) for r in rows]
    with dest.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(MANIFEST_COLUMNS))
        writer.writeheader()
        writer.writerows(normalized)
    return len(normalized)


def extract_dataset_version_id(upload: dict[str, Any]) -> str:
    ver = upload.get("dataset_version_id") or upload.get("version_id")
    if not ver and isinstance(upload.get("version"), dict):
        ver = upload["version"].get("dataset_version_id") or upload["version"].get("version_id")
    return str(ver or "").strip()


def extract_dataset_id(upload: dict[str, Any]) -> str:
    ds = upload.get("dataset_id")
    if not ds and isinstance(upload.get("dataset"), dict):
        ds = upload["dataset"].get("dataset_id") or upload["dataset"].get("id")
    return str(ds or "").strip()


def extract_version_label(upload: dict[str, Any]) -> str:
    label = upload.get("version") or upload.get("dataset_version")
    if not label and isinstance(upload.get("version"), dict):
        label = upload["version"].get("version")
    return str(label or "").strip()


def lineage_item_from_version_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    """Hub ingest: ``name`` = dataset name, ``version`` = vN label (not version_id UUID)."""
    if not isinstance(row, dict):
        return None
    ds_name = str(row.get("dataset_name") or row.get("name") or "").strip()
    ver_label = str(row.get("version") or "").strip()
    if not ds_name or not ver_label:
        return None
    item: dict[str, Any] = {
        "name": ds_name,
        "version": ver_label,
        "source_type": str(row.get("source_type") or "csv_import"),
    }
    if row.get("uri"):
        item["uri"] = row["uri"]
    if row.get("checksum"):
        item["checksum"] = row["checksum"]
    rc = row.get("record_count")
    if rc is not None:
        item["size"] = int(rc)
    return item


def lineage_item_from_publish(pub: dict[str, Any] | None) -> dict[str, Any] | None:
    if not pub:
        return None
    ds_name = str(pub.get("dataset_name") or "").strip()
    ver_label = str(pub.get("version_label") or "").strip()
    if not ver_label:
        upload = pub.get("upload") if isinstance(pub.get("upload"), dict) else {}
        ver_label = extract_version_label(upload)
    if not ds_name or not ver_label:
        return None
    item: dict[str, Any] = {
        "name": ds_name,
        "version": ver_label,
        "source_type": "csv_import",
    }
    upload = pub.get("upload") if isinstance(pub.get("upload"), dict) else {}
    if upload.get("uri"):
        item["uri"] = upload["uri"]
    if upload.get("checksum"):
        item["checksum"] = upload["checksum"]
    rc = pub.get("record_count")
    if rc is not None:
        item["size"] = int(rc)
    return item


def build_split_lineage_ingest(
    *,
    source_version_row: dict[str, Any] | None,
    detected_pub: dict[str, Any] | None,
    not_detected_pub: dict[str, Any] | None,
) -> dict[str, list[dict[str, Any]]]:
    """Task ``cv_yolo_split``: input → detected, not-detected."""
    source_item = lineage_item_from_version_row(source_version_row)
    outputs: list[dict[str, Any]] = []
    for pub in (detected_pub, not_detected_pub):
        item = lineage_item_from_publish(pub)
        if item:
            outputs.append(item)
    if not source_item or not outputs:
        return {}
    return {"inputs": [source_item], "outputs": outputs}


def build_merge_lineage_ingest(
    *,
    detected_pub: dict[str, Any] | None,
    not_detected_pub: dict[str, Any] | None,
    train_pub: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Task ``cv_yolo_detect``: detected + not-detected → train-ready."""
    inputs: list[dict[str, Any]] = []
    for pub in (detected_pub, not_detected_pub):
        item = lineage_item_from_publish(pub)
        if item:
            inputs.append(item)
    train_item = lineage_item_from_publish(train_pub)
    if not inputs or not train_item:
        return {}
    return {"inputs": inputs, "outputs": [train_item]}


def build_detect_lineage_ingests(
    *,
    source_version_row: dict[str, Any] | None,
    detected_pub: dict[str, Any] | None,
    not_detected_pub: dict[str, Any] | None,
    train_pub: dict[str, Any],
) -> list[dict[str, list[dict[str, Any]]]]:
    """Legacy monolithic detect: split + merge blocks (two task_ids if used together)."""
    ingests: list[dict[str, list[dict[str, Any]]]] = []
    split = build_split_lineage_ingest(
        source_version_row=source_version_row,
        detected_pub=detected_pub,
        not_detected_pub=not_detected_pub,
    )
    if split:
        ingests.append(split)
    merge = build_merge_lineage_ingest(
        detected_pub=detected_pub,
        not_detected_pub=not_detected_pub,
        train_pub=train_pub,
    )
    if merge:
        ingests.append(merge)
    return ingests


def publish_split_datasets(
    client: "DatasetClient",
    *,
    source_version_id: str,
    labeled_rows: list[dict[str, Any]],
    missing_rows: list[dict[str, Any]],
    run_id: str,
) -> dict[str, Any]:
    """Upload ``detected`` + ``not-detected`` only (task split)."""
    detected_name = settings.mlair_detected_dataset_name
    not_detected_name = settings.mlair_not_detected_dataset_name

    detected_pub = publish_manifest_rows(client, detected_name, labeled_rows, run_id=run_id)
    not_detected_pub = publish_manifest_rows(client, not_detected_name, missing_rows, run_id=run_id)

    if not detected_pub and not not_detected_pub:
        raise RuntimeError("split produced no dataset versions (no rows with job_id)")

    source_row: dict[str, Any] | None = None
    try:
        source_row = client.get_version(source_version_id)
    except Exception as exc:
        logger.warning("could not load source version %s for lineage: %s", source_version_id, exc)

    lineage = build_split_lineage_ingest(
        source_version_row=source_row,
        detected_pub=detected_pub,
        not_detected_pub=not_detected_pub,
    )
    return {
        "detected_dataset_name": detected_name,
        "not_detected_dataset_name": not_detected_name,
        "detected_dataset_id": detected_pub.get("dataset_id") if detected_pub else None,
        "detected_dataset_version_id": detected_pub.get("dataset_version_id") if detected_pub else None,
        "detected_version_label": detected_pub.get("version_label") if detected_pub else None,
        "not_detected_dataset_id": not_detected_pub.get("dataset_id") if not_detected_pub else None,
        "not_detected_dataset_version_id": (
            not_detected_pub.get("dataset_version_id") if not_detected_pub else None
        ),
        "not_detected_version_label": (
            not_detected_pub.get("version_label") if not_detected_pub else None
        ),
        "detected_pub": detected_pub,
        "not_detected_pub": not_detected_pub,
        "lineage_ingests": [lineage] if lineage else [],
        "lineage": lineage,
    }


def publish_train_ready_merge(
    client: "DatasetClient",
    *,
    detected_pub: dict[str, Any] | None,
    not_detected_pub: dict[str, Any] | None,
    train_rows: list[dict[str, Any]],
    run_id: str,
) -> dict[str, Any]:
    """Upload ``train-ready`` after infer (task detect)."""
    train_name = settings.mlair_train_ready_dataset_name
    train_pub = publish_manifest_rows(client, train_name, train_rows, run_id=run_id)
    if not train_pub:
        raise RuntimeError("train-ready dataset publish produced no version (empty manifest)")

    lineage = build_merge_lineage_ingest(
        detected_pub=detected_pub,
        not_detected_pub=not_detected_pub,
        train_pub=train_pub,
    )
    train_vid = str(train_pub["dataset_version_id"])
    return {
        "train_ready_dataset_name": train_name,
        "train_dataset_id": train_pub.get("dataset_id"),
        "train_dataset_version_id": train_vid,
        "train_ready_record_count": train_pub.get("record_count"),
        "lineage_ingests": [lineage] if lineage else [],
        "lineage": lineage,
    }


def rows_ready_for_train(
    rows: list[dict[str, Any]],
    *,
    store: Any,
    cache: dict[str, dict[str, list[dict[str, Any]]]] | None = None,
) -> list[dict[str, Any]]:
    cache = cache if cache is not None else {}
    ready: list[dict[str, Any]] = []
    for row in rows:
        job_id = str(row.get("job_id") or "").strip()
        frame_index = str(row.get("frame_index") or "").strip()
        if not job_id or not frame_index:
            continue
        if job_id not in cache:
            cache[job_id] = load_job_detections(job_id, store)
        key = normalize_frame_key(frame_index)
        dets = cache[job_id].get(key) or cache[job_id].get(key.lstrip("0") or "0")
        if detections_usable(dets):
            ready.append(row)
    return ready


def merge_train_manifest_rows(
    labeled_rows: list[dict[str, Any]],
    missing_rows: list[dict[str, Any]],
    *,
    store: Any,
    cache: dict[str, dict[str, list[dict[str, Any]]]] | None = None,
) -> list[dict[str, Any]]:
    cache = cache if cache is not None else {}
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in labeled_rows + missing_rows:
        job_id = str(row.get("job_id") or "").strip()
        frame_index = normalize_frame_key(str(row.get("frame_index") or ""))
        key = (job_id, frame_index)
        if key in seen:
            continue
        ready = rows_ready_for_train([row], store=store, cache=cache)
        if ready:
            merged.append(row)
            seen.add(key)
    return merged


def publish_manifest_rows(
    client: "DatasetClient",
    dataset_name: str,
    rows: list[dict[str, Any]],
    *,
    run_id: str,
) -> dict[str, Any] | None:
    if not rows:
        return None
    with tempfile.TemporaryDirectory(prefix="cv-detect-publish-") as tmp:
        csv_path = Path(tmp) / f"{dataset_name}-{run_id[:8]}.csv"
        count = write_manifest_csv(rows, csv_path)
        if count <= 0:
            return None
        upload = client.upload_manifest_csv(csv_path, dataset_name)
        vid = extract_dataset_version_id(upload)
        if not vid:
            raise RuntimeError(f"upload did not return dataset_version_id for dataset={dataset_name}")
        ver_label = extract_version_label(upload)
        return {
            "dataset_name": dataset_name,
            "dataset_id": extract_dataset_id(upload),
            "dataset_version_id": vid,
            "version_label": ver_label,
            "record_count": count,
            "upload": upload,
        }


def publish_detect_split_and_train_ready(
    client: "DatasetClient",
    *,
    source_version_id: str,
    labeled_rows: list[dict[str, Any]],
    missing_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    run_id: str,
    model_id: str,
) -> dict[str, Any]:
    detected_name = settings.mlair_detected_dataset_name
    not_detected_name = settings.mlair_not_detected_dataset_name
    train_name = settings.mlair_train_ready_dataset_name

    detected_pub = publish_manifest_rows(client, detected_name, labeled_rows, run_id=run_id)
    not_detected_pub = publish_manifest_rows(client, not_detected_name, missing_rows, run_id=run_id)
    train_pub = publish_manifest_rows(client, train_name, train_rows, run_id=run_id)
    if not train_pub:
        raise RuntimeError("train-ready dataset publish produced no version (empty manifest)")

    train_vid = str(train_pub["dataset_version_id"])
    source_row: dict[str, Any] | None = None
    try:
        source_row = client.get_version(source_version_id)
    except Exception as exc:
        logger.warning("could not load source version %s for lineage: %s", source_version_id, exc)

    lineage_ingests = build_detect_lineage_ingests(
        source_version_row=source_row,
        detected_pub=detected_pub,
        not_detected_pub=not_detected_pub,
        train_pub=train_pub,
    )
    return {
        "detected_dataset_name": detected_name,
        "not_detected_dataset_name": not_detected_name,
        "train_ready_dataset_name": train_name,
        "detected_dataset_id": detected_pub.get("dataset_id") if detected_pub else None,
        "detected_dataset_version_id": detected_pub.get("dataset_version_id") if detected_pub else None,
        "not_detected_dataset_id": not_detected_pub.get("dataset_id") if not_detected_pub else None,
        "not_detected_dataset_version_id": (
            not_detected_pub.get("dataset_version_id") if not_detected_pub else None
        ),
        "train_dataset_id": train_pub.get("dataset_id"),
        "train_dataset_version_id": train_vid,
        "train_ready_record_count": train_pub.get("record_count"),
        "lineage_ingests": lineage_ingests,
        "lineage": lineage_ingests[0] if lineage_ingests else {},
    }
