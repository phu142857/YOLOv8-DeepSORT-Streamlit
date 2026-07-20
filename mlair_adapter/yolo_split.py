"""Lifecycle task 1 — check labeled / split input manifest into Hub datasets."""

from __future__ import annotations

import logging
from typing import Any

from mlair_adapter.detect_dataset_publish import filter_rows_with_job_id, publish_split_datasets
from mlair_adapter.yolo_detect import scan_manifest_gaps
from shared.artifacts import ArtifactStore
from shared.settings import settings

logger = logging.getLogger(__name__)


def run_yolo_split(
    version_id: str,
    *,
    context: dict[str, Any] | None = None,
    store: ArtifactStore | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    """
    Download input version, split into labeled vs not-labeled rows,
    publish Hub ``detected`` and ``not-detected`` (no YOLO).
    """
    from mlair_adapter.dataset_client import DatasetClient

    ctx = context or {}
    store = store or ArtifactStore()
    client = client or DatasetClient()
    if not client.enabled:
        return {"ok": False, "reason": "mlair_not_configured"}

    run_id = str(ctx.get("run_id") or "manual").strip()

    raw = client.download_version_csv(version_id)
    rows = client.parse_version_manifest(raw.decode("utf-8"))
    if not rows:
        raise ValueError(f"empty dataset version manifest for {version_id}")

    max_frames = settings.mlair_detect_max_frames
    if max_frames <= 0:
        max_frames = settings.mlair_train_max_frames

    labeled, missing, _cache = scan_manifest_gaps(rows, store=store, max_frames=max_frames or 0)
    labeled = filter_rows_with_job_id(labeled)
    missing = filter_rows_with_job_id(missing)
    total = len(labeled) + len(missing)

    logger.info(
        "split: version=%s total=%d labeled=%d not_labeled=%d",
        version_id,
        total,
        len(labeled),
        len(missing),
    )

    publish = publish_split_datasets(
        client,
        source_version_id=version_id,
        labeled_rows=labeled,
        missing_rows=missing,
        run_id=run_id,
    )

    return {
        "ok": True,
        "step": "split",
        "source_dataset_version_id": version_id,
        "total_frames": total,
        "already_labeled": len(labeled),
        "not_labeled": len(missing),
        "lineage_ingests": publish.get("lineage_ingests") or [],
        "lineage": publish.get("lineage"),
        **publish,
    }
