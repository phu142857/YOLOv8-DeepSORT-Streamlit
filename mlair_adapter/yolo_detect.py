"""Incremental YOLO detect for lifecycle train — only frames missing job detections."""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

from mlair_adapter.job_detections import (
    detections_usable,
    load_job_detections,
    normalize_frame_key,
    write_frame_detection,
)
from shared.artifacts import ArtifactStore
from shared.settings import settings

logger = logging.getLogger(__name__)


def _frame_has_detections(
    job_id: str,
    frame_index: str,
    *,
    store: ArtifactStore,
    cache: dict[str, dict[str, list[dict[str, Any]]]],
) -> bool:
    if not job_id:
        return False
    if job_id not in cache:
        cache[job_id] = load_job_detections(job_id, store)
    key = normalize_frame_key(frame_index)
    dets = cache[job_id].get(key) or cache[job_id].get(key.lstrip("0") or "0")
    return detections_usable(dets)


def scan_manifest_gaps(
    rows: list[dict[str, Any]],
    *,
    store: ArtifactStore | None = None,
    max_frames: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, list[dict[str, Any]]]]]:
    """
    Split manifest rows into already-labeled vs missing detections.

    Returns (labeled_rows, missing_rows, job_det_cache).
    """
    store = store or ArtifactStore()
    cache: dict[str, dict[str, list[dict[str, Any]]]] = {}
    labeled: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    for row in rows:
        if max_frames > 0 and len(labeled) + len(missing) >= max_frames:
            break
        frame_index = str(row.get("frame_index") or "").strip()
        job_id = str(row.get("job_id") or "").strip()
        if _frame_has_detections(job_id, frame_index, store=store, cache=cache):
            labeled.append(row)
        else:
            missing.append(row)

    return labeled, missing, cache


def _resolve_row_image(row: dict[str, Any], dest: Path, client: Any) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if client._copy_frame_from_job_artifacts(row, dest):
        return True
    uri = str(row.get("image_uri") or row.get("uri") or "").strip()
    return client._fetch_uri_to_file(uri, dest)


def run_incremental_detect(
    version_id: str,
    *,
    context: dict[str, Any] | None = None,
    store: ArtifactStore | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    """
    Download dataset manifest, detect only frames missing usable job detections.

    Reuses the same YOLO inference path as CV Execution (``predict_frame``).
    """
    from mlair_adapter.dataset_client import DatasetClient
    from mlair_adapter.yolo_train_pipeline import _resolve_base_weights
    from inference.engine import load_model, predict_frame
    from shared.image_io import load_image_bgr

    ctx = context or {}
    store = store or ArtifactStore()
    client = client or DatasetClient()
    if not client.enabled:
        return {"ok": False, "reason": "mlair_not_configured"}

    raw = client.download_version_csv(version_id)
    rows = client.parse_version_manifest(raw.decode("utf-8"))
    if not rows:
        raise ValueError(f"empty dataset version manifest for {version_id}")

    max_frames = settings.mlair_detect_max_frames
    if max_frames <= 0:
        max_frames = settings.mlair_train_max_frames

    labeled, missing, _cache = scan_manifest_gaps(rows, store=store, max_frames=max_frames or 0)
    total = len(labeled) + len(missing)

    if not missing:
        logger.info(
            "detect: all %d frame(s) already labeled for version=%s — skipping inference",
            len(labeled),
            version_id,
        )
        return {
            "ok": True,
            "skipped": True,
            "reason": "all_frames_labeled",
            "dataset_version_id": version_id,
            "total_frames": total,
            "already_labeled": len(labeled),
            "detected": 0,
            "failed": 0,
        }

    logger.info(
        "detect: version=%s total=%d already_labeled=%d to_detect=%d",
        version_id,
        total,
        len(labeled),
        len(missing),
    )

    weights = _resolve_base_weights(ctx)
    model = load_model(str(weights))
    conf = settings.mlair_detect_confidence

    work_dir = Path(tempfile.mkdtemp(prefix="cv-detect-"))
    detected = 0
    failed = 0
    empty = 0

    try:
        for row in missing:
            frame_index = str(row.get("frame_index") or "").strip()
            job_id = str(row.get("job_id") or "").strip()
            if not frame_index:
                failed += 1
                continue

            img_path = work_dir / f"{normalize_frame_key(frame_index)}.jpg"
            if not _resolve_row_image(row, img_path, client):
                logger.warning(
                    "detect: cannot fetch frame_index=%s job_id=%s",
                    frame_index,
                    job_id or "(none)",
                )
                failed += 1
                continue

            try:
                frame = load_image_bgr(img_path)
            except (FileNotFoundError, RuntimeError) as exc:
                logger.warning("detect: cannot read %s: %s", img_path, exc)
                failed += 1
                continue

            _plot, detections, _cin, _cout = predict_frame(model, frame, conf)
            if not detections:
                empty += 1
                logger.debug("detect: no boxes frame_index=%s job_id=%s", frame_index, job_id)

            if job_id:
                write_frame_detection(job_id, frame_index, detections, store=store)
            else:
                logger.warning(
                    "detect: frame_index=%s has no job_id — detections not persisted to artifacts",
                    frame_index,
                )
                failed += 1
                continue

            detected += 1
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    if detected == 0 and failed > 0:
        raise RuntimeError(
            f"detect failed for all {failed} missing frame(s) — check artifacts/jobs or CV_API_BASE_URL"
        )

    return {
        "ok": True,
        "skipped": False,
        "dataset_version_id": version_id,
        "weights": str(weights),
        "confidence": conf,
        "total_frames": total,
        "already_labeled": len(labeled),
        "detected": detected,
        "empty_detections": empty,
        "failed": failed,
    }
