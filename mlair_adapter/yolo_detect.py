"""Lifecycle task 2 — YOLO on not-labeled rows, merge train-ready (split is a separate task)."""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

from mlair_adapter.detect_dataset_publish import (
    filter_rows_with_job_id,
    merge_train_manifest_rows,
    publish_train_ready_merge,
)
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
    """Split rows into already-labeled vs missing detections."""
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


def _load_version_rows(client: Any, version_id: str) -> list[dict[str, Any]]:
    raw = client.download_version_csv(version_id)
    rows = client.parse_version_manifest(raw.decode("utf-8"))
    return filter_rows_with_job_id(rows)


def _run_inference_on_missing(
    missing: list[dict[str, Any]],
    *,
    context: dict[str, Any],
    store: ArtifactStore,
    client: Any,
) -> tuple[int, int, int, str, float]:
    from mlair_adapter.yolo_train_pipeline import resolve_detect_teacher_weights
    from inference.engine import load_model, predict_frame
    from shared.image_io import load_image_bgr

    weights = resolve_detect_teacher_weights(context)
    model = load_model(str(weights))
    conf = settings.mlair_detect_confidence

    work_dir = Path(tempfile.mkdtemp(prefix="cv-detect-"))
    detected = 0
    failed = 0
    empty = 0

    from mlair_adapter.worker_task_runtime import raise_if_cancelled

    try:
        for row in missing:
            raise_if_cancelled()
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

            if job_id:
                write_frame_detection(job_id, frame_index, detections, store=store)
            else:
                failed += 1
                continue

            detected += 1
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    return detected, failed, empty, str(weights), conf


def run_yolo_detect_merge(context: dict[str, Any]) -> dict[str, Any]:
    """
    After ``cv_yolo_split``: infer on not-detected manifest, merge → ``train-ready``.
    """
    from mlair_adapter.dataset_client import DatasetClient
    from mlair_adapter.run_workspace import load_state, require_keys

    ctx = context or {}
    store = ArtifactStore()
    client = DatasetClient()
    if not client.enabled:
        return {"ok": False, "reason": "mlair_not_configured"}

    run_id = str(ctx.get("run_id") or "manual").strip()
    model_id = str(ctx.get("model_id") or ctx.get("mlair_model_id") or "").strip()
    if not model_id:
        return {"ok": False, "reason": "model_id_required"}

    state = load_state(run_id)
    require_keys(
        state,
        (
            "split_ok",
            "source_dataset_version_id",
        ),
        step="detect",
    )

    detected_vid = str(state.get("detected_dataset_version_id") or "").strip()
    not_detected_vid = str(state.get("not_detected_dataset_version_id") or "").strip()
    if not detected_vid and not not_detected_vid:
        raise ValueError("detect requires split output: detected or not-detected version id")

    labeled: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    if detected_vid:
        labeled = _load_version_rows(client, detected_vid)
    if not_detected_vid:
        missing = _load_version_rows(client, not_detected_vid)

    skipped_infer = not missing
    detected_count = 0
    failed = 0
    empty = 0
    weights = ""
    conf = settings.mlair_detect_confidence

    if skipped_infer:
        logger.info("detect: no not-detected frames — skip YOLO infer")
    else:
        logger.info("detect: infer on %d not-detected frame(s)", len(missing))
        detected_count, failed, empty, weights, conf = _run_inference_on_missing(
            missing, context=ctx, store=store, client=client
        )
        if detected_count == 0 and failed > 0 and not labeled:
            raise RuntimeError(
                f"detect failed for all {failed} missing frame(s) — check artifacts/jobs or CV_API_BASE_URL"
            )

    cache: dict[str, dict[str, list[dict[str, Any]]]] = {}
    train_rows = merge_train_manifest_rows(labeled, missing, store=store, cache=cache)
    if not train_rows:
        raise RuntimeError(
            "no frames with usable detections for train-ready merge "
            f"(labeled={len(labeled)} missing={len(missing)} failed={failed})"
        )

    def _pub_from_version(dataset_name: str, version_id: str, label_fallback: str) -> dict[str, Any]:
        row = client.get_version(version_id)
        ver = str(row.get("version") or label_fallback or "")
        return {
            "dataset_name": dataset_name,
            "dataset_version_id": version_id,
            "version_label": ver,
            "upload": {
                "version": ver,
                "uri": row.get("uri"),
                "checksum": row.get("checksum"),
            },
        }

    detected_pub = (
        _pub_from_version(
            settings.mlair_detected_dataset_name,
            detected_vid,
            str(state.get("detected_version_label") or ""),
        )
        if detected_vid
        else None
    )
    not_detected_pub = (
        _pub_from_version(
            settings.mlair_not_detected_dataset_name,
            not_detected_vid,
            str(state.get("not_detected_version_label") or ""),
        )
        if not_detected_vid
        else None
    )

    publish = publish_train_ready_merge(
        client,
        detected_pub=detected_pub,
        not_detected_pub=not_detected_pub,
        train_rows=train_rows,
        run_id=run_id,
    )

    train_vid = str(publish["train_dataset_version_id"])
    return {
        "ok": True,
        "step": "detect",
        "skipped": skipped_infer,
        "source_dataset_version_id": state.get("source_dataset_version_id"),
        "dataset_version_id": train_vid,
        "train_dataset_version_id": train_vid,
        "detected": detected_count,
        "empty_detections": empty,
        "failed": failed,
        "already_labeled": len(labeled),
        "not_detected_frames": len(missing),
        "train_ready_record_count": publish.get("train_ready_record_count"),
        "weights": weights or None,
        "confidence": conf,
        "lineage_ingests": publish.get("lineage_ingests") or [],
        "lineage": publish.get("lineage"),
        **publish,
    }


def run_incremental_detect(
    version_id: str,
    *,
    context: dict[str, Any] | None = None,
    store: ArtifactStore | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    """Legacy single-step API: split + detect + merge (local scripts)."""
    from mlair_adapter.run_workspace import save_state
    from mlair_adapter.yolo_split import run_yolo_split

    ctx = context or {}
    run_id = str(ctx.get("run_id") or "manual")
    split = run_yolo_split(version_id, context={**ctx, "run_id": run_id}, store=store, client=client)
    if not split.get("ok"):
        return split
    save_state(
        run_id,
        {
            "split_ok": True,
            "source_dataset_version_id": version_id,
            "detected_dataset_version_id": split.get("detected_dataset_version_id"),
            "not_detected_dataset_version_id": split.get("not_detected_dataset_version_id"),
            "detected_version_label": split.get("detected_version_label"),
            "not_detected_version_label": split.get("not_detected_version_label"),
        },
    )
    detect = run_yolo_detect_merge({**ctx, "run_id": run_id})
    if detect.get("ok"):
        detect.setdefault("total_frames", split.get("total_frames"))
    return detect
