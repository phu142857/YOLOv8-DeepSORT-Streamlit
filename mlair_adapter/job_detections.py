"""Read/write CV job detection artifacts (no YOLO/torch dependency)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from shared.artifacts import ArtifactStore


def normalize_frame_key(frame_index: str) -> str:
    raw = str(frame_index or "").strip()
    if not raw:
        return ""
    if raw.isdigit():
        return raw.zfill(6)
    return raw


def detections_usable(detections: list[dict[str, Any]] | None) -> bool:
    if not detections:
        return False
    for det in detections:
        xyxy = det.get("xyxy")
        if xyxy and len(xyxy) == 4:
            return True
    return False


def load_job_detections(job_id: str, store: ArtifactStore | None = None) -> dict[str, list[dict[str, Any]]]:
    """Map frame_index stem → detections list from ``artifacts/jobs/{job_id}/detections.*``."""
    store = store or ArtifactStore()
    path = store.resolve_artifact(job_id, "detections")
    if path is None or not path.is_file():
        return {}

    out: dict[str, list[dict[str, Any]]] = {}
    if path.suffix == ".json":
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        frame = str(row.get("frame_index") or row.get("frame") or "0")
        key = normalize_frame_key(frame)
        if key:
            out[key] = row.get("detections") or []
        return out

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        frame = str(row.get("frame_index") or row.get("frame") or "")
        dets = row.get("detections") or []
        if not frame:
            continue
        key = normalize_frame_key(frame)
        if dets:
            out[key] = dets
    return out


def load_existing_detections_map(job_id: str, store: ArtifactStore) -> dict[str, list[dict[str, Any]]]:
    return load_job_detections(job_id, store)


def write_frame_detection(
    job_id: str,
    frame_index: str,
    detections: list[dict[str, Any]],
    *,
    store: ArtifactStore | None = None,
) -> Path:
    """Merge one frame's detections into job artifacts (json for single-frame, jsonl for multi)."""
    store = store or ArtifactStore()
    layout = store.job_layout(job_id)
    det_dir = layout["detections"]
    key = normalize_frame_key(frame_index)

    merged = load_existing_detections_map(job_id, store)
    merged[key] = detections

    jsonl_path = det_dir / "detections.jsonl"
    json_path = det_dir / "detections.json"

    if len(merged) == 1 and json_path.is_file() and not jsonl_path.is_file():
        frame_val: str | int = int(key) if key.isdigit() else key
        store.write_json(
            json_path,
            {"frame": frame_val, "frame_index": key, "detections": detections},
        )
        return json_path

    if json_path.is_file():
        json_path.unlink()
    lines: list[str] = []
    for frame_key in sorted(merged.keys()):
        frame_val = int(frame_key) if frame_key.isdigit() else frame_key
        payload = {
            "frame": frame_val,
            "frame_index": frame_key,
            "detections": merged[frame_key],
        }
        lines.append(json.dumps(payload, default=str))
    jsonl_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return jsonl_path
