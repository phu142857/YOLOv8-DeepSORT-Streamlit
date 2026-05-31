"""Real Ultralytics YOLO training for MLAir dataset versions (CV frame manifests)."""

from __future__ import annotations

import json
import logging
import random
import shutil
from pathlib import Path
from typing import Any

import cv2
import yaml

from mlair_adapter.torch_compat import apply_torch_checkpoint_compat
from mlair_adapter.train_device import resolve_train_device, run_ultralytics_train_with_device_policy

apply_torch_checkpoint_compat()

from ultralytics import YOLO

from inference.engine import load_model, predict_frame
from mlair_adapter.dataset_client import DatasetClient
from shared.artifacts import ArtifactStore
from shared.image_io import load_image_bgr
from mlair_adapter.model_client import ModelClient
from shared.model_resolve import (
    ensure_registry_weights,
    is_registry_model,
    parse_model_spec,
    registry_model_id,
    resolve_model_path,
)
from shared.weights_catalog import PRETRAINED_VERSION, find_weights_in_dir, version_dir
from shared.settings import settings

logger = logging.getLogger(__name__)


def _resolve_train_checkpoint(
    model: Any,
    results: Any,
    work_dir: Path,
    *,
    run_name: str = "train",
) -> tuple[Path, Path]:
    """
    Locate best/last .pt after model.train().

    Vendored ultralytics returns None from train(); pip 8.x may return a Results object.
    """
    save_dir: Path | None = None
    if results is not None:
        sd = getattr(results, "save_dir", None)
        if sd:
            save_dir = Path(sd)

    trainer = getattr(model, "trainer", None)
    if save_dir is None and trainer is not None:
        sd = getattr(trainer, "save_dir", None)
        if sd:
            save_dir = Path(sd)
    if save_dir is None:
        save_dir = work_dir / "runs" / run_name

    candidates: list[Path] = []
    if trainer is not None:
        for attr in ("best", "last"):
            p = getattr(trainer, attr, None)
            if p:
                candidates.append(Path(p))
    candidates.extend([save_dir / "weights" / "best.pt", save_dir / "weights" / "last.pt"])

    best_pt: Path | None = None
    for path in candidates:
        if path.is_file():
            best_pt = path
            break

    if best_pt is None:
        found = _find_checkpoint_under_runs(work_dir, preferred_name=run_name)
        if found:
            best_pt, save_dir = found

    if best_pt is None:
        raise FileNotFoundError(f"no checkpoint under {save_dir}/weights (train may have aborted)")
    return best_pt, save_dir


def _find_checkpoint_under_runs(work_dir: Path, *, preferred_name: str = "train") -> tuple[Path, Path] | None:
    """Fallback when model.trainer is cleared but weights were written to disk."""
    runs_root = work_dir / "runs"
    if not runs_root.is_dir():
        return None

    def _pick(save_dir: Path) -> tuple[Path, Path] | None:
        for name in ("best.pt", "last.pt"):
            pt = save_dir / "weights" / name
            if pt.is_file():
                return pt, save_dir
        return None

    preferred = runs_root / preferred_name
    hit = _pick(preferred)
    if hit:
        return hit

    subdirs = sorted(
        (p for p in runs_root.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for sd in subdirs:
        hit = _pick(sd)
        if hit:
            return hit
    return None


def _metrics_from_train_results(results: Any, model: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    if results is not None and hasattr(results, "results_dict"):
        raw = results.results_dict or {}
        out = {k: float(v) for k, v in raw.items() if isinstance(v, (int, float))}
    if out:
        return out
    trainer = getattr(model, "trainer", None)
    metrics = getattr(trainer, "metrics", None) if trainer else None
    if isinstance(metrics, dict):
        return {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
    return out


# Traffic workload class names (contiguous 0..nc-1 in data.yaml — not raw COCO ids).
_TRAFFIC_CLASS_NAMES: tuple[str, ...] = (
    "bicycle",
    "bus",
    "car",
    "motorcycle",
    "person",
    "train",
    "truck",
)
_NAME_TO_CONTIGUOUS_ID: dict[str, int] = {n: i for i, n in enumerate(_TRAFFIC_CLASS_NAMES)}
# COCO pretrained ids from detector — used only to resolve missing class_name.
_COCO_ID_TO_NAME: dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    6: "train",
    7: "truck",
}


def _detection_contiguous_class_id(det: dict[str, Any]) -> int | None:
    name = str(det.get("class_name") or "").lower().strip()
    if name in _NAME_TO_CONTIGUOUS_ID:
        return _NAME_TO_CONTIGUOUS_ID[name]
    if det.get("class_id") is not None:
        try:
            coco_name = _COCO_ID_TO_NAME.get(int(det["class_id"]))
        except (TypeError, ValueError):
            coco_name = None
        if coco_name:
            return _NAME_TO_CONTIGUOUS_ID.get(coco_name)
    return None


def _resolve_base_weights(context: dict[str, Any]) -> Path:
    artifact_uri = str(context.get("artifact_uri") or "").strip()
    model_id = str(context.get("model_id") or context.get("mlair_model_id") or "").strip()

    if artifact_uri.startswith("file://") or artifact_uri.startswith("/"):
        client = ModelClient()
        dest = settings.mlair_weights_cache_dir / "train-base" / "weights.pt"
        return client.download_artifact(artifact_uri, dest)

    if model_id:
        return ensure_registry_weights(model_id, stage=settings.mlair_promote_stage)

    spec = settings.mlair_train_base_model_spec
    if is_registry_model(spec):
        return ensure_registry_weights(registry_model_id(spec))
    parsed = parse_model_spec(spec)
    if parsed:
        model, version = parsed
        if version == "base":
            root = settings.detection_model_dir
            try:
                pre = find_weights_in_dir(version_dir(root, model, PRETRAINED_VERSION))
            except ValueError:
                pre = None
            if pre is not None:
                logger.info("train base weights: %s (COCO pretrained, not production base)", pre)
                return pre
    return resolve_model_path(spec)


def _pseudo_label_samples(
    samples: list[tuple[Path, str | None]],
    *,
    context: dict[str, Any] | None = None,
) -> list[tuple[Path, str | None, list[dict[str, Any]]]]:
    ctx = context or {}
    base_weights = _resolve_base_weights(ctx)
    logger.warning(
        "prepare: no detections in job artifacts — pseudo-labeling %d frame(s) with %s",
        len(samples),
        base_weights,
    )
    model = load_model(str(base_weights))
    conf = settings.mlair_prepare_pseudo_confidence
    labeled: list[tuple[Path, str | None, list[dict[str, Any]]]] = []
    for img_path, job_id in samples:
        try:
            frame = load_image_bgr(img_path)
        except (FileNotFoundError, RuntimeError):
            logger.warning("prepare: cannot read image %s", img_path)
            continue
        _, detections, _, _ = predict_frame(model, frame, conf)
        if detections:
            labeled.append((img_path, job_id, detections))
    return labeled


def _load_job_detections(job_id: str, store: ArtifactStore | None = None) -> dict[str, list[dict[str, Any]]]:
    """frame_index (stem) -> detections list from job artifacts."""
    store = store or ArtifactStore()
    path = store.resolve_artifact(job_id, "detections")
    if path is None or not path.is_file():
        return {}

    out: dict[str, list[dict[str, Any]]] = {}
    if path.suffix == ".json":
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            frame = str(row.get("frame", "0"))
            out[frame.zfill(6) if frame.isdigit() else frame] = row.get("detections") or []
            return out
        except json.JSONDecodeError:
            return {}

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        frame = str(row.get("frame", row.get("frame_index", "")))
        dets = row.get("detections") or []
        if frame and dets:
            out[frame.zfill(6) if frame.isdigit() else frame] = dets
    return out


def _xyxy_to_yolo_line(xyxy: list[float], img_w: int, img_h: int, class_id: int) -> str:
    x1, y1, x2, y2 = xyxy
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def _write_label_file(label_path: Path, detections: list[dict[str, Any]], img_w: int, img_h: int) -> bool:
    lines: list[str] = []
    for det in detections:
        cls_id = _detection_contiguous_class_id(det)
        if cls_id is None:
            continue
        xyxy = det.get("xyxy")
        if not xyxy or len(xyxy) != 4:
            continue
        lines.append(_xyxy_to_yolo_line(xyxy, img_w, img_h, cls_id))
    if not lines:
        return False
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def _build_yolo_dataset(
    version_id: str,
    work_dir: Path,
    *,
    val_ratio: float = 0.2,
    context: dict[str, Any] | None = None,
) -> tuple[Path, int, int]:
    """Download frames + pseudo-labels from job detections; return (data_yaml, n_train, n_val)."""
    client = DatasetClient()
    images_all = work_dir / "images_all"
    images_all.mkdir(parents=True, exist_ok=True)
    _, count = client.fetch_version_frames(version_id, images_all, max_frames=settings.mlair_train_max_frames)

    samples: list[tuple[Path, str | None]] = []
    manifest_csv = work_dir / "pulled_manifest.csv"
    artifact_store = ArtifactStore()
    job_det_cache: dict[str, dict[str, list[dict[str, Any]]]] = {}

    if manifest_csv.is_file():
        manifest_text = manifest_csv.read_text(encoding="utf-8")
        for row in DatasetClient.parse_version_manifest(manifest_text):
            frame_index = str(row.get("frame_index") or "").strip()
            job_id = str(row.get("job_id") or "").strip()
            img = images_all / f"{frame_index}.jpg"
            if not img.is_file():
                img = images_all / f"{frame_index.zfill(6)}.jpg"
            if not img.is_file():
                continue
            samples.append((img, job_id if job_id else None))

    if not samples:
        for img in sorted(images_all.glob("*.jpg")):
            samples.append((img, None))

    names = list(_TRAFFIC_CLASS_NAMES)
    labeled_samples: list[tuple[Path, str | None, list[dict[str, Any]] | None]] = []
    for img_path, job_id in samples:
        stem = img_path.stem
        dets: list[dict[str, Any]] | None = None
        if job_id:
            if job_id not in job_det_cache:
                job_det_cache[job_id] = _load_job_detections(job_id, artifact_store)
            dets = job_det_cache[job_id].get(stem) or job_det_cache[job_id].get(stem.lstrip("0"))
        if dets:
            labeled_samples.append((img_path, job_id, dets))

    if not labeled_samples:
        if settings.mlair_prepare_fallback_pseudo and samples:
            labeled_samples = _pseudo_label_samples(samples, context=context)
        if not labeled_samples:
            raise ValueError(
                "no labeled frames for training — run CV Execution with a working detector "
                "(use yolov8s/base or restore pretrained; empty detections.jsonl gives no labels)"
            )

    random.shuffle(labeled_samples)
    split = max(1, int(len(labeled_samples) * (1.0 - val_ratio)))
    train_labeled = labeled_samples[:split]
    val_labeled = labeled_samples[split:] or labeled_samples[:1]

    for split_name, subset in (("train", train_labeled), ("val", val_labeled)):
        img_dir = work_dir / "images" / split_name
        lbl_dir = work_dir / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for img_path, _job_id, dets in subset:
            stem = img_path.stem
            dest_img = img_dir / f"{stem}.jpg"
            shutil.copy2(img_path, dest_img)
            h, w = load_image_bgr(dest_img).shape[:2]
            _write_label_file(lbl_dir / f"{stem}.txt", dets or [], w, h)

    data_yaml = work_dir / "data.yaml"
    data = {
        "path": str(work_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: n for i, n in enumerate(names)},
        "nc": len(names),
    }
    data_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return data_yaml, len(train_labeled), len(val_labeled)


def run_yolo_training(context: dict[str, Any], *, work_root: Path | None = None) -> dict[str, Any]:
    """
    Train YOLO from MLAir plugin_context + import new model version.

    Required context keys: dataset_version_id, model_id (or mlair_model_id).
    Optional: dataset_id, artifact_uri, run_id.
    """
    version_id = str(context.get("dataset_version_id") or "").strip()
    model_id = str(context.get("model_id") or context.get("mlair_model_id") or "").strip()
    if not version_id:
        raise ValueError("dataset_version_id is required")
    if not model_id:
        raise ValueError("model_id is required")

    run_id = str(context.get("run_id") or "manual")
    work_dir = (work_root or settings.artifact_root / "mlair_train" / run_id).resolve()
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    base_weights = _resolve_base_weights(context)
    data_yaml, n_train, n_val = _build_yolo_dataset(version_id, work_dir / "dataset", context=context)

    logger.info(
        "Starting YOLO train run_id=%s version=%s base=%s train=%s val=%s",
        run_id,
        version_id,
        base_weights,
        n_train,
        n_val,
    )

    model = YOLO(str(base_weights))
    results, train_device = run_ultralytics_train_with_device_policy(
        model,
        batch=settings.mlair_train_batch,
        data=str(data_yaml),
        epochs=settings.mlair_train_epochs,
        imgsz=settings.mlair_train_imgsz,
        project=str(work_dir / "runs"),
        name="train",
        exist_ok=True,
        verbose=True,
    )
    logger.info("YOLO train finished device=%s", train_device)

    best_pt, save_dir = _resolve_train_checkpoint(model, results, work_dir)

    model_client = ModelClient()
    imported = model_client.import_version(
        model_id,
        best_pt,
        stage=settings.mlair_train_import_stage,
    )

    metrics = _metrics_from_train_results(results, model)

    return {
        "ok": True,
        "model_id": model_id,
        "dataset_version_id": version_id,
        "checkpoint": str(best_pt),
        "imported_version": imported,
        "metrics": metrics,
        "train_images": n_train,
        "val_images": n_val,
    }
