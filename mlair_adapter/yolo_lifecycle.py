"""Phase B — MLAir lifecycle DAG: prepare → train → eval → gate (+ hard-example mine)."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from mlair_adapter.torch_compat import apply_torch_checkpoint_compat

apply_torch_checkpoint_compat()

from ultralytics import YOLO

from mlair_adapter.dataset_client import DatasetClient
from mlair_adapter.model_client import ModelClient
from mlair_adapter.model_sync import ModelSyncService
from mlair_adapter.run_workspace import load_state, require_keys, save_state, workspace_dir
from mlair_adapter.train_device import resolve_train_device, run_ultralytics_train_with_device_policy
from mlair_adapter.worker_task_runtime import capture_active_monitor_sample
from mlair_adapter.yolo_train_pipeline import (
    _build_yolo_dataset,
    _metrics_from_train_results,
    _resolve_base_weights,
    _resolve_train_checkpoint,
)
from shared.settings import settings

logger = logging.getLogger(__name__)


def _extract_map50(metrics: Any) -> float | None:
    if metrics is None:
        return None
    box = getattr(metrics, "box", None)
    if box is not None and hasattr(box, "map50"):
        try:
            return float(box.map50)
        except (TypeError, ValueError):
            pass
    if hasattr(metrics, "results_dict"):
        rd = metrics.results_dict or {}
        for key in ("metrics/mAP50(B)", "mAP50", "map50"):
            if key in rd:
                try:
                    return float(rd[key])
                except (TypeError, ValueError):
                    continue
    return None


def _context_ids(context: dict[str, Any]) -> tuple[str, str, str]:
    version_id = str(context.get("dataset_version_id") or "").strip()
    model_id = str(context.get("model_id") or context.get("mlair_model_id") or "").strip()
    if not model_id:
        from shared.unified_catalog import resolve_mlair_model_id

        for key in ("model_spec", "model_name", "model"):
            spec = str(context.get(key) or "").strip()
            if spec:
                model_id = resolve_mlair_model_id(spec) or ""
                if model_id:
                    break
    run_id = str(context.get("run_id") or "manual").strip()
    if not version_id:
        raise ValueError("dataset_version_id is required")
    if not model_id:
        raise ValueError("model_id is required (set in pipeline context or register model on Hub)")
    return run_id, model_id, version_id


def run_prepare(context: dict[str, Any]) -> dict[str, Any]:
    """Download dataset version, pseudo-label, train/val split → data.yaml on disk."""
    run_id, model_id, version_id = _context_ids(context)
    logger.info(
        "prepare run_id=%s model_id=%s dataset_version_id=%s",
        run_id,
        model_id,
        version_id,
    )
    work_dir = workspace_dir(run_id)
    dataset_dir = work_dir / "dataset"
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    data_yaml, n_train, n_val = _build_yolo_dataset(version_id, dataset_dir, context=context)
    state = save_state(
        run_id,
        {
            "model_id": model_id,
            "dataset_version_id": version_id,
            "work_dir": str(work_dir),
            "data_yaml": str(data_yaml),
            "n_train": n_train,
            "n_val": n_val,
            "prepare_ok": True,
        },
    )
    return {
        "ok": True,
        "step": "prepare",
        "run_id": run_id,
        "data_yaml": str(data_yaml),
        "train_images": n_train,
        "val_images": n_val,
        "workspace": str(work_dir),
        "metrics": {"train_images": float(n_train), "val_images": float(n_val)},
        "state": state,
    }


def run_train_step(context: dict[str, Any]) -> dict[str, Any]:
    """Fine-tune YOLO; import registry version as staging (not production)."""
    run_id, model_id, version_id = _context_ids(context)
    state = load_state(run_id)
    require_keys(state, ("data_yaml", "work_dir"), step="train")

    work_dir = Path(state["work_dir"])
    data_yaml = Path(state["data_yaml"])
    base_weights = _resolve_base_weights(context)

    logger.info("lifecycle train run_id=%s model=%s version=%s", run_id, model_id, version_id)
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
    logger.info("lifecycle train finished device=%s", train_device)
    capture_active_monitor_sample()

    best_pt, save_dir = _resolve_train_checkpoint(model, results, work_dir)

    import_stage = settings.mlair_lifecycle_import_stage
    imported = ModelClient().import_version(model_id, best_pt, stage=import_stage)
    version_num = int(imported.get("version") or 0)
    train_metrics = _metrics_from_train_results(results, model)
    logger.info("train checkpoint=%s save_dir=%s", best_pt, save_dir)

    save_state(
        run_id,
        {
            "checkpoint": str(best_pt),
            "imported_version": version_num,
            "import_stage": import_stage,
            "train_metrics": train_metrics,
            "train_ok": True,
            "train_device": str(train_device),
        },
    )
    return {
        "ok": True,
        "step": "train",
        "checkpoint": str(best_pt),
        "imported_version": version_num,
        "import_stage": import_stage,
        "metrics": train_metrics,
        "train_images": state.get("n_train"),
        "val_images": state.get("n_val"),
    }


def run_eval(context: dict[str, Any]) -> dict[str, Any]:
    """Validate candidate checkpoint on held-out split."""
    run_id, model_id, _version_id = _context_ids(context)
    state = load_state(run_id)
    require_keys(state, ("checkpoint", "data_yaml"), step="eval")

    ckpt = Path(str(state["checkpoint"]))
    data_yaml = str(state["data_yaml"])
    eval_device = resolve_train_device(batch=settings.mlair_train_batch)
    logger.info("lifecycle eval device=%s", eval_device)
    metrics = YOLO(str(ckpt)).val(data=data_yaml, device=eval_device, verbose=False)
    capture_active_monitor_sample()
    map50 = _extract_map50(metrics)
    eval_out = {
        "mAP50": map50,
        "metrics": {},
    }
    if hasattr(metrics, "results_dict"):
        eval_out["metrics"] = {
            k: float(v) for k, v in (metrics.results_dict or {}).items() if isinstance(v, (int, float))
        }

    save_state(run_id, {"eval_map50": map50, "eval_metrics": eval_out, "eval_ok": True})
    return {
        "ok": True,
        "step": "eval",
        "model_id": model_id,
        "mAP50": map50,
        "metrics": eval_out.get("metrics") or {},
        "imported_version": state.get("imported_version"),
    }


def run_gate(context: dict[str, Any]) -> dict[str, Any]:
    """Compare candidate vs production on same val split; promote if improved."""
    run_id, model_id, _version_id = _context_ids(context)
    state = load_state(run_id)
    require_keys(state, ("checkpoint", "data_yaml", "imported_version"), step="gate")

    data_yaml = str(state["data_yaml"])
    candidate_map = state.get("eval_map50")
    if candidate_map is None:
        eval_res = run_eval(context)
        candidate_map = eval_res.get("mAP50")

    prod_weights = _resolve_base_weights({**context, "model_id": model_id})
    gate_device = resolve_train_device(batch=settings.mlair_train_batch)
    logger.info("lifecycle gate eval device=%s", gate_device)
    prod_metrics = YOLO(str(prod_weights)).val(data=data_yaml, device=gate_device, verbose=False)
    capture_active_monitor_sample()
    prod_map = _extract_map50(prod_metrics)

    min_delta = settings.mlair_gate_min_map_delta
    passed = False
    if candidate_map is not None and prod_map is not None:
        passed = float(candidate_map) >= float(prod_map) + float(min_delta)
    elif candidate_map is not None and prod_map is None:
        passed = True

    gate_result = {
        "passed": passed,
        "candidate_map50": candidate_map,
        "production_map50": prod_map,
        "min_delta": min_delta,
        "imported_version": state.get("imported_version"),
    }
    save_state(run_id, {"gate": gate_result, "gate_ok": passed})

    promoted: dict[str, Any] | None = None
    if passed and settings.mlair_lifecycle_auto_promote:
        version_num = int(state["imported_version"])
        promoted = ModelClient().promote(model_id, version_num, stage=settings.mlair_promote_stage)
        gate_result["promoted"] = promoted
        if settings.mlair_sync_on_hub_promote:
            try:
                ModelSyncService().apply_mlair_promotion_to_local(model_id, version_num)
            except Exception as exc:
                logger.warning("post-promote local sync failed: %s", exc)

    if not passed:
        raise RuntimeError(
            f"quality gate failed: candidate mAP50={candidate_map} production mAP50={prod_map} "
            f"(required delta >= {min_delta})"
        )

    return {
        "ok": True,
        "step": "gate",
        "gate": gate_result,
        "promoted": promoted is not None,
        "metrics": {"mAP50": candidate_map, "production_mAP50": prod_map},
    }


def run_hard_example_mine(context: dict[str, Any]) -> dict[str, Any]:
    """
    Scan a dataset version with production weights; append low-confidence frames to hard-example buffer.
    """
    run_id = str(context.get("run_id") or "mine").strip()
    version_id = str(context.get("dataset_version_id") or "").strip()
    model_id = str(context.get("model_id") or context.get("mlair_model_id") or "").strip()
    if not model_id:
        from shared.unified_catalog import resolve_mlair_model_id

        for key in ("model_spec", "model_name", "model"):
            spec = str(context.get(key) or "").strip()
            if spec:
                model_id = resolve_mlair_model_id(spec) or ""
                if model_id:
                    break
    if not version_id:
        raise ValueError("dataset_version_id is required for hard-example mining")
    if not model_id:
        raise ValueError("model_id is required for hard-example mining")

    client = DatasetClient()
    if not client.enabled:
        return {"ok": False, "reason": "mlair_not_configured"}

    work_dir = workspace_dir(run_id) / "mine"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    frames_dir = work_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    _, total = client.fetch_version_frames(version_id, frames_dir, max_frames=settings.mlair_hard_example_max_scan)

    weights = _resolve_base_weights({**context, "model_id": model_id})
    model = YOLO(str(weights))
    max_conf = settings.mlair_hard_example_max_conf
    rows: list[dict[str, Any]] = []
    cache = settings.artifact_root / "mlair_hard_examples" / run_id
    cache.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(frames_dir.glob("*.jpg")):
        results = model.predict(str(img_path), verbose=False, conf=0.01)
        best = 0.0
        for r in results:
            if r.boxes is not None and len(r.boxes):
                confs = r.boxes.conf.cpu().numpy()
                if len(confs):
                    best = max(best, float(confs.max()))
        if best <= max_conf:
            cached = cache / img_path.name
            shutil.copy2(img_path, cached)
            rows.append(
                {
                    "image_uri": f"file://{cached.resolve()}",
                    "frame_index": img_path.stem,
                    "job_id": run_id,
                    "source_file": "hard_example_mine",
                    "artifact_path": img_path.name,
                }
            )

    if not rows:
        return {
            "ok": True,
            "step": "hard_example_mine",
            "scanned": total,
            "appended": 0,
            "reason": "no_hard_examples_found",
        }

    out = client.append_buffer_rows_by_name(
        settings.mlair_hard_example_dataset_name,
        rows=rows,
        source_type="hard_example_mine",
        execution_id=run_id,
    )
    out["scanned"] = total
    out["appended"] = len(rows)
    out["step"] = "hard_example_mine"
    save_state(run_id, {"hard_example_mine": out})
    return out


def run_legacy_monolithic_train(context: dict[str, Any], *, work_root: Path | None = None) -> dict[str, Any]:
    """Backward-compatible single-step train (cv-yolo-vehicle-train v1)."""
    from mlair_adapter.yolo_train_pipeline import run_yolo_training

    return run_yolo_training(context, work_root=work_root)
