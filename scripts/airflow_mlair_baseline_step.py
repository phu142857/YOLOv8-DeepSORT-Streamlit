#!/usr/bin/env python3
"""
Airflow + MLflow E2 baseline: one CV lifecycle step using MLAir Hub datasets.

Reuses ``mlair_adapter.yolo_lifecycle`` (same plugins as ``cv-yolo-lifecycle-train``).
State workspace: ``artifacts/airflow_baseline/{baseline_run_id}/state.json``.

Env:
  BASELINE_RUN_ID — correlate Airflow dag_run (required)
  BASELINE_MODE — ``train_ready`` (default) or ``full``
  MLFLOW_TRACKING_URI — e.g. http://127.0.0.1:5000
  MLFLOW_EXPERIMENT_NAME — default cv-yolo-lifecycle-baseline-e2
  AIRFLOW_DAG_RUN_ID — optional tag on MLflow run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlair_adapter.torch_compat import apply_torch_checkpoint_compat

apply_torch_checkpoint_compat()

from mlair_adapter.yolo_lifecycle import (
    run_detect,
    run_eval,
    run_gate,
    run_prepare,
    run_split,
    run_train_step,
)

STEP_HANDLERS = {
    "split": run_split,
    "detect": run_detect,
    "prepare": run_prepare,
    "train": run_train_step,
    "eval": run_eval,
    "gate": run_gate,
}

MLFLOW_RUN_ID_FILE = "mlflow_run_id.txt"


def _mlflow_run_dir(baseline_run_id: str) -> Path:
    from shared.settings import settings

    return settings.artifact_root / "airflow_baseline" / baseline_run_id


def _ensure_mlflow_run(baseline_run_id: str) -> str | None:
    uri = (os.getenv("MLFLOW_TRACKING_URI") or "").strip()
    if not uri:
        return None
    import mlflow

    mlflow.set_tracking_uri(uri)
    exp = (os.getenv("MLFLOW_EXPERIMENT_NAME") or "cv-yolo-lifecycle-baseline-e2").strip()
    mlflow.set_experiment(exp)

    run_dir = _mlflow_run_dir(baseline_run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    id_path = run_dir / MLFLOW_RUN_ID_FILE
    if id_path.is_file():
        return id_path.read_text(encoding="utf-8").strip()

    run_name = f"af-{baseline_run_id}"[:250]
    if mlflow.active_run():
        mlflow.end_run()
    run = mlflow.start_run(run_name=run_name)
    run_id = run.info.run_id
    id_path.write_text(run_id, encoding="utf-8")
    mlflow.set_tag("baseline_run_id", baseline_run_id)
    af = (os.getenv("AIRFLOW_DAG_RUN_ID") or baseline_run_id).strip()
    mlflow.set_tag("airflow.dag_run_id", af)
    mlflow.set_tag("baseline.mode", (os.getenv("BASELINE_MODE") or "train_ready").strip())
    mlflow.end_run()
    return run_id


def _mlflow_metric_name(step: str, key: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_./: -]", "_", str(key))
    return f"{step}.{safe}"


def _log_step_to_mlflow(baseline_run_id: str, step: str, result: dict) -> None:
    run_id = _ensure_mlflow_run(baseline_run_id)
    if not run_id:
        return
    import mlflow

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    if mlflow.active_run():
        mlflow.end_run()
    with mlflow.start_run(run_id=run_id):
        mlflow.set_tag(f"task.{step}.status", "ok" if result.get("ok") else "fail")
        metrics = result.get("metrics") or {}
        for key, val in metrics.items():
            if isinstance(val, (int, float)):
                mlflow.log_metric(_mlflow_metric_name(step, key), float(val), step=0)
        if result.get("mAP50") is not None:
            mlflow.log_metric(f"{step}.mAP50", float(result["mAP50"]), step=0)
        if result.get("train_dataset_version_id"):
            mlflow.log_param(f"{step}.train_dataset_version_id", str(result["train_dataset_version_id"]))


def _build_context(args: argparse.Namespace) -> dict:
    baseline_run_id = (os.getenv("BASELINE_RUN_ID") or args.baseline_run_id or "baseline").strip()
    mode = (os.getenv("BASELINE_MODE") or args.mode or "train_ready").strip().lower()
    model_id = (args.model_id or os.getenv("MLAIR_MODEL_ID") or "").strip()
    if not model_id:
        raise ValueError("model_id required (--model-id or MLAIR_MODEL_ID)")

    input_vid = (args.dataset_version_id or os.getenv("MLAIR_INPUT_VERSION_ID") or "").strip()
    train_ready_vid = (
        args.train_ready_version_id or os.getenv("MLAIR_TRAIN_READY_VERSION_ID") or ""
    ).strip()

    ctx: dict = {
        "run_id": baseline_run_id,
        "model_id": model_id,
        "mlair_model_id": model_id,
    }

    if mode == "full":
        if args.step == "split":
            if not input_vid:
                raise ValueError("full mode split requires --dataset-version-id (MLAir input version)")
            ctx["dataset_version_id"] = input_vid
        elif args.step in ("detect", "prepare", "train", "eval", "gate"):
            pass
    else:
        vid = train_ready_vid or input_vid
        if args.step in ("prepare", "train", "eval", "gate"):
            if not vid:
                raise ValueError(
                    "train_ready mode requires --train-ready-version-id or MLAIR_TRAIN_READY_VERSION_ID"
                )
            ctx["dataset_version_id"] = vid
            ctx["train_dataset_version_id"] = vid

    return ctx


def _should_skip_step(step: str) -> bool:
    stop_after = (os.getenv("E2_STOP_AFTER") or "").strip().lower()
    if not stop_after:
        return False
    order = ("split", "detect", "prepare", "train", "eval", "gate")
    if stop_after not in order:
        return False
    return order.index(step) > order.index(stop_after)


def main() -> None:
    parser = argparse.ArgumentParser(description="Airflow baseline: one MLAir lifecycle step")
    parser.add_argument("--step", required=True, choices=sorted(STEP_HANDLERS.keys()))
    parser.add_argument("--mode", choices=("train_ready", "full"), default=None)
    parser.add_argument("--dataset-version-id", default=None, help="MLAir input version (full: split)")
    parser.add_argument("--train-ready-version-id", default=None, help="MLAir train-ready version")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--baseline-run-id", default=None)
    args = parser.parse_args()

    baseline_run_id = (os.getenv("BASELINE_RUN_ID") or args.baseline_run_id or "baseline").strip()
    if _should_skip_step(args.step):
        result = {"ok": True, "skipped": True, "step": args.step, "reason": "E2_STOP_AFTER"}
    else:
        handler = STEP_HANDLERS[args.step]
        ctx = _build_context(args)
        baseline_run_id = str(ctx.get("run_id") or baseline_run_id)
        result = handler(ctx)
    _log_step_to_mlflow(baseline_run_id, args.step, result)

    out = {"step": args.step, "baseline_run_id": baseline_run_id, **result}
    print(json.dumps(out, indent=2, default=str))
    if not result.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
