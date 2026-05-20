#!/usr/bin/env python3
"""
Register MLAir pipeline + training policy for real CV YOLO training.

Follows ml-air docs:
  - configure-data-readiness-gating.md (training policy, inputs)
  - model-centric-pipeline-mapping-and-trigger.md (pipeline-mapping)
  - http-pipeline-tasks.md (HTTP task to cv-api)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlair_adapter.base_client import MLAirClient
from mlair_adapter.model_client import ModelClient
from mlair_adapter.pipeline_config import load_cv_yolo_pipeline_config
from shared.settings import settings


def _prefix(client: MLAirClient) -> str:
    return client._prefix()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--required-size", type=int, default=100, help="Training policy required_size")
    parser.add_argument("--pipeline-id", default=settings.mlair_train_pipeline_id)
    parser.add_argument("--train-url", default="http://cv-api:8000/api/v1/mlair/train/execute")
    parser.add_argument(
        "--mode",
        choices=("http", "plugin"),
        default="http",
        help="http = executor HTTP task (default); plugin = external worker + cv_yolo_train",
    )
    parser.add_argument("--map-models", action="store_true", help="Map all registry models to this pipeline")
    args = parser.parse_args()

    client = MLAirClient()
    if not client.enabled:
        print("Set CV_MLAIR_API_URL and CV_MLAIR_TOKEN", file=sys.stderr)
        return 1

    pfx = _prefix(client)
    pipeline_id = args.pipeline_id
    config = load_cv_yolo_pipeline_config(mode=args.mode, cv_train_url=args.train_url)

    # 1) Validate pipeline contract (shift-left per run-pipeline.md)
    val = client.post("/v1/pipelines/validate", json={"config": config})
    print("validate:", json.dumps(val, indent=2)[:500])

    # 2) Create pipeline version
    ver = client.post(
        f"{pfx}/pipelines/{pipeline_id}/versions",
        json={"config": config},
    )
    print("pipeline_version:", ver.get("version_id"), "pipeline:", pipeline_id)

    # 3) Ensure dataset exists + training policy
    datasets = client.get(f"{pfx}/datasets", params={"limit": 50})
    items = datasets.get("items") if isinstance(datasets, dict) else datasets
    ds_row = None
    for row in items or []:
        if row.get("name") == settings.mlair_dataset_name:
            ds_row = row
            break
    if not ds_row:
        print(f"Dataset {settings.mlair_dataset_name!r} not found — run CV Execution with ingest first.", file=sys.stderr)
        return 1
    dataset_id = str(ds_row["dataset_id"])

    policy_body = {
        "trigger_mode": "manual",
        "required_size": args.required_size,
        "freshness_hours": 168,
        "validation_rules": [],
    }
    try:
        policy = client.post(f"{pfx}/datasets/{dataset_id}/training-policies", json=policy_body)
    except Exception:
        policies = client.get(f"{pfx}/datasets/{dataset_id}/training-policies")
        items = policies.get("items") if isinstance(policies, dict) else []
        if items:
            pid = items[0].get("policy_id")
            policy = client.put(
                f"{pfx}/datasets/{dataset_id}/training-policies",
                json={**policy_body, "policy_id": pid},
            )
        else:
            raise
    print("training_policy:", policy.get("policy_id") or policy)

    # 4) Map models → pipeline
    if args.map_models:
        mc = ModelClient()
        for m in mc.list_models():
            mid = m.get("model_id")
            if not mid:
                continue
            mapped = client.put(
                f"{pfx}/models/{mid}/pipeline-mapping",
                json={"pipeline_id": pipeline_id},
            )
            print(f"  mapped {m.get('name')} -> {pipeline_id}", mapped.get("pipeline_id", "ok"))

    print("\nDone. Hub: Train with model on dataset", settings.mlair_dataset_name)
    print("CV auto-train: set CV_MLAIR_AUTO_TRAIN=1 and CV_MLAIR_MODEL_ID=<uuid>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
