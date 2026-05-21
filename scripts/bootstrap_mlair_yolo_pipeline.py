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

from mlair_adapter.pipeline_bootstrap import ensure_cv_yolo_pipeline
from shared.settings import settings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--required-size", type=int, default=100, help="Training policy required_size")
    parser.add_argument("--pipeline-id", default=settings.mlair_train_pipeline_id)
    parser.add_argument("--train-url", default="http://cv-api:8000/api/v1/mlair/train/execute")
    parser.add_argument(
        "--mode",
        choices=("http", "plugin"),
        default=settings.mlair_pipeline_mode,
        help="plugin = Hub Train UI (default); http = executor HTTP only",
    )
    parser.add_argument("--force", action="store_true", help="Publish new version even if config matches")
    parser.add_argument("--map-models", action="store_true", help="Map all registry models to this pipeline")
    args = parser.parse_args()

    out = ensure_cv_yolo_pipeline(
        map_models=args.map_models,
        required_size=args.required_size,
        train_url=args.train_url,
        mode=args.mode,
        force_republish=args.force,
    )
    print(json.dumps(out, indent=2))
    if not out.get("ok"):
        print("Set CV_MLAIR_API_URL and CV_MLAIR_TOKEN", file=sys.stderr)
        return 1
    if out.get("skipped"):
        print("Pipeline already registered:", out.get("pipeline_id"))

    print("\nDone. Hub → Pipelines should list", settings.mlair_train_pipeline_id)
    print("CV auto-train: set CV_MLAIR_AUTO_TRAIN=1 and CV_MLAIR_MODEL_ID=<uuid>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
