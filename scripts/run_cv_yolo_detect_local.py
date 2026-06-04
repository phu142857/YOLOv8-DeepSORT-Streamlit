#!/usr/bin/env python3
"""Run cv_yolo_split + cv_yolo_detect locally (no MLAir worker lease)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlair_adapter.torch_compat import apply_torch_checkpoint_compat

apply_torch_checkpoint_compat()

from mlair_adapter.yolo_lifecycle import run_detect, run_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Local split + detect lifecycle steps")
    parser.add_argument("--dataset-version-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--run-id", default="local-detect")
    args = parser.parse_args()

    ctx = {
        "run_id": args.run_id,
        "dataset_version_id": args.dataset_version_id,
        "model_id": args.model_id,
    }
    print(
        f"MLAIR_API={os.getenv('CV_MLAIR_API_URL', '')} "
        f"CV_API={os.getenv('CV_API_BASE_URL', 'http://127.0.0.1:8000')}",
        flush=True,
    )
    split_out = run_split(ctx)
    print(json.dumps({"split": split_out}, indent=2, default=str))
    if not split_out.get("ok"):
        raise SystemExit(1)
    detect_out = run_detect(ctx)
    print(json.dumps({"detect": detect_out}, indent=2, default=str))
    if not detect_out.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
