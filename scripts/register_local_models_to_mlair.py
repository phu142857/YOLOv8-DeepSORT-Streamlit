#!/usr/bin/env python3
"""Register local YOLO checkpoints in MLAir (delegates to ModelSyncService)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlair_adapter.model_sync import ModelSyncService
from shared.settings import settings
from shared.weights_catalog import scan_detection_models


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, default=settings.detection_model_dir)
    parser.add_argument("--only", action="append", default=[], help="e.g. yolov8n/base")
    args = parser.parse_args()

    svc = ModelSyncService()
    if not svc.enabled:
        print("Set CV_MLAIR_API_URL and CV_MLAIR_TOKEN", file=sys.stderr)
        return 1

    root = Path(args.weights_dir)
    if args.only:
        entries = scan_detection_models(root)
        want = set(args.only)
        entries = [e for e in entries if e.spec in want]
        results = [svc.sync_local_entry(e) for e in entries]
        print({"total": len(entries), "results": results})
    else:
        print(svc.sync_all_local_models(root=root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
