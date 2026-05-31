#!/usr/bin/env python3
"""Trigger one lifecycle train run and print task resource usage (QA)."""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

import requests

API = os.getenv("API_BASE_URL", "http://localhost:8080").rstrip("/")
TENANT = os.getenv("TENANT_ID", "default")
PROJECT = os.getenv("PROJECT_ID", "default_project")
TOKEN = os.getenv("TOKEN", os.getenv("ML_AIR_TRACKING_TOKEN", "admin-token"))
DATASET = os.getenv("DATASET_NAME", "cv-traffic-frames")
DATASET_VERSION = os.getenv("DATASET_VERSION", "v3")  # e.g. v3
PIPELINE = os.getenv("PIPELINE_ID", "cv-yolo-lifecycle-train")
POLL_SEC = float(os.getenv("POLL_SEC", "5"))
TIMEOUT_SEC = int(os.getenv("RUN_TIMEOUT_SEC", "600"))


def hdr() -> dict[str, str]:
    t = TOKEN.strip()
    auth = t if t.lower().startswith("bearer ") else f"Bearer {t}"
    return {"Authorization": auth, "Content-Type": "application/json", "Accept": "application/json"}


def get(path: str, **kw: Any) -> Any:
    r = requests.get(f"{API}{path}", headers=hdr(), timeout=kw.pop("timeout", 30), **kw)
    r.raise_for_status()
    return r.json() if r.content else None


def post(path: str, body: dict, **kw: Any) -> Any:
    r = requests.post(f"{API}{path}", headers=hdr(), json=body, timeout=kw.pop("timeout", 60), **kw)
    r.raise_for_status()
    return r.json() if r.content else None


def items(payload: Any, *keys: str) -> list[dict]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for k in keys or ("items", "data", "results"):
            v = payload.get(k)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
    return []


def find_dataset_version_id() -> tuple[str, str, str]:
    datasets = get(f"/v1/tenants/{TENANT}/projects/{PROJECT}/datasets")
    dataset_id = None
    for d in items(datasets, "items", "datasets"):
        if d.get("name") == DATASET:
            dataset_id = d.get("dataset_id") or d.get("id")
            break
    if not dataset_id:
        raise SystemExit(f"dataset not found: {DATASET}")

    versions = get(
        f"/v1/tenants/{TENANT}/projects/{PROJECT}/datasets/{dataset_id}/versions"
    )
    target = None
    for v in items(versions, "items", "versions"):
        ver = str(v.get("version") or "")
        if ver == str(DATASET_VERSION) or ver == f"v{DATASET_VERSION.lstrip('v')}":
            target = v
            break
    if not target:
        raise SystemExit(f"dataset {DATASET} version {DATASET_VERSION} not found")

    vid = (
        target.get("dataset_version_id")
        or target.get("version_id")
        or target.get("id")
    )
    if not vid:
        raise SystemExit("missing dataset_version_id on version row")
    return str(dataset_id), str(vid), str(DATASET_VERSION)


def pick_model_id() -> str:
    mid = os.getenv("MODEL_ID", "").strip()
    if mid:
        return mid
    models = get(f"/v1/tenants/{TENANT}/projects/{PROJECT}/models")
    for m in items(models, "items", "models"):
        x = m.get("model_id") or m.get("id")
        if x:
            return str(x)
    raise SystemExit("no model_id — set MODEL_ID env")


def fmt_usage(u: dict | None) -> str:
    if not u:
        return "—"
    parts = [
        f"cpu_peak={u.get('cpu_pct_peak')}",
        f"mem_peak={u.get('memory_mb_peak')}MB",
        f"gpu_peak={u.get('gpu_util_pct_peak')}%",
        f"gpu_mem_peak={u.get('gpu_memory_mb_peak')}MB",
        f"samples={u.get('sample_count')}",
    ]
    return " ".join(str(p) for p in parts)


def main() -> int:
    dataset_id, version_id, ver_label = find_dataset_version_id()
    model_id = pick_model_id()
    print(f"dataset={DATASET} v{ver_label} version_id={version_id} model_id={model_id}")

    body = {
        "pipeline_id": PIPELINE,
        "dataset_version_id": version_id,
        "context": {
            "dataset": DATASET,
            "dataset_id": dataset_id,
            "dataset_version_id": version_id,
            "model_id": model_id,
        },
    }
    run = post(f"/v1/tenants/{TENANT}/projects/{PROJECT}/runs", body)
    run_id = run.get("run_id") or run.get("id")
    if not run_id:
        print("FAIL: no run_id in response", run, file=sys.stderr)
        return 1
    print(f"run_id={run_id} status={run.get('status')}")

    deadline = time.time() + TIMEOUT_SEC
    terminal = {"SUCCESS", "SUCCEEDED", "FAILED", "FAILURE", "CANCELLED", "CANCELED"}
    last_status = None
    while time.time() < deadline:
        detail = get(f"/v1/tenants/{TENANT}/projects/{PROJECT}/runs/{run_id}")
        status = str(detail.get("status") or "").upper()
        if status != last_status:
            print(f"  run status={status}")
            last_status = status
        if status in terminal:
            break
        time.sleep(POLL_SEC)
    else:
        print("TIMEOUT waiting for run", file=sys.stderr)
        return 2

    tasks = get(f"/v1/tenants/{TENANT}/projects/{PROJECT}/runs/{run_id}/tasks")
    task_rows = items(tasks, "items", "tasks")
    print(f"\n=== Tasks ({len(task_rows)}) ===")
    ok = True
    for t in task_rows:
        tid = t.get("task_id") or t.get("id")
        plugin = t.get("plugin") or t.get("task_type") or "?"
        st = t.get("status")
        usage = None
        try:
            bundle = get(f"/v1/tenants/{TENANT}/projects/{PROJECT}/tasks/{tid}/usage")
            usage = bundle.get("usage") if isinstance(bundle, dict) else bundle
        except Exception as exc:
            usage = {"error": str(exc)}
        line = fmt_usage(usage if isinstance(usage, dict) else None)
        print(f"  {plugin:20} {st:10} {line}")
        if plugin == "cv_yolo_train" and isinstance(usage, dict):
            gpu = usage.get("gpu_util_pct_peak")
            mem = usage.get("memory_mb_peak")
            if gpu is None or (isinstance(gpu, (int, float)) and gpu <= 0):
                print("    WARN: train gpu_util_pct_peak missing/zero", file=sys.stderr)
                ok = False
            if mem is None:
                print("    WARN: train memory_mb_peak missing", file=sys.stderr)
                ok = False

    live = get(f"/v1/tenants/{TENANT}/projects/{PROJECT}/runs/{run_id}/usage")
    print(f"\n=== Run usage summary ===")
    print(json.dumps(live, indent=2, default=str)[:2000])

    print(f"\n{'PASS' if ok and status in ('SUCCESS', 'SUCCEEDED') else 'CHECK'} run={run_id}")
    return 0 if ok and status in ("SUCCESS", "SUCCEEDED") else 3


if __name__ == "__main__":
    raise SystemExit(main())
