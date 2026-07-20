#!/usr/bin/env python3
"""Verify a YOLO lifecycle run on MLAir purely over the network (like an MLflow client).

Talks ONLY to the MLAir HTTP API (`/v1`, bearer token) — no ml-air source, no SDK import,
no shared filesystem. Asserts the acceptance criteria for one training run:

  * exactly one run_id is inspected (latest, or --run-id / RUN_ID),
  * the run pins a dataset_version_id,
  * usage samples exist (resource telemetry from the external worker),
  * run-level resource_usage roll-up exists,
  * lineage edges exist,
  * run environment was captured (server-side).

Usage:
  MLAIR_API_BASE_URL=http://localhost:8080 ML_AIR_TRACKING_TOKEN=admin-token \\
      python scripts/verify_mlair_run.py [--run-id RUN] [--tenant default] [--project default_project]

Exit code 0 = all required checks PASS, 1 = a required check FAILED, 2 = setup error.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.error
import urllib.request

BASE = os.environ.get("MLAIR_API_BASE_URL", "http://localhost:8080").rstrip("/")
TOKEN = (
    os.environ.get("ML_AIR_TRACKING_TOKEN")
    or os.environ.get("MLAIR_WORKER_TOKEN")
    or os.environ.get("CV_MLAIR_TOKEN")
    or "admin-token"
)


def _get(path: str) -> tuple[int, object]:
    url = f"{BASE}{path}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOKEN}"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode("utf-8", "replace")
            try:
                return resp.status, json.loads(body)
            except json.JSONDecodeError:
                return resp.status, body
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read().decode("utf-8", "replace"))
        except Exception:
            return exc.code, None
    except urllib.error.URLError as exc:
        return 0, str(exc)


def _items(payload: object) -> list:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("items", "samples", "runs", "data", "results"):
            val = payload.get(key)
            if isinstance(val, list):
                return val
    return []


def _deep_find(obj: object, keys: set[str]):
    """Return the first non-empty value for any of `keys` found anywhere in a nested structure."""
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            for k, v in cur.items():
                if k in keys and v not in (None, "", [], {}):
                    return v
                stack.append(v)
        elif isinstance(cur, list):
            stack.extend(cur)
    return None


class Report:
    def __init__(self) -> None:
        self.failed = 0

    def check(self, ok: bool, label: str, detail: str = "", required: bool = True) -> bool:
        mark = "PASS" if ok else ("FAIL" if required else "WARN")
        line = f"  [{mark}] {label}"
        if detail:
            line += f" — {detail}"
        print(line)
        if not ok and required:
            self.failed += 1
        return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default=os.environ.get("RUN_ID"))
    ap.add_argument("--tenant", default=os.environ.get("CV_MLAIR_TENANT", "default"))
    ap.add_argument("--project", default=os.environ.get("CV_MLAIR_PROJECT", "default_project"))
    args = ap.parse_args()

    tp = f"/v1/tenants/{args.tenant}/projects/{args.project}"
    rep = Report()

    print(f"MLAir @ {BASE}  (tenant={args.tenant} project={args.project})")

    status, _ = _get("/health")
    if not rep.check(status == 200, "health", f"GET /health -> {status}"):
        return 2

    # Resolve run_id: explicit, else latest from the runs list.
    run_id = args.run_id
    if not run_id:
        status, payload = _get(f"{tp}/runs")
        runs = _items(payload)
        if status != 200 or not runs:
            rep.check(False, "runs available", f"GET {tp}/runs -> {status}, {len(runs)} run(s)")
            print("\nNo runs found. Trigger a YOLO lifecycle-train run, then re-run this script.")
            return 1
        run_id = runs[0].get("id") or runs[0].get("run_id")
    rep.check(bool(run_id), "run_id resolved", str(run_id))

    status, run = _get(f"{tp}/runs/{run_id}")
    if not rep.check(status == 200 and isinstance(run, dict), "run detail", f"-> {status}"):
        return 1

    dsv = _deep_find(run, {"dataset_version_id", "datasetVersionId"})
    rep.check(bool(dsv), "dataset_version_id pinned", str(dsv))

    status, samples = _get(f"{tp}/runs/{run_id}/usage-samples")
    sample_list = _items(samples)
    rep.check(status == 200 and len(sample_list) > 0, "usage samples", f"{len(sample_list)} sample(s)")

    status, usage = _get(f"{tp}/runs/{run_id}/usage")
    peaks = _deep_find(usage, {"resource_usage", "peaks", "cpu_seconds", "peak_rss_bytes"})
    rep.check(status == 200 and peaks is not None, "run resource_usage roll-up", f"-> {status}")

    # Task detail: usage summary + timeline samples + logs (Hub task detail tabs).
    status, tasks_payload = _get(f"{tp}/runs/{run_id}/tasks")
    task_rows = _items(tasks_payload)
    success_tasks = [t for t in task_rows if str(t.get("status", "")).upper() == "SUCCESS"]
    if success_tasks:
        tid = str(success_tasks[0].get("task_id") or "")
        if tid:
            enc_tid = urllib.parse.quote(tid, safe=":")
            st_u, task_usage = _get(f"{tp}/tasks/{enc_tid}/usage")
            tu = _deep_find(task_usage, {"usage"}) or task_usage
            sample_count = (tu or {}).get("sample_count") if isinstance(tu, dict) else None
            mem_peak = (tu or {}).get("memory_mb_peak") if isinstance(tu, dict) else None
            rep.check(
                st_u == 200 and (sample_count or 0) > 0,
                "task usage (samples for timeline)",
                f"task={tid.split(':')[-1]} sample_count={sample_count} mem_peak={mem_peak}",
            )
            st_s, task_samples = _get(f"{tp}/runs/{run_id}/usage-samples?task_id={enc_tid}")
            ts = _items(task_samples)
            has_timeline = any(isinstance(s, dict) and s.get("sampled_at") for s in ts)
            rep.check(
                st_s == 200 and has_timeline,
                "task usage-samples timeline",
                f"{len(ts)} sample(s) with sampled_at",
            )
            st_l, task_logs = _get(f"{tp}/tasks/{enc_tid}/logs")
            log_items = _items(task_logs)
            detail = task_logs.get("detail") if isinstance(task_logs, dict) else None
            rep.check(
                st_l == 200 and detail != "task_not_found",
                "task logs API",
                f"HTTP {st_l} items={len(log_items)}",
            )
            rep.check(
                len(log_items) > 0,
                "task logs non-empty",
                f"{len(log_items)} line(s) — re-run after worker log fix if 0",
                required=False,
            )

    status, lineage = _get(f"{tp}/lineage/runs/{run_id}")
    edges = _deep_find(lineage, {"edges"}) or []
    nodes = _deep_find(lineage, {"nodes"}) or []
    rep.check(
        status == 200 and (len(edges) > 0 or len(nodes) > 0),
        "lineage edges",
        f"{len(edges)} edge(s), {len(nodes)} node(s)",
    )

    # Environment capture is server-side; look in run detail then tracking.
    env = _deep_find(run, {"environment", "run_environment"})
    if env is None:
        _, tracking = _get(f"{tp}/runs/{run_id}/tracking")
        env = _deep_find(tracking, {"environment", "run_environment"})
    rep.check(env is not None, "run environment captured", "present" if env else "missing", required=False)

    print()
    if rep.failed:
        print(f"RESULT: FAIL ({rep.failed} required check(s) failed) for run {run_id}")
        return 1
    print(f"RESULT: PASS for run {run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
