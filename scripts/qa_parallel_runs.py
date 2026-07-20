#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests


def _now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _env(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name, default)
    if v is None or v == "":
        raise SystemExit(f"Missing env var: {name}")
    return v


def _auth_header(token: str) -> Dict[str, str]:
    token = token.strip()
    if token.lower().startswith("bearer "):
        return {"Authorization": token}
    return {"Authorization": f"Bearer {token}"}


def _urljoin(base: str, path: str) -> str:
    return base.rstrip("/") + "/" + path.lstrip("/")


def _get_json(session: requests.Session, url: str, *, timeout: int = 10) -> Any:
    r = session.get(url, timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"GET {url} -> {r.status_code}: {r.text}")
    if not r.content:
        return None
    return r.json()


def _post_json(session: requests.Session, url: str, body: Dict[str, Any], *, timeout: int = 20) -> Any:
    r = session.post(url, json=body, timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"POST {url} -> {r.status_code}: {r.text}")
    if not r.content:
        return None
    return r.json()


def _run_cmd(cmd: List[str], *, cwd: Optional[str] = None, timeout: int = 8) -> Tuple[int, str]:
    try:
        p = subprocess.run(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
        return p.returncode, p.stdout.strip()
    except FileNotFoundError:
        return 127, f"not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def _guess_executor_replicas(repo_root: str) -> Dict[str, Any]:
    """
    Best-effort:
    - docker compose: count running containers whose service name is 'executor'
    - otherwise: report unknown
    """
    info: Dict[str, Any] = {"method": "unknown", "replicas": None, "raw": None}

    # Prefer `docker compose ps --format json` when available (compose v2).
    code, out = _run_cmd(["docker", "compose", "ps", "--format", "json"], cwd=repo_root)
    if code == 0:
        try:
            rows = json.loads(out) if out else []
            # rows fields vary; common: Service, State, Name
            replicas = 0
            for row in rows:
                svc = (row.get("Service") or row.get("service") or "").strip()
                state = (row.get("State") or row.get("state") or "").strip().lower()
                if svc == "executor" and (state.startswith("running") or state == "running"):
                    replicas += 1
            info.update({"method": "docker compose ps --format json", "replicas": replicas, "raw": rows})
            return info
        except Exception:
            pass

    # Fallback: parse plain `docker compose ps`.
    code, out = _run_cmd(["docker", "compose", "ps"], cwd=repo_root)
    if code == 0 and out:
        replicas = 0
        for line in out.splitlines():
            if "executor" in line and ("Up" in line or "running" in line.lower()):
                replicas += 1
        info.update({"method": "docker compose ps (parsed)", "replicas": replicas, "raw": out})
        return info

    info.update({"method": "docker compose ps (unavailable)", "raw": out})
    return info


@dataclass(frozen=True)
class RunTick:
    t_sec: int
    statuses: Dict[str, str]  # run_id -> status

    @property
    def running_count(self) -> int:
        return sum(1 for s in self.statuses.values() if s == "RUNNING")


def _pick_pipeline_ids(pipelines_payload: Any) -> Tuple[str, str]:
    if not pipelines_payload:
        raise ValueError("no pipelines found")

    items: List[Dict[str, Any]]
    if isinstance(pipelines_payload, list):
        items = pipelines_payload
    elif isinstance(pipelines_payload, dict):
        for k in ("items", "pipelines", "data", "results"):
            if isinstance(pipelines_payload.get(k), list):
                items = pipelines_payload[k]
                break
        else:
            raise ValueError("unexpected pipelines payload shape")
    else:
        raise ValueError("unexpected pipelines payload type")

    ids: List[str] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        pid = it.get("pipeline_id") or it.get("id")
        if isinstance(pid, str) and pid and pid not in ids:
            ids.append(pid)

    if len(ids) < 2:
        raise ValueError("no pipelines found")
    return ids[0], ids[1]


def _find_dataset(datasets_payload: Any, dataset_name: str) -> Tuple[Optional[str], bool]:
    """
    Return (dataset_id, exists_by_name).
    dataset_id may be None if API doesn't expose it in list response.
    """
    if not datasets_payload:
        return None, False

    items: List[Dict[str, Any]]
    if isinstance(datasets_payload, list):
        items = datasets_payload
    elif isinstance(datasets_payload, dict):
        for k in ("items", "datasets", "data", "results"):
            if isinstance(datasets_payload.get(k), list):
                items = datasets_payload[k]
                break
        else:
            return None, False
    else:
        return None, False

    for it in items:
        if not isinstance(it, dict):
            continue
        if it.get("name") == dataset_name:
            did = it.get("dataset_id") or it.get("id")
            if isinstance(did, str) and did:
                return did, True
            return None, True
    return None, False


def _pick_dataset_version_id(versions_payload: Any) -> Optional[str]:
    if not versions_payload:
        return None

    items: List[Dict[str, Any]]
    if isinstance(versions_payload, list):
        items = versions_payload
    elif isinstance(versions_payload, dict):
        for k in ("items", "versions", "data", "results"):
            if isinstance(versions_payload.get(k), list):
                items = versions_payload[k]
                break
        else:
            return None
    else:
        return None

    def _score(it: Dict[str, Any]) -> Tuple[int, str]:
        # Prefer highest numeric version, else newest created_at/updated_at lexicographically (ISO timestamps sort).
        v_raw = it.get("version")
        v_num = -1
        try:
            if v_raw is not None and str(v_raw).strip() != "":
                v_num = int(v_raw)
        except Exception:
            v_num = -1
        ts = str(it.get("created_at") or it.get("updated_at") or it.get("createdAt") or it.get("updatedAt") or "")
        return v_num, ts

    best: Optional[Dict[str, Any]] = None
    for it in items:
        if not isinstance(it, dict):
            continue
        if best is None or _score(it) > _score(best):
            best = it

    if not best:
        return None

    vid = best.get("dataset_version_id") or best.get("version_id") or best.get("id")
    return vid if isinstance(vid, str) and vid else None


def _pick_model_id(models_payload: Any) -> Optional[str]:
    if not models_payload:
        return None

    items: List[Dict[str, Any]]
    if isinstance(models_payload, list):
        items = models_payload
    elif isinstance(models_payload, dict):
        for k in ("items", "models", "data", "results"):
            if isinstance(models_payload.get(k), list):
                items = models_payload[k]
                break
        else:
            return None
    else:
        return None

    for it in items:
        if not isinstance(it, dict):
            continue
        mid = it.get("model_id") or it.get("id")
        if isinstance(mid, str) and mid:
            return mid
    return None


def main() -> int:
    api_base = _env("API_BASE_URL", "http://localhost:8080")
    tenant_id = _env("TENANT_ID", "default")
    project_id = _env("PROJECT_ID", "default_project")
    token = _env("TOKEN")
    dataset_name = os.getenv("DATASET_NAME", "cv-traffic-frames")
    check_task_concurrency = os.getenv("CHECK_TASK_CONCURRENCY", "0").strip() in ("1", "true", "yes", "on")
    # Optional: provide model ids to keep tasks running longer (avoid early failure).
    # - MODEL_ID: single model id used for all runs (legacy)
    # - MODEL_ID_A / MODEL_ID_B: allow testing two different models across runs
    model_id_env = os.getenv("MODEL_ID", "").strip() or None
    model_id_a_env = os.getenv("MODEL_ID_A", "").strip() or None
    model_id_b_env = os.getenv("MODEL_ID_B", "").strip() or None

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/json",
            "Content-Type": "application/json",
            **_auth_header(token),
        }
    )

    # 1) Health
    health_url = _urljoin(api_base, "/health")
    try:
        r = session.get(health_url, timeout=5)
        if r.status_code != 200:
            print(f"FAIL: healthcheck {health_url} -> {r.status_code}", file=sys.stderr)
            return 2
    except Exception as e:
        print(f"FAIL: healthcheck {health_url} error: {e}", file=sys.stderr)
        return 2

    # 2) Pipelines
    pipelines_url = _urljoin(
        api_base, f"/v1/tenants/{tenant_id}/projects/{project_id}/pipelines"
    )
    try:
        pipelines_payload = _get_json(session, pipelines_url, timeout=15)
        pipeline_a, pipeline_b = _pick_pipeline_ids(pipelines_payload)
    except ValueError as e:
        print(str(e))
        return 3
    except Exception as e:
        print(f"FAIL: list pipelines error: {e}", file=sys.stderr)
        return 3

    # 3) Datasets
    datasets_url = _urljoin(
        api_base, f"/v1/tenants/{tenant_id}/projects/{project_id}/datasets"
    )
    try:
        datasets_payload = _get_json(session, datasets_url, timeout=30)
        dataset_id, dataset_exists = _find_dataset(datasets_payload, dataset_name)
        if not dataset_exists:
            print("dataset not found")
            return 4
    except Exception as e:
        print(f"FAIL: list datasets error: {e}", file=sys.stderr)
        return 4

    # dataset_version_id required in this stack (ML_AIR_STRICT_DATASET_VERSION_ALL_POST_RUNS=1)
    dataset_version_id: Optional[str] = None
    if dataset_id:
        try:
            versions_url = _urljoin(
                api_base,
                f"/v1/tenants/{tenant_id}/projects/{project_id}/datasets/{dataset_id}/versions",
            )
            versions_payload = _get_json(session, versions_url, timeout=30)
            dataset_version_id = _pick_dataset_version_id(versions_payload)
        except Exception as e:
            print(f"FAIL: list dataset versions error: {e}", file=sys.stderr)
            return 4
    if not dataset_version_id:
        print("FAIL: dataset_version_id required but no dataset versions found. Materialize a version for this dataset first.")
        return 4

    # Best-effort: pick a model_id (many pipelines require it; without it tasks may fail quickly)
    model_id: Optional[str] = model_id_env
    if not model_id and not (model_id_a_env and model_id_b_env):
        try:
            models_url = _urljoin(
                api_base, f"/v1/tenants/{tenant_id}/projects/{project_id}/models"
            )
            models_payload = _get_json(session, models_url, timeout=20)
            model_id = _pick_model_id(models_payload)
        except Exception:
            model_id = None

    # 4) Concurrency config (best-effort notes)
    exec_info = _guess_executor_replicas(repo_root)
    quotas_url = _urljoin(api_base, f"/v1/tenants/{tenant_id}/quotas")
    quotas_payload = None
    try:
        quotas_payload = _get_json(session, quotas_url, timeout=15)
    except Exception:
        quotas_payload = None

    # 5) Trigger 3 runs near-simultaneously
    runs_url = _urljoin(api_base, f"/v1/tenants/{tenant_id}/projects/{project_id}/runs")
    ts = _now_ts()

    def build_context(mid: Optional[str]) -> Dict[str, Any]:
        ctx: Dict[str, Any] = {"dataset": dataset_name, "dataset_version_id": dataset_version_id}
        if dataset_id:
            ctx["dataset_id"] = dataset_id
        if mid:
            ctx["model_id"] = mid
        return ctx

    model_for_a1 = model_id_a_env or model_id
    model_for_a2 = model_id_b_env or model_id
    model_for_b1 = model_id_a_env or model_id

    bodies = [
        {
            "pipeline_id": pipeline_a,
            "idempotency_key": f"parallel-A-1-{ts}",
            "dataset_version_id": dataset_version_id,
            "context": build_context(model_for_a1),
        },
        {
            "pipeline_id": pipeline_a,
            "idempotency_key": f"parallel-A-2-{ts}",
            "dataset_version_id": dataset_version_id,
            "context": build_context(model_for_a2),
        },
        {
            "pipeline_id": pipeline_b,
            "idempotency_key": f"parallel-B-1-{ts}",
            "dataset_version_id": dataset_version_id,
            "context": build_context(model_for_b1),
        },
    ]

    run_ids: List[str] = []
    trigger_errors: List[str] = []
    for body in bodies:
        try:
            payload = _post_json(session, runs_url, body, timeout=30)
            rid = None
            if isinstance(payload, dict):
                rid = payload.get("run_id") or payload.get("id")
            if not isinstance(rid, str) or not rid:
                raise RuntimeError(f"unexpected create run response: {payload}")
            run_ids.append(rid)
        except Exception as e:
            trigger_errors.append(f"{body.get('idempotency_key')}: {e}")

    if trigger_errors:
        print("FAIL: trigger runs error(s):", file=sys.stderr)
        for err in trigger_errors:
            print(f"- {err}", file=sys.stderr)
        return 5

    # 6) Poll 60s, 1s interval
    def get_status(run_id: str) -> str:
        url = _urljoin(
            api_base, f"/v1/tenants/{tenant_id}/projects/{project_id}/runs/{run_id}"
        )
        payload = _get_json(session, url, timeout=10)
        if isinstance(payload, dict):
            s = payload.get("status") or payload.get("state")
            if isinstance(s, str):
                return s
        return "UNKNOWN"

    ticks: List[RunTick] = []
    pass_parallel = False
    pass_task_parallel = False
    task_timeline: List[str] = []

    def get_tasks(rid: str) -> list[dict[str, Any]]:
        url = _urljoin(
            api_base,
            f"/v1/tenants/{tenant_id}/projects/{project_id}/runs/{rid}/tasks",
        )
        payload = _get_json(session, url, timeout=15)
        if isinstance(payload, dict) and isinstance(payload.get("items"), list):
            return [it for it in payload["items"] if isinstance(it, dict)]
        if isinstance(payload, list):
            return [it for it in payload if isinstance(it, dict)]
        return []

    start = time.monotonic()
    for t_sec in range(0, 61):
        statuses: Dict[str, str] = {}
        for rid in run_ids:
            try:
                statuses[rid] = get_status(rid)
            except Exception:
                statuses[rid] = "ERROR"
        tick = RunTick(t_sec=t_sec, statuses=statuses)
        ticks.append(tick)
        if tick.running_count >= 2:
            pass_parallel = True

        if check_task_concurrency:
            running_tasks_total = 0
            per_run_running: list[str] = []
            for rid in run_ids:
                try:
                    items = get_tasks(rid)
                except Exception:
                    items = []
                running = [it for it in items if it.get("status") == "RUNNING"]
                running_tasks_total += len(running)
                if running:
                    per_run_running.append(
                        f"{rid[:8]}:{','.join(sorted({str(it.get('plugin') or '').strip() or it.get('task_id','').split(':')[-1] for it in running}))}"
                    )
            if running_tasks_total >= 2:
                pass_task_parallel = True
            # sample every second; cap output to keep it short
            if t_sec % 2 == 0 or running_tasks_total >= 2:
                task_timeline.append(
                    f"t+{t_sec:02d}s  task_running_total={running_tasks_total}  " + "  ".join(per_run_running)
                )

        # sleep to keep ~1s cadence
        elapsed = time.monotonic() - start
        next_target = t_sec + 1
        sleep_for = next_target - elapsed
        if sleep_for > 0 and t_sec < 60:
            time.sleep(min(1.0, sleep_for))

    # Build short timeline 10–20 lines: keep lines where statuses change, plus any tick with >=2 RUNNING
    def timeline_lines() -> List[str]:
        lines: List[str] = []
        prev: Optional[Dict[str, str]] = None
        for tick in ticks:
            changed = prev is None or tick.statuses != prev
            highlight = tick.running_count >= 2
            # Also include every 5s to avoid too few lines when nothing changes.
            periodic = tick.t_sec % 5 == 0
            if changed or highlight or periodic:
                parts = [f"{rid[:8]}={tick.statuses[rid]}" for rid in run_ids]
                lines.append(
                    f"t+{tick.t_sec:02d}s  running={tick.running_count}  " + "  ".join(parts)
                )
                prev = dict(tick.statuses)
        # Cap to ~20 lines by trimming middle if needed.
        if len(lines) > 20:
            return lines[:10] + ["..."] + lines[-9:]
        return lines

    print("Selected pipelines:")
    print(f"- pipelineA: {pipeline_a}")
    print(f"- pipelineB: {pipeline_b}")
    print("")
    print("Triggered runs:")
    for i, rid in enumerate(run_ids, start=1):
        print(f"- run_id{i}: {rid}")
    print("")
    print("Timeline (sampled):")
    for line in timeline_lines():
        print(line)
    print("")

    if check_task_concurrency:
        print("Task timeline (sampled):")
        for line in task_timeline[:20]:
            print(line)
        if len(task_timeline) > 20:
            print("...")
        print("")

    if pass_parallel:
        if check_task_concurrency and not pass_task_parallel:
            print("RESULT: PASS (run-level concurrency observed), but FAIL task-level (never observed >=2 tasks RUNNING)")
            print("Hint: scale worker replicas / enable autoscaling (HPA/KEDA) and ensure worker_id is unique per replica.")
            return 11
        print("RESULT: PASS (>=2 runs RUNNING concurrently observed)")
        return 0

    # 7) Diagnostics on FAIL
    print("RESULT: FAIL (never observed >=2 runs RUNNING concurrently in first 60s)")
    print("")
    print("Diagnostics:")
    print("- executor replicas/process (best-effort):")
    print(json.dumps(exec_info, indent=2, ensure_ascii=False))
    print("")
    print("- quotas (GET /v1/tenants/{TENANT_ID}/quotas) response (best-effort):")
    if quotas_payload is None:
        print("null")
    else:
        print(json.dumps(quotas_payload, indent=2, ensure_ascii=False))
    print("")

    for rid in run_ids:
        tasks_url = _urljoin(
            api_base,
            f"/v1/tenants/{tenant_id}/projects/{project_id}/runs/{rid}/tasks",
        )
        print(f"- run {rid} tasks:")
        try:
            tasks_payload = _get_json(session, tasks_url, timeout=30)
            print(json.dumps(tasks_payload, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"  error fetching tasks: {e}")
        print("")

    # Simple bottleneck conclusion heuristic.
    concl: List[str] = []
    replicas = exec_info.get("replicas")
    if isinstance(replicas, int) and replicas <= 1:
        concl.append("(a) executor=1 (likely serial execution)")
    if quotas_payload is not None:
        qtxt = json.dumps(quotas_payload, ensure_ascii=False).lower()
        if "concurrent" in qtxt and ("1" in qtxt or "max" in qtxt):
            concl.append("(b) quota may limit concurrent RUNNING runs")
    if not concl:
        concl.append("(d) other: check scheduler/executor logs and run state transitions")
    print("Conclusion (heuristic):")
    for c in concl:
        print(f"- {c}")

    return 10


if __name__ == "__main__":
    raise SystemExit(main())
