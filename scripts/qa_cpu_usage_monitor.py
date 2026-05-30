#!/usr/bin/env python3
"""QA: compare ResourceMonitor CPU % vs independent process-tree measurement (3s windows)."""

from __future__ import annotations

import os
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MLAIR_RESOURCE_SAMPLE_INTERVAL_SEC", "3")

from mlair_adapter.resource_monitor import ResourceMonitor
from mlair_adapter.task_resource_monitor import TaskResourceMonitor, cpu_percent_from_delta
from mlair_adapter.usage_cost_math import cpu_delta_to_machine_percent, effective_logical_cpus

try:
    import psutil
except ImportError:
    print("FAIL: psutil required")
    raise SystemExit(1)


def _cpu_time_tree(root_pid: int) -> float:
    mon = TaskResourceMonitor()
    return mon._cpu_time_seconds_tree(root_pid)


def reference_tree_machine_percent(root_pid: int, window: float = 3.0) -> float:
    """Independent ground truth: cpu_times delta over ``window`` seconds."""
    c0 = _cpu_time_tree(root_pid)
    t0 = time.perf_counter()
    time.sleep(window)
    c1 = _cpu_time_tree(root_pid)
    t1 = time.perf_counter()
    raw = cpu_percent_from_delta(c1 - c0, t1 - t0)
    return cpu_delta_to_machine_percent(raw)


def _cpu_burn(stop: threading.Event, duty: float = 0.55) -> None:
    while not stop.is_set():
        deadline = time.perf_counter() + duty * 0.05
        while time.perf_counter() < deadline:
            _ = sum(i * i for i in range(8000))
        time.sleep(0.05 * (1.0 - duty))


def main() -> int:
    interval = float(os.getenv("MLAIR_RESOURCE_SAMPLE_INTERVAL_SEC", "3"))
    duration = float(os.getenv("QA_CPU_DURATION_SEC", "18"))
    cpus = effective_logical_cpus()
    print(f"=== CPU usage QA (sample_interval={interval}s, duration={duration}s, logical_cpus={cpus}) ===")

    stop = threading.Event()
    burn = threading.Thread(target=_cpu_burn, args=(stop, 0.55), name="cpu-burn", daemon=True)
    burn.start()

    monitor_samples: list[float] = []
    reference_samples: list[float] = []
    system_samples: list[float] = []

    with ResourceMonitor(task_id="qa-cpu", flush_interval_seconds=0, interval_seconds=interval) as monitor:
        root_pid = monitor._inner.root_pid or os.getpid()
        windows = max(1, int(duration / interval))
        print(f"{'window':>6}  {'monitor%':>10}  {'ref_tree%':>10}  {'system%':>10}  {'delta':>8}")
        print("-" * 52)

        for w in range(1, windows + 1):
            ref = reference_tree_machine_percent(root_pid, window=interval)
            reference_samples.append(ref)
            sys_pct = psutil.cpu_percent(interval=interval)
            system_samples.append(sys_pct)

            usage = monitor.latest_heartbeat_usage() or {}
            mon_pct = float(usage.get("cpu_percent") or 0.0)
            if mon_pct <= 0 and monitor._inner._samples:
                mon_pct = float(monitor._inner._samples[-1].get("cpu_percent") or 0.0)
            monitor_samples.append(mon_pct)
            delta = mon_pct - ref
            print(f"{w:>6}  {mon_pct:>10.2f}  {ref:>10.2f}  {sys_pct:>10.2f}  {delta:>+8.2f}")

    stop.set()
    burn.join(timeout=2)

    bundle = monitor.complete_bundle()
    summary = monitor.summary()
    peaks = [float(s.get("cpu_percent") or 0) for s in bundle.get("usage_samples") or []]

    def _avg(vals: list[float]) -> float:
        return statistics.mean(vals) if vals else 0.0

    mon_avg = _avg(monitor_samples[1:] if len(monitor_samples) > 1 else monitor_samples)
    ref_avg = _avg(reference_samples)
    sys_avg = _avg(system_samples)
    mon_peak = max(peaks) if peaks else summary.get("cpu_percent_peak", 0)

    print("-" * 52)
    print(f"monitor avg (windows): {mon_avg:.2f}%  peak: {mon_peak:.2f}%")
    print(f"reference tree avg:    {ref_avg:.2f}%")
    print(f"system-wide avg:       {sys_avg:.2f}%")
    print(f"contract summary peak: {summary.get('cpu_percent_peak')}")

    err = abs(mon_avg - ref_avg)
    ok = err <= 5.0 or (ref_avg > 0 and err / ref_avg <= 0.25)
    print(f"\n{'PASS' if ok else 'WARN'}: monitor vs reference avg delta={err:.2f} pp")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
