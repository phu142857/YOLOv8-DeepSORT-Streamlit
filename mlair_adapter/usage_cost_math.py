"""Resource sample aggregation (aligned with ml-air sdk/usage_cost_math.py)."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]


def effective_logical_cpus() -> int:
    """Logical CPUs available to this process (respects Docker/cgroup affinity)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    if psutil is not None:
        return max(1, int(psutil.cpu_count(logical=True) or 1))
    return 1


def parse_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    raw = str(value).strip()
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def normalize_resource_usage(raw: Any) -> dict[str, float | int | None]:
    out: dict[str, float | int | None] = {
        "duration_ms": None,
        "cpu_time_seconds": None,
        "memory_rss_kb": None,
        "gpu_seconds": None,
        "gpu_memory_mb_seconds": None,
        "disk_read_bytes": None,
        "disk_write_bytes": None,
    }
    if not isinstance(raw, dict):
        return out
    if raw.get("duration_seconds") is not None and out["duration_ms"] is None:
        try:
            out["duration_ms"] = int(max(0.0, float(raw["duration_seconds"])) * 1000)
        except (TypeError, ValueError):
            pass
    for key in out:
        val = raw.get(key)
        if val is None:
            continue
        try:
            if key in ("memory_rss_kb", "duration_ms", "disk_read_bytes", "disk_write_bytes"):
                out[key] = int(val)
            else:
                out[key] = float(val)
        except (TypeError, ValueError):
            continue
    return out


def _avg_peak(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return sum(values) / len(values), max(values)


def cpu_delta_to_machine_percent(raw: float | None, *, logical_cpus: int | None = None) -> float:
    """Convert delta/cpu_times sampling (100% = one full core) to 0–100 machine utilization."""
    if raw is None:
        return 0.0
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if val <= 0:
        return 0.0
    cpus = logical_cpus if logical_cpus is not None else effective_logical_cpus()
    return min(100.0, round(val / max(1, int(cpus)), 2))


def normalize_cpu_tree_percent(raw: float | None, *, logical_cpus: int | None = None) -> float | None:
    """Ingest: legacy summed tree (>100) → machine scale; contract v1 samples pass through."""
    if raw is None:
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    if val <= 0:
        return 0.0
    if val <= 100:
        return min(100.0, val)
    cpus = logical_cpus if logical_cpus is not None else effective_logical_cpus()
    return min(100.0, max(0.0, val / max(1, int(cpus))))


def aggregate_samples(
    samples: list[tuple[Any, ...]],
    *,
    runtime_seconds: float,
    fallback_memory_mb: float | None,
) -> dict[str, float | int | None]:
    cpu_vals: list[float] = []
    mem_vals: list[float] = []
    gpu_util_vals: list[float] = []
    gpu_mem_vals: list[float] = []

    parsed_mem: list[tuple[datetime, float]] = []
    peak_mb = fallback_memory_mb or 0.0

    for row in samples:
        ts = row[0] if isinstance(row[0], datetime) else parse_ts(row[0])
        cpu = row[1] if len(row) > 1 else None
        mem = row[2] if len(row) > 2 else None
        gpu_util = row[3] if len(row) > 3 else None
        gpu_mem = row[4] if len(row) > 4 else None

        if cpu is not None:
            normalized = normalize_cpu_tree_percent(float(cpu))
            if normalized is not None:
                cpu_vals.append(normalized)
        if mem is not None:
            mem_f = float(mem)
            mem_vals.append(mem_f)
            peak_mb = max(peak_mb, mem_f)
            if ts is not None:
                parsed_mem.append((ts, mem_f))
        if gpu_util is not None:
            gpu_util_vals.append(float(gpu_util))
        if gpu_mem is not None:
            gpu_mem_vals.append(float(gpu_mem))

    cpu_avg, cpu_peak = _avg_peak(cpu_vals)
    mem_avg, mem_peak = _avg_peak(mem_vals)
    if mem_peak is None and peak_mb > 0:
        mem_peak = peak_mb
    if mem_avg is None and mem_peak is not None:
        mem_avg = mem_peak

    gpu_util_avg, gpu_util_peak = _avg_peak(gpu_util_vals)
    gpu_mem_avg, gpu_mem_peak = _avg_peak(gpu_mem_vals)

    mem_mb_seconds = 0.0
    parsed_mem.sort(key=lambda x: x[0])
    if len(parsed_mem) == 1:
        mem_mb_seconds = parsed_mem[0][1] * max(runtime_seconds, 0.0)
    elif len(parsed_mem) >= 2:
        for i in range(len(parsed_mem) - 1):
            t0, m0 = parsed_mem[i]
            t1, m1 = parsed_mem[i + 1]
            delta = max(0.0, (t1 - t0).total_seconds())
            mem_mb_seconds += ((m0 + m1) / 2.0) * delta
        _last_t, last_m = parsed_mem[-1]
        avg_gap = runtime_seconds / max(len(parsed_mem), 1)
        mem_mb_seconds += last_m * max(avg_gap * 0.5, 0.0)
    elif peak_mb > 0:
        mem_mb_seconds = peak_mb * max(runtime_seconds, 0.0)

    return {
        "memory_rss_peak_kb": int(peak_mb * 1024) if peak_mb > 0 else 0,
        "memory_mb_seconds": mem_mb_seconds,
        "sample_count": len(samples),
        "cpu_pct_avg": cpu_avg,
        "cpu_pct_peak": cpu_peak,
        "memory_mb_avg": mem_avg,
        "memory_mb_peak": mem_peak,
        "gpu_util_pct_avg": gpu_util_avg,
        "gpu_util_pct_peak": gpu_util_peak,
        "gpu_memory_mb_avg": gpu_mem_avg,
        "gpu_memory_mb_peak": gpu_mem_peak,
    }
