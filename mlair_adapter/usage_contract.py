"""MLAir Resource Usage Contract v1 — worker-agnostic peaks and totals."""

from __future__ import annotations

from typing import Any

from mlair_adapter.usage_cost_math import (
    aggregate_samples,
    normalize_cpu_tree_percent,
    normalize_resource_usage,
    parse_ts,
)

CONTRACT_VERSION = "v1"

CONTRACT_SUMMARY_KEYS = (
    "duration_seconds",
    "cpu_time_seconds",
    "cpu_percent_peak",
    "memory_mb_peak",
    "gpu_percent_peak",
    "gpu_memory_mb_peak",
    "disk_read_bytes",
    "disk_write_bytes",
)

CONTRACT_HEARTBEAT_KEYS = (
    "cpu_percent",
    "memory_mb",
    "gpu_util_percent",
    "gpu_memory_mb",
)


def sample_dicts_to_rows(samples: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        rows.append(
            (
                parse_ts(sample.get("sampled_at")),
                sample.get("cpu_percent"),
                sample.get("memory_mb"),
                sample.get("gpu_util_percent"),
                sample.get("gpu_memory_mb"),
            )
        )
    return rows


def contract_summary_from_report(report: dict[str, Any]) -> dict[str, Any]:
    ru = report.get("resource_usage") if isinstance(report.get("resource_usage"), dict) else {}
    samples = report.get("usage_samples") if isinstance(report.get("usage_samples"), list) else []

    duration_seconds: float | None = None
    if ru.get("duration_ms") is not None:
        duration_seconds = max(0.0, float(ru["duration_ms"]) / 1000.0)

    fallback_mb = (float(ru["memory_rss_kb"]) / 1024.0) if ru.get("memory_rss_kb") else None
    agg = aggregate_samples(
        sample_dicts_to_rows(samples),
        runtime_seconds=duration_seconds or 0.0,
        fallback_memory_mb=fallback_mb,
    )

    summary: dict[str, Any] = {
        "duration_seconds": duration_seconds,
        "cpu_time_seconds": ru.get("cpu_time_seconds"),
        "cpu_percent_peak": agg.get("cpu_pct_peak"),
        "memory_mb_peak": agg.get("memory_mb_peak"),
        "gpu_percent_peak": agg.get("gpu_util_pct_peak"),
        "gpu_memory_mb_peak": agg.get("gpu_memory_mb_peak"),
        "disk_read_bytes": ru.get("disk_read_bytes"),
        "disk_write_bytes": ru.get("disk_write_bytes"),
    }
    return {k: v for k, v in summary.items() if v is not None}


def contract_complete_resource_usage(report: dict[str, Any]) -> dict[str, Any]:
    ru = dict(report.get("resource_usage") or {}) if isinstance(report.get("resource_usage"), dict) else {}
    summary = contract_summary_from_report(report)
    out: dict[str, Any] = {**normalize_resource_usage(ru), **ru}
    for key in CONTRACT_SUMMARY_KEYS:
        val = summary.get(key)
        if val is not None:
            out[key] = val
    if out.get("duration_seconds") and not out.get("duration_ms"):
        out["duration_ms"] = int(float(out["duration_seconds"]) * 1000)
    if summary.get("memory_mb_peak") is not None and out.get("memory_rss_kb") is None:
        out["memory_rss_kb"] = int(float(summary["memory_mb_peak"]) * 1024)
    return {k: v for k, v in out.items() if v is not None}


def contract_heartbeat_from_sample(sample: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(sample, dict) or not sample:
        return None
    out: dict[str, Any] = {}
    for key in CONTRACT_HEARTBEAT_KEYS:
        val = sample.get(key)
        if val is None:
            continue
        if key == "cpu_percent":
            cpu = normalize_cpu_tree_percent(float(val)) if val is not None else None
            if cpu is not None:
                out[key] = cpu
        else:
            out[key] = val
    return out or None
