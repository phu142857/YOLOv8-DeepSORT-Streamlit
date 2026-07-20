"""Extract MLAir task metrics + artifacts from CV plugin results.

Resource usage for external tasks is sampled in-worker by the official SDK
(``sdk.resource_monitor``) and sent via
``sdk.worker_client.post_task_complete_from_bundle`` / ``post_task_fail`` — MLAir
cannot measure a separate worker container itself. This module only maps plugin
results to Contract metrics/artifacts.
"""

from __future__ import annotations

from typing import Any

# Per-sample fields for Hub usage timeline chart.
_V1_SAMPLE_KEYS = (
    "sampled_at",
    "cpu_percent",
    "memory_mb",
    "gpu_util_percent",
    "gpu_memory_mb",
    "network_rx_bytes",
    "network_tx_bytes",
    "gpu_power_w",
    "gpu_temp_c",
    "device_id",
)


def normalize_usage_bundle_for_complete(bundle: dict[str, Any]) -> dict[str, Any]:
    """Contract v1 complete payload — peaks + timeline samples without duplicate mirrors.

    MLAir's ``contract_complete_resource_usage`` merges legacy ingest fields with v1
    peaks (``duration_ms`` + ``duration_seconds``, ``memory_rss_kb`` + ``memory_mb_peak``).
    Hub task detail may render both; drop redundant legacy mirrors when the v1 peak
    is present. Samples are trimmed to contract keys so timeline series stay consistent.
    """
    from sdk.usage_contract import contract_complete_resource_usage

    raw_ru = bundle.get("resource_usage") if isinstance(bundle.get("resource_usage"), dict) else {}
    samples = bundle.get("usage_samples") if isinstance(bundle.get("usage_samples"), list) else []
    report = {"resource_usage": raw_ru, "usage_samples": samples}
    ru = contract_complete_resource_usage(report)

    # Drop legacy mirrors when v1 peak exists (avoids duplicate cards in Hub).
    if ru.get("duration_seconds") is not None:
        ru.pop("duration_ms", None)
    if ru.get("memory_mb_peak") is not None:
        ru.pop("memory_rss_kb", None)

    clean_samples: list[dict[str, Any]] = []
    for s in samples:
        if not isinstance(s, dict):
            continue
        row = {k: s[k] for k in _V1_SAMPLE_KEYS if s.get(k) is not None}
        if row:
            clean_samples.append(row)

    return {"resource_usage": ru, "usage_samples": clean_samples}


def metrics_from_plugin_result(result: dict[str, Any]) -> dict[str, Any]:
    """Raw metric keys — MLAir adds ``{plugin}.`` prefix in ``_persist_run_plugin_tracking``."""
    metrics: dict[str, Any] = dict(result.get("metrics") or {})
    for key in ("mAP50", "production_map50", "train_images", "val_images", "imported_version"):
        if result.get(key) is not None:
            metrics[key] = result[key]
    gate = result.get("gate")
    if isinstance(gate, dict):
        for k in ("candidate_map50", "production_map50", "passed", "promoted"):
            if gate.get(k) is not None:
                metrics[k] = gate[k]
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            out[key] = 1.0 if value else 0.0
        elif isinstance(value, (int, float)):
            out[key] = float(value)
    return out


def artifacts_from_plugin_result(result: dict[str, Any], *, plugin: str) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    step = str(result.get("step") or plugin)

    checkpoint = str(result.get("checkpoint") or "").strip()
    if checkpoint:
        uri = checkpoint if checkpoint.startswith("file://") else f"file://{checkpoint}"
        items.append({"path": f"{step}/checkpoint", "uri": uri})

    data_yaml = str(result.get("data_yaml") or "").strip()
    if data_yaml:
        uri = data_yaml if data_yaml.startswith("file://") else f"file://{data_yaml}"
        items.append({"path": f"{step}/data.yaml", "uri": uri})

    workspace = str(result.get("workspace") or "").strip()
    if workspace and not data_yaml:
        items.append({"path": f"{step}/workspace", "uri": f"file://{workspace}"})

    return items
