"""Single catalog: one row per ``weights/detection/{model}``, MLAir linked by name."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mlair_adapter.model_client import ModelClient
from mlair_adapter.model_sync import ModelSyncService
from mlair_adapter.sync_metadata import read_version_sync_metadata
from shared.model_resolve import REGISTRY_PREFIX, is_registry_model
from shared.schemas import UnifiedModelOption, UnifiedModelsResponse
from shared.settings import settings
from shared.weights_catalog import (
    find_weights_in_dir,
    list_detection_model_names,
    model_spec,
    parse_model_spec,
    pick_canonical_local_entry,
    version_dir,
)


def model_name_from_job_spec(spec: str) -> str:
    raw = str(spec or "").strip()
    if is_registry_model(raw):
        return ""
    parsed = parse_model_spec(raw)
    if parsed:
        return parsed[0]
    return raw


def resolve_mlair_model_id(
    model_spec_or_name: str,
    *,
    client: ModelClient | None = None,
    state: dict[str, Any] | None = None,
) -> str | None:
    """Map job ``model/version`` (or ``registry:{id}``) → MLAir ``model_id``."""
    raw = str(model_spec_or_name or "").strip()
    if not raw:
        return None
    if raw.startswith(REGISTRY_PREFIX):
        return raw[len(REGISTRY_PREFIX) :].strip() or None

    model = model_name_from_job_spec(raw)
    if not model:
        return None

    client = client or ModelClient()
    if state is None:
        state = ModelSyncService(client).load_sync_state()

    for key in (model_spec(model, "base"), model_spec(model, "production"), raw):
        row = state.get(key) if isinstance(state, dict) else None
        if isinstance(row, dict) and row.get("model_id"):
            return str(row["model_id"])

    parsed = parse_model_spec(raw)
    if parsed:
        m, ver = parsed
        meta = read_version_sync_metadata(version_dir(settings.detection_model_dir, m, ver))
        if meta and meta.mlair_model_id:
            return meta.mlair_model_id

    if client.enabled:
        hub = client.find_model_by_name(model)
        if hub and hub.get("model_id"):
            return str(hub["model_id"])

    if settings.mlair_model_id:
        return settings.mlair_model_id
    return None


def list_unified_detection_models(
    *,
    root: Path | None = None,
    client: ModelClient | None = None,
) -> UnifiedModelsResponse:
    root = Path(root or settings.detection_model_dir)
    client = client or ModelClient()
    svc = ModelSyncService(client)
    state = svc.load_sync_state()

    registry_by_name: dict[str, dict[str, Any]] = {}
    if client.enabled:
        for row in client.list_models():
            name = str(row.get("name") or "")
            if name and name not in registry_by_name:
                registry_by_name[name] = row

    items: list[UnifiedModelOption] = []
    for model_name in list_detection_model_names(root):
        entry = pick_canonical_local_entry(root, model_name)
        if entry is None:
            continue

        base_weights = find_weights_in_dir(version_dir(root, model_name, "base"))
        if base_weights is not None:
            spec = model_spec(model_name, "base")
            weights_path = base_weights
        else:
            spec = entry.spec
            weights_path = entry.weights_path

        st = state.get(model_spec(model_name, "base")) or state.get(entry.spec) or {}
        model_id = str(st.get("model_id") or "") or None
        prod_ver = st.get("registry_version")
        aligned = bool(st.get("aligned"))

        reg = registry_by_name.get(model_name)
        if reg:
            if not model_id:
                model_id = str(reg.get("model_id") or "") or None
            if prod_ver is None and client.enabled and model_id:
                resolved = client.resolve_version_row(model_id, stage=settings.mlair_sync_stage)
                if resolved and resolved.get("version") is not None:
                    prod_ver = int(resolved["version"])

        label = model_name
        if prod_ver is not None:
            label = f"{model_name} (MLAir v{prod_ver})"
        elif reg:
            label = f"{model_name} (MLAir)"

        items.append(
            UnifiedModelOption(
                model=model_name,
                spec=spec,
                label=label,
                weights_path=str(weights_path),
                mlair_model_id=model_id,
                mlair_production_version=int(prod_ver) if prod_ver is not None else None,
                aligned=aligned,
            )
        )

    return UnifiedModelsResponse(
        root=str(root),
        mlair_configured=client.enabled,
        items=items,
    )
