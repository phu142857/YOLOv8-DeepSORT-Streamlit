"""POST Hub ``/lineage/ingest`` so dataset version edges appear in the Lineage UI."""

from __future__ import annotations

import logging
from typing import Any

from mlair_adapter.base_client import MLAirClient

logger = logging.getLogger(__name__)


def ingest_lineage(
    client: MLAirClient,
    *,
    run_id: str,
    task_id: str,
    lineage: dict[str, Any],
) -> dict[str, Any]:
    """Persist ``lineage_edges`` for one task (inputs/outputs use dataset **name** + **vN** label)."""
    if not client.enabled:
        return {"ingested": False, "reason": "mlair_not_configured"}
    rid = str(run_id or "").strip()
    tid = str(task_id or "").strip()
    if not rid or not tid:
        return {"ingested": False, "reason": "run_id_or_task_id_missing"}
    if not isinstance(lineage, dict):
        return {"ingested": False, "reason": "invalid_lineage"}
    ins = lineage.get("inputs") or []
    outs = lineage.get("outputs") or []
    if not ins and not outs:
        return {"ingested": False, "reason": "empty_lineage"}

    path = f"{client._prefix()}/lineage/ingest"
    try:
        out = client.post(
            path,
            json={"run_id": rid, "task_id": tid, "lineage": lineage},
        )
        if isinstance(out, dict):
            logger.info(
                "lineage_ingest run_id=%s task_id=%s edges=%s ingested=%s",
                rid,
                tid,
                out.get("edges"),
                out.get("ingested"),
            )
            return out
        return {"ingested": True, "edges": 0, "response": out}
    except Exception as exc:
        logger.warning("lineage_ingest_failed run_id=%s task_id=%s err=%s", rid, tid, exc)
        return {"ingested": False, "error": str(exc)}


def ingest_lineage_blocks(
    client: MLAirClient,
    *,
    run_id: str,
    task_id: str,
    blocks: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for block in blocks or []:
        if not isinstance(block, dict):
            continue
        results.append(
            ingest_lineage(client, run_id=run_id, task_id=task_id, lineage=block)
        )
    return results
