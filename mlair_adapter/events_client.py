"""Semantic lifecycle events (local log + optional MLAir audit)."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from shared.artifacts import ArtifactStore

logger = logging.getLogger(__name__)


def emit_event(
    event_type: str,
    payload: dict[str, Any],
    *,
    job_id: str | None = None,
    store: ArtifactStore | None = None,
) -> None:
    """Record semantic event locally; Phase 4 can forward to Redis / webhooks."""
    record = {
        "type": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    logger.info("semantic_event %s %s", event_type, json.dumps(payload, default=str))
    if job_id and store:
        store.append_log(job_id, f"EVENT {event_type} {json.dumps(payload, default=str)}")
