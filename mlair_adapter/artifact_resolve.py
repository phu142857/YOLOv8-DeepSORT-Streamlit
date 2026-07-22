"""Resolve MLAir ``file://`` model artifacts on the CV worker (split deploy).

MLAir stores ``artifact_uri`` paths on the controller volume. External workers often
lack that mount; this module falls back to ``weights/detection/{model}/`` on the worker
host (see ``CV_DETECTION_MODEL_DIR``) without changing the MLAir framework.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from urllib.parse import urlparse

from shared.settings import settings
from shared.weights_catalog import PRETRAINED_VERSION, find_weights_in_dir, version_dir

logger = logging.getLogger(__name__)

_MLAIR_MODELS_MARKER = "/models/"
_VERSION_DIR_RE = re.compile(r"^v(\d+)$", re.IGNORECASE)


def parse_mlair_model_artifact_uri(artifact_uri: str) -> tuple[str | None, int | None]:
    """
  Parse Hub default layout::

      file:///mlair/artifacts/models/{tenant}/{project}/{model_name}/v{N}

  Returns ``(model_name, version_number)`` or ``(None, None)``.
  """
    raw = str(artifact_uri or "").strip()
    if not raw:
        return None, None
    parsed = urlparse(raw)
    path = parsed.path if parsed.scheme == "file" else raw
    marker = _MLAIR_MODELS_MARKER
    idx = path.find(marker)
    if idx < 0:
        return None, None
    tail = [p for p in path[idx + len(marker) :].split("/") if p]
    if len(tail) < 3:
        return None, None
    model_name = tail[-2]
    version: int | None = None
    m = _VERSION_DIR_RE.match(tail[-1])
    if m:
        version = int(m.group(1))
    return model_name, version


def resolve_local_detection_weights(
    *,
    model_name: str,
    version: int | None = None,
    detection_root: Path | None = None,
) -> Path | None:
    """Find ``weights.pt`` under ``weights/detection/{model_name}/``."""
    name = str(model_name or "").strip()
    if not name:
        return None
    root = Path(detection_root or settings.detection_model_dir)
    version_labels: list[str] = []
    if version is not None:
        version_labels.append(f"v{int(version)}")
    version_labels.extend(["base", "production", PRETRAINED_VERSION])
    seen: set[str] = set()
    for label in version_labels:
        if label in seen:
            continue
        seen.add(label)
        try:
            found = find_weights_in_dir(version_dir(root, name, label))
        except ValueError:
            continue
        if found is not None:
            return found
    return None


def resolve_local_weights_from_artifact_uri(artifact_uri: str) -> Path | None:
    """Map controller ``file://`` URI → local ``weights/detection`` when mount is absent."""
    model_name, version = parse_mlair_model_artifact_uri(artifact_uri)
    if not model_name:
        return None
    return resolve_local_detection_weights(model_name=model_name, version=version)


def resolve_local_weights_for_hub_model(model_id: str, client: object | None = None) -> Path | None:
    """Resolve by Hub registry ``name`` (matches ``weights/detection/{name}/``)."""
    mid = str(model_id or "").strip()
    if not mid:
        return None
    from mlair_adapter.model_client import ModelClient

    mc = client if client is not None else ModelClient()
    if not getattr(mc, "enabled", True):
        return None
    try:
        row = mc.get_model(mid)
    except Exception as exc:
        logger.debug("hub model lookup failed for %s: %s", mid, exc)
        return None
    name = str(row.get("name") or "").strip()
    if not name:
        return None
    return resolve_local_detection_weights(model_name=name)
