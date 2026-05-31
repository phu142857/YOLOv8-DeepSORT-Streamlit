"""Resolve YOLO train device: GPU/MPS when available, else CPU."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def resolve_train_device(*, batch: int = 0) -> str | int:
    """
    Return Ultralytics ``device`` argument.

    ``CV_MLAIR_TRAIN_DEVICE``:
      - ``auto`` (default): CUDA GPU 0, else Apple MPS, else ``cpu``
      - ``cpu`` / ``0`` / ``cuda:0`` / ``mps``: passed through when valid
    """
    raw = (
        os.getenv("CV_MLAIR_TRAIN_DEVICE")
        or os.getenv("MLAIR_TRAIN_DEVICE")
        or "auto"
    ).strip().lower()

    if raw in ("", "auto"):
        return _auto_device(batch=batch)

    if raw == "cpu":
        return "cpu"
    if raw == "mps":
        return "mps"
    if raw.startswith("cuda:"):
        return raw.split(":", 1)[1] if _cuda_available() else "cpu"
    if raw.isdigit():
        return int(raw) if _cuda_available() else "cpu"
    return raw


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _mps_available() -> bool:
    try:
        import torch

        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        return False


def _auto_device(*, batch: int = 0) -> str | int:
    del batch  # batch divisibility handled by Ultralytics when device is set
    if _cuda_available():
        try:
            import torch

            name = torch.cuda.get_device_name(0)
            logger.info("train device: cuda:0 (%s)", name)
        except Exception:
            logger.info("train device: cuda:0")
        return 0
    if _mps_available():
        logger.info("train device: mps")
        return "mps"
    logger.info("train device: cpu")
    return "cpu"
