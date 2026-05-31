"""Train device policy: auto GPU/CPU, optional CPU after consecutive GPU train failures."""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

_streak_lock = threading.Lock()
_consecutive_gpu_train_failures = 0


def gpu_fail_streak_threshold() -> int:
    raw = (
        os.getenv("CV_MLAIR_GPU_FAIL_STREAK_FOR_CPU")
        or os.getenv("MLAIR_GPU_FAIL_STREAK_FOR_CPU")
        or "3"
    ).strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 3


def consecutive_gpu_train_failures() -> int:
    with _streak_lock:
        return _consecutive_gpu_train_failures


def reset_gpu_train_failures() -> None:
    global _consecutive_gpu_train_failures
    with _streak_lock:
        if _consecutive_gpu_train_failures:
            logger.info("reset GPU train failure streak (was %s)", _consecutive_gpu_train_failures)
        _consecutive_gpu_train_failures = 0


def record_gpu_train_failure() -> int:
    global _consecutive_gpu_train_failures
    with _streak_lock:
        _consecutive_gpu_train_failures += 1
        n = _consecutive_gpu_train_failures
    logger.warning("GPU train failure streak=%s/%s", n, gpu_fail_streak_threshold())
    return n


def is_gpu_device(device: str | int) -> bool:
    if isinstance(device, int):
        return device >= 0
    d = str(device).strip().lower()
    return d not in ("", "cpu", "mps") and not d.startswith("cpu")


def resolve_train_device(*, batch: int = 0) -> str | int:
    """
    Ultralytics ``device`` for eval/inference-style calls.

    ``auto``: CUDA → MPS → CPU (no failure streak).
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


def pick_lifecycle_train_device(*, batch: int = 0) -> str | int:
    """
    Device for lifecycle ``train`` step only.

    - After ``CV_MLAIR_GPU_FAIL_STREAK_FOR_CPU`` consecutive GPU train failures → CPU.
    - Otherwise same as ``resolve_train_device`` (GPU if available, else CPU).
    """
    threshold = gpu_fail_streak_threshold()
    with _streak_lock:
        streak = _consecutive_gpu_train_failures
    if streak >= threshold:
        logger.info(
            "train device: cpu (%s consecutive GPU train failures, threshold=%s)",
            streak,
            threshold,
        )
        return "cpu"
    return resolve_train_device(batch=batch)


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
    del batch
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
    logger.info("train device: cpu (no GPU)")
    return "cpu"


def run_ultralytics_train_with_device_policy(
    model: Any,
    *,
    batch: int | None = None,
    **train_kwargs: Any,
) -> tuple[Any, str | int]:
    """
    Run ``model.train`` with lifecycle train device policy.

    On GPU failure: increment streak; after threshold, retry once on CPU in the same call.
    """
    batch = batch if batch is not None else int(train_kwargs.get("batch") or 0)
    device = pick_lifecycle_train_device(batch=batch)
    threshold = gpu_fail_streak_threshold()

    try:
        results = model.train(device=device, **train_kwargs)
    except Exception as exc:
        if not is_gpu_device(device):
            raise
        failures = record_gpu_train_failure()
        if failures < threshold:
            raise
        logger.warning(
            "GPU train failed %s times in a row; retrying same train on CPU",
            failures,
        )
        try:
            from mlair_adapter.worker_task_runtime import refresh_task_memory_baseline

            refresh_task_memory_baseline()
            results = model.train(device="cpu", **train_kwargs)
            logger.info("train succeeded on CPU after GPU failure streak")
            return results, "cpu"
        except Exception as cpu_exc:
            raise cpu_exc from exc

    if is_gpu_device(device):
        reset_gpu_train_failures()
    return results, device
