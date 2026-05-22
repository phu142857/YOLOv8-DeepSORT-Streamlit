"""PyTorch 2.6+ checkpoint loading for Ultralytics YOLO .pt files."""

from __future__ import annotations

import functools
import inspect


def apply_torch_checkpoint_compat() -> None:
    """
    PyTorch 2.6 defaults torch.load(..., weights_only=True), which breaks YOLO checkpoints
    (pickle contains ultralytics.nn.tasks.DetectionModel). Force weights_only=False when unset.
    """
    import torch

    if getattr(torch.load, "_mlair_patched", False):
        return
    if "weights_only" not in inspect.signature(torch.load).parameters:
        return

    _orig_load = torch.load

    @functools.wraps(_orig_load)
    def _load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _orig_load(*args, **kwargs)

    _load._mlair_patched = True  # type: ignore[attr-defined]
    torch.load = _load  # type: ignore[assignment]
