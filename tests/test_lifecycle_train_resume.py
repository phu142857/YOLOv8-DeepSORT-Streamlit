"""Resume kwargs after worker OOM / pod restart."""

from __future__ import annotations

from pathlib import Path

from mlair_adapter.yolo_train_pipeline import lifecycle_train_extra_kwargs


def test_lifecycle_train_extra_kwargs_empty_without_checkpoint(tmp_path: Path) -> None:
    assert lifecycle_train_extra_kwargs(tmp_path) == {}


def test_lifecycle_train_extra_kwargs_resume_when_last_pt_exists(tmp_path: Path) -> None:
    last_pt = tmp_path / "runs" / "train" / "weights" / "last.pt"
    last_pt.parent.mkdir(parents=True)
    last_pt.write_bytes(b"stub")
    assert lifecycle_train_extra_kwargs(tmp_path) == {"resume": str(last_pt)}
