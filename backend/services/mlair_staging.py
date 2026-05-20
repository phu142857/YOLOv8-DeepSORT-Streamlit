"""Stage MLAir dataset version frames into a job before inference."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from mlair_adapter.dataset_client import DatasetClient
from shared.artifacts import ArtifactStore

logger = logging.getLogger(__name__)


def stage_mlair_version(
    job_id: str,
    dataset_version_id: str,
    store: ArtifactStore,
    *,
    max_frames: int = 0,
) -> Path:
    """
    Pull frames from MLAir into job source/ and return path to first image
    (or a synthetic video placeholder — we run frame-by-frame from frames/).
    """
    client = DatasetClient()
    if not client.enabled:
        raise RuntimeError("MLAir is not configured (CV_MLAIR_API_URL / CV_MLAIR_TOKEN)")

    layout = store.job_layout(job_id)
    pull_dir = layout["source"] / "mlair_frames"
    if pull_dir.exists():
        shutil.rmtree(pull_dir)
    pull_dir.mkdir(parents=True, exist_ok=True)

    _, count = client.fetch_version_frames(dataset_version_id, pull_dir, max_frames=max_frames)

    # Pipeline expects a single source file OR we adapt to use frames dir
    # Copy first frame as reference + symlink all into frames for detection pass
    frames_out = layout["frames"]
    frames_out.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(sorted(pull_dir.glob("*.jpg"))):
        dest = frames_out / src.name
        shutil.copy2(src, dest)

    marker = layout["source"] / "mlair_pull.json"
    store.write_json(
        marker,
        {
            "dataset_version_id": dataset_version_id,
            "frames_pulled": count,
            "pull_dir": store.relative(pull_dir),
        },
    )

    # Use first frame as nominal source for worker path validation
    first = next(iter(sorted(pull_dir.glob("*.jpg"))), None)
    if first is None:
        raise RuntimeError("no frames pulled from MLAir")
    staged = layout["source"] / first.name
    shutil.copy2(first, staged)
    return staged
