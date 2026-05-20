"""Video/image processing pipeline with artifact emission."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

import cv2

import config
from inference.engine import load_model, predict_frame, resolve_model_path
from shared.artifacts import ArtifactStore
from shared.schemas import ArtifactManifest
from shared.settings import settings

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]
CancelCallback = Callable[[], bool]


def _open_video_writer(out_path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    for codec in ("avc1", "mp4v", "XVID"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, size)
        if writer.isOpened():
            return writer
    raise RuntimeError(f"Cannot open video writer for {out_path}")


def process_video_job(
    job_id: str,
    video_path: Path,
    model_name: str,
    confidence: float,
    store: ArtifactStore,
    on_progress: ProgressCallback | None = None,
    should_cancel: CancelCallback | None = None,
) -> ArtifactManifest:
    layout = store.job_layout(job_id)
    model_path = resolve_model_path(model_name)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = load_model(str(model_path))

    config.OBJECT_COUNTER = None
    config.OBJECT_COUNTER1 = None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    max_frames = settings.max_video_frames

    out_path = layout["output"] / "processed.mp4"
    writer = _open_video_writer(out_path, fps, (width, height))

    detections_path = layout["detections"] / "detections.jsonl"
    tracking_path = layout["tracking"] / "tracks.jsonl"
    frame_idx = 0
    frames_extracted = 0
    started = time.perf_counter()

    try:
        while True:
            if should_cancel and should_cancel():
                raise InterruptedError("job cancelled")

            ok, frame = cap.read()
            if not ok:
                break

            if max_frames > 0 and frame_idx >= max_frames:
                logger.info("Job %s hit max_video_frames=%d", job_id, max_frames)
                break

            plotted, detections, counters_in, counters_out = predict_frame(model, frame, confidence)
            writer.write(plotted)

            store.append_jsonl(detections_path, {"frame": frame_idx, "detections": detections})
            store.append_jsonl(
                tracking_path,
                {"frame": frame_idx, "counters_in": counters_in, "counters_out": counters_out},
            )

            if frame_idx % settings.frame_extract_interval == 0:
                cv2.imwrite(str(layout["frames"] / f"{frame_idx:06d}.jpg"), frame)
                frames_extracted += 1

            frame_idx += 1
            if on_progress:
                if total_frames > 0:
                    on_progress(frame_idx / total_frames, f"frame {frame_idx}/{total_frames}")
                else:
                    on_progress(min(0.99, frame_idx / 1000), f"frame {frame_idx}")
    finally:
        cap.release()
        writer.release()

    elapsed = time.perf_counter() - started
    aggregates = {
        "frames_processed": frame_idx,
        "frames_extracted": frames_extracted,
        "counters_in": dict(config.OBJECT_COUNTER1 or {}),
        "counters_out": dict(config.OBJECT_COUNTER or {}),
        "model": model_name,
        "confidence": confidence,
        "fps": fps,
        "elapsed_sec": round(elapsed, 2),
    }
    aggregates_path = layout["aggregates"] / "counts.json"
    store.write_json(aggregates_path, aggregates)

    meta_path = layout["aggregates"] / "pipeline_meta.json"
    meta = {
        "job_id": job_id,
        "source": store.relative(video_path),
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames_reported": total_frames,
        "frames_processed": frame_idx,
        "elapsed_sec": round(elapsed, 2),
    }
    store.write_json(meta_path, meta)

    manifest = ArtifactManifest(
        job_id=job_id,
        source_video=store.relative(video_path),
        processed_video=store.relative(out_path),
        detections_json=store.relative(detections_path),
        tracking_json=store.relative(tracking_path),
        aggregates_json=store.relative(aggregates_path),
        frames_dir=store.relative(layout["frames"]),
        frame_count=frame_idx,
        frames_extracted=frames_extracted,
        pipeline_meta_json=store.relative(meta_path),
    )
    store.save_manifest(job_id, manifest)
    return manifest


def process_frames_dir_job(
    job_id: str,
    frames_dir: Path,
    model_name: str,
    confidence: float,
    store: ArtifactStore,
    on_progress: ProgressCallback | None = None,
    should_cancel: CancelCallback | None = None,
) -> ArtifactManifest:
    """Process pre-staged frames (e.g. pulled from MLAir dataset version)."""
    layout = store.job_layout(job_id)
    model_path = resolve_model_path(model_name)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = load_model(str(model_path))

    config.OBJECT_COUNTER = None
    config.OBJECT_COUNTER1 = None

    frame_paths = sorted(Path(frames_dir).glob("*.jpg"))
    if not frame_paths:
        raise RuntimeError(f"No frames in {frames_dir}")

    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise RuntimeError("Cannot read first frame")
    height, width = first.shape[:2]
    fps = 25.0
    out_path = layout["output"] / "processed.mp4"
    writer = _open_video_writer(out_path, fps, (width, height))

    detections_path = layout["detections"] / "detections.jsonl"
    tracking_path = layout["tracking"] / "tracks.jsonl"
    total = len(frame_paths)
    started = time.perf_counter()

    try:
        for frame_idx, frame_path in enumerate(frame_paths):
            if should_cancel and should_cancel():
                raise InterruptedError("job cancelled")
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue
            plotted, detections, counters_in, counters_out = predict_frame(model, frame, confidence)
            writer.write(plotted)
            store.append_jsonl(detections_path, {"frame": frame_idx, "detections": detections})
            store.append_jsonl(
                tracking_path,
                {"frame": frame_idx, "counters_in": counters_in, "counters_out": counters_out},
            )
            if on_progress:
                on_progress((frame_idx + 1) / total, f"frame {frame_idx + 1}/{total}")
    finally:
        writer.release()

    elapsed = time.perf_counter() - started
    aggregates_path = layout["aggregates"] / "counts.json"
    store.write_json(
        aggregates_path,
        {
            "frames_processed": len(frame_paths),
            "counters_in": dict(config.OBJECT_COUNTER1 or {}),
            "counters_out": dict(config.OBJECT_COUNTER or {}),
            "model": model_name,
            "confidence": confidence,
            "source": "mlair_frames",
            "elapsed_sec": round(elapsed, 2),
        },
    )

    manifest = ArtifactManifest(
        job_id=job_id,
        source_image=store.relative(frames_dir),
        processed_video=store.relative(out_path),
        detections_json=store.relative(detections_path),
        tracking_json=store.relative(tracking_path),
        aggregates_json=store.relative(aggregates_path),
        frames_dir=store.relative(frames_dir),
        frame_count=len(frame_paths),
        frames_extracted=len(frame_paths),
    )
    store.save_manifest(job_id, manifest)
    return manifest


def process_image_job(
    job_id: str,
    image_path: Path,
    model_name: str,
    confidence: float,
    store: ArtifactStore,
) -> ArtifactManifest:
    layout = store.job_layout(job_id)
    model_path = resolve_model_path(model_name)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = load_model(str(model_path))

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    plotted, detections, counters_in, counters_out = predict_frame(model, frame, confidence)
    out_path = layout["output"] / "processed.jpg"
    cv2.imwrite(str(out_path), plotted)

    frame_path = layout["frames"] / "000000.jpg"
    cv2.imwrite(str(frame_path), frame)

    detections_path = layout["detections"] / "detections.json"
    store.write_json(detections_path, {"frame": 0, "detections": detections})
    tracking_path = layout["tracking"] / "tracks.jsonl"
    store.append_jsonl(
        tracking_path,
        {"frame": 0, "counters_in": counters_in, "counters_out": counters_out},
    )
    aggregates_path = layout["aggregates"] / "counts.json"
    store.write_json(
        aggregates_path,
        {
            "frames_processed": 1,
            "counters_in": counters_in,
            "counters_out": counters_out,
            "model": model_name,
            "confidence": confidence,
        },
    )

    manifest = ArtifactManifest(
        job_id=job_id,
        source_image=store.relative(image_path),
        processed_video=store.relative(out_path),
        detections_json=store.relative(detections_path),
        tracking_json=store.relative(tracking_path),
        aggregates_json=store.relative(aggregates_path),
        frames_dir=store.relative(layout["frames"]),
        frame_count=1,
        frames_extracted=1,
    )
    store.save_manifest(job_id, manifest)
    return manifest
