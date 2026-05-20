"""Client-facing inference: run pipeline via API and persist frames to MLAir dataset."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from frontend.api_client import CVApiClient
from shared.schemas import JobCreate, JobResponse, JobStatus, SourceType
from shared.settings import settings


def _source_type_for_filename(filename: str) -> SourceType:
    ext = Path(filename).suffix.lower()
    if ext in settings.video_extensions:
        return SourceType.VIDEO
    return SourceType.IMAGE


def run_pipeline_job(
    api: CVApiClient,
    file_bytes: bytes,
    filename: str,
    model_name: str,
    confidence: float,
    *,
    on_progress=None,
) -> JobResponse:
    upload = api.upload(file_bytes, filename)
    job = api.create_job(
        JobCreate(
            source_type=_source_type_for_filename(filename),
            model_name=model_name,
            confidence=confidence,
            upload_id=upload["upload_id"],
        )
    )
    api.start_job(job.id)
    return api.wait_for_job(job.id, on_tick=on_progress)


def show_job_results(api: CVApiClient, job: JobResponse) -> None:
    if job.status != JobStatus.COMPLETED:
        st.error(job.error or job.message or "Processing failed")
        return

    st.markdown("**Vehicle In**")
    st.write(job.counters_in or {})
    st.markdown("**Vehicle Out**")
    st.write(job.counters_out or {})

    processed = job.artifact_manifest.get("processed_video", "")
    suffix = Path(processed).suffix.lower() if processed else ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        api.download_artifact(job.id, "processed", Path(tmp.name))
        if suffix in settings.image_extensions:
            st.image(tmp.name, caption="Detected", width="stretch")
        else:
            st.video(tmp.name)


def show_dataset_feedback(api: CVApiClient, job: JobResponse) -> None:
    """Brief user message: buffer ingest + optional new dataset version."""
    if not settings.client_save_to_dataset:
        return
    if not job.mlair_dataset_id:
        st.caption("Dataset sync disabled or MLAir not configured.")
        return

    st.success("Frames saved to MLAir dataset buffer.")
    if job.mlair_dataset_version_id:
        st.info(f"New dataset version created: `{job.mlair_dataset_version_id}`")
    else:
        try:
            buf = api.mlair_get_buffer(job.mlair_dataset_id)
            current = buf.get("current_size", buf.get("record_count", 0))
            threshold = buf.get("target_threshold", "—")
            st.caption(
                f"Buffer: **{current}** / threshold **{threshold}** — "
                f"a version is created when accumulation reaches the threshold."
            )
        except Exception:
            pass

    hub = settings.mlair_hub_url.rstrip("/")
    st.markdown(
        f"Manage lifecycle, readiness, and training in **[MLAir Hub]({hub})**."
    )


def render_image_client(api: CVApiClient, model_name: str, confidence: float) -> None:
    source_img = st.sidebar.file_uploader(
        label="Choose an image...",
        type=("jpg", "jpeg", "png", "bmp", "webp"),
    )
    col1, col2 = st.columns(2)

    with col1:
        if source_img:
            st.image(source_img, caption="Uploaded Image", width="stretch")

    if source_img and st.button("Execution", type="primary"):
        if not settings.client_save_to_dataset:
            st.warning("Dataset save is disabled (`CV_CLIENT_SAVE_TO_DATASET=0`).")
            return

        progress = st.progress(0.0, text="Running detection…")

        def on_tick(job: JobResponse) -> None:
            progress.progress(job.progress, text=job.message or job.current_step)

        with col2:
            try:
                data = source_img.getvalue()
                job = run_pipeline_job(
                    api, data, source_img.name, model_name, confidence, on_progress=on_tick
                )
                progress.progress(1.0, text="Done")
                show_job_results(api, job)
                show_dataset_feedback(api, job)
            except Exception as exc:
                st.error(str(exc))


def render_video_client(api: CVApiClient, model_name: str, confidence: float) -> None:
    source_video = st.sidebar.file_uploader(label="Choose a video...")
    if source_video:
        st.video(source_video)

    if source_video and st.button("Execution", type="primary"):
        progress = st.progress(0.0, text="Processing video…")

        def on_tick(job: JobResponse) -> None:
            progress.progress(job.progress, text=job.message or job.current_step)

        try:
            data = source_video.getvalue()
            job = run_pipeline_job(
                api, data, source_video.name, model_name, confidence, on_progress=on_tick
            )
            progress.progress(1.0, text="Done")
            show_job_results(api, job)
            show_dataset_feedback(api, job)
        except Exception as exc:
            st.error(str(exc))


def render_webcam_client(api: CVApiClient, model_name: str, confidence: float) -> None:
    st.caption("Capture from your camera, then run **Execution** to detect and save to the dataset.")
    snapshot = st.camera_input("Webcam")
    if snapshot and st.button("Execution", type="primary"):
        progress = st.progress(0.0, text="Processing capture…")

        def on_tick(job: JobResponse) -> None:
            progress.progress(job.progress, text=job.message or job.current_step)

        try:
            job = run_pipeline_job(
                api,
                snapshot.getvalue(),
                "webcam.jpg",
                model_name,
                confidence,
                on_progress=on_tick,
            )
            progress.progress(1.0, text="Done")
            show_job_results(api, job)
            show_dataset_feedback(api, job)
        except Exception as exc:
            st.error(str(exc))
