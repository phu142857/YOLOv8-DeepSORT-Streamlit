#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Client UI — vehicle detection for end users. Lifecycle / MLAir ops: MLAir Hub (:38080)."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

import config
from frontend.api_client import CVApiClient
from frontend.client_inference import (
    render_image_client,
    render_video_client,
    render_webcam_client,
)
from shared.settings import settings

st.set_page_config(
    page_title="Vehicle Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

hub_url = settings.mlair_hub_url.rstrip("/")

st.title("Vehicle Detection & Counting")
st.caption("YOLOv8 + DeepSORT — upload image, video, or use webcam, then run **Execution**.")

api = CVApiClient()
health = api.health()
api_online = health is not None

st.sidebar.header("Model")
model_type = st.sidebar.selectbox("Select Model", config.DETECTION_MODEL_LIST)
confidence = float(st.sidebar.slider("Confidence", 30, 100, 50)) / 100

st.sidebar.header("Source")
source_selectbox = st.sidebar.selectbox("Select Source", config.SOURCES_LIST)

st.sidebar.divider()
st.sidebar.markdown("**Operations & lifecycle**")
st.sidebar.markdown(
    f"Datasets, versions, readiness, and training are managed in "
    f"[MLAir Hub]({hub_url}) (port 38080)."
)
st.sidebar.caption(
    f"API: `{'online' if api_online else 'offline'}` · "
    f"Auto-save to dataset: `{settings.client_save_to_dataset and settings.mlair_auto_ingest}`"
)

if not api_online:
    st.error("CV API is offline. Start the stack: `./scripts/docker-up.sh` or `./scripts/run_api.sh`")
    st.stop()

if not settings.client_save_to_dataset or not settings.mlair_auto_ingest:
    st.warning(
        "Saving to MLAir dataset is disabled. Set `CV_CLIENT_SAVE_TO_DATASET=1` and "
        "`CV_MLAIR_AUTO_INGEST=1`, and configure MLAir API credentials."
    )

model_path = Path(config.DETECTION_MODEL_DIR, str(model_type))
if not model_path.exists():
    st.error(f"Model not found: {model_path}")
    st.stop()

if source_selectbox == config.SOURCES_LIST[0]:
    render_image_client(api, model_type, confidence)
elif source_selectbox == config.SOURCES_LIST[1]:
    render_video_client(api, model_type, confidence)
else:
    render_webcam_client(api, model_type, confidence)
