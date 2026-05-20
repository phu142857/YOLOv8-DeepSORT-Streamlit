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
model_options: list[str] = []
model_value_map: dict[str, str] = {}
mlair_configured = False
mlair_unlinked = False

if api_online:
    try:
        unified = api.list_unified_models()
        mlair_configured = unified.mlair_configured
        if unified.mlair_configured and unified.items and not any(i.mlair_model_id for i in unified.items):
            mlair_unlinked = True
        for item in unified.items:
            label = item.label or item.model
            model_options.append(label)
            model_value_map[label] = item.spec
    except Exception:
        pass

if not model_options:
    for spec in config.DETECTION_MODEL_LIST:
        model_options.append(spec)
        model_value_map[spec] = spec

if not model_options:
    st.error("No detection models found under weights/detection/{model}/{version}/")
    st.stop()

model_label = st.sidebar.selectbox("Select Model", model_options)
model_type = model_value_map.get(model_label, model_label)
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
if mlair_unlinked:
    st.sidebar.info(
        f"MLAir has no model linked yet for folders under `weights/detection/`. "
        f"cv-api auto-sync pushes those checkpoints to Hub (same name). "
        f"Or import on [MLAir Hub]({hub_url})."
    )
elif mlair_configured:
    st.sidebar.caption(
        "Inference uses `weights/detection/<model>/base/weights.pt` — kept identical to MLAir production."
    )

if not api_online:
    st.error("CV API is offline. Start the stack: `./scripts/docker-up.sh` or `./scripts/run_api.sh`")
    st.stop()

if not settings.client_save_to_dataset or not settings.mlair_auto_ingest:
    st.warning(
        "Saving to MLAir dataset is disabled. Set `CV_CLIENT_SAVE_TO_DATASET=1` and "
        "`CV_MLAIR_AUTO_INGEST=1`, and configure MLAir API credentials."
    )

try:
    from shared.weights_catalog import resolve_local_weights

    resolve_local_weights(config.DETECTION_MODEL_DIR, model_type)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.caption(
        f"Expected layout: `{config.DETECTION_MODEL_DIR}/<model>/base/weights.pt` "
        "(synced from MLAir production after training)."
    )
    st.stop()

if source_selectbox == config.SOURCES_LIST[0]:
    render_image_client(api, model_type, confidence)
elif source_selectbox == config.SOURCES_LIST[1]:
    render_video_client(api, model_type, confidence)
else:
    render_webcam_client(api, model_type, confidence)
