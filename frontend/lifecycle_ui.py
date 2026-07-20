"""Streamlit helpers for lifecycle timeline and MLAir readiness (Hub-style)."""

from __future__ import annotations

import streamlit as st

from shared.schemas import JobLifecycleResponse


_STATUS_ICON = {
    "done": "✅",
    "failed": "❌",
    "blocked": "⚠️",
    "pending": "○",
}


def render_timeline(lifecycle: JobLifecycleResponse) -> None:
    st.markdown(f"**Job status:** `{lifecycle.job_status}`")
    for step in lifecycle.steps:
        icon = _STATUS_ICON.get(step.status, "•")
        line = f"{icon} **{step.label}**"
        if step.detail:
            line += f" — {step.detail[:120]}"
        st.markdown(line)


def render_readiness_hub(
    readiness: dict,
    *,
    dataset_id: str | None = None,
    version_id: str | None = None,
) -> None:
    """Hub-style readiness card (status, ready flag, blocking reasons)."""
    ready = readiness.get("ready", False)
    status = readiness.get("status") or "UNKNOWN"
    reasons = readiness.get("reasons") or []

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Status", status)
    with c2:
        st.metric("Ready for training", "Yes" if ready else "No")
    with c3:
        st.metric("Blocking items", len(reasons))

    if ready:
        st.success("Dataset version is **READY** for training under current policy.")
    else:
        st.warning("Dataset version is **NOT READY** — resolve blocking reasons below.")

    if dataset_id:
        st.caption(f"Dataset `{dataset_id[:16]}…`" + (f" · version `{version_id[:16]}…`" if version_id else ""))

    if reasons:
        st.markdown("**Blocking reasons**")
        for r in reasons:
            st.markdown(f"- {r}")
    else:
        st.caption("No blocking reasons returned.")

    with st.expander("Raw readiness payload"):
        st.json(readiness.get("raw", readiness))


def render_job_mlair_summary(job) -> None:
    """Compact MLAir block after job completion."""
    if not job.mlair_dataset_id and not job.mlair_readiness:
        return
    st.markdown("#### MLAir linkage")
    if job.mlair_dataset_id:
        st.write(f"Dataset: `{job.mlair_dataset_id}`")
    if job.mlair_dataset_version_id:
        st.write(f"Version: `{job.mlair_dataset_version_id}`")
    if job.mlair_readiness:
        render_readiness_hub(
            job.mlair_readiness,
            dataset_id=job.mlair_dataset_id,
            version_id=job.mlair_dataset_version_id,
        )
