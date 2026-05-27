"""CV UI — upload ZIP of images → MLAir dataset version."""

from __future__ import annotations

import streamlit as st

from frontend.api_client import CVApiClient
from shared.settings import settings


def render_dataset_zip_import(api: CVApiClient, *, hub_url: str) -> None:
    st.sidebar.markdown("**Import dataset (ZIP)**")
    with st.sidebar.expander("Upload images ZIP → MLAir", expanded=False):
        st.caption(
            "ZIP chỉ chứa ảnh (.jpg, .png, …). Tạo **dataset version** trên MLAir để Train trên Hub."
        )
        dataset_name = st.text_input(
            "Dataset name",
            value=settings.mlair_dataset_name,
            key="dataset_zip_name",
        )
        zip_file = st.file_uploader(
            "ZIP file",
            type=["zip"],
            key="dataset_zip_file",
        )
        if st.button("Import to MLAir", type="primary", disabled=zip_file is None):
            if not zip_file:
                st.warning("Chọn file ZIP trước.")
                return
            if not str(dataset_name or "").strip():
                st.warning("Nhập tên dataset.")
                return
            with st.spinner("Đang giải nén và tạo dataset trên MLAir…"):
                try:
                    result = api.import_dataset_zip(
                        zip_file.getvalue(),
                        zip_file.name or "upload.zip",
                        str(dataset_name).strip(),
                    )
                except Exception as exc:
                    st.error(f"Import failed: {exc}")
                    return
            st.success(
                f"Đã import **{result.image_count}** ảnh → dataset `{result.dataset_name}`."
            )
            if result.dataset_version_id:
                st.info(f"Version ID: `{result.dataset_version_id}`")
            st.markdown(
                f"Tiếp theo: [MLAir Hub]({hub_url}) → pin version này khi **Train with model**."
            )
