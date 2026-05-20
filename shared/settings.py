"""Application settings (env-overridable)."""

import os
from dataclasses import dataclass
from pathlib import Path


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_bool(key: str, default: bool = True) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    artifact_root: Path = Path(_env("CV_ARTIFACT_ROOT", "artifacts"))
    api_host: str = _env("CV_API_HOST", "0.0.0.0")
    api_port: int = int(_env("CV_API_PORT", "8000"))
    api_base_url: str = _env("CV_API_BASE_URL", "http://127.0.0.1:8000")
    detection_model_dir: Path = Path(_env("CV_DETECTION_MODEL_DIR", "weights/detection"))
    default_model: str = _env("CV_DEFAULT_MODEL", "yolov8n.pt")
    frame_extract_interval: int = int(_env("CV_FRAME_EXTRACT_INTERVAL", "30"))
    max_upload_mb: int = int(_env("CV_MAX_UPLOAD_MB", "500"))
    max_video_frames: int = int(_env("CV_MAX_VIDEO_FRAMES", "0"))  # 0 = no limit
    persist_jobs: bool = _env_bool("CV_PERSIST_JOBS", True)
    job_poll_timeout_sec: float = float(_env("CV_JOB_POLL_TIMEOUT_SEC", "600"))
    log_level: str = _env("CV_LOG_LEVEL", "INFO")
    trace_header: str = _env("CV_TRACE_HEADER", "X-Trace-Id")
    mlair_api_url: str = _env("CV_MLAIR_API_URL", "")
    mlair_token: str = _env("CV_MLAIR_TOKEN", "")
    mlair_tenant: str = _env("CV_MLAIR_TENANT", "default")
    mlair_project: str = _env("CV_MLAIR_PROJECT", "default_project")
    mlair_dataset_name: str = _env("CV_MLAIR_DATASET_NAME", "cv-traffic-frames")
    mlair_auto_ingest: bool = _env_bool("CV_MLAIR_AUTO_INGEST", True)
    mlair_auto_materialize: bool = _env_bool("CV_MLAIR_AUTO_MATERIALIZE", True)
    mlair_hub_url: str = _env("CV_MLAIR_HUB_URL", "http://localhost:38080")
    client_save_to_dataset: bool = _env_bool("CV_CLIENT_SAVE_TO_DATASET", True)

    video_extensions: frozenset[str] = frozenset(
        {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
    )
    image_extensions: frozenset[str] = frozenset(
        {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )


settings = Settings()
