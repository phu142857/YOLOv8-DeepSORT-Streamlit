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
    default_model: str = _env("CV_DEFAULT_MODEL", "yolov8n/base")
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
    # Ignored at runtime — threshold/strategy come from MLAir Hub buffer API.
    mlair_buffer_threshold: int = int(_env("CV_MLAIR_BUFFER_THRESHOLD", "0"))
    mlair_hub_url: str = _env("CV_MLAIR_HUB_URL", "http://localhost:38080")
    client_save_to_dataset: bool = _env_bool("CV_CLIENT_SAVE_TO_DATASET", True)
    mlair_model_id: str = _env("CV_MLAIR_MODEL_ID", "")
    mlair_auto_train: bool = _env_bool("CV_MLAIR_AUTO_TRAIN", False)
    mlair_auto_promote: bool = _env_bool("CV_MLAIR_AUTO_PROMOTE", False)
    mlair_train_fail_pipeline: bool = _env_bool("CV_MLAIR_TRAIN_FAIL_PIPELINE", False)
    mlair_train_poll_timeout_sec: float = float(_env("CV_MLAIR_TRAIN_POLL_TIMEOUT_SEC", "3600"))
    mlair_train_poll_interval_sec: float = float(_env("CV_MLAIR_TRAIN_POLL_INTERVAL_SEC", "10"))
    mlair_promote_stage: str = _env("CV_MLAIR_PROMOTE_STAGE", "production")
    mlair_weights_cache_dir: Path = Path(_env("CV_MLAIR_WEIGHTS_CACHE_DIR", "weights/registry"))
    # Must match ml-air-api model artifact root (docker volume at /mlair/artifacts/models).
    mlair_model_artifact_mount: str = _env("CV_MLAIR_MODEL_ARTIFACT_MOUNT", "/mlair/artifacts")
    # Auto push local weights/detection → MLAir registry; pull after train/promote.
    mlair_auto_sync_models: bool = _env_bool("CV_MLAIR_AUTO_SYNC_MODELS", True)
    mlair_sync_on_startup: bool = _env_bool("CV_MLAIR_SYNC_ON_STARTUP", True)
    mlair_sync_after_train: bool = _env_bool("CV_MLAIR_SYNC_AFTER_TRAIN", True)
    mlair_sync_stage: str = _env("CV_MLAIR_SYNC_STAGE", "production")
    mlair_sync_interval_sec: float = float(_env("CV_MLAIR_SYNC_INTERVAL_SEC", "300"))
    mlair_mirror_registry_to_local: bool = _env_bool("CV_MLAIR_MIRROR_REGISTRY_TO_LOCAL", True)
    mlair_sync_state_path: Path = Path(_env("CV_MLAIR_SYNC_STATE_PATH", "artifacts/.mlair_model_sync_state.json"))
    # MLAir pipeline `cv-yolo-vehicle-train` (HTTP task → /api/v1/mlair/train/execute)
    mlair_train_pipeline_id: str = _env("CV_MLAIR_TRAIN_PIPELINE_ID", "cv-yolo-vehicle-train")
    mlair_train_callback_token: str = _env("CV_MLAIR_TRAIN_CALLBACK_TOKEN", "admin-token")
    mlair_train_base_model_spec: str = _env("CV_MLAIR_TRAIN_BASE_MODEL", "yolov8n/base")
    mlair_train_epochs: int = int(_env("CV_MLAIR_TRAIN_EPOCHS", "10"))
    mlair_train_batch: int = int(_env("CV_MLAIR_TRAIN_BATCH", "8"))
    mlair_train_imgsz: int = int(_env("CV_MLAIR_TRAIN_IMGSZ", "640"))
    mlair_train_max_frames: int = int(_env("CV_MLAIR_TRAIN_MAX_FRAMES", "500"))
    mlair_train_import_stage: str = _env("CV_MLAIR_TRAIN_IMPORT_STAGE", "production")

    video_extensions: frozenset[str] = frozenset(
        {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
    )
    image_extensions: frozenset[str] = frozenset(
        {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )


settings = Settings()
