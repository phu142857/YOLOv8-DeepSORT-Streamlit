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
    # Copy frames to MLAir dataset volume (duplicate on EFS). Default off: train reads artifacts/jobs via shared EFS or cv-api URL.
    mlair_persist_ingest_frames: bool = _env_bool("CV_MLAIR_PERSIST_INGEST_FRAMES", False)
    mlair_runtime_frames_subdir: str = _env("CV_MLAIR_RUNTIME_FRAMES_SUBDIR", "cv-runtime-frames")
    api_host: str = _env("CV_API_HOST", "0.0.0.0")
    api_port: int = int(_env("CV_API_PORT", "8000"))
    api_base_url: str = _env("CV_API_BASE_URL", "http://127.0.0.1:8000")
    detection_model_dir: Path = Path(_env("CV_DETECTION_MODEL_DIR", "weights/detection"))
    # Comma-separated Ultralytics names to seed on empty disk (EFS after destroy, first deploy).
    detection_bootstrap_models: str = _env(
        "CV_DETECTION_BOOTSTRAP_MODELS", "yolov8n,yolov8s,yolov8m,yolov8l,yolov8x"
    )
    default_model: str = _env("CV_DEFAULT_MODEL", "yolov8s/base")
    # Extra dropdown rows ``{model} (COCO pretrained)`` when base != pretrained (recovery).
    catalog_include_pretrained: bool = _env_bool("CV_CATALOG_INCLUDE_PRETRAINED", False)
    # S3 source of truth for promoted checkpoints (inference always reads local cache).
    s3_models_bucket: str = _env("CV_MODELS_S3_BUCKET", "")
    s3_models_prefix: str = _env("CV_MODELS_S3_PREFIX", "ml-models")
    s3_models_region: str = _env("CV_MODELS_S3_REGION", "") or _env("AWS_REGION", "ap-southeast-1")
    s3_models_sync_on_startup: bool = _env_bool("CV_MODELS_S3_SYNC_ON_STARTUP", True)
    s3_models_upload_on_promote: bool = _env_bool("CV_MODELS_S3_UPLOAD_ON_PROMOTE", True)
    frame_extract_interval: int = int(_env("CV_FRAME_EXTRACT_INTERVAL", "30"))
    max_upload_mb: int = int(_env("CV_MAX_UPLOAD_MB", "500"))
    dataset_zip_max_mb: int = int(_env("CV_DATASET_ZIP_MAX_MB", "2048"))
    # Safety rails for ZIP dataset import (avoid zip-bombs / EFS overload).
    dataset_zip_max_images: int = int(_env("CV_DATASET_ZIP_MAX_IMAGES", "100000"))
    dataset_zip_max_unzipped_mb: int = int(_env("CV_DATASET_ZIP_MAX_UNZIPPED_MB", "20480"))
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
    # MLAir pipeline input + training-policy minimum record count (Hub readiness).
    mlair_pipeline_required_size: int = int(_env("CV_MLAIR_PIPELINE_REQUIRED_SIZE", "50"))
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
    # Vet-AI-style periodic registry resync (defaults 120s); falls back to sync_interval if unset.
    mlair_registry_resync_seconds: float = float(
        _env("CV_MLAIR_REGISTRY_RESYNC_SECONDS")
        or _env("CV_MLAIR_SYNC_INTERVAL_SEC", "120")
    )
    mlair_registry_sync_at_startup: bool = _env_bool("CV_MLAIR_REGISTRY_SYNC_AT_STARTUP", True)
    mlair_disk_sync_mode: str = _env("CV_MLAIR_DISK_SYNC_MODE", "metadata")  # metadata | state
    mlair_disk_import_all_versions: bool = _env_bool("CV_MLAIR_DISK_IMPORT_ALL_VERSIONS", False)
    # Hub = catalog/control plane. ``sync-full`` pull all models (bootstrap); keep off by default.
    mlair_mirror_registry_to_local: bool = _env_bool("CV_MLAIR_MIRROR_REGISTRY_TO_LOCAL", False)
    # Align ``base/`` with Hub ``production_version`` on promote webhook + registry resync (default on).
    mlair_sync_production_from_hub: bool = _env_bool("CV_MLAIR_SYNC_PRODUCTION_FROM_HUB", True)
    mlair_sync_state_path: Path = Path(_env("CV_MLAIR_SYNC_STATE_PATH", "artifacts/.mlair_model_sync_state.json"))
    # MLAir pipeline cv-yolo-vehicle-train: plugin (Hub Train UI) | http (executor-only, Hub blocks train)
    mlair_pipeline_mode: str = _env("CV_MLAIR_PIPELINE_MODE", "plugin").strip().lower()
    # Phase B lifecycle DAG (prepare → train → eval → gate)
    mlair_train_pipeline_id: str = _env("CV_MLAIR_TRAIN_PIPELINE_ID", "cv-yolo-lifecycle-train")
    mlair_legacy_train_pipeline_id: str = _env("CV_MLAIR_LEGACY_TRAIN_PIPELINE_ID", "cv-yolo-vehicle-train")
    mlair_hard_example_pipeline_id: str = _env("CV_MLAIR_HARD_EXAMPLE_PIPELINE_ID", "cv-hard-example-mine")
    mlair_hard_example_dataset_name: str = _env("CV_MLAIR_HARD_EXAMPLE_DATASET", "cv-traffic-hard-examples")
    mlair_lifecycle_import_stage: str = _env("CV_MLAIR_LIFECYCLE_IMPORT_STAGE", "staging")
    mlair_lifecycle_auto_promote: bool = _env_bool("CV_MLAIR_LIFECYCLE_AUTO_PROMOTE", True)
    mlair_gate_min_map_delta: float = float(_env("CV_MLAIR_GATE_MIN_MAP_DELTA", "0.0"))
    mlair_hard_example_max_conf: float = float(_env("CV_MLAIR_HARD_EXAMPLE_MAX_CONF", "0.35"))
    mlair_hard_example_max_scan: int = int(_env("CV_MLAIR_HARD_EXAMPLE_MAX_SCAN", "500"))
    mlair_bootstrap_pipeline_on_startup: bool = _env_bool("CV_MLAIR_BOOTSTRAP_PIPELINE", True)
    mlair_train_callback_token: str = _env("CV_MLAIR_TRAIN_CALLBACK_TOKEN", "admin-token")
    # MLAir Hub promote → pull production into weights/detection (webhook + periodic pull).
    mlair_sync_on_hub_promote: bool = _env_bool("CV_MLAIR_SYNC_ON_HUB_PROMOTE", True)
    mlair_promote_webhook_token: str = _env(
        "CV_MLAIR_PROMOTE_WEBHOOK_TOKEN",
        _env("CV_MLAIR_TRAIN_CALLBACK_TOKEN", "admin-token"),
    )
    mlair_train_base_model_spec: str = _env("CV_MLAIR_TRAIN_BASE_MODEL", "yolov8s/pretrained")
    mlair_train_epochs: int = int(_env("CV_MLAIR_TRAIN_EPOCHS", "10"))
    mlair_train_batch: int = int(_env("CV_MLAIR_TRAIN_BATCH", "8"))
    mlair_train_workers: int = int(_env("CV_MLAIR_TRAIN_WORKERS", "4"))
    mlair_train_imgsz: int = int(_env("CV_MLAIR_TRAIN_IMGSZ", "640"))
    # 0 = use entire dataset version manifest (no cap). Set 500 etc. for quick QA only.
    mlair_train_max_frames: int = int(_env("CV_MLAIR_TRAIN_MAX_FRAMES", "0"))
    mlair_train_import_stage: str = _env("CV_MLAIR_TRAIN_IMPORT_STAGE", "production")
    mlair_prepare_fallback_pseudo: bool = _env_bool("CV_MLAIR_PREPARE_FALLBACK_PSEUDO", True)
    mlair_prepare_pseudo_confidence: float = float(_env("CV_MLAIR_PREPARE_PSEUDO_CONF", "0.25"))
    mlair_detect_confidence: float = float(_env("CV_MLAIR_DETECT_CONF", "0.25"))
    # Auto-labeling (detect) teacher model. MUST be a strong general detector — the
    # model selected for training may start from an untrained checkpoint and produce
    # zero pseudo-labels. Defaults to the COCO-pretrained yolov8s on the host weights
    # mount (survives `podman system reset`). Set to "" to reuse the run's base model.
    mlair_detect_base_model_spec: str = _env("CV_MLAIR_DETECT_BASE_MODEL", "yolov8s/pretrained")
    # 0 = use CV_MLAIR_TRAIN_MAX_FRAMES cap (same as prepare/train pull)
    mlair_detect_max_frames: int = int(_env("CV_MLAIR_DETECT_MAX_FRAMES", "0"))
    # Lifecycle detect: split source → Hub datasets, merge for prepare/train
    mlair_detected_dataset_name: str = _env("CV_MLAIR_DETECTED_DATASET", "detected")
    mlair_not_detected_dataset_name: str = _env("CV_MLAIR_NOT_DETECTED_DATASET", "not-detected")
    mlair_train_ready_dataset_name: str = _env("CV_MLAIR_TRAIN_READY_DATASET", "train-ready")
    # Lineage: Hub complete ingests body.lineage. extra = POST only lineage_ingests[1:]; off | all
    mlair_lineage_post_ingest: str = _env("CV_MLAIR_LINEAGE_POST_INGEST", "extra").strip().lower()

    video_extensions: frozenset[str] = frozenset(
        {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
    )
    image_extensions: frozenset[str] = frozenset(
        {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif", ".heic", ".heif"}
    )


settings = Settings()
