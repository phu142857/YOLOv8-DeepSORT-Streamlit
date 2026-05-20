"""API and job contracts."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SourceType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    WEBCAM = "webcam"
    MLAIR = "mlair"


class JobCreate(BaseModel):
    source_type: SourceType = SourceType.VIDEO
    model_name: str = "yolov8n/base"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    upload_id: str | None = None
    mlair_dataset_id: str | None = None
    mlair_dataset_version_id: str | None = None


class JobResponse(BaseModel):
    id: str
    status: JobStatus
    source_type: SourceType
    model_name: str
    confidence: float
    progress: float = 0.0
    message: str = ""
    current_step: str = ""
    source_filename: str = ""
    error: str | None = None
    created_at: datetime
    updated_at: datetime
    artifact_manifest: dict[str, Any] = Field(default_factory=dict)
    counters_in: dict[str, int] = Field(default_factory=dict)
    counters_out: dict[str, int] = Field(default_factory=dict)
    mlair_dataset_id: str | None = None
    mlair_dataset_version_id: str | None = None
    mlair_readiness: dict[str, Any] = Field(default_factory=dict)
    mlair_ingest: dict[str, Any] = Field(default_factory=dict)
    mlair_training: dict[str, Any] = Field(default_factory=dict)
    mlair_model_version: dict[str, Any] = Field(default_factory=dict)


class UploadResponse(BaseModel):
    upload_id: str
    filename: str
    path: str
    size_bytes: int


class ArtifactManifest(BaseModel):
    job_id: str
    source_video: str | None = None
    source_image: str | None = None
    processed_video: str | None = None
    detections_json: str | None = None
    tracking_json: str | None = None
    aggregates_json: str | None = None
    frames_dir: str | None = None
    frame_count: int = 0
    frames_extracted: int = 0
    pipeline_meta_json: str | None = None


class ArtifactEntry(BaseModel):
    name: str
    path: str
    size_bytes: int
    kind: str  # file | dir


class JobArtifactsResponse(BaseModel):
    job_id: str
    artifacts: list[ArtifactEntry]


class MLAirDatasetSummary(BaseModel):
    dataset_id: str
    name: str = ""
    current_size: int | None = None


class MLAirReadinessResponse(BaseModel):
    dataset_id: str
    dataset_version_id: str | None = None
    status: str = ""
    ready: bool = False
    reasons: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


class LifecycleStep(BaseModel):
    id: str
    label: str
    status: str = "pending"  # pending | done | failed | blocked
    detail: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)


class JobLifecycleResponse(BaseModel):
    job_id: str
    job_status: str
    steps: list[LifecycleStep] = Field(default_factory=list)


class RegistryModelOption(BaseModel):
    model_id: str
    name: str = ""
    registry_value: str
    production_version: int | None = None
    artifact_uri: str | None = None


class RegistryModelsResponse(BaseModel):
    configured: bool = False
    items: list[RegistryModelOption] = Field(default_factory=list)


class LocalModelOption(BaseModel):
    """Checkpoint under ``weights/detection/{model}/{version}/``."""

    model: str
    version: str
    spec: str  # model/version — stored in job.model_name
    label: str = ""
    weights_path: str = ""


class LocalModelsResponse(BaseModel):
    root: str = ""
    items: list[LocalModelOption] = Field(default_factory=list)
