"""Worker contract — each step is MLAir-executor friendly."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from shared.artifacts import ArtifactStore


@dataclass
class WorkerContext:
    job_id: str
    source_path: Path
    model_name: str
    confidence: float
    store: ArtifactStore
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkerResult:
    ok: bool
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class Worker(Protocol):
    name: str

    def run(self, ctx: WorkerContext) -> WorkerResult: ...
