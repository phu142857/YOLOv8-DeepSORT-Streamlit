"""Filesystem artifact store for lineage, replay, and MLAir ingestion."""

from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from shared.schemas import ArtifactEntry, ArtifactManifest
from shared.settings import settings

logger = logging.getLogger(__name__)


class ArtifactStore:
    """Persists workload artifacts under ``artifacts/jobs/{job_id}/``."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or settings.artifact_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.uploads_dir = self.root / "uploads"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

    def new_upload_id(self) -> str:
        return str(uuid.uuid4())

    def upload_path(self, upload_id: str, filename: str) -> Path:
        dest = self.uploads_dir / upload_id
        dest.mkdir(parents=True, exist_ok=True)
        safe_name = Path(filename).name
        return dest / safe_name

    def job_dir(self, job_id: str) -> Path:
        path = self.root / "jobs" / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def job_layout(self, job_id: str) -> dict[str, Path]:
        base = self.job_dir(job_id)
        layout = {
            "base": base,
            "source": base / "source",
            "frames": base / "frames",
            "detections": base / "detections",
            "tracking": base / "tracking",
            "output": base / "output",
            "aggregates": base / "aggregates",
            "steps": base / "steps",
            "logs": base / "logs",
        }
        for key, p in layout.items():
            if key != "base":
                p.mkdir(parents=True, exist_ok=True)
        return layout

    def save_upload_to_job(self, job_id: str, upload_path: Path) -> Path:
        layout = self.job_layout(job_id)
        dest = layout["source"] / upload_path.name
        shutil.copy2(upload_path, dest)
        return dest

    def write_json(self, path: Path, payload: Any) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)
        return path

    def append_jsonl(self, path: Path, record: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def append_log(self, job_id: str, line: str) -> None:
        log_path = self.job_layout(job_id)["logs"] / "pipeline.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line.rstrip() + "\n")

    def save_step(self, job_id: str, step_name: str, payload: dict[str, Any]) -> Path:
        path = self.job_layout(job_id)["steps"] / f"{step_name}.json"
        return self.write_json(path, payload)

    def save_manifest(self, job_id: str, manifest: ArtifactManifest) -> Path:
        path = self.job_layout(job_id)["base"] / "manifest.json"
        self.write_json(path, manifest.model_dump())
        return path

    def load_manifest(self, job_id: str) -> ArtifactManifest | None:
        path = self.job_dir(job_id) / "manifest.json"
        if not path.exists():
            return None
        with path.open(encoding="utf-8") as f:
            return ArtifactManifest(**json.load(f))

    def relative(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.root))
        except ValueError:
            return str(path)

    def list_artifacts(self, job_id: str) -> list[ArtifactEntry]:
        base = self.job_dir(job_id)
        if not base.exists():
            return []

        entries: list[ArtifactEntry] = []

        def walk(directory: Path, prefix: str = "") -> None:
            if not directory.exists():
                return
            for item in sorted(directory.iterdir()):
                rel_name = f"{prefix}/{item.name}" if prefix else item.name
                if item.is_file():
                    entries.append(
                        ArtifactEntry(
                            name=rel_name,
                            path=self.relative(item),
                            size_bytes=item.stat().st_size,
                            kind="file",
                        )
                    )
                elif item.is_dir():
                    entries.append(
                        ArtifactEntry(
                            name=rel_name + "/",
                            path=self.relative(item),
                            size_bytes=0,
                            kind="dir",
                        )
                    )
                    walk(item, rel_name)

        walk(base)
        return entries

    def resolve_artifact(self, job_id: str, artifact_type: str) -> Path | None:
        layout = self.job_layout(job_id)
        if artifact_type == "manifest":
            p = layout["base"] / "manifest.json"
            return p if p.exists() else None
        if artifact_type == "export":
            p = layout["base"] / "export.json"
            return p if p.exists() else None
        if artifact_type == "preview":
            preview = layout["output"] / "preview.jpg"
            return preview if preview.is_file() else None
        if artifact_type == "processed":
            manifest = self.load_manifest(job_id)
            if manifest and manifest.processed_video:
                candidate = self.root / manifest.processed_video
                if candidate.is_file():
                    return candidate
            for name in ("processed.mp4", "processed.jpg", "processed.png"):
                candidate = layout["output"] / name
                if candidate.is_file():
                    return candidate
            for ext in ("*.mp4", "*.avi", "*.mov", "*.jpg", "*.jpeg", "*.png"):
                matches = sorted(layout["output"].glob(ext))
                if matches:
                    return matches[0]
            return None
        if artifact_type == "detections":
            for name in ("detections.jsonl", "detections.json"):
                p = layout["detections"] / name
                if p.exists():
                    return p
        if artifact_type == "tracking":
            p = layout["tracking"] / "tracks.jsonl"
            return p if p.exists() else None
        if artifact_type == "aggregates":
            p = layout["aggregates"] / "counts.json"
            return p if p.exists() else None
        if artifact_type == "log":
            p = layout["logs"] / "pipeline.log"
            return p if p.exists() else None
        return None
