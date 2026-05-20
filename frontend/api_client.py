"""HTTP client for CV Lifecycle API."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import httpx

from shared.schemas import (
    JobArtifactsResponse,
    JobCreate,
    JobLifecycleResponse,
    JobResponse,
    JobStatus,
)
from shared.settings import settings


class CVApiClient:
    def __init__(self, base_url: str | None = None, timeout: float = 300.0) -> None:
        self.base_url = (base_url or settings.api_base_url).rstrip("/")
        self.timeout = timeout

    def _headers(self, trace_id: str | None = None) -> dict[str, str]:
        h: dict[str, str] = {}
        if trace_id:
            h[settings.trace_header] = trace_id
        return h

    def health(self) -> dict[str, Any] | None:
        try:
            r = httpx.get(f"{self.base_url}/health", timeout=5.0)
            return r.json() if r.status_code == 200 else None
        except httpx.HTTPError:
            return None

    def upload(self, file_bytes: bytes, filename: str) -> dict[str, Any]:
        files = {"file": (filename, file_bytes)}
        r = httpx.post(f"{self.base_url}/api/v1/uploads", files=files, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def create_job(self, spec: JobCreate) -> JobResponse:
        r = httpx.post(
            f"{self.base_url}/api/v1/jobs",
            json=spec.model_dump(mode="json"),
            timeout=30.0,
        )
        r.raise_for_status()
        return JobResponse(**r.json())

    def list_jobs(self, limit: int = 20) -> list[JobResponse]:
        r = httpx.get(f"{self.base_url}/api/v1/jobs", params={"limit": limit}, timeout=30.0)
        r.raise_for_status()
        return [JobResponse(**j) for j in r.json()]

    def start_job(self, job_id: str, force: bool = False) -> JobResponse:
        r = httpx.post(
            f"{self.base_url}/api/v1/jobs/{job_id}/start",
            params={"force": force} if force else None,
            timeout=30.0,
        )
        r.raise_for_status()
        return JobResponse(**r.json())

    def cancel_job(self, job_id: str) -> JobResponse:
        r = httpx.post(f"{self.base_url}/api/v1/jobs/{job_id}/cancel", timeout=30.0)
        r.raise_for_status()
        return JobResponse(**r.json())

    def get_job(self, job_id: str) -> JobResponse:
        r = httpx.get(f"{self.base_url}/api/v1/jobs/{job_id}", timeout=30.0)
        r.raise_for_status()
        return JobResponse(**r.json())

    def list_artifacts(self, job_id: str) -> JobArtifactsResponse:
        r = httpx.get(f"{self.base_url}/api/v1/jobs/{job_id}/artifacts", timeout=30.0)
        r.raise_for_status()
        return JobArtifactsResponse(**r.json())

    def wait_for_job(
        self,
        job_id: str,
        poll_interval: float = 1.0,
        timeout: float | None = None,
        on_tick: Callable[[JobResponse], None] | None = None,
    ) -> JobResponse:
        deadline = time.time() + (timeout or settings.job_poll_timeout_sec)
        while time.time() < deadline:
            job = self.get_job(job_id)
            if on_tick:
                on_tick(job)
            if job.status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}:
                return job
            time.sleep(poll_interval)
        raise TimeoutError(f"job {job_id} did not finish within {timeout}s")

    def artifact_url(self, job_id: str, artifact_type: str) -> str:
        return f"{self.base_url}/api/v1/jobs/{job_id}/artifacts/{artifact_type}"

    def mlair_status(self) -> dict[str, Any]:
        r = httpx.get(f"{self.base_url}/api/v1/mlair/status", timeout=10.0)
        r.raise_for_status()
        return r.json()

    def mlair_list_datasets(self) -> list[dict[str, Any]]:
        r = httpx.get(f"{self.base_url}/api/v1/mlair/datasets", timeout=30.0)
        r.raise_for_status()
        return r.json().get("items", [])

    def mlair_list_versions(self, dataset_id: str) -> list[dict[str, Any]]:
        r = httpx.get(f"{self.base_url}/api/v1/mlair/datasets/{dataset_id}/versions", timeout=30.0)
        r.raise_for_status()
        return r.json().get("items", [])

    def mlair_get_buffer(self, dataset_id: str) -> dict[str, Any]:
        r = httpx.get(f"{self.base_url}/api/v1/mlair/datasets/{dataset_id}/buffer", timeout=30.0)
        r.raise_for_status()
        return r.json()

    def get_job_lifecycle(self, job_id: str) -> JobLifecycleResponse:
        r = httpx.get(f"{self.base_url}/api/v1/jobs/{job_id}/lifecycle", timeout=30.0)
        r.raise_for_status()
        return JobLifecycleResponse(**r.json())

    def mlair_get_readiness(self, dataset_id: str, version_id: str | None = None) -> dict[str, Any]:
        params = {}
        if version_id:
            params["dataset_version_id"] = version_id
        r = httpx.get(
            f"{self.base_url}/api/v1/mlair/datasets/{dataset_id}/readiness",
            params=params,
            timeout=30.0,
        )
        r.raise_for_status()
        return r.json()

    def mlair_evaluate_readiness(self, dataset_id: str, version_id: str | None = None) -> dict[str, Any]:
        params = {}
        if version_id:
            params["dataset_version_id"] = version_id
        r = httpx.post(
            f"{self.base_url}/api/v1/mlair/datasets/{dataset_id}/readiness/evaluate",
            params=params,
            timeout=60.0,
        )
        r.raise_for_status()
        return r.json()

    def mlair_materialize(self, dataset_id: str) -> dict[str, Any]:
        r = httpx.post(f"{self.base_url}/api/v1/mlair/datasets/{dataset_id}/materialize", timeout=60.0)
        r.raise_for_status()
        return r.json()

    def download_artifact(self, job_id: str, artifact_type: str, dest: Path) -> Path:
        url = self.artifact_url(job_id, artifact_type)
        with httpx.stream("GET", url, timeout=self.timeout) as r:
            r.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
        return dest
