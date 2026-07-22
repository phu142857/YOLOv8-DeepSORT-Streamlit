#!/usr/bin/env python3
"""Import a ZIP of images into MLAir via cv-api, with progress bars."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx
from tqdm import tqdm


class _ProgressReader:
    """Wrap a file so httpx multipart upload can drive a tqdm bar."""

    def __init__(self, path: Path, bar: tqdm) -> None:
        self._file = path.open("rb")
        self._bar = bar

    def read(self, size: int = -1) -> bytes:
        chunk = self._file.read(size)
        if chunk:
            self._bar.update(len(chunk))
        return chunk

    def close(self) -> None:
        self._bar.close()
        self._file.close()


def import_via_api(zip_path: Path, dataset_name: str, api_base: str, timeout: float) -> dict:
    api_base = api_base.rstrip("/")
    total = zip_path.stat().st_size
    upload_bar = tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc="Upload ZIP",
    )
    reader = _ProgressReader(zip_path, upload_bar)
    try:
        files = {"file": (zip_path.name, reader, "application/zip")}
        data = {"dataset_name": dataset_name}
        with httpx.Client(timeout=timeout) as client:
            process_bar = tqdm(total=100, desc="Server: extract + MLAir", unit="%")
            process_bar.set_postfix_str("waiting…")
            r = client.post(f"{api_base}/api/v1/datasets/import-zip", files=files, data=data)
            process_bar.n = 100
            process_bar.set_postfix_str("done")
            process_bar.refresh()
            process_bar.close()
        r.raise_for_status()
        return r.json()
    finally:
        reader.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Import image ZIP → MLAir dataset (with progress).")
    parser.add_argument("--zip", required=True, type=Path, help="Path to .zip file")
    parser.add_argument(
        "--dataset-name",
        default="cv-traffic-frames",
        help="MLAir dataset name (default: cv-traffic-frames)",
    )
    parser.add_argument("--api", default="http://127.0.0.1:8000", help="cv-api base URL")
    parser.add_argument("--timeout", type=float, default=3600.0, help="HTTP timeout seconds")
    args = parser.parse_args()

    zip_path = args.zip.expanduser().resolve()
    if not zip_path.is_file():
        print(f"ZIP not found: {zip_path}", file=sys.stderr)
        return 1
    if zip_path.suffix.lower() != ".zip":
        print("Expected a .zip file", file=sys.stderr)
        return 1

    print(f"Dataset: {args.dataset_name}")
    print(f"ZIP:     {zip_path} ({zip_path.stat().st_size / (1024 * 1024):.1f} MB)")

    try:
        result = import_via_api(zip_path, args.dataset_name, args.api, args.timeout)
    except httpx.HTTPStatusError as exc:
        print(f"API error {exc.response.status_code}: {exc.response.text}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Import failed: {exc}", file=sys.stderr)
        return 1

    print()
    print("Done.")
    print(f"  images:          {result.get('image_count', '?')}")
    print(f"  import_job_id:   {result.get('import_job_id', '')}")
    print(f"  dataset_id:      {result.get('dataset_id', '')}")
    print(f"  version_id:      {result.get('dataset_version_id', '')}")
    if result.get("hub_url"):
        print(f"  hub:             {result['hub_url']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
