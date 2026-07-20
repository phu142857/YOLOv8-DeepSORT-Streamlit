"""S3 source of truth for detection weights — download once, serve from local disk."""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

from shared.settings import settings
from shared.weights_catalog import ensure_version_layout, find_weights_in_dir

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "production.json"
S3_CHECKPOINT_NAMES = ("best.pt", "weights.pt", "model.pt", "last.pt")


def s3_enabled() -> bool:
    return bool(settings.s3_models_bucket.strip())


def _client():
    import boto3

    kwargs: dict[str, Any] = {}
    region = settings.s3_models_region.strip()
    if region:
        kwargs["region_name"] = region
    return boto3.client("s3", **kwargs)


def _prefix() -> str:
    return settings.s3_models_prefix.strip("/")


def manifest_key(model: str) -> str:
    return f"{_prefix()}/detection/{model}/{MANIFEST_FILENAME}"


def version_object_prefix(model: str, version: str) -> str:
    return f"{_prefix()}/detection/{model}/{version}"


def _parse_manifest(body: bytes) -> str | None:
    try:
        data = json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    version = data.get("version")
    if version is None:
        return None
    return str(version).strip() or None


def read_production_version(model: str) -> str | None:
    if not s3_enabled():
        return None
    bucket = settings.s3_models_bucket
    key = manifest_key(model)
    try:
        resp = _client().get_object(Bucket=bucket, Key=key)
        return _parse_manifest(resp["Body"].read())
    except Exception as exc:
        code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
        if code in {"NoSuchKey", "404", "NotFound"}:
            logger.debug("no production manifest for %s at s3://%s/%s", model, bucket, key)
            return None
        logger.warning("failed to read s3://%s/%s: %s", bucket, key, exc)
        return None


def _download_object(bucket: str, key: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    _client().download_file(bucket, key, str(dest))


def _resolve_s3_checkpoint_key(model: str, version: str) -> str | None:
    bucket = settings.s3_models_bucket
    base = version_object_prefix(model, version)
    for name in S3_CHECKPOINT_NAMES:
        key = f"{base}/{name}"
        try:
            _client().head_object(Bucket=bucket, Key=key)
            return key
        except Exception:
            continue
    return None


def download_version_checkpoint(model: str, version: str, dest: Path) -> Path:
    """Download ``{prefix}/detection/{model}/{version}/*.pt`` to ``dest``."""
    if not s3_enabled():
        raise RuntimeError("S3 models bucket not configured")
    bucket = settings.s3_models_bucket
    key = _resolve_s3_checkpoint_key(model, version)
    if key is None:
        raise FileNotFoundError(
            f"no checkpoint under s3://{bucket}/{version_object_prefix(model, version)}/"
            f" ({', '.join(S3_CHECKPOINT_NAMES)})"
        )
    _download_object(bucket, key, dest)
    logger.info("downloaded s3://%s/%s -> %s", bucket, key, dest)
    return dest


def write_production_manifest(model: str, version: str, *, sha256: str | None = None) -> None:
    if not s3_enabled():
        return
    bucket = settings.s3_models_bucket
    payload: dict[str, Any] = {"version": version}
    if sha256:
        payload["sha256"] = sha256
    body = json.dumps(payload, indent=2).encode("utf-8")
    key = manifest_key(model)
    _client().put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
    logger.info("updated s3://%s/%s -> version=%s", bucket, key, version)


def upload_checkpoint(
    model: str,
    version: str,
    local_path: Path,
    *,
    filename: str = "best.pt",
    set_production: bool = False,
    sha256: str | None = None,
) -> str:
    """Upload local ``.pt`` to S3 version folder; optionally update production manifest."""
    if not s3_enabled():
        raise RuntimeError("S3 models bucket not configured")
    local_path = Path(local_path)
    if not local_path.is_file():
        raise FileNotFoundError(local_path)
    bucket = settings.s3_models_bucket
    key = f"{version_object_prefix(model, version)}/{filename}"
    _client().upload_file(str(local_path), bucket, key)
    logger.info("uploaded %s -> s3://%s/%s", local_path, bucket, key)
    if set_production:
        write_production_manifest(model, version, sha256=sha256)
    return key


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_replace(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    shutil.copy2(src, tmp)
    tmp.replace(dest)


def _rotate_previous(base_weights: Path, root: Path, model: str) -> None:
    previous_dir = ensure_version_layout(root, model, "previous")
    previous_pt = previous_dir / "weights.pt"
    if base_weights.is_file():
        _atomic_replace(base_weights, previous_pt)


def apply_local_production_layout(
    root: Path,
    model: str,
    checkpoint: Path,
    *,
    also_current: bool = True,
) -> dict[str, Path]:
    """
    Install checkpoint into canonical local folders (base + production + optional current).

    Rotates existing base → previous for instant rollback.
    """
    root = Path(root)
    checkpoint = Path(checkpoint)
    base_dir = ensure_version_layout(root, model, "base")
    prod_dir = ensure_version_layout(root, model, "production")
    base_pt = base_dir / "weights.pt"
    prod_pt = prod_dir / "weights.pt"

    existing = find_weights_in_dir(base_dir)
    if existing is not None and existing.resolve() != checkpoint.resolve():
        _rotate_previous(existing, root, model)

    _atomic_replace(checkpoint, base_pt)
    _atomic_replace(checkpoint, prod_pt)
    out: dict[str, Path] = {"base": base_pt, "production": prod_pt}
    if also_current:
        current_dir = ensure_version_layout(root, model, "current")
        current_pt = current_dir / "weights.pt"
        _atomic_replace(checkpoint, current_pt)
        out["current"] = current_pt
    return out


def sync_model_from_s3(model: str, root: Path | None = None) -> Path | None:
    """
    Read production.json from S3, download checkpoint, install under ``weights/detection``.

    Returns path to ``base/weights.pt`` or None if S3 disabled / no manifest / download failed.
    """
    if not s3_enabled():
        return None
    root = Path(root or settings.detection_model_dir)
    version = read_production_version(model)
    if not version:
        return None

    with tempfile.TemporaryDirectory(prefix="cv-s3-model-") as tmp:
        tmp_pt = Path(tmp) / "checkpoint.pt"
        try:
            download_version_checkpoint(model, version, tmp_pt)
        except FileNotFoundError:
            logger.warning("S3 production version %s for %s has no checkpoint", version, model)
            return None

        manifest_sha: str | None = None
        try:
            bucket = settings.s3_models_bucket
            resp = _client().get_object(Bucket=bucket, Key=manifest_key(model))
            data = json.loads(resp["Body"].read().decode("utf-8"))
            if str(data.get("version", "")).strip() == version:
                manifest_sha = str(data.get("sha256") or "") or None
        except Exception:
            manifest_sha = None

        local_sha = _sha256_file(tmp_pt)
        if manifest_sha and manifest_sha != local_sha:
            logger.error(
                "S3 checksum mismatch for %s/%s (manifest=%s local=%s)",
                model,
                version,
                manifest_sha[:12],
                local_sha[:12],
            )
            return None

        paths = apply_local_production_layout(root, model, tmp_pt)
        logger.info(
            "S3 sync %s production=%s -> %s",
            model,
            version,
            paths["base"],
        )
        return paths["base"]


def hub_version_tag(version: int) -> str:
    return f"v{int(version)}"
