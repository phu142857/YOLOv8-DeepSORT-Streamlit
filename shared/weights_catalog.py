"""Local detection weights layout: weights/detection/{model}/{version}/*.pt"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

# Preferred checkpoint names inside a version folder (training exports).
WEIGHT_FILENAMES = ("weights.pt", "best.pt", "last.pt", "model.pt")

# API / job field: ``yolov8n/base`` (not a filesystem path).
MODEL_SPEC_SEP = "/"

_SAFE_SEGMENT = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")


@dataclass(frozen=True)
class LocalModelEntry:
    model: str
    version: str
    weights_path: Path

    @property
    def spec(self) -> str:
        return model_spec(self.model, self.version)

    @property
    def label(self) -> str:
        return f"{self.model} / {self.version}"


def model_spec(model: str, version: str) -> str:
    return f"{model.strip()}/{version.strip()}"


def parse_model_spec(spec: str) -> tuple[str, str] | None:
    """Return (model, version) when spec is ``model/version``."""
    raw = str(spec or "").strip()
    if not raw or MODEL_SPEC_SEP not in raw:
        return None
    model, version = raw.split(MODEL_SPEC_SEP, 1)
    model, version = model.strip(), version.strip()
    if not model or not version:
        return None
    return model, version


def version_dir(root: Path, model: str, version: str) -> Path:
    if not _SAFE_SEGMENT.match(model) or not _SAFE_SEGMENT.match(version):
        raise ValueError(f"invalid model or version name: {model!r} / {version!r}")
    return root / model / version


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def weights_files_equivalent(a: Path, b: Path) -> bool:
    """True when both paths exist and contain the same checkpoint bytes."""
    a, b = Path(a), Path(b)
    if not a.is_file() or not b.is_file():
        return False
    if a.stat().st_size != b.stat().st_size:
        return False
    return sha256_file(a) == sha256_file(b)


def find_weights_in_dir(version_path: Path) -> Path | None:
    """Resolve checkpoint inside ``.../{model}/{version}/``."""
    version_path = Path(version_path)
    if not version_path.is_dir():
        return None
    for name in WEIGHT_FILENAMES:
        candidate = version_path / name
        if candidate.is_file():
            return candidate
    pts = sorted(version_path.glob("*.pt"))
    if len(pts) == 1:
        return pts[0]
    if len(pts) > 1:
        # Prefer explicit names if multiple .pt files exist.
        for name in WEIGHT_FILENAMES:
            for p in pts:
                if p.name == name:
                    return p
        return pts[0]
    return None


# Active inference slot: mirrors MLAir ``production`` (promote webhook / production pull).
CANONICAL_LOCAL_VERSIONS = ("base", "production")
# Immutable COCO seed; never overwritten by promote.
PRETRAINED_VERSION = "pretrained"
# Per-train archives on disk: ``v1/``, ``v2/``, … (kept when production rolls back to an older vN).


def pick_canonical_local_entry(root: Path, model: str) -> LocalModelEntry | None:
    """One representative checkpoint per model: base → production → highest vN."""
    root = Path(root)
    for version in CANONICAL_LOCAL_VERSIONS:
        try:
            weights = find_weights_in_dir(version_dir(root, model, version))
        except ValueError:
            weights = None
        if weights is not None:
            return LocalModelEntry(model=model, version=version, weights_path=weights)

    best: LocalModelEntry | None = None
    best_num = -1
    model_path = root / model
    if not model_path.is_dir():
        return None
    for version_dir_path in model_path.iterdir():
        if not version_dir_path.is_dir():
            continue
        name = version_dir_path.name
        if name in CANONICAL_LOCAL_VERSIONS:
            continue
        if name.startswith("v") and name[1:].isdigit():
            num = int(name[1:])
        else:
            continue
        weights = find_weights_in_dir(version_dir_path)
        if weights is None:
            continue
        if num > best_num:
            best_num = num
            best = LocalModelEntry(model=model, version=name, weights_path=weights)
    return best


def list_detection_model_names(root: Path) -> list[str]:
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(
        d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")
    )


def scan_detection_models(root: Path) -> list[LocalModelEntry]:
    """
    Discover ``root/{model}/{version}/*.pt``.

    Ignores hidden dirs and flat ``*.pt`` at root (legacy — still resolved separately).
    """
    root = Path(root)
    if not root.is_dir():
        return []

    entries: list[LocalModelEntry] = []
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("."):
            continue
        for version_dir in sorted(model_dir.iterdir()):
            if not version_dir.is_dir() or version_dir.name.startswith("."):
                continue
            weights = find_weights_in_dir(version_dir)
            if weights is None:
                continue
            entries.append(
                LocalModelEntry(
                    model=model_dir.name,
                    version=version_dir.name,
                    weights_path=weights,
                )
            )
    return entries


def resolve_local_weights(root: Path, model_name: str) -> Path:
    """
    Resolve weights path from:
    - ``model/version`` (catalog layout)
    - legacy flat ``yolov8n.pt`` under root
    - absolute/relative path if file exists
    """
    root = Path(root)
    raw = str(model_name or "").strip()
    if not raw:
        raise FileNotFoundError("empty model_name")

    parsed = parse_model_spec(raw)
    if parsed:
        model, version = parsed
        weights = find_weights_in_dir(version_dir(root, model, version))
        if weights is not None:
            return weights
        raise FileNotFoundError(
            f"no weights in {version_dir(root, model, version)} "
            f"(expected one of {', '.join(WEIGHT_FILENAMES)} or a single *.pt)"
        )

    direct = Path(raw)
    if direct.is_file():
        return direct

    legacy = root / raw
    if legacy.is_file():
        return legacy

    # Bare model name without version → prefer version ``base``, then ``production``.
    for version in ("base", "production", "latest"):
        weights = find_weights_in_dir(version_dir(root, raw, version))
        if weights is not None:
            return weights

    raise FileNotFoundError(
        f"model not found: {raw} (use model/version under {root}, e.g. yolov8n/base)"
    )


def ensure_version_layout(root: Path, model: str, version: str) -> Path:
    """Return version directory, creating it if missing (for training export scripts)."""
    path = version_dir(root, model, version)
    path.mkdir(parents=True, exist_ok=True)
    return path
