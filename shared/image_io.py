"""Load images for OpenCV/YOLO (JPEG/PNG/WebP + AVIF/HEIC via Pillow)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_image_bgr(path: Path | str) -> np.ndarray:
    """
    Return BGR uint8 image. OpenCV first; Pillow fallback for AVIF/HEIC and misnamed files.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"image not found: {path}")

    frame = cv2.imread(str(path))
    if frame is not None:
        return frame

    data = path.read_bytes()
    if data:
        decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if decoded is not None:
            return decoded

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(f"Cannot read image: {path}") from exc

    try:
        with Image.open(path) as im:
            rgb = im.convert("RGB")
            arr = np.asarray(rgb)
    except Exception as exc:
        raise RuntimeError(f"Cannot read image: {path}") from exc

    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def normalize_image_file(path: Path) -> Path:
    """
    Rewrite to a true JPEG on disk when OpenCV cannot decode (e.g. AVIF named .jpg).
    Returns path to readable file (same or new).
    """
    path = Path(path)
    if cv2.imread(str(path)) is not None:
        return path

    bgr = load_image_bgr(path)
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        out = path
    else:
        out = path.with_suffix(".jpg")
    cv2.imwrite(str(out), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if out.resolve() != path.resolve() and path.is_file():
        path.unlink(missing_ok=True)
    return out
