"""Browser-playable video export (H.264) via ffmpeg."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def transcode_mp4_for_browser(video_path: Path) -> Path:
    """
    Re-mux/re-encode to H.264 + yuv420p so HTML5 video players (Streamlit, Chrome) can play it.
    Replaces the file in place on success.
    """
    video_path = Path(video_path)
    if not video_path.is_file() or video_path.stat().st_size == 0:
        return video_path

    if not ffmpeg_available():
        logger.warning("ffmpeg not found — processed video may not play in browser")
        return video_path

    tmp = video_path.with_name(video_path.stem + "_h264.mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=600)
        if tmp.is_file() and tmp.stat().st_size > 0:
            tmp.replace(video_path)
            logger.info("Transcoded %s to H.264 for browser playback", video_path.name)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        logger.warning("ffmpeg transcode failed for %s: %s", video_path, exc)
        tmp.unlink(missing_ok=True)
    return video_path
