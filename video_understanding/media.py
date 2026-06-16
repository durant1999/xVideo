from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .downloaders import download_url as download_remote_url
from .downloaders.utils import extract_first_url
from .utils import PipelineError, ensure_dir, require_binary, run_command


def is_url(source: str) -> bool:
    return extract_first_url(source) is not None


def safe_stem(value: str) -> str:
    extracted_url = extract_first_url(value)
    stem = Path(urlparse(extracted_url).path).stem if extracted_url else Path(value).stem
    stem = stem or "video"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "video"


def download_video(
    source: str,
    output_dir: str | Path,
    download_config: dict[str, Any] | None = None,
) -> Path:
    if not is_url(source):
        path = Path(source)
        if not path.exists():
            raise PipelineError(f"Input video not found: {path}")
        return path

    result = download_remote_url(source, output_dir, config=download_config)
    return result.primary_video_path


def probe_duration(video_path: str | Path) -> float:
    require_binary("ffprobe")
    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(video_path),
        ],
        label="ffprobe duration",
    )
    payload = json.loads(result.stdout)
    try:
        return float(payload["format"]["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PipelineError(f"Unable to read duration from ffprobe output for {video_path}") from exc


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    minutes, second = divmod(total_seconds, 60)
    hours, minute = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minute}m {second}s"
    if minute:
        return f"{minute}m {second}s"
    return f"{second}s"


def enforce_duration_limit(
    duration_seconds: float,
    max_duration_seconds: float | int | None,
    *,
    video_path: str | Path | None = None,
) -> None:
    if max_duration_seconds is None:
        return
    limit = float(max_duration_seconds)
    if limit <= 0:
        return
    if duration_seconds <= limit:
        return

    target = f" for {video_path}" if video_path else ""
    raise PipelineError(
        "Video duration"
        f"{target} is {format_duration(duration_seconds)} ({duration_seconds:.1f}s), "
        f"which exceeds the configured limit of {format_duration(limit)} ({limit:.1f}s). "
        "Refusing to analyze this video."
    )


def split_windows(duration: float, window_seconds: float) -> list[tuple[float, float]]:
    if duration <= 0:
        return []
    windows: list[tuple[float, float]] = []
    count = int(math.ceil(duration / window_seconds))
    for index in range(count):
        start = index * window_seconds
        end = min(duration, start + window_seconds)
        if end > start:
            windows.append((start, end))
    return windows


def extract_audio(video_path: str | Path, output_wav: str | Path) -> Path:
    require_binary("ffmpeg")
    output = Path(output_wav)
    ensure_dir(output.parent)
    run_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(output),
        ],
        label="ffmpeg audio extraction",
    )
    return output


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    fps: float,
    start: float,
    end: float,
    max_side: int,
    jpeg_quality: int,
) -> list[dict[str, float | int | str]]:
    require_binary("ffmpeg")
    if fps <= 0:
        raise PipelineError("fps must be greater than 0")
    if end <= start:
        raise PipelineError("frame extraction end must be after start")

    target_dir = ensure_dir(output_dir)
    for stale in target_dir.glob("frame_*.jpg"):
        stale.unlink()

    duration = end - start
    scale_filter = f"scale={max_side}:{max_side}:force_original_aspect_ratio=decrease"
    pattern = str(target_dir / "frame_%06d.jpg")
    run_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(video_path),
            "-t",
            f"{duration:.3f}",
            "-vf",
            f"fps={fps},{scale_filter}",
            "-q:v",
            str(jpeg_quality),
            pattern,
        ],
        label="ffmpeg frame extraction",
    )

    frames: list[dict[str, float | int | str]] = []
    for index, frame_path in enumerate(sorted(target_dir.glob("frame_*.jpg"))):
        timestamp = start + index / fps
        if timestamp > end:
            timestamp = end
        frames.append(
            {
                "index": index,
                "timestamp": round(timestamp, 3),
                "path": str(frame_path),
            }
        )
    if not frames:
        raise PipelineError(f"No frames extracted for {video_path} [{start:.1f}, {end:.1f}]")
    return frames
