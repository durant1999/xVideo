from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .utils import PipelineError


DEFAULT_CONFIG: dict[str, Any] = {
    "workdir": "runs",
    "download": {
        "order": ["yt-dlp", "twitter-video-downloader", "ideaflow"],
        "yt-dlp": {
            "enabled": True,
            "no_playlist": True,
        },
        "twitter-video-downloader": {
            "enabled": True,
            "base_url": "https://twittervideodownloader.com/en/",
            "timeout_seconds": 120,
        },
        "ideaflow": {
            "enabled": True,
            "base_url": "https://parse.ideaflow.top/",
            "timeout_seconds": 120,
        },
    },
    "video": {
        "fps": 1.0,
        "segment_seconds": 45,
        "max_side": 960,
        "jpeg_quality": 3,
        "max_duration_seconds": 1800,
    },
    "vl": {
        "base_url": "http://127.0.0.1:8000/v1",
        "api_key": "EMPTY",
        "model": "Qwen3-VL-32B-Instruct-AWQ",
        "temperature": 0.0,
        "max_tokens": 1800,
        "timeout_seconds": 600,
    },
    "asr": {
        "backend": "faster-whisper",
        "model": "large-v3",
        "device": "cuda",
        "device_index": 1,
        "compute_type": "float16",
        "language": "zh",
        "beam_size": 5,
    },
    "fusion": {
        "window_seconds": 45,
    },
    "summary": {
        "base_url": "http://127.0.0.1:8000/v1",
        "api_key": "EMPTY",
        "model": "Qwen3-VL-32B-Instruct-AWQ",
        "temperature": 0.1,
        "max_tokens": 2400,
        "timeout_seconds": 600,
    },
    "ab_eval": {
        "judge_base_url": "http://127.0.0.1:8000/v1",
        "judge_api_key": "EMPTY",
        "judge_model": "Qwen3-VL-32B-Instruct-AWQ",
        "max_tokens": 2400,
        "timeout_seconds": 600,
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        default_path = Path("configs/pipeline.yaml")
        path = default_path if default_path.exists() else None
    if path is None:
        return copy.deepcopy(DEFAULT_CONFIG)

    config_path = Path(path)
    if not config_path.exists():
        raise PipelineError(f"Config file not found: {config_path}")

    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        loaded = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise PipelineError("PyYAML is required to read YAML config files.") from exc
        loaded = yaml.safe_load(text) or {}

    if not isinstance(loaded, dict):
        raise PipelineError(f"Config root must be an object: {config_path}")
    return deep_merge(DEFAULT_CONFIG, loaded)
