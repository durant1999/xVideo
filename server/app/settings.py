"""Environment-driven configuration for the BFF.

Kept dependency-free (stdlib only) so Phase 0 runs with just FastAPI + Uvicorn.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


DEFAULT_REPO_ROOT = str(Path(__file__).resolve().parents[2])


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # Bind loopback only — the server is reached via the SSH LocalForward, never
    # exposed publicly. (See ARCHITECTURE.md §6.)
    host: str = "127.0.0.1"
    port: int = 8788

    # Bearer token required on every route except /healthz. Generate once with
    # e.g. `python -c "import secrets; print(secrets.token_urlsafe(32))"`.
    api_token: str = ""

    # Phase 1: flip to 1 on the server (where `video_understanding` is importable)
    # to mount the job routes that wrap MCPJobManager.
    enable_jobs: bool = False

    # Mirrors mcp_server.build_manager() so phone + Claude share one job history.
    repo_root: str = DEFAULT_REPO_ROOT
    job_root: str = "runs/mcp_jobs"
    config_path: str = "configs/pipeline.yaml"
    keep_media: bool = False

    @property
    def has_token(self) -> bool:
        return bool(self.api_token)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        host=os.getenv("XVIDEO_HOST", "127.0.0.1"),
        port=int(os.getenv("XVIDEO_PORT", "8788")),
        api_token=os.getenv("XVIDEO_API_TOKEN", ""),
        enable_jobs=_as_bool(os.getenv("XVIDEO_ENABLE_JOBS"), default=False),
        repo_root=os.getenv("XVIDEO_REPO_ROOT", DEFAULT_REPO_ROOT),
        job_root=os.getenv("XVIDEO_JOB_ROOT", "runs/mcp_jobs"),
        config_path=os.getenv("XVIDEO_CONFIG_PATH", "configs/pipeline.yaml"),
        keep_media=_as_bool(os.getenv("XVIDEO_KEEP_MEDIA"), default=False),
    )
