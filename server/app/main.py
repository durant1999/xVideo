"""xVideo App BFF — FastAPI application factory.

Phase 0 ships only /healthz (open) and /ping (authed) so the phone -> Mac -> SSH
tunnel -> server link can be verified in isolation. The Phase 1 job routes are
written in app/jobs.py and mounted only when XVIDEO_ENABLE_JOBS=1, so they never
break Phase 0 (e.g. when running off-server where `video_understanding` is absent).
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from typing import Any, AsyncIterator

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .auth import require_auth
from .devices import add_device, notify_job_done
from .settings import get_settings


_PUSH_TERMINAL_STATES = {"succeeded", "failed", "cancelled"}


async def _watch_completions(manager: Any, poll_seconds: float = 5.0) -> None:
    seen: set[str] = set()
    try:
        for job in manager.list_jobs(limit=100):
            job_id = job.get("job_id")
            if job_id and job.get("status") in _PUSH_TERMINAL_STATES:
                seen.add(str(job_id))
    except Exception as exc:  # pragma: no cover - startup should continue without push history.
        print(f"[push] seed error: {exc}", flush=True)

    while True:
        await asyncio.sleep(poll_seconds)
        try:
            for job in manager.list_jobs(limit=100):
                job_id = job.get("job_id")
                if (
                    job_id
                    and job.get("status") in _PUSH_TERMINAL_STATES
                    and job_id not in seen
                ):
                    seen.add(str(job_id))
                    await asyncio.to_thread(notify_job_done, job)
        except Exception as exc:  # pragma: no cover - watcher should not crash the BFF.
            print(f"[push] watcher error: {exc}", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.settings = settings
    app.state.job_manager = None
    app.state.push_watcher = None

    if settings.enable_jobs:
        from .jobs import create_manager

        app.state.job_manager = create_manager(settings)
        app.state.push_watcher = asyncio.create_task(
            _watch_completions(app.state.job_manager)
        )

    try:
        yield
    finally:
        watcher = getattr(app.state, "push_watcher", None)
        if watcher:
            watcher.cancel()
            with suppress(asyncio.CancelledError):
                await watcher
        manager = getattr(app.state, "job_manager", None)
        if manager and hasattr(manager, "shutdown"):
            manager.shutdown()


app = FastAPI(title="xVideo App BFF", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz", tags=["meta"])
async def healthz() -> dict[str, str]:
    """Unauthenticated liveness probe — used to smoke-test the SSH tunnel."""
    return {"status": "ok"}


@app.get("/ping", tags=["meta"], dependencies=[Depends(require_auth)])
async def ping() -> dict[str, bool]:
    """Authenticated probe — confirms the Bearer token reaches the server."""
    return {"authenticated": True}


class DeviceReg(BaseModel):
    push_token: str = Field(..., min_length=1, max_length=512)
    platform: str = Field(default="unknown", max_length=64)


@app.post("/devices", tags=["devices"], dependencies=[Depends(require_auth)])
async def register_device(body: DeviceReg) -> dict[str, bool]:
    await asyncio.to_thread(add_device, body.push_token, body.platform)
    return {"ok": True}


# --- Phase 1: job routes wrapping MCPJobManager (disabled until enabled) -------
_settings = get_settings()
if _settings.enable_jobs:
    from .jobs import router as jobs_router

    app.include_router(jobs_router)
