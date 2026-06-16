"""xVideo App BFF — FastAPI application factory.

Phase 0 ships only /healthz (open) and /ping (authed) so the phone -> Mac -> SSH
tunnel -> server link can be verified in isolation. The Phase 1 job routes are
written in app/jobs.py and mounted only when XVIDEO_ENABLE_JOBS=1, so they never
break Phase 0 (e.g. when running off-server where `video_understanding` is absent).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import Depends, FastAPI

from .auth import require_auth
from .settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.settings = settings
    app.state.job_manager = None

    if settings.enable_jobs:
        from .jobs import create_manager

        app.state.job_manager = create_manager(settings)

    try:
        yield
    finally:
        manager = getattr(app.state, "job_manager", None)
        if manager and hasattr(manager, "shutdown"):
            manager.shutdown()


app = FastAPI(title="xVideo App BFF", version="0.1.0", lifespan=lifespan)


@app.get("/healthz", tags=["meta"])
async def healthz() -> dict[str, str]:
    """Unauthenticated liveness probe — used to smoke-test the SSH tunnel."""
    return {"status": "ok"}


@app.get("/ping", tags=["meta"], dependencies=[Depends(require_auth)])
async def ping() -> dict[str, bool]:
    """Authenticated probe — confirms the Bearer token reaches the server."""
    return {"authenticated": True}


# --- Phase 1: job routes wrapping MCPJobManager (disabled until enabled) -------
_settings = get_settings()
if _settings.enable_jobs:
    from .jobs import router as jobs_router

    app.include_router(jobs_router)
