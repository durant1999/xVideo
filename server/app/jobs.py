"""Phase 1 — REST/SSE routes that wrap the existing MCPJobManager.

This is a thin adapter: it does NOT reimplement queuing, persistence, cancellation
or QA — all of that already lives in `video_understanding.mcp_jobs.MCPJobManager`,
the same class the repo's MCP server uses. We just translate HTTP <-> manager calls.

Mounted only when XVIDEO_ENABLE_JOBS=1 (see app/main.py), because importing
`video_understanding` only works on the GPU server's `vedio_understand` env.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, NoReturn

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .auth import require_auth
from .settings import get_settings

router = APIRouter(prefix="/jobs", tags=["jobs"], dependencies=[Depends(require_auth)])

TERMINAL_STATES = {"succeeded", "failed", "cancelled", "unknown"}
HEARTBEAT_SECONDS = 15.0
logger = logging.getLogger(__name__)


def create_manager(settings: Any | None = None) -> Any:
    """Build a shared MCPJobManager during FastAPI lifespan startup."""
    try:
        from video_understanding.mcp_jobs import MCPJobManager  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on server env
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Could not import video_understanding.mcp_jobs. Run the BFF inside the "
                f"'vedio_understand' conda env on the GPU server. Original error: {exc}"
            ),
        ) from exc

    settings = settings or get_settings()
    return MCPJobManager(
        repo_root=settings.repo_root,
        job_root=settings.job_root,
        config_path=settings.config_path,
        max_workers=1,
    )


def get_manager(request: Request) -> Any:
    manager = getattr(request.app.state, "job_manager", None)
    if manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Job routes are enabled but the job manager is not initialized.",
        )
    return manager


def _is_pipeline_error(exc: Exception) -> bool:
    return exc.__class__.__name__ == "PipelineError"


def _raise_manager_error(
    exc: Exception,
    *,
    client_status: int,
    server_detail: str,
) -> NoReturn:
    if isinstance(exc, HTTPException):
        raise exc
    if isinstance(exc, ValueError) or _is_pipeline_error(exc):
        raise HTTPException(status_code=client_status, detail=str(exc)) from exc
    logger.exception(server_detail)
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=server_detail,
    ) from exc


class JobSubmit(BaseModel):
    source: str = Field(..., description="Video URL, local path, or text.")
    question: str | None = None
    skip_summary: bool = False
    extra_prompt: str | None = None
    fps: float | None = Field(default=None, ge=0.05, le=5.0)
    segment_seconds: float | None = Field(default=None, ge=5, le=300)
    max_side: int | None = Field(default=None, ge=320, le=2048)


class AskBody(BaseModel):
    question: str = Field(..., min_length=1)
    max_chars: int = Field(default=20000, ge=1, le=200000)


def _state_or_404(manager: Any, job_id: str) -> dict[str, Any]:
    try:
        return manager.public_state(job_id)
    except Exception as exc:
        _raise_manager_error(
            exc,
            client_status=status.HTTP_404_NOT_FOUND,
            server_detail="Unexpected error while reading job state.",
        )


@router.post("")
async def submit_job(
    body: JobSubmit, manager: Any = Depends(get_manager)
) -> dict[str, Any]:
    try:
        return await run_in_threadpool(
            manager.submit_video_job,
            source=body.source,
            question=body.question,
            skip_summary=body.skip_summary,
            extra_prompt=body.extra_prompt,
            fps=body.fps,
            segment_seconds=body.segment_seconds,
            max_side=body.max_side,
        )
    except Exception as exc:
        _raise_manager_error(
            exc,
            client_status=status.HTTP_400_BAD_REQUEST,
            server_detail="Unexpected error while submitting job.",
        )


@router.get("")
async def list_jobs(
    limit: int = Query(default=20, ge=1, le=100),
    manager: Any = Depends(get_manager),
) -> list[dict[str, Any]]:
    return await run_in_threadpool(manager.list_jobs, limit=limit)


@router.get("/{job_id}")
async def get_job(job_id: str, manager: Any = Depends(get_manager)) -> dict[str, Any]:
    return await run_in_threadpool(_state_or_404, manager, job_id)


@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str, manager: Any = Depends(get_manager)) -> dict[str, Any]:
    try:
        return await run_in_threadpool(manager.cancel_job, job_id)
    except Exception as exc:
        _raise_manager_error(
            exc,
            client_status=status.HTTP_404_NOT_FOUND,
            server_detail="Unexpected error while cancelling job.",
        )


@router.post("/{job_id}/ask")
async def ask_job(
    job_id: str,
    body: AskBody,
    manager: Any = Depends(get_manager),
) -> dict[str, Any]:
    """Follow-up Q&A against the cached fused context — slow (runs a subprocess)."""
    try:
        return await run_in_threadpool(
            manager.ask_video, job_id, body.question, max_chars=body.max_chars
        )
    except Exception as exc:
        _raise_manager_error(
            exc,
            client_status=status.HTTP_400_BAD_REQUEST,
            server_detail="Unexpected error while asking over job context.",
        )


@router.get("/{job_id}/artifact/{name}")
async def read_artifact(
    job_id: str,
    name: str,
    max_chars: int = Query(default=20000, ge=1, le=200000),
    manager: Any = Depends(get_manager),
) -> dict[str, Any]:
    """Read a text artifact (summary / context / *.jsonl / job.log) via the allow-list."""
    try:
        return await run_in_threadpool(
            manager.read_artifact, job_id, name, max_chars=max_chars
        )
    except Exception as exc:
        _raise_manager_error(
            exc,
            client_status=status.HTTP_400_BAD_REQUEST,
            server_detail="Unexpected error while reading job artifact.",
        )


@router.get("/{job_id}/events")
async def job_events(
    job_id: str,
    request: Request,
    manager: Any = Depends(get_manager),
) -> StreamingResponse:
    """SSE stream: poll public_state and push on change until a terminal state."""
    async def event_stream():
        last_signature: str | None = None
        last_heartbeat = time.monotonic()
        while True:
            if await request.is_disconnected():
                break
            try:
                state = await run_in_threadpool(manager.public_state, job_id)
            except Exception as exc:
                yield f"event: error\ndata: {json.dumps({'detail': str(exc)})}\n\n"
                break

            signature = f"{state.get('status')}|{state.get('updated_at')}"
            if signature != last_signature:
                last_signature = signature
                last_heartbeat = time.monotonic()
                yield f"data: {json.dumps(state, ensure_ascii=False)}\n\n"
            elif time.monotonic() - last_heartbeat >= HEARTBEAT_SECONDS:
                last_heartbeat = time.monotonic()
                heartbeat = {
                    "job_id": job_id,
                    "status": state.get("status"),
                    "updated_at": state.get("updated_at"),
                }
                payload = json.dumps(heartbeat, ensure_ascii=False)
                yield f"event: heartbeat\ndata: {payload}\n\n"

            if state.get("status") in TERMINAL_STATES:
                break
            await asyncio.sleep(1.5)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
