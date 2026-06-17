from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .mcp_jobs import MCPJobManager, write_mcp_readme
from .utils import PipelineError


INSTRUCTIONS = """Video Understanding MCP exposes async tools for Chinese short-video analysis.
Use submit_video_job first; it returns quickly with a job_id while GPU work continues on the server.
Poll get_job_status, then read summary/context artifacts. Do not submit many concurrent jobs unless the server owner configured capacity.
Artifacts are restricted to each job directory; tools cannot read arbitrary server paths."""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_manager(args: argparse.Namespace) -> MCPJobManager:
    root = repo_root()
    job_root = Path(args.job_root)
    write_mcp_readme(root / job_root if not job_root.is_absolute() else job_root)
    return MCPJobManager(
        repo_root=root,
        job_root=job_root,
        config_path=args.config,
        python_executable=args.python,
        max_workers=args.max_workers,
        cleanup_media_on_success=not args.keep_media,
    )


def build_server(args: argparse.Namespace):
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise PipelineError(
            "The MCP SDK is not installed. Install with `pip install -e '.[mcp]'`."
        ) from exc

    manager = build_manager(args)
    mcp = FastMCP(
        "video-understanding",
        instructions=INSTRUCTIONS,
        host=args.host,
        port=args.port,
        streamable_http_path=args.path,
        stateless_http=args.stateless_http,
        log_level=args.log_level,
    )

    @mcp.tool()
    def get_server_info() -> dict[str, Any]:
        """Return server configuration and operational guidance."""
        return {
            "name": "video-understanding",
            "transport_path": args.path,
            "job_root": str(manager.job_root),
            "config_path": str(manager.config_path) if manager.config_path else None,
            "max_workers": args.max_workers,
            "python": manager.python_executable,
            "cleanup_media_on_success": manager.cleanup_media_on_success,
            "workflow": [
                "submit_video_job(source)",
                "poll get_job_status(job_id)",
                "read get_job_artifact(job_id, 'summary') or get_job_artifact(job_id, 'context')",
                "optionally ask_video(job_id, question)",
            ],
            "security": (
                "The server is intended to bind to localhost and be reached from a Mac through "
                "SSH/VPN. Do not expose it publicly without an authenticated reverse proxy."
            ),
        }

    @mcp.tool()
    def submit_video_job(
        source: str,
        question: str | None = None,
        skip_summary: bool = False,
        extra_prompt: str | None = None,
        fps: float | None = None,
        segment_seconds: float | None = None,
        max_side: int | None = None,
    ) -> dict[str, Any]:
        """Submit a URL, share text, or server-local video path for async analysis."""
        return manager.submit_video_job(
            source=source,
            question=question,
            skip_summary=skip_summary,
            extra_prompt=extra_prompt,
            fps=fps,
            segment_seconds=segment_seconds,
            max_side=max_side,
        )

    @mcp.tool()
    def get_job_status(job_id: str) -> dict[str, Any]:
        """Get status, timestamps, log path, and artifact paths for a submitted job."""
        return manager.public_state(job_id)

    @mcp.tool()
    def list_jobs(limit: int = 20) -> list[dict[str, Any]]:
        """List recent jobs, newest first."""
        return manager.list_jobs(limit=limit)

    @mcp.tool()
    def get_job_artifact(
        job_id: str,
        artifact: str = "summary",
        max_chars: int = 20000,
    ) -> dict[str, Any]:
        """Read an allowed job artifact such as summary, context, visual, asr, fused, or log."""
        return manager.read_artifact(job_id, artifact, max_chars=max_chars)

    @mcp.tool()
    def ask_video(job_id: str, question: str, max_chars: int = 20000) -> dict[str, Any]:
        """Answer a question over an already fused job context."""
        return manager.ask_video(job_id, question, max_chars=max_chars)

    @mcp.tool()
    def cancel_job(job_id: str) -> dict[str, Any]:
        """Request cancellation for a queued or running job."""
        return manager.cancel_job(job_id)

    return mcp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Video Understanding MCP server.")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="streamable-http",
        help="MCP transport. Use streamable-http for Mac Codex over SSH tunnel.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind host.")
    parser.add_argument("--port", type=int, default=9000, help="HTTP bind port.")
    parser.add_argument("--path", default="/mcp", help="Streamable HTTP MCP path.")
    parser.add_argument("--job-root", default="runs/mcp_jobs", help="Directory for MCP jobs.")
    parser.add_argument("--config", default="configs/pipeline.yaml", help="Pipeline config path.")
    parser.add_argument("--python", default=sys.executable, help="Python executable for pipeline jobs.")
    parser.add_argument("--max-workers", type=int, default=1, help="Maximum concurrent pipeline jobs.")
    parser.add_argument(
        "--keep-media",
        action="store_true",
        help="Keep source video, frames, and audio.wav after successful jobs.",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--stateless-http", action="store_true", help="Use stateless HTTP mode.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.transport == "streamable-http" and args.host not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        print(
            "warning: binding MCP HTTP to a non-local host. Put this behind SSH/VPN or an "
            "authenticated reverse proxy.",
            file=sys.stderr,
        )
    try:
        server = build_server(args)
        server.run(transport=args.transport)
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
