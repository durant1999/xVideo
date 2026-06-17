from __future__ import annotations

import json
import os
import signal
import shutil
import subprocess
import sys
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .utils import PipelineError, ensure_dir, read_text, write_text


ALLOWED_ARTIFACTS = {
    "summary": "work/summary.md",
    "summary.md": "work/summary.md",
    "context": "work/context.md",
    "context.md": "work/context.md",
    "visual": "work/visual.jsonl",
    "visual.jsonl": "work/visual.jsonl",
    "asr": "work/asr.jsonl",
    "asr.jsonl": "work/asr.jsonl",
    "fused": "work/fused.jsonl",
    "fused.jsonl": "work/fused.jsonl",
    "log": "job.log",
    "job.log": "job.log",
    "download_metadata": "work/source/download_metadata.json",
    "download_metadata.json": "work/source/download_metadata.json",
}

MEDIA_CLEANUP_RELATIVE_PATHS = (
    "work/frames",
    "work/audio.wav",
)
SOURCE_KEEP_FILES = {"download_metadata.json"}


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def new_job_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:10]}"


def validate_job_id(job_id: str) -> str:
    if not job_id or len(job_id) > 80:
        raise PipelineError("job_id is required and must be at most 80 characters")
    if not all(char.isalnum() or char in "-_" for char in job_id):
        raise PipelineError(f"Invalid job_id: {job_id}")
    return job_id


def clamp_optional_float(
    value: float | None,
    *,
    name: str,
    minimum: float,
    maximum: float,
) -> float | None:
    if value is None:
        return None
    value = float(value)
    if value < minimum or value > maximum:
        raise PipelineError(f"{name} must be between {minimum} and {maximum}")
    return value


def clamp_optional_int(
    value: int | None,
    *,
    name: str,
    minimum: int,
    maximum: int,
) -> int | None:
    if value is None:
        return None
    value = int(value)
    if value < minimum or value > maximum:
        raise PipelineError(f"{name} must be between {minimum} and {maximum}")
    return value


def path_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file() or path.is_symlink():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file() or child.is_symlink():
            try:
                total += child.stat().st_size
            except OSError:
                continue
    return total


class MCPJobManager:
    """Persistent, filesystem-backed job manager for the MCP server.

    The manager deliberately keeps job workdirs under a single root and only
    exposes a small artifact allow list. That prevents MCP callers from choosing
    arbitrary output paths or reading arbitrary server files.
    """

    def __init__(
        self,
        *,
        repo_root: str | Path,
        job_root: str | Path = "runs/mcp_jobs",
        config_path: str | Path | None = "configs/pipeline.yaml",
        python_executable: str | Path | None = None,
        max_workers: int = 1,
        execute_jobs: bool = True,
        cleanup_media_on_success: bool = True,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        job_root_path = Path(job_root)
        self.job_root = (
            (self.repo_root / job_root_path).resolve()
            if not job_root_path.is_absolute()
            else job_root_path.resolve()
        )
        self.config_path = (
            None
            if config_path is None
            else (self.repo_root / config_path).resolve()
            if not Path(config_path).is_absolute()
            else Path(config_path).resolve()
        )
        self.python_executable = str(python_executable or sys.executable)
        self.execute_jobs = execute_jobs
        self.cleanup_media_on_success = bool(cleanup_media_on_success)
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=max(1, int(max_workers)))
        self._futures: dict[str, Future[Any]] = {}
        self._processes: dict[str, subprocess.Popen[str]] = {}
        ensure_dir(self.job_root)
        self._mark_stale_active_jobs()

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=False)

    def _job_dir(self, job_id: str) -> Path:
        validate_job_id(job_id)
        return self.job_root / job_id

    def _state_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "state.json"

    def _read_state_unlocked(self, job_id: str) -> dict[str, Any]:
        path = self._state_path(job_id)
        if not path.exists():
            raise PipelineError(f"Job not found: {job_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def read_state(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            return self._read_state_unlocked(job_id)

    def _write_state_unlocked(self, state: dict[str, Any]) -> dict[str, Any]:
        state["updated_at"] = utc_now()
        job_id = validate_job_id(str(state["job_id"]))
        job_dir = ensure_dir(self._job_dir(job_id))
        target = job_dir / "state.json"
        temp = job_dir / "state.json.tmp"
        temp.write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temp.replace(target)
        return state

    def _update_state(self, job_id: str, **updates: Any) -> dict[str, Any]:
        with self._lock:
            state = self._read_state_unlocked(job_id)
            state.update(updates)
            return self._write_state_unlocked(state)

    def _mark_stale_active_jobs(self) -> None:
        for state_path in self.job_root.glob("*/state.json"):
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if state.get("status") in {"queued", "running", "cancelling"}:
                state["status"] = "unknown"
                state["error"] = (
                    "MCP server restarted while this job was active; inspect job.log and "
                    "rerun the job if needed."
                )
                state["updated_at"] = utc_now()
                state_path.write_text(
                    json.dumps(state, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )

    def build_run_command(
        self,
        *,
        source: str,
        workdir: Path,
        question: str | None,
        skip_summary: bool,
        extra_prompt: str | None,
        fps: float | None,
        segment_seconds: float | None,
        max_side: int | None,
    ) -> list[str]:
        command = [
            self.python_executable,
            "-m",
            "video_understanding",
            "run",
            source,
            "--workdir",
            str(workdir),
        ]
        if self.config_path and self.config_path.exists():
            command.extend(["--config", str(self.config_path)])
        if question:
            command.extend(["--question", question])
        if skip_summary:
            command.append("--skip-summary")
        if extra_prompt:
            command.extend(["--prompt", extra_prompt])
        if fps is not None:
            command.extend(["--fps", str(fps)])
        if segment_seconds is not None:
            command.extend(["--segment-seconds", str(segment_seconds)])
        if max_side is not None:
            command.extend(["--max-side", str(max_side)])
        return command

    def submit_video_job(
        self,
        *,
        source: str,
        question: str | None = None,
        skip_summary: bool = False,
        extra_prompt: str | None = None,
        fps: float | None = None,
        segment_seconds: float | None = None,
        max_side: int | None = None,
    ) -> dict[str, Any]:
        source = str(source).strip()
        if not source:
            raise PipelineError("source is required")
        if len(source) > 8000:
            raise PipelineError("source is too long; pass a URL, share text, or server-local path")
        fps = clamp_optional_float(fps, name="fps", minimum=0.05, maximum=5.0)
        segment_seconds = clamp_optional_float(
            segment_seconds,
            name="segment_seconds",
            minimum=5.0,
            maximum=300.0,
        )
        max_side = clamp_optional_int(max_side, name="max_side", minimum=320, maximum=2048)

        job_id = new_job_id()
        job_dir = ensure_dir(self._job_dir(job_id))
        workdir = ensure_dir(job_dir / "work")
        log_path = job_dir / "job.log"
        command = self.build_run_command(
            source=source,
            workdir=workdir,
            question=question,
            skip_summary=skip_summary,
            extra_prompt=extra_prompt,
            fps=fps,
            segment_seconds=segment_seconds,
            max_side=max_side,
        )
        state = {
            "job_id": job_id,
            "status": "queued",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "source": source,
            "question": question,
            "skip_summary": bool(skip_summary),
            "workdir": str(workdir),
            "log_path": str(log_path),
            "command": command,
            "cleanup_media_on_success": self.cleanup_media_on_success,
            "artifacts": self.expected_artifacts(job_id),
        }
        with self._lock:
            self._write_state_unlocked(state)
            if self.execute_jobs:
                self._futures[job_id] = self._executor.submit(
                    self._run_job,
                    job_id,
                    command,
                    log_path,
                )
        return self.public_state(job_id)

    def _run_job(self, job_id: str, command: list[str], log_path: Path) -> None:
        self._update_state(job_id, status="running", started_at=utc_now(), pid=None)
        ensure_dir(log_path.parent)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"$ {' '.join(command)}\n")
            log.flush()
            try:
                process = subprocess.Popen(
                    command,
                    cwd=str(self.repo_root),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                with self._lock:
                    self._processes[job_id] = process
                self._update_state(job_id, pid=process.pid)
                returncode = process.wait()
            except Exception as exc:  # noqa: BLE001 - persist operational failure for caller.
                self._update_state(job_id, status="failed", error=str(exc), finished_at=utc_now())
                return
            finally:
                with self._lock:
                    self._processes.pop(job_id, None)

        state = self.read_state(job_id)
        if state.get("status") == "cancelling":
            self._update_state(job_id, status="cancelled", returncode=returncode, finished_at=utc_now())
        elif returncode == 0:
            cleanup_result: dict[str, Any] = {"enabled": False}
            if self.cleanup_media_on_success:
                try:
                    cleanup_result = self.cleanup_media_assets(job_id)
                except Exception as exc:  # noqa: BLE001 - cleanup should not fail a good job.
                    cleanup_result = {
                        "enabled": True,
                        "error": str(exc),
                        "deleted_paths": [],
                        "bytes_freed": 0,
                    }
            self._update_state(
                job_id,
                status="succeeded",
                returncode=returncode,
                finished_at=utc_now(),
                artifacts=self.expected_artifacts(job_id),
                media_cleanup=cleanup_result,
            )
        else:
            self._update_state(
                job_id,
                status="failed",
                returncode=returncode,
                finished_at=utc_now(),
                error=f"Pipeline exited with status {returncode}; inspect job.log.",
                artifacts=self.expected_artifacts(job_id),
            )

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        state = self.read_state(job_id)
        if state.get("status") in {"succeeded", "failed", "cancelled", "unknown"}:
            return self.public_state(job_id)
        self._update_state(job_id, status="cancelling")
        with self._lock:
            future = self._futures.get(job_id)
            process = self._processes.get(job_id)
        if process and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        elif future and future.cancel():
            self._update_state(job_id, status="cancelled", finished_at=utc_now())
        return self.public_state(job_id)

    def expected_artifacts(self, job_id: str) -> dict[str, str]:
        job_dir = self._job_dir(job_id)
        return {name: str(job_dir / relative) for name, relative in ALLOWED_ARTIFACTS.items()}

    def public_state(self, job_id: str) -> dict[str, Any]:
        state = self.read_state(job_id)
        status = {
            key: state.get(key)
            for key in (
                "job_id",
                "status",
                "created_at",
                "updated_at",
                "started_at",
                "finished_at",
                "pid",
                "returncode",
                "source",
                "question",
                "skip_summary",
                "workdir",
                "log_path",
                "error",
                "cleanup_media_on_success",
                "media_cleanup",
            )
            if key in state
        }
        status["artifacts"] = {
            name: path for name, path in state.get("artifacts", {}).items() if name in ALLOWED_ARTIFACTS
        }
        return status

    def list_jobs(self, *, limit: int = 20) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 100))
        states: list[dict[str, Any]] = []
        for state_path in self.job_root.glob("*/state.json"):
            try:
                states.append(json.loads(state_path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                continue
        states.sort(
            key=lambda row: (str(row.get("created_at", "")), str(row.get("job_id", ""))),
            reverse=True,
        )
        return [self.public_state(str(state["job_id"])) for state in states[:limit] if state.get("job_id")]

    def _safe_job_path(self, job_id: str, relative_path: str) -> Path:
        job_dir = self._job_dir(job_id).resolve()
        path = (job_dir / relative_path).resolve()
        if not path.is_relative_to(job_dir):
            raise PipelineError("Resolved cleanup path escaped the job directory")
        return path

    def _delete_path(self, path: Path) -> int:
        if not path.exists():
            return 0
        size = path_size_bytes(path)
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
        return size

    def cleanup_media_assets(self, job_id: str) -> dict[str, Any]:
        """Delete heavy intermediate media while preserving text artifacts and metadata."""
        deleted_paths: list[str] = []
        bytes_freed = 0

        for relative_path in MEDIA_CLEANUP_RELATIVE_PATHS:
            path = self._safe_job_path(job_id, relative_path)
            if path.exists():
                bytes_freed += self._delete_path(path)
                deleted_paths.append(str(path))

        source_dir = self._safe_job_path(job_id, "work/source")
        if source_dir.exists() and source_dir.is_dir():
            for child in sorted(source_dir.iterdir()):
                if child.name in SOURCE_KEEP_FILES:
                    continue
                bytes_freed += self._delete_path(child)
                deleted_paths.append(str(child))

        return {
            "enabled": True,
            "deleted_paths": deleted_paths,
            "bytes_freed": bytes_freed,
        }

    def artifact_path(self, job_id: str, artifact: str) -> Path:
        artifact_key = str(artifact).strip()
        if artifact_key not in ALLOWED_ARTIFACTS:
            allowed = ", ".join(sorted(ALLOWED_ARTIFACTS))
            raise PipelineError(f"Unsupported artifact '{artifact}'. Allowed values: {allowed}")
        path = (self._job_dir(job_id) / ALLOWED_ARTIFACTS[artifact_key]).resolve()
        job_dir = self._job_dir(job_id).resolve()
        if not path.is_relative_to(job_dir):
            raise PipelineError("Resolved artifact path escaped the job directory")
        return path

    def read_artifact(self, job_id: str, artifact: str, *, max_chars: int = 20000) -> dict[str, Any]:
        max_chars = max(1, min(int(max_chars), 200000))
        path = self.artifact_path(job_id, artifact)
        if not path.exists():
            return {
                "job_id": job_id,
                "artifact": artifact,
                "path": str(path),
                "exists": False,
                "content": "",
                "truncated": False,
                "size_bytes": 0,
            }
        size_bytes = path.stat().st_size
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            text = handle.read(max_chars + 1)
        truncated = len(text) > max_chars
        return {
            "job_id": job_id,
            "artifact": artifact,
            "path": str(path),
            "exists": True,
            "content": text[:max_chars],
            "truncated": truncated,
            "size_bytes": size_bytes,
        }

    def ask_video(self, job_id: str, question: str, *, max_chars: int = 20000) -> dict[str, Any]:
        question = str(question).strip()
        if not question:
            raise PipelineError("question is required")
        context_path = self.artifact_path(job_id, "context.md")
        if not context_path.exists():
            raise PipelineError(f"context.md does not exist for job {job_id}; wait for fusion to finish")
        output_name = f"qa_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}.md"
        output_path = self._job_dir(job_id) / "work" / output_name
        command = [
            self.python_executable,
            "-m",
            "video_understanding",
            "summarize",
            "--context",
            str(context_path),
            "--output",
            str(output_path),
            "--question",
            question,
        ]
        if self.config_path and self.config_path.exists():
            command.extend(["--config", str(self.config_path)])
        result = subprocess.run(
            command,
            cwd=str(self.repo_root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise PipelineError(f"QA failed: {detail}")
        content = output_path.read_text(encoding="utf-8") if output_path.exists() else result.stdout
        truncated = len(content) > max_chars
        return {
            "job_id": job_id,
            "question": question,
            "path": str(output_path),
            "content": content[:max_chars],
            "truncated": truncated,
        }


def write_mcp_readme(job_root: str | Path) -> None:
    target = Path(job_root) / "README.txt"
    if target.exists():
        return
    write_text(
        target,
        "Video Understanding MCP job directory. Each job is isolated under its own "
        "subdirectory with state.json, job.log, and work/ artifacts.\n",
    )
