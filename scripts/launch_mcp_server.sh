#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONDA_ENV="${CONDA_ENV:-}"
TRANSPORT="${TRANSPORT:-streamable-http}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-9000}"
PATH_PREFIX="${PATH_PREFIX:-/mcp}"
JOB_ROOT="${JOB_ROOT:-runs/mcp_jobs}"
CONFIG="${CONFIG:-configs/pipeline.yaml}"
MAX_WORKERS="${MAX_WORKERS:-1}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
PYTHON_BIN="${PYTHON_BIN:-python}"

args=(
  -m video_understanding.mcp_server
  --transport "$TRANSPORT"
  --host "$HOST"
  --port "$PORT"
  --path "$PATH_PREFIX"
  --job-root "$JOB_ROOT"
  --config "$CONFIG"
  --max-workers "$MAX_WORKERS"
  --log-level "$LOG_LEVEL"
)

if [[ "${STATELESS_HTTP:-0}" == "1" ]]; then
  args+=(--stateless-http)
fi

if [[ "${KEEP_MEDIA:-0}" == "1" ]]; then
  args+=(--keep-media)
fi

cd "$ROOT_DIR"

if [[ -n "$CONDA_ENV" ]]; then
  exec conda run --no-capture-output -n "$CONDA_ENV" "$PYTHON_BIN" "${args[@]}"
fi

exec "$PYTHON_BIN" "${args[@]}"
