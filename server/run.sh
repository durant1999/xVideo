#!/usr/bin/env bash
# Run the BFF on the GPU server inside the existing conda env.
# Usage:  ./run.sh        (foreground)
#         tmux new -s bff './run.sh'   (keep alive in a tmux session)
set -euo pipefail

cd "$(dirname "$0")"
SERVER_DIR="$(pwd)"
DEFAULT_REPO_ROOT="$(cd "$SERVER_DIR/.." && pwd)"

# Load .env (KEY=VALUE lines) into the environment.
set -a
[ -f .env ] && . ./.env
set +a

HOST="${XVIDEO_HOST:-127.0.0.1}"
PORT="${XVIDEO_PORT:-8788}"
CONDA_ENV="${CONDA_ENV:-vedio_understand}"
REPO_ROOT="${XVIDEO_REPO_ROOT:-$DEFAULT_REPO_ROOT}"
PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH

# Loopback-only by design — the SSH LocalForward bridges it to the Mac.
exec conda run --no-capture-output -n "$CONDA_ENV" \
  uvicorn app.main:app --host "$HOST" --port "$PORT"
