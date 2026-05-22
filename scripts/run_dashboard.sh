#!/usr/bin/env bash
# Thin Bash wrapper around scripts/run_dashboard.py for muscle memory.
# All real logic (platform detection, env vars, uvicorn path) lives in
# the Python launcher so Linux / macOS / Windows share one source of truth.
#
# Usage:
#   bash scripts/run_dashboard.sh                   # defaults
#   bash scripts/run_dashboard.sh --port 9000
#   bash scripts/run_dashboard.sh --stream-device cuda

set -euo pipefail
cd "$(dirname "$0")/.."

# Prefer the venv's Python so platform detection sees the same torch
# the FastAPI app will use.
if [[ -x "venv/bin/python" ]]; then
    PY="venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PY="python3"
else
    PY="python"
fi

exec "$PY" scripts/run_dashboard.py "$@"
