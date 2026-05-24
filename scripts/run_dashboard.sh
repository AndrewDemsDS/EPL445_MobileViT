#!/usr/bin/env bash
# Launch the FastAPI dashboard with the ROCm environment variables that
# the Radeon 780M (gfx1103) needs to load MobileViT and YOLO without
# segfaulting at startup.
#
# Usage:
#   bash scripts/run_dashboard.sh            # foreground, default port 8000
#   bash scripts/run_dashboard.sh 9000       # override the port
#   bash scripts/run_dashboard.sh 8000 cpu   # force stream to CPU

set -euo pipefail

PORT="${1:-8000}"
STREAM_DEVICE="${2:-cpu}"   # cpu (recommended for stability) or cuda

# gfx1103 is missing from rocBLAS's TensileLibrary. Spoof gfx1100.
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export HSA_XNACK=1

# Pin the live stream to CPU so a stream session and an offline job
# don't fight for the same iGPU VRAM (offline jobs still use the GPU).
export STREAM_DEVICE="$STREAM_DEVICE"

echo "═════════════════════════════════════════════════════════"
echo "  MobileViT Dashboard"
echo "  http://localhost:${PORT}"
echo "  Stream device: ${STREAM_DEVICE}   (offline jobs still use GPU)"
echo "═════════════════════════════════════════════════════════"

# Use the venv's uvicorn so we never accidentally pick up a system Python.
exec ./venv/bin/uvicorn src.app.main:app --host 0.0.0.0 --port "${PORT}"
