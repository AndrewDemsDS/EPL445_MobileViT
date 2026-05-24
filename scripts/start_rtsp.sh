#!/usr/bin/env bash
# Start a local RTSP server (mediamtx) that loops data/raw/traffic_long.mp4
# at rtsp://localhost:8554/traffic for the dashboard live-stream demo.
#
# Usage:
#   bash scripts/start_rtsp.sh           # foreground
#   bash scripts/start_rtsp.sh --detach  # background
#
# Stop with Ctrl+C in foreground, or `pkill mediamtx` after --detach.

set -euo pipefail

cd "$(dirname "$0")/.."

BIN="scripts/bin/mediamtx"
CFG="scripts/mediamtx.yml"
VID="data/raw/traffic_long.mp4"

if [[ ! -x "$BIN" ]]; then
  echo "mediamtx binary not found at $BIN"
  echo "Download once:"
  echo "  curl -sL -o /tmp/m.tgz https://github.com/bluenviron/mediamtx/releases/download/v1.9.0/mediamtx_v1.9.0_linux_amd64.tar.gz"
  echo "  mkdir -p scripts/bin && tar -xz -C scripts/bin -f /tmp/m.tgz mediamtx"
  exit 1
fi

if [[ ! -f "$VID" ]]; then
  echo "Demo video not found at $VID"
  echo "Build it with:"
  echo "  bash scripts/build_long_traffic.sh   # if you have one"
  echo "Or copy any local H.264 MP4 there and re-run."
  exit 1
fi

echo "════════════════════════════════════════════════════"
echo "  MediaMTX → rtsp://localhost:8554/traffic"
echo "  Source: $VID"
echo "  Stop:   Ctrl+C   (or pkill mediamtx if detached)"
echo "════════════════════════════════════════════════════"

if [[ "${1:-}" == "--detach" ]]; then
  nohup "$BIN" "$CFG" > /tmp/mediamtx.log 2>&1 &
  disown
  echo "Started in background (PID $!). Logs: /tmp/mediamtx.log"
else
  exec "$BIN" "$CFG"
fi
