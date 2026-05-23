#!/usr/bin/env bash
# Download the demo traffic clips from the GitHub release.
# Skips files already on disk.
set -euo pipefail
cd "$(dirname "$0")/.."

REPO="AndrewDemsDS/EPL445_MobileViT"
TAG="media-v1"

mkdir -p data/raw outputs/predictions

fetch() {
    local asset="$1" dest="$2"
    if [[ -f "$dest" ]]; then
        echo "✓ $dest already present, skipping"
        return
    fi
    echo "↓ $asset → $dest"
    curl -L --fail --progress-bar \
        -o "$dest" \
        "https://github.com/${REPO}/releases/download/${TAG}/${asset}"
}

fetch sample_traffic.mp4       data/raw/sample_traffic.mp4
fetch traffic_long.mp4         data/raw/traffic_long.mp4
fetch web_annotated_output.mp4 outputs/predictions/web_annotated_output.mp4

echo "Done."
