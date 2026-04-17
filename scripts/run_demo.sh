#!/usr/bin/env bash
# Run video inference demo
set -euo pipefail

echo "═══ MobileViT Traffic Demo — Video Inference ═══"
python -m src.inference.predict_video --config configs/demo.yaml "$@"

# Aggregate counts
echo ""
echo "═══ Aggregating class counts ═══"
python -m src.inference.aggregate_counts \
    --csv outputs/predictions/frame_predictions.csv \
    --output outputs/predictions/class_counts.json
