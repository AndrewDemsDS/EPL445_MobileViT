#!/usr/bin/env bash
# Train the MobileViT traffic classifier
set -euo pipefail

echo "═══ Training MobileViT Traffic Classifier ═══"
python -m src.training.train --config configs/train.yaml "$@"
