#!/usr/bin/env bash
# Evaluate the trained model on the test set
set -euo pipefail

export HSA_OVERRIDE_GFX_VERSION=11.0.0

echo "═══ Evaluating MobileViT Traffic Classifier ═══"
python -m src.evaluation.evaluate --config configs/eval.yaml "$@"
