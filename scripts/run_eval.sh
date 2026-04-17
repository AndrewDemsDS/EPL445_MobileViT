#!/usr/bin/env bash
# Evaluate the trained model on the test set
set -euo pipefail

echo "═══ Evaluating MobileViT Traffic Classifier ═══"
python -m src.evaluation.evaluate --config configs/eval.yaml "$@"
