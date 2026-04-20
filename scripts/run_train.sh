#!/usr/bin/env bash
# Train the MobileViT traffic classifier
set -euo pipefail

# gfx1103 (Radeon 780M) is not in the rocBLAS TensileLibrary bundled with PyTorch.
# Override to the nearest supported RDNA 3 target (gfx1100) so matrix ops work.
export HSA_OVERRIDE_GFX_VERSION=11.0.0
# Help ROCm allocator cope with shared-memory iGPU (Radeon 780M)
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
# Cap visible GTT memory so the kernel driver doesn't starve the rest of the system
export HSA_XNACK=1

echo "═══ Training MobileViT Traffic Classifier ═══"
python -m src.training.train --config configs/train.yaml "$@"
