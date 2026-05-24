# Train the MobileViT traffic classifier (Windows / PowerShell).
#
# Windows users typically have NVIDIA CUDA or CPU — the HSA_* env vars from
# the Linux wrapper are AMD-ROCm-on-Linux only, so we skip them here.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (Test-Path "venv/Scripts/python.exe") {
    $Py = ".\venv\Scripts\python.exe"
} else {
    $Py = "python"
}

Write-Host "=== Training MobileViT Traffic Classifier ==="
& $Py -m src.training.train --config configs/train.yaml @args
