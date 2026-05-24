# Run video inference demo (Windows / PowerShell).
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (Test-Path "venv/Scripts/python.exe") {
    $Py = ".\venv\Scripts\python.exe"
} else {
    $Py = "python"
}

Write-Host "=== MobileViT Traffic Demo - Video Inference ==="
& $Py -m src.inference.predict_video --config configs/demo.yaml @args

Write-Host ""
Write-Host "=== Aggregating class counts ==="
& $Py -m src.inference.aggregate_counts `
    --csv outputs/predictions/frame_predictions.csv `
    --output outputs/predictions/class_counts.json
