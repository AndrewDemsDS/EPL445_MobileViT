# Evaluate the trained model on the test set (Windows / PowerShell).
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (Test-Path "venv/Scripts/python.exe") {
    $Py = ".\venv\Scripts\python.exe"
} else {
    $Py = "python"
}

Write-Host "=== Evaluating MobileViT Traffic Classifier ==="
& $Py -m src.evaluation.evaluate --config configs/eval.yaml @args
