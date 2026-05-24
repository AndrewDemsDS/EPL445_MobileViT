# Windows / PowerShell wrapper around scripts/run_dashboard.py.
# All platform detection (CUDA / DirectML / CPU) lives in the Python launcher.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (Test-Path "venv/Scripts/python.exe") {
    $Py = ".\venv\Scripts\python.exe"
} else {
    $Py = "python"
}

& $Py scripts/run_dashboard.py @args
