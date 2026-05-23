# Download the demo traffic clips from the GitHub release (Windows / PowerShell).
# Skips files already on disk.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$Repo = "AndrewDemsDS/EPL445_MobileViT"
$Tag  = "media-v1"

New-Item -ItemType Directory -Force -Path "data/raw"            | Out-Null
New-Item -ItemType Directory -Force -Path "outputs/predictions" | Out-Null

function Get-Asset($Asset, $Dest) {
    if (Test-Path $Dest) {
        Write-Host "OK $Dest already present, skipping"
        return
    }
    $Url = "https://github.com/$Repo/releases/download/$Tag/$Asset"
    Write-Host "Downloading $Asset -> $Dest"
    # Invoke-WebRequest is much slower than curl on PS5 because of the progress
    # bar; suppress it and prefer curl.exe when present (ships with Windows 10+).
    $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($curl) {
        & curl.exe -L --fail --progress-bar -o $Dest $Url
    } else {
        $ProgressPreference = "SilentlyContinue"
        Invoke-WebRequest -Uri $Url -OutFile $Dest
    }
}

Get-Asset "sample_traffic.mp4"       "data/raw/sample_traffic.mp4"
Get-Asset "traffic_long.mp4"         "data/raw/traffic_long.mp4"
Get-Asset "web_annotated_output.mp4" "outputs/predictions/web_annotated_output.mp4"

Write-Host "Done."
