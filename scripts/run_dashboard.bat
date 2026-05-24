@echo off
REM Launch the MobileViT traffic dashboard at http://localhost:8000
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_dashboard.ps1" %*
