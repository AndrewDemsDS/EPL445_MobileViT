@echo off
REM Run video inference demo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_demo.ps1" %*
