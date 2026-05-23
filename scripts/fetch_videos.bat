@echo off
REM Download the demo traffic clips from the GitHub release.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0fetch_videos.ps1" %*
