@echo off
REM Train the MobileViT traffic classifier.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_train.ps1" %*
