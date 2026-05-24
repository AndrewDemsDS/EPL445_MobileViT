@echo off
REM Evaluate the trained model on the test set.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_eval.ps1" %*
