@echo off
cd /d "%~dp0"
title KizCaption
echo Menjalankan KizCaption by kenewjr 2026...
.\.venv\Scripts\python.exe main.py
if %errorlevel% neq 0 (
    echo.
    echo Aplikasi berhenti dengan error code %errorlevel%.
    pause
)
