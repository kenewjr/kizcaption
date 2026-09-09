@echo off
cd /d "%~dp0"
title LumaCaption
echo Menjalankan LumaCaption...
.\.venv\Scripts\python.exe main.py
if %errorlevel% neq 0 (
    echo.
    echo Aplikasi berhenti dengan error code %errorlevel%.
    pause
)
