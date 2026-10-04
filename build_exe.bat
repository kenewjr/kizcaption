@echo off
cd /d "%~dp0"
title Build KizCaption Standalone .EXE
echo ======================================================
echo  Membangun KizCaption.exe dengan PyInstaller...
echo ======================================================
echo.

.\.venv\Scripts\pyinstaller.exe KizCaption.spec --noconfirm
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Gagal melakukan build PyInstaller.
    pause
    exit /b %errorlevel%
)

echo.
echo Menyalin file template dan aset runtime ke folder dist\KizCaption...
if not exist "dist\KizCaption\output" mkdir "dist\KizCaption\output"
if not exist "dist\KizCaption\models" mkdir "dist\KizCaption\models"

copy /y "output\overlay.html" "dist\KizCaption\output\" >nul
copy /y "models\silero_vad.onnx" "dist\KizCaption\models\" >nul
copy /y "vocabulary.json" "dist\KizCaption\" >nul
copy /y "config.example.json" "dist\KizCaption\" >nul

if exist "dist\KizCaption\_internal\nvidia\cublas\bin\cublas64_12.dll" (
    copy /y "dist\KizCaption\_internal\nvidia\cublas\bin\cublas64_12.dll" "dist\KizCaption\" >nul
    copy /y "dist\KizCaption\_internal\nvidia\cublas\bin\cublasLt64_12.dll" "dist\KizCaption\" >nul
    echo [GPU] cublas64_12.dll dan cublasLt64_12.dll berhasil disalin ke folder KizCaption!
) else if exist ".venv\Lib\site-packages\nvidia\cublas\bin\cublas64_12.dll" (
    copy /y ".venv\Lib\site-packages\nvidia\cublas\bin\cublas64_12.dll" "dist\KizCaption\" >nul
    copy /y ".venv\Lib\site-packages\nvidia\cublas\bin\cublasLt64_12.dll" "dist\KizCaption\" >nul
    echo [GPU] cublas64_12.dll dan cublasLt64_12.dll berhasil disalin dari .venv!
)

echo.
echo ======================================================
echo  BUILD SUKSES!
echo  Executable berada di: dist\KizCaption\KizCaption.exe
echo ======================================================
echo.
pause
