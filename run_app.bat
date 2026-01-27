@echo off
REM Voice Intercom Streamlit App - Windows Startup Script
setlocal enabledelayedexpansion

echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║       🎤 Voice Intercom - Streamlit Application           ║
echo ╚════════════════════════════════════════════════════════════╝
echo.

REM Get script directory
cd /d "%~dp0"

echo 📍 Working directory: %cd%

REM Check if virtual environment exists
if not exist "venv" (
    echo ⚠️  Virtual environment not found!
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if requirements are installed
echo 📦 Checking dependencies...
pip install -q -r requirements.txt

REM Check if models exist
if not exist "models" (
    echo ❌ Error: models directory not found
    exit /b 1
)

for /f %%A in ('dir /b models\*.pth 2^>nul ^| find /c /v ""') do set MODEL_COUNT=%%A

if %MODEL_COUNT% equ 0 (
    echo ❌ Error: No trained models found in .\models\
    echo    Please ensure model checkpoints (*.pth files) are in the models directory
    exit /b 1
)

echo ✓ Found %MODEL_COUNT% trained models

echo.
echo ✓ Starting Streamlit application...
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo 🌐 Open your browser to: http://localhost:8501
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.

streamlit run app.py --client.toolbarMode=viewer

pause
