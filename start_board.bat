@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
)

echo Starting GrowingNN Board at http://127.0.0.1:8765
echo Press Ctrl+C to stop.
echo.

python "%~dp0growingnn_board\run_server.py"
if errorlevel 1 (
    echo.
    echo Board failed to start. Install deps: pip install -r growingnn_board\requirements.txt
    pause
)
