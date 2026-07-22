@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
    set "PYTHON=py"
) else (
    set "PYTHON=python"
)

%PYTHON% -c "import pyvis" >nul 2>nul
if errorlevel 1 (
    echo Installing the PyVis graph dependency...
    %PYTHON% -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo Dependency installation failed.
        pause
        exit /b 1
    )
)

echo Building GrowingNN documentation...
%PYTHON% build.py
if errorlevel 1 (
    echo.
    echo Build failed. Make sure Python 3.9 or newer is installed.
    pause
    exit /b 1
)

echo.
echo Documentation is available at http://localhost:8000
echo Press Ctrl+C to stop the server.
start "" "http://localhost:8000"
%PYTHON% -m http.server 8000 --directory dist
