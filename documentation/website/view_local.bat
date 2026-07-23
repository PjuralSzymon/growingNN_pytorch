@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
    set "PYTHON=py"
) else (
    set "PYTHON=python"
)

where node >nul 2>nul
if errorlevel 1 (
    set "NODE_VERSION=24.18.0"
    set "NODE_HOME=%LOCALAPPDATA%\GrowingNN\node-v24.18.0-win-x64"
    if not exist "%LOCALAPPDATA%\GrowingNN\node-v24.18.0-win-x64\node.exe" (
        echo Installing portable Node.js 24...
        powershell -NoProfile -ExecutionPolicy Bypass -Command "$root='%LOCALAPPDATA%\GrowingNN'; New-Item -ItemType Directory -Force -Path $root | Out-Null; Invoke-WebRequest 'https://nodejs.org/dist/v24.18.0/node-v24.18.0-win-x64.zip' -OutFile ($root + '\node.zip'); Expand-Archive ($root + '\node.zip') $root -Force; Remove-Item ($root + '\node.zip')"
        if errorlevel 1 (
            echo Node.js installation failed.
            pause
            exit /b 1
        )
    )
    set "PATH=%LOCALAPPDATA%\GrowingNN\node-v24.18.0-win-x64;%PATH%"
)

%PYTHON% -c "import pyvis" >nul 2>nul
if errorlevel 1 (
    echo Installing PyVis...
    %PYTHON% -m pip install -r requirements.txt
    if errorlevel 1 (
        echo PyVis installation failed.
        pause
        exit /b 1
    )
)

cd app
if not exist node_modules (
    echo Installing Angular dependencies...
    call npm ci
    if errorlevel 1 (
        echo Angular dependency installation failed.
        pause
        exit /b 1
    )
)

echo.
echo Starting Angular documentation at http://localhost:4200
echo Press Ctrl+C to stop the server.
start "" "http://localhost:4200"
call npm start
