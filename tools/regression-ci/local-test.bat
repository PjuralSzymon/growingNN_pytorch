@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem Start the regression CI image on this PC and check /healthz.
rem Then follow LOCAL_TEST.md: POST a SHA+PR, watch the GitHub comment and http://127.0.0.1:8080
rem
rem GitHub Actions cannot reach localhost. This script does not trigger a GitHub workflow.
rem You POST /v1/jobs yourself. The container still comments on a real PR.

docker info >nul 2>&1
if errorlevel 1 (
    echo Docker is not running.
    echo Start Docker Desktop, wait until it is ready, then run this script again.
    exit /b 1
)

if not exist ".env" (
    copy /y ".env.example" ".env" >nul
    echo Created .env from .env.example
    echo Fill GITHUB_TOKEN, GITHUB_REPO, CI_SHARED_SECRET, DASHBOARD_PASSWORD
    echo then run this script again.
    exit /b 1
)

echo Building and starting growingnn-regression-ci on port 8080 ...
docker compose up -d --build
if errorlevel 1 (
    echo docker compose up failed.
    exit /b 1
)

echo Waiting for /healthz ...
set "OK="
for /L %%I in (1,1,30) do (
    curl.exe -fsS http://127.0.0.1:8080/healthz >nul 2>&1
    if not errorlevel 1 (
        set "OK=1"
        goto :healthy
    )
    timeout /t 2 /nobreak >nul
)

:healthy
if not defined OK (
    echo Container did not become healthy. Logs:
    docker compose logs --tail 80
    exit /b 1
)

curl.exe -fsS http://127.0.0.1:8080/healthz
echo.
echo.
echo Site:     http://127.0.0.1:8080
echo Logs:     docker compose logs -f
echo Next:     .\trigger-job.bat  (asks for SHA and PR)
echo Stop:     docker compose down
exit /b 0
