@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

rem Ask for commit SHA and PR number, read CI_SHARED_SECRET from .env, POST /v1/jobs.

if not exist ".env" (
    echo Missing .env in tools\regression-ci
    echo Copy .env.example to .env and fill it first.
    exit /b 1
)

set "CI_SHARED_SECRET="
for /f "usebackq eol=# tokens=1,* delims==" %%A in (".env") do (
    if /i "%%A"=="CI_SHARED_SECRET" set "CI_SHARED_SECRET=%%B"
)
if "!CI_SHARED_SECRET!"=="" (
    echo CI_SHARED_SECRET is empty in .env
    exit /b 1
)

curl.exe -fsS http://127.0.0.1:8080/healthz >nul 2>&1
if errorlevel 1 (
    echo Worker is not up at http://127.0.0.1:8080
    echo Run local-test.bat first.
    exit /b 1
)

set /p SHA="Commit SHA: "
set /p PR="PR number: "
if "!SHA!"=="" (
    echo SHA is required.
    exit /b 1
)
if "!PR!"=="" (
    echo PR number is required.
    exit /b 1
)

echo Posting job for sha=!SHA! pr=!PR! ...
curl.exe -fsS -X POST http://127.0.0.1:8080/v1/jobs ^
  -H "Authorization: Bearer !CI_SHARED_SECRET!" ^
  -H "Content-Type: application/json" ^
  -d "{\"sha\":\"!SHA!\",\"pr\":!PR!}"
echo.
echo.
echo Watch logs:  docker compose logs -f
echo Timeline:    http://127.0.0.1:8080
exit /b 0
