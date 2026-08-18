@echo off
setlocal EnableExtensions

rem =============================================================================
rem Build GrowingNN regression CI image on this machine, then copy it to Hostinger.
rem Image name: growingnn-regression-ci
rem Archive:    growingnn-regression-ci.tar.gz  (no secrets inside)
rem
rem --- After this script finishes, on Hostinger (SSH) ---
rem
rem 1. Copy the archive to the VPS (run this on your PC, not on Hostinger):
rem      scp growingnn-regression-ci.tar.gz root@YOUR_VPS_IP:/root/
rem
rem 2. Also put the compose folder on the VPS (clone the repo, or copy
rem    tools/regression-ci/ including docker-compose.yml and .env.example).
rem      git clone https://github.com/OWNER/REPO.git
rem      cd REPO
rem
rem 3. Load the image (no docker build on the VPS):
rem      gzip -dc /root/growingnn-regression-ci.tar.gz | docker load
rem      docker images growingnn-regression-ci
rem
rem 4. Secrets stay on the VPS, never in the image:
rem      cd tools/regression-ci
rem      cp .env.example .env
rem      nano .env
rem        CI_SHARED_SECRET=...          same as GitHub secret GROWINGNN_CI_SECRET
rem        GITHUB_TOKEN=...              contents:read + pull-requests:write
rem        GITHUB_REPO=owner/repo
rem        DASHBOARD_PASSWORD=...
rem
rem 5. Start and leave it running (use the loaded image, do not --build):
rem      docker compose up -d
rem      docker compose ps
rem      curl http://127.0.0.1:8080/healthz
rem    You want {"status":"ok"}. restart: unless-stopped keeps it after reboot.
rem
rem 6. HTTPS: point something.hstgr.cloud at the VPS, proxy to port 8080.
rem    Origin must be https://something.hstgr.cloud  (no trailing slash)
rem
rem 7. GitHub repo Settings - Secrets and variables - Actions:
rem      GROWINGNN_CI_URL     = https://something.hstgr.cloud
rem      GROWINGNN_CI_SECRET  = same string as CI_SHARED_SECRET in .env
rem
rem 8. Test: open a PR to main. Trigger action should go green in seconds.
rem    Later a PR comment appears. Log in on the site with DASHBOARD_PASSWORD.
rem
rem Logs if the action is green but no comment:
rem      cd tools/regression-ci
rem      docker compose logs -f
rem =============================================================================

cd /d "%~dp0"

set "IMAGE=growingnn-regression-ci"
set "ARCHIVE=%~dp0growingnn-regression-ci.tar.gz"
set "TAR=%~dp0growingnn-regression-ci.tar"

echo Building %IMAGE% ...
docker compose build
if errorlevel 1 (
    echo docker compose build failed.
    exit /b 1
)

echo Saving %IMAGE% ...
if exist "%TAR%" del /f /q "%TAR%"
if exist "%ARCHIVE%" del /f /q "%ARCHIVE%"
docker save "%IMAGE%" -o "%TAR%"
if errorlevel 1 (
    echo docker save failed.
    exit /b 1
)

echo Compressing to gzip ...
where gzip >nul 2>&1
if not errorlevel 1 (
    gzip -f "%TAR%"
) else (
    tar -czf "%ARCHIVE%" -C "%~dp0" growingnn-regression-ci.tar
    if exist "%TAR%" del /f /q "%TAR%"
)

if not exist "%ARCHIVE%" (
    echo gzip failed. Install Git for Windows (gzip) or use tar.
    exit /b 1
)

echo.
echo Done. Archive:
echo   %ARCHIVE%
echo Copy it to Hostinger, then follow the rem comments at the top of this file.
exit /b 0
