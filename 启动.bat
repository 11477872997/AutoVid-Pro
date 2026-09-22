@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title AutoVidDub Backend - Keep This Window Open
cls
echo ============================================================
echo   AutoVidDub Backend
echo   Keep this terminal open. Closing it stops the program.
echo ============================================================
echo.
echo Checking port 7860...
for /f "usebackq delims=" %%P in (`powershell -NoProfile -Command "$c=Get-NetTCPConnection -LocalPort 7860 -State Listen -ErrorAction SilentlyContinue; if($c){$c.OwningProcess | Sort-Object -Unique}"`) do (
  echo Stopping previous backend process PID %%P...
  taskkill /PID %%P /T /F >nul 2>nul
)
powershell -NoProfile -Command "Start-Sleep -Milliseconds 500; if(Get-NetTCPConnection -LocalPort 7860 -State Listen -ErrorAction SilentlyContinue){exit 1}else{exit 0}"
if errorlevel 1 (
  echo [ERROR] Port 7860 could not be released.
  pause
  exit /b 1
)
echo Port 7860 is ready.
echo.
if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Runtime environment not found.
  echo Run 一键启动.bat first.
  echo.
  pause
  exit /b 1
)
set "GRADIO_ANALYTICS_ENABLED=False"
call ".venv\Scripts\activate.bat"
echo Building React frontend...
if not exist "frontend\node_modules" (
  pushd frontend
  call npm install
  if errorlevel 1 (
    popd
    echo [ERROR] Frontend dependency installation failed.
    pause
    exit /b 1
  )
  popd
)
pushd frontend
call npm run build
if errorlevel 1 (
  popd
  echo [ERROR] React frontend build failed.
  pause
  exit /b 1
)
popd
echo [%date% %time%] Starting backend...
echo.
python -u server.py
set "APP_EXIT_CODE=%errorlevel%"
echo.
echo ============================================================
echo Backend stopped. Exit code: %APP_EXIT_CODE%
echo Review the messages above before closing this terminal.
echo ============================================================
pause
exit /b %APP_EXIT_CODE%
