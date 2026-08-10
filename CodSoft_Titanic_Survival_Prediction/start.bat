@echo off
cd /d "%~dp0"
title Titanic Survival Prediction Dashboard

echo ================================================
echo    Titanic Survival Prediction Dashboard
echo    Author: Jishnu Vardhan Kancharla
echo ================================================
echo.

REM --- Check Python launcher (python or py) ---
set "PY_CMD=python"
where python >nul 2>nul
if errorlevel 1 (
    where py >nul 2>nul
    if not errorlevel 1 (
        set "PY_CMD=py"
    ) else (
        echo [ERROR] Python was not found in PATH or as py launcher.
        echo Please install Python 3.8 or newer and make sure it is added to PATH.
        echo.
        pause
        exit /b 1
    )
)

echo Using Python command: %PY_CMD%

REM --- Is port 5000 already in use? ---
powershell -NoProfile -ExecutionPolicy Bypass -Command "if (Get-NetTCPConnection -LocalPort 5000 -State Listen -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }" >nul 2>nul
if errorlevel 1 goto START_SERVER

echo.
echo [NOTICE] Port 5000 is already in use - Dashboard may already be running.
echo.
echo   1. Open existing dashboard in Web Browser
echo   2. Stop existing server and start fresh
echo   3. Exit
echo.
set /p CHOICE="Select an option 1, 2, or 3 [Default: 1]: "

if "%CHOICE%"=="2" goto STOP_AND_START
if "%CHOICE%"=="3" goto END

echo Opening http://127.0.0.1:5000 ...
start "" "http://127.0.0.1:5000"
echo.
pause
goto END

:STOP_AND_START
echo Stopping existing dashboard process...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$p = Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and $_.CommandLine -like '*app.py*' }; if ($p) { Stop-Process -Id $p.ProcessId -Force }" >nul 2>nul
timeout /t 2 >nul

:START_SERVER
echo.
echo Starting the Titanic Survival Prediction dashboard server...
echo   - Model training will take around 10-20 seconds at startup.
echo   - The web browser will open automatically once ready.
echo   - Keep this CMD window open to maintain the server.
echo.
echo -------------------------------------------------
echo.

%PY_CMD% app.py

echo.
echo ================================================
echo Server stopped.
echo ================================================
pause

:END
