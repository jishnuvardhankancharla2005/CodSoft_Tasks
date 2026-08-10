@echo off
setlocal
cd /d "%~dp0"
title Titanic Survival Prediction - Console Demo

echo ================================================
echo   Titanic Survival Prediction - Standalone Demo
echo   Author: Jishnu Vardhan Kancharla
echo ================================================
echo.

set "PY_CMD="
where python >nul 2>nul
if not errorlevel 1 (
    set "PY_CMD=python"
) else (
    where py >nul 2>nul
    if not errorlevel 1 (
        set "PY_CMD=py"
    ) else (
        echo [ERROR] Python was not found in PATH or as 'py' launcher.
        pause
        exit /b 1
    )
)

echo Starting standalone prediction and chart visualization...
echo.

%PY_CMD% predict_with_visualization.py

echo.
echo Demo finished.
pause
