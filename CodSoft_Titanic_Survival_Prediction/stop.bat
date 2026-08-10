@echo off
setlocal
cd /d "%~dp0"
title Titanic Survival Dashboard - Stopper

echo ================================================
echo Stopping Titanic Survival Prediction Dashboard
echo ================================================
echo.

powershell -NoProfile -ExecutionPolicy Bypass -Command "$p = Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and $_.CommandLine -like '*app.py*' }; if ($p) { $p | ForEach-Object { Stop-Process -Id $_.ProcessId -Force; 'Stopped dashboard PID ' + $_.ProcessId } } else { 'Dashboard server is not currently running.' }"

echo.
echo Done.
pause
