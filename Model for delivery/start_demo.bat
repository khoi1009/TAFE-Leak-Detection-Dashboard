@echo off
:: TAFE Leak Detection - Demo Mode (No Backend Required)
:: Runs dashboard directly without authentication

echo ============================================
echo   TAFE Leak Detection - DEMO MODE
echo ============================================
echo.
echo Running dashboard without backend (demo mode)
echo.

cd /d "%~dp0frontend"

set DEMO_MODE=true
set DASHBOARD_PORT=8050

echo Starting Dashboard in Demo Mode on port 8050...
echo Access at: http://127.0.0.1:8050
echo.

python app.py

