@echo off
:: TAFE Leak Detection - Start Dashboard
:: Main dashboard with GIS Map

echo ============================================
echo   TAFE Leak Detection - Dashboard
echo ============================================
echo.

cd /d "%~dp0frontend"

echo Starting Dashboard on port 8051...
echo Access at: http://127.0.0.1:8051
echo.

set DASHBOARD_PORT=8051
set API_URL=http://localhost:8000/api/v1
set LOGIN_URL=http://localhost:8050

python app.py
