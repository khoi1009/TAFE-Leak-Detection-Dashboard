@echo off
:: TAFE Leak Detection - Start Login Portal
:: Authenticates users before accessing dashboard

echo ============================================
echo   TAFE Leak Detection - Login Portal
echo ============================================
echo.

cd /d "%~dp0frontend"

echo Starting Login Portal on port 8050...
echo Access at: http://127.0.0.1:8050
echo.

set LOGIN_PORT=8050
set DASHBOARD_URL=http://localhost:8051
set API_URL=http://localhost:8000/api/v1

python login_app.py
