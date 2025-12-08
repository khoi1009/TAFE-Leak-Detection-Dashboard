@echo off
:: TAFE Leak Detection - Start All Services
:: Starts Backend API, Login Portal, and Dashboard

echo ============================================
echo   TAFE Leak Detection - Complete System
echo ============================================
echo.
echo This will start 3 services:
echo   1. Backend API    (port 8000)
echo   2. Login Portal   (port 8050) 
echo   3. Dashboard      (port 8051)
echo.
echo Press any key to start all services...
pause > nul

:: Start Backend in new window
echo Starting Backend API...
start "TAFE API Backend" cmd /k "%~dp0start_backend.bat"
timeout /t 5 /nobreak > nul

:: Start Dashboard in new window
echo Starting Dashboard...
start "TAFE Dashboard" cmd /k "%~dp0start_dashboard.bat"
timeout /t 3 /nobreak > nul

:: Start Login Portal in new window
echo Starting Login Portal...
start "TAFE Login" cmd /k "%~dp0start_login.bat"

echo.
echo ============================================
echo   All services started!
echo ============================================
echo.
echo Access Points:
echo   Login Portal: http://127.0.0.1:8050
echo   Dashboard:    http://127.0.0.1:8051
echo   API Docs:     http://127.0.0.1:8000/docs
echo.
echo Default Credentials:
echo   Admin:    admin / admin123
echo   Operator: operator / operator123
echo.
echo Press any key to open Login Portal...
pause > nul
start http://127.0.0.1:8050
