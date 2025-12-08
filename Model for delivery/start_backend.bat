@echo off
:: TAFE Leak Detection - Start Backend API
:: Run this first before starting the dashboard

echo ============================================
echo   TAFE Leak Detection - Backend API
echo ============================================
echo.

cd /d "%~dp0backend"

echo Installing dependencies...
pip install -r requirements.txt -q

echo.
echo Starting FastAPI backend on port 8000...
echo API Docs: http://127.0.0.1:8000/docs
echo.

uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
