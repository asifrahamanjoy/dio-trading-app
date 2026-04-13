@echo off
title Dio Trading App
echo ============================================================
echo   DIO TRADING APP - EUR/USD Analysis ^& Signals
echo ============================================================
echo.
cd /d "%~dp0dio_trading_app"
call "%~dp0.venv\Scripts\activate.bat"
python start.py
pause
