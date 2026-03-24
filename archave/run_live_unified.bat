@echo off
title StockWise UNIFIED Engine (Short/Mid/Long)

echo ===================================================
echo   STOCKWISE GEN-12: UNIFIED LIVE ENGINE
echo ===================================================
echo.
echo Launching Single Process for All Timeframes...
echo (Sniper=1h, Tactical=1d, Strategic=1wk)
echo.

:: python live_trading_engine.py --mode PAPER --unified
".\.venv\Scripts\python.exe" live_trading_engine.py --mode PAPER --unified

echo.
echo [WARNING] Engine stopped or crashed.
pause
