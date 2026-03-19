@echo off
title StockWise FULL LAUNCHER (Scanner + Live)
cd /d "%~dp0"

echo ===================================================
echo   STOCKWISE GEN-13: FULL SYSTEM LAUNCHER
echo ===================================================
echo.
echo   [1] Nightly Scanner (~90 min for ~4000 symbols)
echo   [2] 3x Live Trading Engines (1h / 1d / 1wk)
echo.
echo   Scanner runs FIRST, then Live engines start.
echo   Safe to run — uses atomic JSON I/O.
echo ===================================================
echo.

echo [STEP 1/2] Running Nightly Scanner...
echo Started at: %date% %time%
echo.

".venv\Scripts\python.exe" stock_hunter.py

echo.
echo [STEP 1/2] Scanner COMPLETE at: %date% %time%
echo.

echo [STEP 2/2] Launching 3 Live Trading Engines...
echo.

:: 1. Short Range (Sniper) - 1h
start "StockWise [SHORT-TERM | 1h]" cmd /k ".\.venv\Scripts\python.exe live_trading_engine.py --mode PAPER --interval 1h"

:: 2. Mid Range (Tactical) - 1d
start "StockWise [MID-TERM | 1d]" cmd /k ".\.venv\Scripts\python.exe live_trading_engine.py --mode PAPER --interval 1d"

:: 3. Long Range (Strategic) - 1wk
start "StockWise [LONG-TERM | 1wk]" cmd /k ".\.venv\Scripts\python.exe live_trading_engine.py --mode PAPER --interval 1wk"

echo.
echo ===================================================
echo   [SUCCESS] Scanner done + 3 engines launched.
echo   VIP list updated. Engines using latest scan data.
echo   Check /logs folder for log files.
echo ===================================================
echo.
pause
