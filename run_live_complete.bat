@echo off
title StockWise COMPLETE Launcher

echo ===================================================
echo   STOCKWISE GEN-12: MULTI-TIMEFRAME LAUNCHER
echo ===================================================
echo.

:: --- NotebookLM Google Drive sync DISABLED (G: drive removed) ---
:: To re-enable, uncomment the line below:
:: start "DataSync" /b ".\.venv\Scripts\python.exe" notebooklm_sync.py
echo.

echo Launching 3 Independent Process Engines...
echo.

:: 1. Short Range (Sniper) - 1h
start "StockWise [SHORT-TERM | 1h]" cmd /k ".\.venv\Scripts\python.exe live_trading_engine.py --mode PAPER --interval 1h"

:: 2. Mid Range (Tactical) - 1d
start "StockWise [MID-TERM | 1d]" cmd /k ".\.venv\Scripts\python.exe live_trading_engine.py --mode PAPER --interval 1d"

:: 3. Long Range (Strategic) - 1wk
start "StockWise [LONG-TERM | 1wk]" cmd /k ".\.venv\Scripts\python.exe live_trading_engine.py --mode PAPER --interval 1wk"

echo.
echo [SUCCESS] All engines launched in separate windows.
echo Check the /logs folder for distinct log files.
echo [NOTE] Google Drive sync disabled. Logs saved to local /logs folder.
echo.
pause
