@echo off
title StockWise Engine (Brain)
:loop
cls
echo [StockWise] Launching Live Trading Engine...
echo.
".\.venv\Scripts\python.exe" live_trading_engine.py --mode PAPER --interval 1h

echo.
echo [WARNING] Engine crashed! Restarting in 5 seconds...
timeout /t 5
goto loop