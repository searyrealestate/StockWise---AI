@echo off
REM StockWise Live Trading Launcher
echo [StockWise] Launching Live Trading Engine...
".\.venv\Scripts\python.exe" live_trading_engine.py
if %errorlevel% neq 0 (
    echo.
    echo [Error] Live Engine crashed or exited.
    pause
)