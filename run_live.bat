@echo off
REM StockWise Launcher
REM Automatically uses the project's virtual environment

echo [StockWise] Launching Live Trading Engine...
".\.venv\Scripts\python.exe" live_trading_engine.py %*

if %errorlevel% neq 0 (
    echo.
    echo [Error] The script exited with an error. 
    echo Please check if you have setup the environment correctly.
    pause
)
