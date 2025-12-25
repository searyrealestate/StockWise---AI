@echo off
REM StockWise Verification Launcher
REM Runs the standalone Sniper Verification Script

echo [StockWise] Verifying Sniper Logic...
".\.venv\Scripts\python.exe" verify_sniper_logic.py

if %errorlevel% neq 0 (
    echo.
    echo [Error] Verification script failed.
    pause
) else (
    echo.
    echo [Success] Verification complete.
    pause
)
