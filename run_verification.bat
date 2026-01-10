@echo off
REM StockWise Verification Launcher
echo [StockWise] Launching Standard Verification (Gen-10)...
".\.venv\Scripts\python.exe" verify_sniper_logic.py
if %errorlevel% neq 0 (
    echo.
    echo [Error] Verification failed.
    pause
) else (
    echo.
    echo [Success] Complete.
    pause
)