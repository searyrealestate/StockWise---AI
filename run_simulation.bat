@echo off
REM StockWise Simulation Launcher
REM Runs the Strict Test (Train on Past -> Trade on Future)
echo [StockWise] Launching Strict Out-of-Sample Test...
".\.venv\Scripts\python.exe" run_strict_test.py
if %errorlevel% neq 0 (
    echo.
    echo [Error] Simulation failed.
    pause
) else (
    echo.
    echo [Success] Simulation complete. Check logs.
    pause
)