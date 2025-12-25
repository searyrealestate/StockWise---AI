@echo off
REM StockWise Simulation Launcher
REM Runs the Continuous Learning Analyzer (Gen-9 Backtest/Simulation)

echo [StockWise] Launching Gen-9 Simulation...
".\.venv\Scripts\python.exe" continuous_learning_analyzer.py

if %errorlevel% neq 0 (
    echo.
    echo [Error] Simulation failed.
    pause
) else (
    echo.
    echo [Success] Simulation complete. Check logs for details.
    pause
)
