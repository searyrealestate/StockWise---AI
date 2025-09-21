"""
🔧 Configuration file for 95% Confidence Trading System
"""

# conftest.py
import pytest
import pandas as pd
import os
from datetime import datetime


# 95% Confidence System Settings
CONFIDENCE_SETTINGS = {
    'target_confidence': 95.0,
    'minimum_trading_confidence': 85.0,
    'high_confidence_threshold': 90.0,
    'ultra_confidence_threshold': 95.0,
    'enable_pre_breakout_detection': True,
    'enable_manual_review_flagging': True
}

# Dynamic Profit Targets
PROFIT_TARGETS = {
    95: 0.08,  # 8% for 95%+ confidence
    90: 0.06,  # 6% for 90-94% confidence  
    85: 0.05,  # 5% for 85-89% confidence
    80: 0.04,  # 4% for 80-84% confidence
    75: 0.037, # 3.7% for 75-79% confidence
    'default': 0.037
}

# Risk Management
RISK_SETTINGS = {
    'max_position_risk': 0.02,  # 2% of account per trade
    'stop_loss_pct': 0.06,      # 6% stop loss
    'trailing_stop_activation': 0.5,  # Activate trailing stop at 50% of profit target
    'max_daily_trades': 5,
    'max_weekly_trades': 15
}

# Model Settings
MODEL_SETTINGS = {
    'ensemble_model_dir': 'models/ensemble/',
    'retrain_frequency_days': 30,
    'min_historical_accuracy': 0.70,
    'require_volume_confirmation': True,
    'market_regime_lookback': 20
}


def pytest_addoption(parser):
    """Adds the --mode command-line option to pytest."""
    parser.addoption("--mode", action="store", default="long", help="Backtest mode: 'short' or 'long'")


def pytest_sessionstart(session):
    """Called after the test session is started."""
    session.test_counter = 0
    session.total_tests_collected = len(session.items)
    session.results = []  # NEW: Initialize a list to store results
    print()


def pytest_runtest_logreport(report, session):
    """Called when a test report is generated."""
    if report.when == 'call':
        session.test_counter += 1
        progress = (session.test_counter / session.total_tests_collected) * 100
        print(f"\n--- Progress: {session.test_counter}/{session.total_tests_collected} ({progress:.0f}%) ---")

        # NEW: Store the metrics attached to the test report
        if hasattr(report, "metrics"):
            session.results.append(report.metrics)


def pytest_sessionfinish(session, exitstatus):
    """Called after the entire test session finishes."""
    # NEW: Save the collected results to a CSV file
    if session.results:
        print("\n--- Saving Backtest Summary ---")
        results_df = pd.DataFrame(session.results)

        # Create a dedicated directory for results
        results_dir = "backtest_results"
        os.makedirs(results_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(results_dir, f"backtest_summary_{timestamp}.csv")

        results_df.to_csv(output_path, index=False, float_format='%.2f')
        print(f"✅ Backtest summary saved to: {output_path}")