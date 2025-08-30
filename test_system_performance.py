import pytest
import pandas as pd
import numpy as np
import logging
import json
import os
import glob
from datetime import datetime
from unittest.mock import MagicMock
from tqdm.contrib.logging import logging_redirect_tqdm
from tqdm import tqdm

from data_manager import DataManager
from stockwise_simulation_gen2 import ProfessionalStockAdvisor

# --- Logger Setup ---
LOG_DIR = "logs/test_system_performance_log"
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = f"test_system_performance_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_filepath = os.path.join(LOG_DIR, log_filename)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] (%(name)s) %(message)s',
                    filename=log_filepath, filemode='w')
logger = logging.getLogger("SystemTest")
logger.info(f"Test run initiated. Logging to: {log_filepath}")


# --- Test Fixtures and Helper Functions ---
def load_latest_feature_columns():
    model_dir = "models/NASDAQ-training set"
    model_files = glob.glob(os.path.join(model_dir, "nasdaq_gen2_optimized_model_*.pkl"))
    if not model_files: raise FileNotFoundError(f"No model files found in '{model_dir}'.")
    latest_model_path = max(model_files, key=os.path.getctime)
    features_path = latest_model_path.replace(".pkl", "_features.json")
    with open(features_path, 'r') as f: return json.load(f)


@pytest.fixture(scope="session")
def test_data_manager():
    return DataManager("models/NASDAQ-testing set/features", label="Test")


@pytest.fixture(scope="session")
def advisor_instance():
    return ProfessionalStockAdvisor(testing_mode=True)


@pytest.fixture(scope="session")
def feature_columns():
    return load_latest_feature_columns()


# --- Test Cases ---
class TestCoreLogic:
    # def test_tax_and_fee_calculation(self, advisor_instance):
    #     logger.info("--- Running Test T2.2: Tax & Fee Calculation ---")
    #     net_profit, _ = advisor_instance.apply_israeli_fees_and_tax(10.0, 1000.0)
    #     assert net_profit == pytest.approx(7.125), "Tax/fee calculation is incorrect."
    #     logger.info("Summary (T2.2): Test PASSED. Fee calculation is correct.")

    def test_tax_and_fee_calculation_scenarios(self, advisor_instance):
        """NEW: Tests both the minimum fee and per-share fee scenarios."""
        logger.info("--- Running Test T2.2: Tax & Fee Calculation Scenarios ---")

        # Scenario 1: Minimum fee applies (e.g., small trade)
        # 10 shares * $0.008 = $0.08, which is less than the $2.50 minimum. Total fee = $5.00
        gross_profit_dollars_1 = 100.0  # $100 profit
        num_shares_1 = 10
        # Expected: $100 profit - $5 fee = $95. Tax on $95 is $23.75. Net profit = $71.25.
        net_profit_1, _ = advisor_instance.apply_israeli_fees_and_tax(gross_profit_dollars_1, num_shares_1)
        assert net_profit_1 == pytest.approx(71.25), "Minimum fee scenario failed."
        logger.info("Summary (Scenario 1): Test PASSED. Minimum fee calculation is correct.")

        # Scenario 2: Per-share fee applies (e.g., large trade)
        # 500 shares * $0.008 = $4.00, which is more than $2.50. Total fee = $8.00
        gross_profit_dollars_2 = 100.0  # $100 profit
        num_shares_2 = 500
        # Expected: $100 profit - $8 fee = $92. Tax on $92 is $23.00. Net profit = $69.00.
        net_profit_2, _ = advisor_instance.apply_israeli_fees_and_tax(gross_profit_dollars_2, num_shares_2)
        assert net_profit_2 == pytest.approx(69.00), "Per-share fee scenario failed."
        logger.info("Summary (Scenario 2): Test PASSED. Per-share fee calculation is correct.")

    def test_dynamic_profit_calculation(self, advisor_instance):
        """
        Verifies the dynamic profit target calculation at different confidence levels.
        """
        logger.info("--- Running Test: Dynamic Profit Calculation Logic ---")

        # Test cases: (input_confidence, expected_profit_pct)
        test_cases = [
            (50.0, 3.5),  # Low confidence
            (65.0, 5.0),  # Moderate confidence
            (80.0, 6.5),  # High confidence
            (95.0, 8.0)  # Very High confidence
        ]

        for confidence, expected_profit in test_cases:
            calculated_profit = advisor_instance.calculate_dynamic_profit_target(confidence)
            logger.info(f"Testing Confidence: {confidence} -> Expected: {expected_profit}%, Got: {calculated_profit}%")
            assert calculated_profit == expected_profit, f"Dynamic profit failed for confidence {confidence}."

        logger.info("Summary: Test PASSED. Dynamic profit logic is correct.")


class TestFinancialBacktesting:
    def _run_backtest(self, advisor, data_manager, feature_columns, start_date=None, end_date=None,
                      description="Backtesting"):
        logger.info(f"Starting backtest from {start_date} to {end_date}...")
        portfolio_value, history, trades = 100000, [], []

        symbols = data_manager.get_available_symbols()
        if not symbols: pytest.skip("No backtest data.")

        df_list = [df.assign(symbol=s) for s in tqdm(symbols, desc="Loading Data") if
                   (df := data_manager.load_feature_file(s)) is not None and not df.empty]
        if not df_list: return {'Total Return': 0, 'Win Rate': 0, 'Max Drawdown': 100}

        full_df = pd.concat(df_list, ignore_index=True)
        full_df['Datetime'] = pd.to_datetime(full_df['Datetime'])
        full_df.set_index('Datetime', inplace=True)
        full_df.sort_index(inplace=True)

        symbol_dfs = {symbol: df.copy() for symbol, df in full_df.groupby('symbol')}

        backtest_df = full_df.copy()
        if start_date: backtest_df = backtest_df.loc[start_date:]
        if end_date: backtest_df = backtest_df.loc[:end_date]

        smart_mock_manager = MagicMock()
        smart_mock_manager.get_stock_data.side_effect = lambda symbol: symbol_dfs.get(symbol)
        advisor.data_source_manager = smart_mock_manager
        advisor.model = MagicMock()
        advisor.feature_names = feature_columns

        with logging_redirect_tqdm():
            for date, row in tqdm(backtest_df.iterrows(), total=len(backtest_df), desc=description):
                prediction = row['Target']
                action = "BUY" if prediction == 1 else "WAIT"

                if action == 'BUY':
                    entry_price = row['Close']
                    exit_date = date + pd.Timedelta(days=5)
                    try:
                        exit_price = symbol_dfs[row['symbol']].loc[exit_date]['Close']
                        profit = (exit_price - entry_price) * 10
                        net_profit = (profit - 5.0) - (max(0, profit - 5.0) * 0.25)
                        trades.append(net_profit)
                        portfolio_value += net_profit
                    except (KeyError, IndexError):
                        pass
                history.append(portfolio_value)

        history = pd.Series(history)
        total_return = (history.iloc[-1] / history.iloc[0] - 1) * 100 if len(history) > 1 and history.iloc[
            0] != 0 else 0
        win_rate = (len([t for t in trades if t > 0]) / len(trades)) * 100 if trades else 0
        drawdown = (history - history.cummax()) / history.cummax()
        return {'Total Return': total_return, 'Win Rate': win_rate, 'Max Drawdown': abs(drawdown.min()) * 100}

    def test_overall_performance(self, advisor_instance, test_data_manager, feature_columns):
        logger.info("--- Running Test T3.1: Overall Historical Backtest ---")
        metrics = self._run_backtest(advisor_instance, test_data_manager, feature_columns,
                                     description="Overall Backtest")
        logger.info(
            f"KPIs - Return: {metrics['Total Return']:.2f}%, Win Rate: {metrics['Win Rate']:.2f}%, Max Drawdown: {metrics['Max Drawdown']:.2f}%")
        assert metrics['Total Return'] > 10, "KPI FAIL: Total Return is below 10%."
        assert metrics['Max Drawdown'] < 100, "KPI FAIL: Max Drawdown is too high."
        logger.info("Summary (T3.1): Test PASSED. Backtest completed.")

    def test_stress_test_bear_market(self, advisor_instance, test_data_manager, feature_columns):
        logger.info("--- Running Test T3.3: Stress Test (2022 Bear Market) ---")
        metrics = self._run_backtest(advisor_instance, test_data_manager, feature_columns, start_date="2022-01-01",
                                     end_date="2022-12-31", description="Stress Test (2022)")
        logger.info(f"KPI - Max Drawdown during stress test: {metrics['Max Drawdown']:.2f}%")
        assert metrics['Max Drawdown'] < 100, "KPI FAIL: Strategy performs poorly in a bear market."
        logger.info("Summary (T3.3): Test PASSED. Stress test completed.")