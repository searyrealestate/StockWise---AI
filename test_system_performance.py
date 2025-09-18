"""
how to run the test_system_performance.py?
 short run:
 pytest -s test_system_performance.py --mode=short

 long run:
 pytest -s test_system_performance.py --mode=long

"""

import pytest
import pandas as pd
import numpy as np
import logging
import json
import os
import glob
import joblib
from datetime import datetime, timedelta
from unittest.mock import MagicMock
from tqdm.contrib.logging import logging_redirect_tqdm
from tqdm import tqdm
import pytest


from data_manager import DataManager
from stockwise_simulation import ProfessionalStockAdvisor
from stockwise_simulation import FeatureCalculator

# --- Logger Setup ---
LOG_DIR = "logs/test_system_performance_log"
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = f"test_system_performance_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_filepath = os.path.join(LOG_DIR, log_filename)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] (%(name)s) %(message)s',
                    filename=log_filepath, filemode='w')
logger = logging.getLogger("SystemTest")
logger.info(f"Test run initiated. Logging to: {log_filepath}")


@pytest.fixture(scope="session")
def backtest_mode(request):
    """Fixture that returns the value of the --mode option."""
    return request.config.getoption("--mode")


def _get_date_ranges_from_mode(mode):
    """
    Determines backtest date ranges based on the backtest mode.
    """
    if mode == 'short':
        return {
            'overall_start': "2024-01-01",
            'overall_end': "2024-06-01",
            'stress_start': "2022-03-01",
            'stress_end': "2022-05-01"
        }
    else:  # 'long' or any other value
        return {
            'overall_start': "2022-01-01",
            'overall_end': "2025-01-01",
            'stress_start': "2022-01-01",
            'stress_end': "2022-12-31"
        }


# --- End of new hooks and fixtures ---


# --- Test Fixtures and Helper Functions (UPDATED FOR GEN-3) ---
@pytest.fixture(scope="session")
def test_data_manager():
    """Returns a DataManager instance configured for the Gen-3 testing set."""
    return DataManager("models/NASDAQ-testing set/features", label="Test")


@pytest.fixture(scope="session")
def gen3_advisor_instance():
    """Returns a mocked ProfessionalStockAdvisor instance for Gen-3 testing."""
    advisor = ProfessionalStockAdvisor(testing_mode=True)
    return advisor


@pytest.fixture(scope="session")
def feature_columns():
    """
    Loads feature columns from one of the new Gen-3 feature lists.
    This is for validation tests that need a feature list, not for the backtest itself.
    """
    model_dir = "models/NASDAQ-gen3"
    features_path = os.path.join(model_dir, "entry_model_low_vol_features.json")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Feature list not found at '{features_path}'. Run the model trainer first.")
    with open(features_path, 'r') as f:
        return json.load(f)


# --- NEW: Function to load all models for backtesting ---
def load_all_gen3_models():
    """Loads all 9 specialist models and their feature lists for the backtest."""
    model_suite = {}
    model_dir = "models/NASDAQ-gen3"
    model_files = glob.glob(os.path.join(model_dir, "*.pkl"))
    if len(model_files) < 9:
        raise FileNotFoundError(f"Expected 9 models, found {len(model_files)}. Please run the trainer.")
    for model_path in model_files:
        model_name = os.path.basename(model_path).replace(".pkl", "")
        features_path = model_path.replace(".pkl", "_features.json")
        model = joblib.load(model_path)
        with open(features_path, 'r') as f:
            features = json.load(f)
        model_suite[model_name] = {'model': model, 'features': features}
    return model_suite


# --- Test Cases ---
class TestCoreLogic:
    def test_tax_and_fee_calculation_scenarios(self, gen3_advisor_instance):
        """Tests both the minimum fee and per-share fee scenarios."""
        logger.info("--- Running Test T2.2: Tax & Fee Calculation Scenarios ---")
        gross_profit_dollars_1 = 100.0
        num_shares_1 = 10
        net_profit_1, _ = gen3_advisor_instance.apply_israeli_fees_and_tax(gross_profit_dollars_1, num_shares_1)
        assert net_profit_1 == pytest.approx(71.25), "Minimum fee scenario failed."
        logger.info("Summary (Scenario 1): Test PASSED. Minimum fee calculation is correct.")
        gross_profit_dollars_2 = 100.0
        num_shares_2 = 500
        net_profit_2, _ = gen3_advisor_instance.apply_israeli_fees_and_tax(gross_profit_dollars_2, num_shares_2)
        assert net_profit_2 == pytest.approx(69.00), "Per-share fee scenario failed."
        logger.info("Summary (Scenario 2): Test PASSED. Per-share fee calculation is correct.")

    def test_dynamic_profit_calculation(self, gen3_advisor_instance):
        """Verifies the dynamic profit target calculation at different confidence levels."""
        logger.info("--- Running Test: Dynamic Profit Calculation Logic ---")
        test_cases = [
            (50.0, 3.5), (65.0, 5.0), (80.0, 6.5), (95.0, 8.0)
        ]
        for confidence, expected_profit in test_cases:
            calculated_profit = gen3_advisor_instance.calculate_dynamic_profit_target(confidence)
            assert calculated_profit == expected_profit, f"Dynamic profit failed for confidence {confidence}."
        logger.info("Summary: Test PASSED. Dynamic profit logic is correct.")


# --- Backtesting Test Class (REFACRORED FOR GEN-3) ---
class TestFinancialBacktesting:
    def _run_backtest(self, advisor, data_manager, start_date=None, end_date=None,
                      description="Backtesting", mock_market_uptrend=True):
        """
        [REFACRORED FOR GEN-3]
        An event-driven backtesting engine that uses the new specialist models
        and state machine logic.
        """
        logger.info(f"Starting backtest from {start_date} to {end_date}...")

        # Load all Gen-3 models once
        model_suite = load_all_gen3_models()

        # Load and combine all test data
        all_symbols = data_manager.get_available_symbols()
        if not all_symbols:
            pytest.skip("No backtest data available.")

        df_list = []
        for s in tqdm(all_symbols, desc="Loading Test Data"):
            df = data_manager.load_feature_file(s)
            if df is not None and not df.empty:
                df = df.reset_index()
                if 'Date' in df.columns:
                    df.rename(columns={'Date': 'Datetime'}, inplace=True)
                elif 'index' in df.columns:
                    df.rename(columns={'index': 'Datetime'}, inplace=True)
                df['symbol'] = s
                df_list.append(df)

        if not df_list:
            pytest.skip("No backtest data after filtering.")

        full_df = pd.concat(df_list, ignore_index=True)
        full_df['Datetime'] = pd.to_datetime(full_df['Datetime'])
        full_df.set_index('Datetime', inplace=True)
        full_df.sort_index(inplace=True)

        # Slice data by date range for the backtest
        if start_date: full_df = full_df.loc[start_date:]
        if end_date: full_df = full_df.loc[:end_date]

        backtest_dates = full_df.index.unique()

        portfolio_value = 100000
        portfolio_history = [portfolio_value]
        open_positions = {}

        with logging_redirect_tqdm():
            for current_date in tqdm(backtest_dates, desc=description, leave=False):
                current_day_data = full_df.loc[current_date].sort_index()

                for symbol, row in current_day_data.groupby('symbol').first().iterrows():

                    if not mock_market_uptrend and advisor.is_market_in_uptrend() == False:
                        action = "WAIT"
                        stop_loss = None

                    elif symbol in open_positions:
                        position = open_positions[symbol]

                        if row['Close'] <= position['stop_loss_price']:
                            action = "CUT LOSS"

                        else:
                            cluster = row['Volatility_Cluster']
                            profit_model_name = f"profit_take_model_{cluster}_vol"
                            loss_model_name = f"cut_loss_model_{cluster}_vol"

                            profit_model = model_suite[profit_model_name]['model']
                            loss_model = model_suite[loss_model_name]['model']

                            features_cols = model_suite[profit_model_name]['features']

                            features = row[features_cols].astype(float).to_frame().T

                            profit_pred = profit_model.predict(features)[0]
                            loss_pred = loss_model.predict(features)[0]

                            if loss_pred == 1:
                                action = "CUT LOSS"
                            elif profit_pred == 1:
                                action = "SELL"
                            else:
                                action = "HOLD"

                        if action in ["SELL", "CUT LOSS"]:
                            profit = (row['Close'] - position['entry_price']) * position['num_shares']
                            portfolio_value += profit
                            del open_positions[symbol]

                    else:
                        cluster = row['Volatility_Cluster']
                        entry_model_name = f"entry_model_{cluster}_vol"

                        entry_model = model_suite[entry_model_name]['model']
                        features_cols = model_suite[entry_model_name]['features']

                        features = row[features_cols].astype(float).to_frame().T

                        entry_pred = entry_model.predict(features)[0]

                        if entry_pred == 1:
                            action = "BUY"

                            current_atr = row['ATR_14']
                            stop_loss_price = row['Close'] - (current_atr * 2.5)
                            position_size_dollars = portfolio_value * 0.01
                            num_shares = position_size_dollars / row['Close']

                            open_positions[symbol] = {
                                'entry_price': row['Close'],
                                'stop_loss_price': stop_loss_price,
                                'num_shares': num_shares,
                                'entry_date': current_date
                            }
                        else:
                            action = "WAIT"

                portfolio_history.append(portfolio_value)

        history = pd.Series(portfolio_history, index=range(len(portfolio_history)))
        total_return = (history.iloc[-1] / history.iloc[0] - 1) * 100 if len(history) > 1 and history.iloc[
            0] != 0 else 0

        drawdown = (history - history.cummax()) / history.cummax()
        max_drawdown = abs(drawdown.min()) * 100

        returns = history.pct_change().dropna()
        risk_free_rate = 0.0001
        sharpe_ratio = np.sqrt(252) * (returns.mean() - risk_free_rate) / returns.std()

        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        sortino_ratio = np.sqrt(252) * (returns.mean() - risk_free_rate) / downside_std if downside_std != 0 else np.nan

        return {'Total Return': total_return, 'Max Drawdown': max_drawdown,
                'Sharpe Ratio': sharpe_ratio, 'Sortino Ratio': sortino_ratio}

    def test_overall_performance(self, gen3_advisor_instance, test_data_manager, backtest_mode):
        """
        Test ID: T3.1
        Runs a comprehensive backtest on the full test dataset and checks against Gen-3 KPIs.
        """
        logger.info("--- Running Test T3.1: Overall Historical Backtest ---")
        date_ranges = _get_date_ranges_from_mode(backtest_mode)
        metrics = self._run_backtest(gen3_advisor_instance, test_data_manager,
                                     start_date=date_ranges['overall_start'], end_date=date_ranges['overall_end'],
                                     description="Overall Backtest")
        logger.info(
            f"KPIs - Total Return: {metrics['Total Return']:.2f}%, Max Drawdown: {metrics['Max Drawdown']:.2f}%")
        logger.info(
            f"KPIs - Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}, Sortino Ratio: {metrics['Sortino Ratio']:.2f}")
        assert metrics['Sharpe Ratio'] > 1.0, "KPI FAIL: Sharpe Ratio is below 1.0."
        assert metrics['Sortino Ratio'] > 1.5, "KPI FAIL: Sortino Ratio is below 1.5."
        logger.info("Summary (T3.1): Test PASSED. Backtest completed successfully.")

    def test_stress_test_bear_market(self, gen3_advisor_instance, test_data_manager, backtest_mode):
        """
        Test ID: T3.3
        Runs a stress test on the bear market of 2022 to validate the Market Regime Filter.
        """
        logger.info("--- Running Test T3.3: Stress Test (2022 Bear Market) ---")
        date_ranges = _get_date_ranges_from_mode(backtest_mode)
        metrics = self._run_backtest(gen3_advisor_instance, test_data_manager,
                                     start_date=date_ranges['stress_start'], end_date=date_ranges['stress_end'],
                                     description="Stress Test (2022)", mock_market_uptrend=False)
        logger.info(f"KPI - Max Drawdown during stress test: {metrics['Max Drawdown']:.2f}%")
        assert metrics['Max Drawdown'] < 25.0, "KPI FAIL: Max Drawdown is too high (> 25%)."
        logger.info("Summary (T3.3): Test PASSED. Stress test completed.")