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
from datetime import datetime
from tqdm import tqdm
from data_manager import DataManager
import random


# --- NOTE: It is best practice to move the function below into a file named 'conftest.py' ---
def pytest_addoption(parser):
    """Adds the --mode command-line option to pytest."""
    parser.addoption("--mode", action="store", default="short", help="Backtest mode: short or long")

# --- Agent Configurations ---
AGENT_CONFIGS = {
    "dynamic": {
        "model_dir": "models/NASDAQ-gen3-dynamic",
        "test_data_dir": "models/NASDAQ-testing set/features/dynamic_profit"
    },
    "2pct": {
        "model_dir": "models/NASDAQ-gen3-2pct",
        "test_data_dir": "models/NASDAQ-testing set/features/2per_profit"
    },
    "3pct": {
        "model_dir": "models/NASDAQ-gen3-3pct",
        "test_data_dir": "models/NASDAQ-testing set/features/3per_profit"
    },
    "4pct": {
        "model_dir": "models/NASDAQ-gen3-4pct",
        "test_data_dir": "models/NASDAQ-testing set/features/4per_profit"
    }
}

# --- Logger Setup ---
# ... (logging setup remains the same) ...
LOG_DIR = "logs/test_system_performance_log"
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = f"test_system_performance_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_filepath = os.path.join(LOG_DIR, log_filename)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] (%(name)s) %(message)s',
                    filename=log_filepath, filemode='w')
logger = logging.getLogger("SystemTest")
logger.info(f"Test run initiated. Logging to: {log_filepath}")


def _get_date_ranges_from_mode(mode):
    """Determines backtest date ranges based on the backtest mode."""
    if mode == 'short':
        return {'overall_start': "2024-01-01", 'overall_end': "2024-06-01", 'stress_start': "2022-03-01", 'stress_end': "2022-05-01"}
    else:  # 'long'
        return {'overall_start': "2022-01-01", 'overall_end': "2025-01-01", 'stress_start': "2022-01-01", 'stress_end': "2022-12-31"}


# MODIFIED: This helper now takes model_dir as an argument
def load_all_gen3_models(model_dir):
    """Loads all 9 specialist models and their feature lists for the backtest."""
    model_suite = {}
    model_files = glob.glob(os.path.join(model_dir, "*.pkl"))
    if len(model_files) < 9:
        pytest.fail(f"Expected 9 models in '{model_dir}', but found {len(model_files)}. Please run the trainer for this agent.")
    for model_path in model_files:
        model_name = os.path.basename(model_path).replace(".pkl", "")
        features_path = model_path.replace(".pkl", "_features.json")
        model = joblib.load(model_path)
        with open(features_path, 'r') as f:
            features = json.load(f)
        model_suite[model_name] = {'model': model, 'features': features}
    return model_suite


# --- Backtesting Test Class (Now Parameterized for all agents) ---
@pytest.mark.parametrize("agent_name, config", AGENT_CONFIGS.items())
class TestFinancialBacktesting:

    # In test_system_performance.py

    # Replace your existing _run_backtest function with this one:
    def _run_backtest(self, agent_name, config, mode, start_date=None, end_date=None, description="Backtesting",
                      mock_market_downtrend=False):
        logger.info(f"--- Starting backtest for AGENT: {agent_name.upper()} ---")

        model_suite = load_all_gen3_models(config['model_dir'])
        data_manager = DataManager(config['test_data_dir'], label="Test")
        all_symbols = data_manager.get_available_symbols()
        if not all_symbols:
            pytest.skip(f"No backtest data available for agent {agent_name} in {config['test_data_dir']}")

        if mode == 'short':
            sample_size = 100
            if len(all_symbols) > sample_size:
                logger.info(f"Using a random sample of {sample_size} symbols for backtest speed.")
                all_symbols = random.sample(all_symbols, sample_size)

        data_frames = []
        for symbol in tqdm(all_symbols, desc=f"Loading data for {agent_name}"):
            df = data_manager.load_feature_file(symbol)
            if df is not None and not df.empty:
                df['symbol'] = symbol
                data_frames.append(df)

        if not data_frames:
            pytest.skip(f"Could not load any valid data for agent {agent_name}")

        full_df = pd.concat(data_frames)
        full_df.columns = [col.lower() for col in full_df.columns]

        # --- Convert types ONCE after loading ---
        for col in full_df.select_dtypes(include=['number']).columns:
            full_df[col] = full_df[col].astype(float)

        full_df.reset_index(inplace=True)
        if 'Date' in full_df.columns:
            full_df.rename(columns={'Date': 'Datetime'}, inplace=True)

        full_df.sort_values(by=['Datetime', 'symbol'], inplace=True)

        if start_date: full_df = full_df[full_df['Datetime'] >= pd.to_datetime(start_date)]
        if end_date: full_df = full_df[full_df['Datetime'] <= pd.to_datetime(end_date)]

        portfolio_value = 100000.0
        cash = portfolio_value
        portfolio_history = []
        open_positions = {}
        last_known_prices = {}

        for current_date, daily_data in tqdm(full_df.groupby('Datetime'), desc=f"{description} ({agent_name})"):
            for symbol, row in daily_data.set_index('symbol').iterrows():
                last_known_prices[symbol] = row['close']
                # --- POSITION MANAGEMENT LOGIC (SELL-SIDE) ---
                if symbol in open_positions:
                    position = open_positions[symbol]
                    if row['low'] <= position['stop_loss_price']:
                        action = "CUT LOSS"
                    else:
                        cluster = row['volatility_cluster']
                        profit_model_name = f"profit_take_model_{cluster}_vol"
                        loss_model_name = f"cut_loss_model_{cluster}_vol"

                        if profit_model_name in model_suite and loss_model_name in model_suite:
                            profit_model = model_suite[profit_model_name]['model']
                            loss_model = model_suite[loss_model_name]['model']

                            # MODIFIED: Convert features to float before prediction
                            features = row[model_suite[profit_model_name]['features']].to_frame().T

                            if loss_model.predict(features)[0] == 1:
                                action = "CUT LOSS"
                            elif profit_model.predict(features)[0] == 1:
                                action = "SELL"
                            else:
                                action = "HOLD"
                        else:
                            action = "HOLD"

                    if action in ["SELL", "CUT LOSS"]:
                        cash += row['close'] * position['num_shares']
                        del open_positions[symbol]

                # --- ENTRY LOGIC (BUY-SIDE) ---
                if symbol not in open_positions:
                    if mock_market_downtrend: continue

                    cluster = row['volatility_cluster']
                    entry_model_name = f"entry_model_{cluster}_vol"
                    if entry_model_name in model_suite:
                        entry_model = model_suite[entry_model_name]['model']

                        # MODIFIED: Convert features to float before prediction
                        features = row[model_suite[entry_model_name]['features']].to_frame().T

                        if entry_model.predict(features)[0] == 1:
                            position_size = cash * 0.1
                            if row['close'] > 0:
                                num_shares = position_size / row['close']
                                if cash >= position_size:
                                    cash -= position_size
                                    open_positions[symbol] = {
                                        'entry_price': row['close'],
                                        'stop_loss_price': row['close'] - (row['atr_14'] * 2.5),
                                        'num_shares': num_shares
                                    }

            current_holdings_value = 0
            for s, pos_data in open_positions.items():
                current_price = last_known_prices.get(s, pos_data['entry_price'])
                current_holdings_value += pos_data['num_shares'] * current_price

            portfolio_history.append(cash + current_holdings_value)

        # --- METRICS CALCULATION ---
        history = pd.Series(portfolio_history)
        if len(history) < 2: return {'Total Return': 0, 'Max Drawdown': 0, 'Sharpe Ratio': 0, 'Sortino Ratio': 0}

        total_return = (history.iloc[-1] / history.iloc[0] - 1) * 100
        drawdown = (history - history.cummax()) / history.cummax()
        max_drawdown = abs(drawdown.min()) * 100
        returns = history.pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() != 0 else 0
        downside_returns = returns[returns < 0]
        sortino_ratio = np.sqrt(
            252) * returns.mean() / downside_returns.std() if not downside_returns.empty and downside_returns.std() != 0 else 0

        return {'Total Return': total_return, 'Max Drawdown': max_drawdown, 'Sharpe Ratio': sharpe_ratio,
                'Sortino Ratio': sortino_ratio}

    def test_overall_performance(self, agent_name, config, request):
        mode = request.config.getoption("--mode")
        date_ranges = _get_date_ranges_from_mode(mode)
        metrics = self._run_backtest(agent_name, config, mode,
                                     start_date=date_ranges['overall_start'], end_date=date_ranges['overall_end'])
        # Add agent name and test type to metrics, then attach to the report
        metrics['agent'] = agent_name
        metrics['test_type'] = 'Overall'
        request.node.add_report_section("call", "metrics", metrics)

        logger.info(f"AGENT [{agent_name.upper()}] Overall KPIs - Return: {metrics['Total Return']:.2f}%, Max Drawdown: {metrics['Max Drawdown']:.2f}%, Sharpe: {metrics['Sharpe Ratio']:.2f}")
        assert metrics['Sharpe Ratio'] > 1.0, f"KPI FAIL [{agent_name}]: Sharpe Ratio is below 1.0."
        logger.info(f"Summary [{agent_name.upper()}]: Overall Test PASSED.")

    def test_stress_test_bear_market(self, agent_name, config, request):
        mode = request.config.getoption("--mode")
        date_ranges = _get_date_ranges_from_mode(mode)
        metrics = self._run_backtest(agent_name, config, mode,
                                     start_date=date_ranges['stress_start'], end_date=date_ranges['stress_end'],
                                     description="Stress Test", mock_market_downtrend=True)

        # Add agent name and test type to metrics, then attach to the report
        metrics['agent'] = agent_name
        metrics['test_type'] = 'Stress Test (Bear Market)'
        request.node.add_report_section("call", "metrics", metrics)

        logger.info(f"AGENT [{agent_name.upper()}] Stress Test KPIs - Max Drawdown: {metrics['Max Drawdown']:.2f}%")
        assert metrics['Max Drawdown'] < 25.0, f"KPI FAIL [{agent_name}]: Max Drawdown is too high (> 25%)."
        logger.info(f"Summary [{agent_name.upper()}]: Stress Test PASSED.")