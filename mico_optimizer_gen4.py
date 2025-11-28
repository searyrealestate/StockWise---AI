# mico_optimizer_gen4.py
"""
Robust Optimizer Engine for StockWise
=====================================
Performs grid search optimization for trading strategies.
Handles data warmup correctly to ensure indicators (SMA200) work from Day 1 of testing.
"""

import itertools
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging

# Setup logging
logger = logging.getLogger("MicoOptimizer")


def run_full_optimization(optimizer_advisors, parameter_grids, symbol, start_date, end_date, pre_fetched_data=None):
    """
    Runs grid search optimization for multiple advisors.
    Returns a DataFrame of the best parameters found.
    """
    results = []

    # 1. Validate Data
    if pre_fetched_data is None or pre_fetched_data.empty:
        logger.error("❌ No data provided for optimization.")
        return pd.DataFrame()

    # Ensure data is sorted and columns lower
    data = pre_fetched_data.copy()
    data.sort_index(inplace=True)
    data.columns = [c.lower() for c in data.columns]

    # Convert dates to timestamps for comparison
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    # 2. Iterate Advisors
    for advisor_name, advisor in optimizer_advisors.items():
        if advisor_name not in parameter_grids:
            continue

        grid = parameter_grids[advisor_name]
        logger.info(f"⚙️ Optimizing {advisor_name}...")

        # Generate all combinations
        keys, values = zip(*grid.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        best_pnl = -np.inf
        best_params = None
        best_stats = {}

        # 3. Grid Search Loop
        # We use tqdm to show progress
        for params in tqdm(combinations, desc=f"Tuning {advisor_name}", leave=False):
            try:
                # Run Simulation
                pnl, win_rate, trades = _simulate_fast_backtest(
                    advisor, data, params, start_dt, end_dt, symbol
                )

                # Filter: Must have at least 5 trades to be valid
                if trades >= 5 and pnl > best_pnl:
                    best_pnl = pnl
                    best_params = params
                    best_stats = {
                        'Total Return': round(pnl, 2),
                        'Win Rate': round(win_rate, 2),
                        'Trades': trades,
                        'Strategy': advisor_name
                    }

            except Exception as e:
                # logger.debug(f"Sim failed for {params}: {e}")
                continue

        if best_params:
            # Merge params and stats
            full_record = {**best_params, **best_stats}
            results.append(full_record)
            logger.info(f"✅ Best {advisor_name}: {best_pnl:.2f}% Return ({best_stats['Trades']} trades)")
        else:
            logger.warning(f"⚠️ No profitable settings found for {advisor_name} (or too few trades).")

    # 4. Return Results
    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)


def _simulate_fast_backtest(advisor, full_data, params, start_dt, end_dt, symbol):
    """
    Fast logic to simulate trades without full overhead.
    Calls advisor.analyze() but manages the loop efficiently.
    """
    balance = 10000
    position = None  # None or dict {'price': x, 'stop': y, 'target': z}
    trades_count = 0
    wins = 0

    # Identify the test window indices to avoid date parsing in loop
    test_indices = full_data.index[(full_data.index >= start_dt) & (full_data.index <= end_dt)]

    if len(test_indices) == 0:
        return -100, 0, 0

    # Main Simulation Loop
    for date in test_indices:
        # Market Data for Today
        # We need to pass the SLICE up to today so indicators calc correctly
        # Optimizing: We assume the advisor handles the slice efficiently or we pass the full df + date

        # Get OHLC for today
        today_row = full_data.loc[date]
        current_price = today_row['close']
        low_price = today_row['low']
        high_price = today_row['high']

        # 1. Manage Open Position
        if position:
            # Check Stop Loss
            if low_price <= position['stop']:
                # Stopped Out
                pnl_pct = (position['stop'] - position['price']) / position['price']
                balance *= (1 + pnl_pct)
                position = None

            # Check Profit Target
            elif high_price >= position['target']:
                # Take Profit
                pnl_pct = (position['target'] - position['price']) / position['price']
                balance *= (1 + pnl_pct)
                wins += 1
                position = None

            continue  # Skip buy check if we are in a trade (simplified)

        # 2. Check for Entry Signal
        # We call the advisor's analyze method
        # Note: We assume analyze() uses the data up to 'date'
        try:
            # We pass full_data; advisor MUST slice it up to 'date'
            decision = advisor.analyze(
                stock_data=full_data,
                symbol=symbol,
                analysis_date=date,
                params=params,
                use_market_filter=False  # Disable market filter for pure strategy test
            )

            if decision.get('signal') == 'BUY':
                entry_price = decision.get('current_price', current_price)
                stop_loss = decision.get('stop_loss_price', entry_price * 0.95)
                target = decision.get('profit_target_price', entry_price * 1.05)

                position = {
                    'price': entry_price,
                    'stop': stop_loss,
                    'target': target
                }
                trades_count += 1

        except Exception:
            pass  # Skip day on error

    # Calculate Metrics
    total_return_pct = (balance - 10000) / 10000 * 100
    win_rate = (wins / trades_count * 100) if trades_count > 0 else 0.0

    return total_return_pct, win_rate, trades_count