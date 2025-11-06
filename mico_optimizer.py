# optimizer.py
"""
Trading System Parameter Optimizer
==================================

This script provides a "grid search" optimization engine designed to be
integrated into a Streamlit application. Its purpose is to find the best-
performing set of parameters for one or more trading advisors based on
historical backtesting.

It is a "brute-force" optimizer that tests every possible combination of
parameters provided in a grid.

How it Works:
-------------
1.  **Grid Search**: The `run_full_optimization` function takes a dictionary of
    advisor instances (e.g., {'MICO': MicoAdvisor()}) and a corresponding
    dictionary of parameter grids (e.g., {'MICO': {'sma_fast': [10, 20], 'sma_slow': [50, 100]}}).
2.  **Iterative Backtesting**: The `_optimize_single_model` function iterates
    through every possible parameter combination using `itertools.product`.
3.  **Simulation**: For each combination, it runs a full backtest by calling the
    advisor's `analyze` method (which must accept a `params` argument) for
    every single day in the specified date range.
4.  **Performance Scoring**: The resulting "BUY" signals are passed to a
    lightweight simulator (`_simulate_trades_for_performance`) which
    calculates a single "Total P/L %" score for that parameter set.
5.  **Finds Best**: It tracks the parameters that generated the highest P/L score.

Outputs:
--------
-   **Streamlit UI**: Displays progress bars, detailed results tables for each
    model, and a final summary of the best-performing model.
-   **JSON File**: Saves the best-performing parameter set for *all* tested
    models into a single `best_params.json` file for future use.

"""


import pandas as pd
import streamlit as st
import itertools
import json
from results_analyzer import run_backtest  # We'll reuse the backtester


def save_best_params(model_name, params):
    """Saves the best parameters for a model to a JSON file."""
    try:
        with open("best_params.json", "r") as f:
            all_params = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_params = {} # Create a new file if it doesn't exist or is empty

    all_params[model_name] = params
    with open("best_params.json", "w") as f:
        json.dump(all_params, f, indent=4)
    st.success(f"Saved best parameters for {model_name} to `best_params.json`.")


def run_full_optimization(optimizer_advisors: dict, parameter_grids: dict, symbol: str, start_date, end_date):
    """
    Runs an optimization for EVERY model, finds the best params for each,
    saves them all, and displays a final summary.
    """
    st.info(f"Starting full system calibration for {symbol}...")

    all_best_results = []

    # Loop through each advisor the user passed in
    for model_name, advisor_instance in optimizer_advisors.items():
        if model_name not in parameter_grids:
            st.warning(f"Skipping {model_name}: No parameter grid defined.")
            continue

        st.markdown(f"---")
        st.subheader(f"Optimizing: {model_name}")

        # Run the optimization for this one model
        best_params, best_performance, results_df = _optimize_single_model(
            advisor_instance, symbol, start_date, end_date,
            parameter_grids[model_name]
        )

        # Display the detailed results for this specific model
        st.dataframe(results_df.style.format({'Profit Factor': '{:.2f}'}), use_container_width=True)

        if best_params:
            st.success(f"🏆 Best for {model_name}: `{best_params}` (Profit Factor: {best_performance:.2f})")
            all_best_results.append({
                "Model": model_name,
                "Best Profit Factor": best_performance,
                "Best Parameters": json.dumps(best_params)
            })
            # Save the params for this model to the file
            save_best_params(model_name, best_params)
        else:
            st.error(f"No successful trades found for {model_name}.")

    st.markdown("---")
    st.header("Calibration Complete")

    if not all_best_results:
        st.error("No successful optimizations were completed.")
        return

    # Display a final summary of all models
    summary_df = pd.DataFrame(all_best_results).sort_values(by="Best Profit Factor", ascending=False)
    st.subheader("Overall Optimization Summary")
    st.dataframe(summary_df, use_container_width=True)
    st.success("All models have been optimized and the best parameters are saved to `best_params.json`.")


def _optimize_single_model(advisor_instance, symbol, start_date, end_date, parameter_grid):
    """Runs a backtest for every combination of parameters FOR A SINGLE MODEL."""

    param_names = parameter_grid.keys()
    param_values = parameter_grid.values()
    param_combinations = list(itertools.product(*param_values))

    st.write(f"Testing {len(param_combinations)} parameter combinations...")

    best_performance = -float('inf')
    best_params = None
    results = []

    progress_bar = st.progress(0)
    for i, params_tuple in enumerate(param_combinations):
        params = dict(zip(param_names, params_tuple))

        all_signals = []
        for analysis_date in pd.date_range(start_date, end_date):
            signal = advisor_instance.analyze(symbol, analysis_date, params=params)
            if signal and signal.get('signal') == 'BUY':
                signal['Analysis Date'] = analysis_date.strftime('%Y-%m-%d')
                signal['Symbol'] = symbol
                signal['Entry Price'] = signal['current_price']
                signal['Profit Target ($)'] = signal['profit_target_price']
                signal['Stop-Loss'] = signal['stop_loss_price']
                all_signals.append(signal)

        if all_signals:
            trades_df = pd.DataFrame(all_signals)
            performance = _simulate_trades_for_performance(trades_df, advisor_instance.dm)
            results.append({**params, 'Profit Factor': performance})

            if performance > best_performance:
                best_performance = performance
                best_params = params

        progress_bar.progress((i + 1) / len(param_combinations))

    progress_bar.empty()

    if not results:
        # Get the model name for a correct error message
        model_name = type(advisor_instance).__name__
        st.error(f"No successful trades found for {model_name}.")
        return None, -float('inf'), pd.DataFrame()

    results_df = pd.DataFrame(results).sort_values(by='Profit Factor', ascending=False)
    return best_params, best_performance, results_df


def _simulate_trades_for_performance(trades_df, data_manager):
    """
    A lightweight backtester that returns a single performance metric (Profit Factor).
    Profit Factor = (Sum of all P/L % from wins) / (Absolute Sum of all P/L % from losses)
    """
    all_pl_percents = []
    for _, trade in trades_df.iterrows():
        entry_date = pd.to_datetime(trade['Analysis Date'])
        df_raw = data_manager.get_stock_data(trade['Symbol'], start_date=entry_date)

        if df_raw.empty or not isinstance(df_raw.index, pd.DatetimeIndex):
            continue

        trade_period_df = df_raw[df_raw.index >= entry_date].copy()

        exit_price = None
        for day_index in range(1, len(trade_period_df)):
            if day_index > 60: break
            current_day = trade_period_df.iloc[day_index]
            if current_day['low'] <= trade['Stop-Loss']:
                exit_price = trade['Stop-Loss']
                break
            elif current_day['high'] >= trade['Profit Target ($)']:
                exit_price = trade['Profit Target ($)']
                break

        if exit_price is None:
            exit_price = trade_period_df['close'].iloc[-1] if not trade_period_df.empty else trade['Entry Price']

        if trade['Entry Price'] > 0:
            pl_percent = ((exit_price - trade['Entry Price']) / trade['Entry Price']) * 100
            all_pl_percents.append(pl_percent)

    if not all_pl_percents:
        return 0  # No trades, so profit factor is 0

    # --- Calculate Profit Factor ---
    total_gains = sum(p for p in all_pl_percents if p > 0)
    total_losses = abs(sum(p for p in all_pl_percents if p < 0))

    if total_losses == 0:
        # If there are no losses, it's an "infinite" profit factor (or 0 if no gains)
        return float('inf') if total_gains > 0 else 0

    return total_gains / total_losses
