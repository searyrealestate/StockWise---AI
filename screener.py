"""
StockWise AI Market Screener
============================

This script provides the core functionality for the market screener feature in the
StockWise AI Trading Advisor application.

It takes an instance of the ProfessionalStockAdvisor, a list of stock symbols
(the "universe"), and an analysis date, then iterates through each stock to find
potential "BUY" opportunities.

The screener is designed to integrate seamlessly with the Streamlit front-end,
providing real-time updates on its progress and displaying the results in a
dynamically updated table as they are found.

Key Functionality:
------------------
-   Iterates through a large list of stocks provided by the user.
-   Calls the `run_analysis` method from the main advisor for each stock.
-   Filters for results that are a "BUY" signal and exceed a minimum confidence
    threshold (default 70%).
-   For each valid opportunity, it calculates a hypothetical net profit based on
    a $1,000 investment.
-   Uses Streamlit's `st.progress` and `st.dataframe` placeholders to update
    the user interface in real-time during the scan.
-   Returns a pandas DataFrame containing all the identified opportunities, sorted
    by the model's confidence level.

"""


import pandas as pd
import streamlit as st
from utils import clean_raw_data
import json


def run_unified_screener(active_advisors: dict, stock_universe: list,
                         analysis_date, investment_amount=1000, debug_mode=False, use_optimized_params=False):
    """
    Scans a universe of stocks. Can now automatically load and use optimized parameters.
    """
    recommended_trades = []
    st.subheader("📈 Unified Screener Results")
    progress_placeholder = st.empty()
    results_placeholder = st.empty()
    total_stocks = len(stock_universe)

    # Load optimized parameters if the user has requested it
    best_params = {}
    if use_optimized_params:
        try:
            with open("best_params.json", "r") as f:
                best_params = json.load(f)
            st.success("✅ Loaded optimized parameters from `best_params.json`.")
        except FileNotFoundError:
            st.warning("⚠️ `best_params.json` not found. Running with default model parameters.")

    for i, symbol in enumerate(stock_universe):
        scan_models = '/'.join(active_advisors.keys())
        progress_text = f"Scanning ({scan_models})... ({i + 1}/{total_stocks}): {symbol}"
        progress_placeholder.progress((i + 1) / total_stocks, text=progress_text)

        for advisor_name, advisor_instance in active_advisors.items():
            # Get the specific optimized params for this model, or an empty dict if none exist
            # Note: The key for the dictionary is the Class Name (e.g., "MichaAdvisor")
            model_class_name = type(advisor_instance).__name__
            params_for_model = best_params.get(model_class_name, {})

            result = advisor_instance.analyze(symbol, analysis_date, params=params_for_model)

            if result and (result.get('signal') or result.get('action')) == 'BUY':
                buy_price = result.get('current_price', 0)
                profit_target_price = result.get('profit_target_price', 0)

                net_profit_dollars = 0
                if buy_price > 0 and profit_target_price > 0:
                    shares = investment_amount / buy_price
                    gross_profit = (profit_target_price - buy_price) * shares
                    net_profit_dollars, _ = st.session_state.advisor.apply_israeli_fees_and_tax(gross_profit, shares)

                trade_info = {
                    'Symbol': symbol,
                    'Source': advisor_name,
                    'Entry Price': buy_price if buy_price > 0 else None,
                    'Profit Target ($)': profit_target_price if profit_target_price > 0 else None,
                    'Stop-Loss': result.get('stop_loss_price'),
                    'Est. Net Profit ($)': net_profit_dollars if net_profit_dollars > 0 else None,
                    # BUG FIX: Always add the Analysis Date, regardless of debug mode.
                    'Analysis Date': analysis_date.strftime('%Y-%m-%d')
                }

                if debug_mode:
                    trade_info['RSI'] = result.get('debug_rsi')

                recommended_trades.append(trade_info)

    progress_placeholder.empty()
    if not recommended_trades:
        results_placeholder.warning("No 'BUY' signals found from the selected models for this date.")
        return pd.DataFrame()

    final_df = pd.DataFrame(recommended_trades)
    formatter = {
        'Entry Price': '${:.2f}', 'Profit Target ($)': '${:.2f}', 'Stop-Loss': '${:.2f}',
        'Est. Net Profit ($)': '${:.2f}', 'RSI': '{:.2f}'
    }
    results_placeholder.dataframe(final_df.style.format(formatter, na_rep='-'), use_container_width=True)

    return final_df
