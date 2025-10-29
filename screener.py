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
from tqdm import tqdm
import streamlit as st
from stockwise_simulation import ProfessionalStockAdvisor
import pandas as pd
import streamlit as st
from stockwise_simulation import ProfessionalStockAdvisor


def find_buy_opportunities(advisor: ProfessionalStockAdvisor, stock_universe: list, analysis_date,
                           confidence_threshold=70, investment_amount=1000):
    """
    Scans a universe of stocks and displays BUY signals in real-time using a Streamlit placeholder.
    """
    recommended_trades = []

    # Create placeholders for real-time UI updates
    progress_placeholder = st.empty()
    results_placeholder = st.empty()

    total_stocks = len(stock_universe)

    for i, symbol in enumerate(stock_universe):
        progress_text = f"Scanning... ({i + 1}/{total_stocks}): {symbol}"
        progress_placeholder.progress((i + 1) / total_stocks, text=progress_text)

        _, result = advisor.run_analysis(symbol, analysis_date)

        if result and result['action'] == "BUY" and result.get('confidence', 0) > confidence_threshold:
            confidence = result.get('confidence', 0)
            buy_price = result.get('current_price', 0)
            est_profit_pct = advisor.calculate_dynamic_profit_target(confidence)

            net_profit_dollars = 0
            profit_target_price = 0  # Initialize profit target price

            if buy_price > 0:
                profit_target_price = buy_price * (1 + est_profit_pct / 100)
                hypothetical_shares = investment_amount / buy_price
                gross_profit_dollars = (profit_target_price - buy_price) * hypothetical_shares
                net_profit_dollars, _ = advisor.apply_israeli_fees_and_tax(gross_profit_dollars, hypothetical_shares)

            trade_info = {
                'Symbol': symbol,
                'Confidence': confidence,
                'Entry Price': buy_price,
                'Profit Target ($)': profit_target_price,
                'Stop-Loss': result.get('stop_loss_price'),
                'Est. Net Profit ($)': net_profit_dollars,
                'Est. Gross Profit (%)': est_profit_pct,
                'Agent': result.get('agent')
            }
            recommended_trades.append(trade_info)

            # Update the results table in real-time
            temp_df = pd.DataFrame(recommended_trades)
            if not temp_df.empty:
                temp_df = temp_df.sort_values(by='Confidence', ascending=False).reset_index(drop=True)
                results_placeholder.dataframe(temp_df.style.format({
                    'Confidence': '{:.1f}%',
                    'Entry Price': '${:.2f}',
                    'Profit Target ($)': '${:.2f}',
                    'Stop-Loss': '${:.2f}',
                    'Est. Net Profit ($)': '${:.2f}',
                    'Est. Gross Profit (%)': '{:.1f}%'
                }), use_container_width=True)

    progress_placeholder.empty()

    if not recommended_trades:
        results_placeholder.warning("No strong 'BUY' signals found for the selected date.")
        return pd.DataFrame()

    final_df = pd.DataFrame(recommended_trades).sort_values(by='Confidence', ascending=False).reset_index(drop=True)
    return final_df