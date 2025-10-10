# mico_system.py  # Micha Stock system
"""
MICO Rule-Based Trading Advisor
===============================

This script implements a rule-based trading advisor and screener inspired by the
"Micho Stock" methodology. It is designed to run in parallel with the AI-based
agents to provide an alternative, logic-based analysis.

The system focuses on identifying stocks that exhibit a combination of strong
upward trends, significant volume spikes (indicating institutional interest),
and favorable entry points that are not overbought.

Core Logic:
-----------
The `MicoAdvisor` class analyzes a stock and generates a "BUY" signal only if all
of the following three strict conditions are met:
1.  **Strong Uptrend**: The stock's current price is above its 50-day moving
    average, which in turn is above its 200-day moving average.
2.  **Volume Spike**: Within the last 5 days, the trading volume has spiked to
    more than double its 30-day average.
3.  **Good Entry Point**: The 14-day Relative Strength Index (RSI) is below 65,
    indicating the stock is not currently in an overbought condition.

If a stock meets all criteria, it is flagged as a "BUY"; otherwise, it is
considered a "HOLD".

Functionality:
--------------
-   `analyze(symbol)`: Analyzes a single stock symbol and returns a dictionary
    containing the signal ('BUY' or 'HOLD') and a human-readable string
    explaining which of the rules were met. Results are cached for 15 minutes
    for performance.
-   `run_screener(stock_universe)`: Iterates through a given list of stock
    symbols, running the `analyze` function on each. It uses a real-time
    progress bar in the Streamlit UI and returns a pandas DataFrame of all
    stocks that triggered a "BUY" signal.

"""

import pandas as pd
import pandas_ta as ta
import streamlit as st
from data_source_manager import DataSourceManager


class MicoAdvisor:
    """
    A rule-based trading advisor inspired by the "Micho Stock" methodology,
    focusing on trend, volume, and momentum to find opportunities.
    """

    def __init__(self, data_manager: DataSourceManager):
        self.dm = data_manager

    @st.cache_data(ttl=900)  # Cache results for 15 minutes
    def analyze(_self, symbol: str) -> dict:
        """
        Analyzes a single stock based on the MICO rule-based system.
        Note: _self is used because st.cache_data hashes the first argument.
        """
        # 1. Fetch data
        df = _self.dm.get_stock_data(symbol, days_back=300)
        if df.empty or len(df) < 200:
            return {'signal': 'WAIT', 'reason': 'Insufficient historical data.'}

        # 2. Calculate indicators
        df.ta.sma(length=50, append=True, col_names='SMA_50')
        df.ta.sma(length=200, append=True, col_names='SMA_200')
        df.ta.rsi(length=14, append=True, col_names='RSI_14')
        df['volume_ma_30'] = df['volume'].rolling(30).mean()

        latest = df.iloc[-1]

        # 3. Apply rules
        is_uptrend = latest['close'] > latest['SMA_50'] and latest['SMA_50'] > latest['SMA_200']

        recent_volume = df.tail(5)
        is_volume_spike = (recent_volume['volume'] > recent_volume['volume_ma_30'] * 2).any()

        is_not_overbought = latest['RSI_14'] < 65

        # 4. Generate signal
        reasons = []
        if is_uptrend:
            reasons.append("✅ Strong Uptrend")
        else:
            reasons.append("❌ Not in Uptrend")

        if is_volume_spike:
            reasons.append("✅ Recent Volume Spike (Smart Money)")
        else:
            reasons.append("❌ No Recent Volume Spike")

        if is_not_overbought:
            reasons.append("✅ Good Entry Point (Not Overbought)")
        else:
            reasons.append("❌ Potentially Overbought (RSI > 65)")

        reason_string = " | ".join(reasons)

        if is_uptrend and is_volume_spike and is_not_overbought:
            return {'signal': 'BUY', 'reason': reason_string}

        return {'signal': 'HOLD', 'reason': reason_string}

    def run_screener(self, stock_universe: list) -> pd.DataFrame:
        """
        Runs the MICO analysis across a list of stocks to find BUY opportunities.
        """
        st.subheader("MICO System Scan")
        opportunities = []
        progress_placeholder = st.empty()

        for i, symbol in enumerate(stock_universe):
            progress_text = f"Scanning with MICO... ({i + 1}/{len(stock_universe)}): {symbol}"
            progress_placeholder.progress((i + 1) / len(stock_universe), text=progress_text)

            result = self.analyze(symbol)
            if result['signal'] == 'BUY':
                opportunities.append({'Symbol': symbol, 'MICO Signal': 'BUY', 'MICO Reason': result['reason']})

        progress_placeholder.empty()
        return pd.DataFrame(opportunities)