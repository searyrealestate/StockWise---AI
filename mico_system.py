# mico_system.py  # Micha Stock system
"""
Micha (MICO) Rule-Based Trading Advisor
========================================

This script implements a dynamic, rule-based trading advisor (Micha System).
It is designed to run in parallel with the AI-based agents and can be
optimized using a grid-search backtester.

The system analyzes stocks based on a flexible set of rules for trend,
momentum, and entry points.

Core Logic:
-----------
The `MichaAdvisor` class analyzes a stock and generates a "BUY" signal if
all of the following conditions (based on *default* parameters) are met:
1.  **Strong Uptrend**: The stock's price is above its 50-day moving
    average, which in turn is above its 200-day moving average.
2.  **Momentum Confirmation**: The MACD line is currently above its signal line.
3.  **Good Entry Point**: The 14-day Relative Strength Index (RSI) is below 70,
    indicating the stock is not in an overbought condition.

All of these parameters (SMAs, RSI, MACD, etc.) are configurable via a
`params` dictionary passed to the `analyze` method.

Functionality:
--------------
-   `analyze(symbol, analysis_date, params=None)`: Analyzes a single stock
    symbol for a specific historical date. It uses the provided `params`
    dictionary to configure its rules. If a "BUY" signal is found, it
    returns a dictionary containing the signal, reasoning, and
    calculated stop-loss and profit-target prices.
-   `run_screener(stock_universe)`: Iterates through a given list of stock
    symbols, running the `analyze` function on each. It uses a real-time
    progress bar in the Streamlit UI and returns a pandas DataFrame of all
    stocks that triggered a "BUY" signal.

"""


import pandas as pd
import pandas_ta as ta
import streamlit as st
from data_source_manager import DataSourceManager
from utils import clean_raw_data


class MichaAdvisor:
    """
    A rule-based trading advisor (MICO/Micha System).
    Analyzes stocks based on a predefined set of technical indicator rules.
    """

    def __init__(self, data_manager: DataSourceManager, logger=None):
        self.dm = data_manager
        # Use a dummy logger if none is provided, prevents crashes
        self.log = logger if logger else lambda msg, level="INFO": None

    def analyze(_self, symbol: str, analysis_date, params: dict = None) -> dict:
        _self.log(f"MICO: Starting analysis for {symbol} on {analysis_date}.", "INFO")
        if params is None: params = {}

        # --- Make all rules dynamic based on params ---
        sma_short = params.get('sma_short', 50)
        sma_long = params.get('sma_long', 200)
        rsi_period = params.get('rsi_period', 14)
        rsi_threshold = params.get('rsi_threshold', 70)
        macd_fast = params.get('macd_fast', 12)
        macd_slow = params.get('macd_slow', 26)
        macd_signal = params.get('macd_signal', 9)
        atr_period = params.get('atr_period', 14)
        atr_mult_stop = params.get('atr_mult_stop', 2.0)
        atr_mult_profit = params.get('atr_mult_profit', 2.0)
        stop_loss_mode = params.get('stop_loss_mode', 'atr')  # 'atr' or 'ma'
        ma_stop_period = params.get('ma_stop_period', 50)  # e.g., use 50-day MA as stop

        # Use sma_long as minimum data requirement
        df_raw = _self.dm.get_stock_data(symbol, days_back=sma_long + 50)
        if df_raw.empty:
            _self.log(f"MICO: Failed to download data for {symbol}.", "WARN")
            return {'signal': 'WAIT', 'reason': f'Failed to download data for {symbol} (Network Error).'}

        df_slice = df_raw[df_raw.index <= pd.to_datetime(analysis_date)]

        # if df_slice.empty or len(df_slice) < sma_long:
        #     return {'signal': 'WAIT', 'reason': 'Insufficient historical data for this date.'}
        if df_slice.empty:
            _self.log(f"MICO: Data error after calculating indicators for {symbol}.", "WARN")
            return {'signal': 'WAIT', 'reason': 'Data error after calculating indicators.'}

        # --- Apply indicators with dynamic lengths ---
        df_slice.ta.sma(length=sma_short, append=True)
        df_slice.ta.sma(length=sma_long, append=True)

        # --- Ensure the MA for the stop-loss is calculated ---
        if stop_loss_mode == 'ma' and f'sma_{ma_stop_period}' not in (f'sma_{sma_short}', f'sma_{sma_long}'):
            _self.log(f"MICO: Calculating extra SMA-{ma_stop_period} for stop-loss.", "DEBUG")
            df_slice.ta.sma(length=ma_stop_period, append=True)

        df_slice.ta.rsi(length=rsi_period, append=True)
        df_slice.ta.macd(fast=macd_fast, slow=macd_slow, signal=macd_signal, append=True)
        df_slice.ta.atr(length=atr_period, append=True)
        df_slice.columns = [col.lower() for col in df_slice.columns]
        df_slice.dropna(inplace=True)

        if df_slice.empty: return {'signal': 'WAIT', 'reason': 'Data error after calculating indicators.'}

        latest = df_slice.iloc[-1]
        reasons, buy_conditions_met = [], 0

        # --- Check rules using dynamic column names ---
        if latest['close'] > latest[f'sma_{sma_short}'] and latest[f'sma_{sma_short}'] > latest[f'sma_{sma_long}']:
            buy_conditions_met += 1
            reasons.append(f"✅ Price > {sma_short}-day SMA & {sma_short} > {sma_long}-day SMA.")
        else:
            _self.log(f"MICO: {symbol} failed trend check.", "DEBUG")
            reasons.append(f"❌ Price not in uptrend (SMA {sma_short}/{sma_long}).")

        if latest[f'rsi_{rsi_period}'] < rsi_threshold:
            buy_conditions_met += 1
            reasons.append(f"✅ RSI ({latest[f'rsi_{rsi_period}']:.1f}) < {rsi_threshold}.")
        else:
            _self.log(f"MICO: {symbol} failed RSI check ({latest[f'rsi_{rsi_period}']:.1f}).", "DEBUG")
            reasons.append(f"❌ RSI ({latest[f'rsi_{rsi_period}']:.1f}) >= {rsi_threshold}.")

        # --- CORRECTED MACD Column Name Check ---
        macd_col = f"macd_{macd_fast}_{macd_slow}_{macd_signal}"
        signal_col = f"macds_{macd_fast}_{macd_slow}_{macd_signal}"  # Note the 's'

        if latest[macd_col] > latest[signal_col]:
            buy_conditions_met += 1
            reasons.append("✅ MACD > signal line.")
        else:
            _self.log(f"MICO: {symbol} failed MACD check.", "DEBUG")
            reasons.append("❌ MACD < signal line.")

        # --- All 3 conditions must be met ---
        if buy_conditions_met == 3:
            _self.log(f"MICO: BUY signal triggered for {symbol}.", "INFO")
            current_price = latest['close']
            atr_value = latest[f'atr_{atr_period}']

            # --- ADD FUNDAMENTALS ---
            pe_ratio, ps_ratio, de_ratio = None, None, None
            try:
                info = _self.dm.get_fundamental_info(symbol)
                if info:
                    pe_ratio = info.get('trailingPE')
                    ps_ratio = info.get('priceToSalesTrailing12Months')
                    de_ratio = info.get('debtToEquity')
                    _self.log(f"MICO: Fetched fundamentals for {symbol}. PE: {pe_ratio}", "DEBUG")
            except Exception as e:
                _self.log(f"MICO: Could not fetch fundamentals for {symbol}. Error: {e}", "WARN")

                # --- START OF LOGIC THAT WAS INDENTED WRONG ---
                # Now it is OUTSIDE the 'except' block

                # --- Dynamic Stop-Loss Logic ---
                if stop_loss_mode == 'ma':
                    stop_loss_price = latest[f'sma_{ma_stop_period}']
                    _self.log(f"MICO: {symbol} using MA-based stop at {stop_loss_price:.2f}", "DEBUG")
                else:  # Default to ATR
                    stop_loss_price = current_price - (atr_value * atr_mult_stop)
                    _self.log(f"MICO: {symbol} using ATR-based stop at {stop_loss_price:.2f}", "DEBUG")

                risk = current_price - stop_loss_price

                # --- Safety check ---
                if risk <= 0:
                    _self.log(f"MICO: {symbol} invalid risk ({risk}). Price is below stop-loss. Defaulting to ATR.",
                              "WARN")
                    stop_loss_price = current_price - (atr_value * atr_mult_stop)  # Default to ATR
                    risk = current_price - stop_loss_price

                    # Profit target should be ABOVE the current price
                profit_target_price = current_price + (risk * atr_mult_profit)  # Use R/R

                return {
                    'signal': 'BUY', 'reason': "\n".join(reasons),
                    'current_price': current_price,
                    'stop_loss_price': stop_loss_price,
                    'profit_target_price': profit_target_price,
                    'debug_rsi': latest[f'rsi_{rsi_period}'],
                    'PE Ratio': pe_ratio,  # Add to output
                    'P/S Ratio': ps_ratio,  # Add to output
                    'Debt/Equity': de_ratio  # Add to output
                }

        _self.log(f"MICO: {symbol} did not meet all conditions. Signal: WAIT.", "DEBUG")
        return {'signal': 'WAIT', 'reason': "\n".join(reasons)}

    # def run_screener(self, stock_universe: list):
    #     """Scans a universe of stocks and displays Micha BUY signals in a Streamlit UI."""
    #     st.subheader("📜 Micha System Screener Results")
    #
    #     recommended_trades = []
    #     progress_placeholder = st.empty()
    #     results_placeholder = st.empty()
    #
    #     for i, symbol in enumerate(stock_universe):
    #         progress_text = f"Scanning... ({i + 1}/{len(stock_universe)}): {symbol}"
    #         progress_placeholder.progress((i + 1) / len(stock_universe), text=progress_text)
    #
    #         result = self.analyze(symbol)
    #         if result and result['signal'] == 'BUY':
    #             trade_info = {
    #                 'Symbol': symbol,
    #                 'Source': 'Micha',  # Identifies the source system
    #                 'Reason': result.get('reason', 'N/A')
    #             }
    #             recommended_trades.append(trade_info)
    #
    #             # Update table in real-time
    #             temp_df = pd.DataFrame(recommended_trades)
    #             results_placeholder.dataframe(temp_df, use_container_width=True)
    #
    #     progress_placeholder.empty()
    #     if not recommended_trades:
    #         results_placeholder.warning("Micha System found no BUY signals in this universe.")

    def run_screener(self, stock_universe: list):
        """
        Scans a universe of stocks and displays Micha BUY signals in a Streamlit UI.
        This function is now DEPRECATED and logic is handled by screener.py
        """
        st.warning("MichaAdvisor.run_screener() is deprecated. Use screener.py.", icon="⚠️")
        return pd.DataFrame()