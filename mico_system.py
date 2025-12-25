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

"""


import pandas as pd
import pandas_ta as ta
import streamlit as st
from data_source_manager import DataSourceManager
from utils import clean_raw_data
from technical_analyzer import TechnicalAnalyzer
import logging
import fundamental_analyzer
import chart_pattern_recognizer
import json
import system_config as cfg
import os


class MichaAdvisor:
    """
    A rule-based trading advisor (MICO/Micha System).
    Analyzes stocks based on a predefined set of technical indicator rules.
    """
    def __init__(self, data_manager: DataSourceManager):
        self.dm = data_manager
        self.log = logging.getLogger(type(self).__name__)
        # --- Instantiate a Ticker object cache ---
        self._ticker_cache = {}

    # --- Helper to get cached ticker ---
    def _get_ticker(self, symbol):
        if symbol not in self._ticker_cache:
            self._ticker_cache[symbol] = fundamental_analyzer.get_ticker_object(symbol)
        return self._ticker_cache[symbol]

    def get_best_params(self, symbol: str) -> dict:
        """
        Attempts to load optimized parameters for a specific stock from models/Gen-4.
        Returns a dictionary of the best parameters found, or empty dict if not found.
        """
        try:
            # Construct path: models/Gen-4/optimization_results_SYMBOL.json
            path = os.path.join("models", "Gen-4", f"optimization_results_{symbol}.json")

            if os.path.exists(path):
                # Load JSON
                with open(path, 'r') as f:
                    data = json.load(f)

                # If it's a list (orient='records'), take the first item (Best Result)
                if isinstance(data, list) and len(data) > 0:
                    best_config = data[0]
                    # Filter out non-parameter fields like 'Total Return'
                    clean_params = {k: v for k, v in best_config.items() if
                                    k not in ['Total Return', 'Win Rate', 'Trades']}
                    self.log.info(f"✅ Loaded optimized params for {symbol}")
                    return clean_params

        except Exception as e:
            self.log.warning(f"Could not load optimized params for {symbol}: {e}")

        return {}

    def _extract_and_validate_params(self, params: dict) -> dict:
        """Extracts, casts, and validates all parameters from the dictionary."""
        if params is None:
            params = {}

        # SMAs
        sma_short = int(params.get('sma_short', cfg.STRATEGY_PARAMS.get('sma_short', 50)))
        sma_long = int(params.get('sma_long', cfg.STRATEGY_PARAMS.get('sma_long', 200)))

        # Standard MA/RSI/MACD indicators (using existing defaults as they are universal TA)
        rsi_period = int(params.get('rsi_period', 14))
        rsi_threshold = int(params.get('rsi_threshold', cfg.STRATEGY_PARAMS.get('rsi_threshold', 70)))
        macd_fast = int(params.get('macd_fast', 12))
        macd_slow = int(params.get('macd_slow', 26))
        macd_signal = int(params.get('macd_signal', 9))

        # ATR / Stop Loss
        atr_period = int(params.get('atr_period', 14))
        ma_stop_period = int(params.get('ma_stop_period', 50))

        # Minimum conditions
        min_conditions_to_buy = int(params.get('min_conditions_to_buy', 3))

        # Earnings
        earnings_lookahead_days = int(params.get('earnings_lookahead_days', 7))

        # --- Correctly cast FLOATS (for multipliers, ratios, decimals) ---
        atr_mult_stop = float(params.get('atr_mult_stop', cfg.STRATEGY_PARAMS.get('atr_mult_stop', 2.0)))
        atr_mult_profit = float(params.get('atr_mult_profit', 2.0))
        max_debt_equity = float(params.get('max_debt_equity', 2.0))
        min_pe_ratio = float(params.get('min_pe_ratio', 5.0))
        max_atr_quantile = float(params.get('max_atr_quantile', 0.75))

        # --- Correctly cast STRINGS and BOOLEANS ---
        stop_loss_mode = str(params.get('stop_loss_mode', 'atr'))
        use_volume_check = bool(params.get('use_volume_check', True))
        use_candlestick_check = bool(params.get('use_candlestick_check', True))
        use_fundamental_filter = bool(params.get('use_fundamental_filter', False))
        use_continuation_check = bool(params.get('use_continuation_check', False))
        use_atr_filter = bool(params.get('use_atr_filter', False))
        use_multi_timeframe = bool(params.get('use_multi_timeframe', False))
        use_earnings_filter = bool(params.get('use_earnings_filter', False))

        # # --- Correctly cast INTEGERS (for pandas-ta lengths, counters) ---
        # sma_150_len = int(params.get('sma_150_len', 150))
        # sma_200_len = int(params.get('sma_200_len', 200))
        # # Note: The original file has redundancy, using only the necessary ones below
        # sma_slope_period = int(params.get('sma_slope_period', 3))
        # sma_short = int(params.get('sma_short', 50))
        # sma_long = int(params.get('sma_long', 200))
        # rsi_period = int(params.get('rsi_period', 14))
        # rsi_threshold = int(params.get('rsi_threshold', 70))
        # macd_fast = int(params.get('macd_fast', 12))
        # macd_slow = int(params.get('macd_slow', 26))
        # macd_signal = int(params.get('macd_signal', 9))
        # atr_period = int(params.get('atr_period', 14))
        # ma_stop_period = int(params.get('ma_stop_period', 50))
        # weekly_sma_short = int(params.get('weekly_sma_short', 10))
        # weekly_sma_long = int(params.get('weekly_sma_long', 40))
        # min_conditions_to_buy = int(params.get('min_conditions_to_buy', 3))
        # earnings_lookahead_days = int(params.get('earnings_lookahead_days', 7))
        #
        # # --- Correctly cast FLOATS (for multipliers, ratios, decimals) ---
        # atr_mult_stop = float(params.get('atr_mult_stop', 2.0))
        # atr_mult_profit = float(params.get('atr_mult_profit', 2.0))
        # max_debt_equity = float(params.get('max_debt_equity', 2.0))
        # min_pe_ratio = float(params.get('min_pe_ratio', 5.0))
        # max_atr_quantile = float(params.get('max_atr_quantile', 0.75))
        #
        # # --- Correctly cast STRINGS and BOOLEANS ---
        # stop_loss_mode = str(params.get('stop_loss_mode', 'atr'))
        # use_volume_check = bool(params.get('use_volume_check', True))
        # use_candlestick_check = bool(params.get('use_candlestick_check', True))
        # use_fundamental_filter = bool(params.get('use_fundamental_filter', False))
        # use_continuation_check = bool(params.get('use_continuation_check', False))
        # use_atr_filter = bool(params.get('use_atr_filter', False))
        # use_multi_timeframe = bool(params.get('use_multi_timeframe', False))
        # use_earnings_filter = bool(params.get('use_earnings_filter', False))

        # Re-pack into a validated dictionary
        validated_params = {k: v for k, v in locals().items() if not k.startswith('_') and k != 'params'}

        return validated_params

    def _run_technical_analysis(self, df_slice: pd.DataFrame, params: dict, symbol: str) -> pd.DataFrame:
        """
        Calculates all technical indicators and runs pre-filters (Multi-Timeframe, ATR).
        """
        weekly_sma_short = int(params.get('weekly_sma_short', cfg.STRATEGY_PARAMS.get('weekly_sma_short', 10)))
        weekly_sma_long = int(params.get('weekly_sma_long', cfg.STRATEGY_PARAMS.get('weekly_sma_long', 40)))

        # --- Multi-Timeframe Check (Early Exit Filter) ---
        if params['use_multi_timeframe']:
            # Resampling and calculation logic is complex but kept here for isolation.
            weekly_df = df_slice.resample('W').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
            })
            weekly_df[f'sma_w_{weekly_sma_short}'] = weekly_df['close'].rolling(window=weekly_sma_short).mean()
            weekly_df[f'sma_w_{weekly_sma_long}'] = weekly_df['close'].rolling(window=weekly_sma_long).mean()
            weekly_df.dropna(inplace=True)

            if weekly_df.empty or not (weekly_df.iloc[-1][f'sma_w_{weekly_sma_short}'] > weekly_df.iloc[-1][
                f'sma_w_{weekly_sma_long}']):
                # Return empty to signify failure, which is handled in the main analyze function.
                self.log.info(f"MICO: {symbol} failed Multi-Timeframe filter.")
                return pd.DataFrame()

        # --- Calculate Indicators ---
        df_slice.columns = [col.lower() for col in df_slice.columns]

        # SMAs
        df_slice.ta.sma(length=params['sma_short'], append=True)
        df_slice.ta.sma(length=params['sma_long'], append=True)
        if params['stop_loss_mode'] == 'ma' and params['ma_stop_period'] not in (
        params['sma_short'], params['sma_long']):
            df_slice.ta.sma(length=params['ma_stop_period'], append=True)

        # RSI, MACD, ATR
        df_slice.ta.rsi(length=params['rsi_period'], append=True)
        df_slice.ta.macd(fast=params['macd_fast'], slow=params['macd_slow'], signal=params['macd_signal'],
                         append=True)
        df_slice.ta.atr(length=params['atr_period'], append=True)

        df_slice.columns = [col.lower() for col in df_slice.columns]
        df_slice.dropna(inplace=True)

        # --- ATR Volatility Filter (Post-Calculation Filter) ---
        if params['use_atr_filter'] and not df_slice.empty:
            atr_col = f'atrr_{params["atr_period"]}'
            atr_threshold = df_slice[atr_col].quantile(params['max_atr_quantile'])
            latest_atr = df_slice[atr_col].iloc[-1]

            if latest_atr > atr_threshold:
                self.log.info(f"MICO: {symbol} failed ATR Volatility filter.")
                return pd.DataFrame()  # Signal filter failure

        return df_slice

        # Add this method to MichaAdvisor class
    def _calculate_risk_management(self, latest: pd.Series, current_price: float, params: dict, analyzer) -> tuple:
        """Calculates Stop-Loss and Profit Target prices."""
        atr_value = latest[f'atrr_{params["atr_period"]}']
        atr_mult_stop = params['atr_mult_stop']

        # --- 1. Stop-Loss Logic ---
        if params['stop_loss_mode'] == 'support':
            sr_levels = analyzer.find_support_resistance()
            nearest_support = sr_levels['support'][-1] if sr_levels['support'] else None
            stop_loss_price = nearest_support * 0.99 if nearest_support else current_price - (
                        atr_value * atr_mult_stop)
        elif params['stop_loss_mode'] == 'ma':
            stop_loss_price = latest[f'sma_{params["ma_stop_period"]}']
        else:  # Default to ATR
            stop_loss_price = current_price - (atr_value * atr_mult_stop)

        risk = current_price - stop_loss_price

        # Safety check for risk calculation
        if risk <= 0:
            stop_loss_price = current_price - (atr_value * atr_mult_stop)
            risk = current_price - stop_loss_price

        # --- 2. Take-Profit Logic ---
        # The original code's logic is kept for profit targeting
        sr_levels = analyzer.find_support_resistance()  # Recalculated if not done in SL logic
        nearest_resistance = sr_levels['resistance'][0] if sr_levels['resistance'] else None

        if nearest_resistance and nearest_resistance > current_price:
            profit_target_price = nearest_resistance
        else:
            profit_target_price = current_price + (risk * params['atr_mult_profit'])

        return stop_loss_price, profit_target_price

    # def analyze(_self, stock_data, symbol: str, analysis_date, params: dict = None,
    #             use_market_filter: bool = True) -> dict:
    #     _self.log.info(f"MICO: Starting analysis for {symbol} on {analysis_date}.")
    #     if params is None: params = {}
    #
    #     # --- 1. Get All Parameters ---
    #     # --- Correctly cast INTEGERS (for pandas-ta lengths, counters) ---
    #     sma_150_len = int(params.get('sma_150_len', 150))
    #     sma_200_len = int(params.get('sma_200_len', 200))
    #     sma_slope_period = int(params.get('sma_slope_period', 3))
    #     # Stop-Loss mode 'sma_150'
    #     stop_loss_mode = str(params.get('stop_loss_mode', 'sma_150'))
    #     atr_mult_stop = float(params.get('atr_mult_stop', 2.0))
    #     # New filter flags
    #     use_fundamental_filter = bool(params.get('use_fundamental_filter', True))
    #     use_pattern_filter = bool(params.get('use_pattern_filter', False))
    #     sma_short = int(params.get('sma_short', 50))
    #     sma_long = int(params.get('sma_long', 200))
    #     rsi_period = int(params.get('rsi_period', 14))
    #     rsi_threshold = int(params.get('rsi_threshold', 70))
    #     macd_fast = int(params.get('macd_fast', 12))
    #     macd_slow = int(params.get('macd_slow', 26))
    #     macd_signal = int(params.get('macd_signal', 9))
    #     atr_period = int(params.get('atr_period', 14))
    #     ma_stop_period = int(params.get('ma_stop_period', 50))
    #     weekly_sma_short = int(params.get('weekly_sma_short', 10))
    #     weekly_sma_long = int(params.get('weekly_sma_long', 40))
    #     min_conditions_to_buy = int(params.get('min_conditions_to_buy', 3))
    #     earnings_lookahead_days = int(params.get('earnings_lookahead_days', 7))
    #
    #     # --- Correctly cast FLOATS (for multipliers, ratios, decimals) ---
    #     atr_mult_stop = float(params.get('atr_mult_stop', 2.0))
    #     atr_mult_profit = float(params.get('atr_mult_profit', 2.0))
    #     max_debt_equity = float(params.get('max_debt_equity', 2.0))
    #     min_pe_ratio = float(params.get('min_pe_ratio', 5.0))
    #     max_atr_quantile = float(params.get('max_atr_quantile', 0.75))
    #
    #     # --- Correctly cast STRINGS (for text-based modes) ---
    #     stop_loss_mode = str(params.get('stop_loss_mode', 'atr'))
    #
    #     # --- Correctly cast BOOLEANS (for True/False flags) ---
    #     use_volume_check = bool(params.get('use_volume_check', True))
    #     use_candlestick_check = bool(params.get('use_candlestick_check', True))
    #     use_fundamental_filter = bool(params.get('use_fundamental_filter', False))
    #     use_continuation_check = bool(params.get('use_continuation_check', False))
    #     use_atr_filter = bool(params.get('use_atr_filter', False))
    #     use_multi_timeframe = bool(params.get('use_multi_timeframe', False))
    #     use_earnings_filter = bool(params.get('use_earnings_filter', False))
    #
    #     # --- 2. Get Data ---
    #     # DATA IS NOW PASSED IN - NO NEED TO RE-FETCH
    #     df_raw = clean_raw_data(stock_data)
    #
    #     # --- DEBUG 1: Log incoming data ---
    #     _self.log.debug(
    #         f"[{symbol}]: MICO received data shape: {df_raw.shape}. Date range: {df_raw.index.min()} to {df_raw.index.max()}")
    #
    #     # --- Remove timezone info before comparing ---
    #     if pd.api.types.is_datetime64_any_dtype(df_raw.index) and df_raw.index.tz is not None:
    #         df_raw.index = df_raw.index.tz_localize(None)
    #         _self.log.debug(f"[{symbol}]: MICO converted data index to tz-naive.")
    #
    #     # Added datetime conversion for slicing ---
    #     analysis_date_dt = pd.to_datetime(analysis_date)
    #
    #     if df_raw.empty or len(df_raw) < sma_200_len:
    #         _self.log.warning(f"MICO: Skipping {symbol}, not enough data.")
    #         return {'signal': 'WAIT', 'reason': f'Not enough data ({len(df_raw)} < {sma_200_len} days).'}
    #
    #     # Check if we have enough data for the long-term moving average
    #     if len(df_raw) < sma_long:
    #         _self.log.warning(f"MICO: Skipping {symbol}, not enough data ({len(df_raw)} < {sma_long} days).")
    #         return {'signal': 'WAIT', 'reason': f'Not enough data ({len(df_raw)} < {sma_long} days).'}
    #
    #     # Sliced data is now created from df_raw using the correct date ---
    #     df_slice = df_raw[df_raw.index <= analysis_date_dt]
    #
    #     if df_slice.empty:
    #         _self.log.warning(f"MICO: Data error after calculating indicators for {symbol}.")
    #         return {'signal': 'WAIT', 'reason': 'Data error after calculating indicators.'}
    #
    #     reasons, buy_conditions_met = [], 0
    #
    #     # --- 3. Multi-Timeframe Check (runs BEFORE daily indicators) ---
    #     if use_multi_timeframe:
    #         weekly_df = df_slice.resample('W').agg({
    #             'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    #         })
    #         try:
    #             # --- Validate BOTH window lengths ---
    #             weekly_sma_short = int(weekly_sma_short)
    #             weekly_sma_long = int(weekly_sma_long)
    #
    #             if weekly_sma_short <= 0:
    #                 weekly_sma_short = 10  # fallback default
    #             if weekly_sma_long <= 0:
    #                 weekly_sma_long = 40  # fallback default
    #
    #         except (TypeError, ValueError):
    #             # Fallback in case a non-integer is passed
    #             weekly_sma_short = 10
    #             weekly_sma_long = 40
    #
    #             # Now both are guaranteed to be integers
    #         weekly_df[f'sma_w_{weekly_sma_short}'] = weekly_df['close'].rolling(window=weekly_sma_short).mean()
    #         weekly_df[f'sma_w_{weekly_sma_long}'] = weekly_df['close'].rolling(window=weekly_sma_long).mean()
    #         weekly_df.dropna(inplace=True)
    #
    #         if weekly_df.empty:
    #             return {'signal': 'WAIT', 'reason': 'Not enough data for weekly analysis.'}
    #
    #         latest_weekly = weekly_df.iloc[-1]
    #         if not (latest_weekly[f'sma_w_{weekly_sma_short}'] > latest_weekly[f'sma_w_{weekly_sma_long}']):
    #             reason = f"❌ Failed Multi-Timeframe check: Weekly SMA {weekly_sma_short} is not above {weekly_sma_long}."
    #             _self.log.info(f"MICO: {symbol} failed filter. {reason}")
    #             return {'signal': 'WAIT', 'reason': reason}
    #
    #         reasons.append("✅ Passed Multi-Timeframe (Weekly) check.")
    #         buy_conditions_met += 1
    #
    #     # --- Earnings and Events Exclusion (Step 5) ---
    #     if use_earnings_filter:
    #         next_earnings_date = _self.dm.get_earnings_calendar(symbol)
    #         if next_earnings_date:
    #             days_to_earnings = (next_earnings_date - analysis_date).days
    #
    #             if 0 <= days_to_earnings <= earnings_lookahead_days:
    #                 reason = f"❌ Failed Earnings check: Earnings are in {days_to_earnings} days (on {next_earnings_date})."
    #                 _self.log.info(f"MICO: {symbol} failed filter. {reason}")
    #                 return {'signal': 'WAIT', 'reason': reason}
    #
    #         # If we pass (or no date is found), we can add it to the reasons
    #         reasons.append("✅ Passed Earnings Calendar check.")
    #
    #     # --- 4. Calculate ALL Daily Indicators ---
    #     df_slice.columns = [col.lower() for col in df_slice.columns]
    #
    #     df_slice.ta.sma(length=sma_short, append=True)
    #     df_slice.ta.sma(length=sma_long, append=True)
    #     if stop_loss_mode == 'ma' and f'sma_{ma_stop_period}' not in (f'sma_{sma_short}', f'sma_{sma_long}'):
    #         df_slice.ta.sma(length=ma_stop_period, append=True)
    #
    #     df_slice.ta.rsi(length=rsi_period, append=True)
    #     df_slice.ta.macd(fast=macd_fast, slow=macd_slow, signal=macd_signal, append=True)
    #
    #     df_slice.ta.atr(length=atr_period, append=True)
    #
    #     # Now we apply lowercase to the NEW indicator columns (e.g., SMA_50 -> sma_50)
    #     df_slice.columns = [col.lower() for col in df_slice.columns]
    #     df_slice.dropna(inplace=True)
    #
    #     if df_slice.empty: return {'signal': 'WAIT', 'reason': 'Data error after calculating indicators.'}
    #
    #     # --- 5. Run ATR Filter (now AFTER ATR is calculated) ---
    #     if use_atr_filter:
    #         atr_threshold = df_slice[f'atrr_{atr_period}'].quantile(max_atr_quantile)  # <-- f'atrr_{atr_period}'
    #         latest_atr = df_slice[f'atrr_{atr_period}'].iloc[-1]  # <-- f'atrr_{atr_period}'
    #
    #         if latest_atr > atr_threshold:
    #             reason = f"❌ Failed ATR Volatility check: Current ATR ({latest_atr:.2f}) > {max_atr_quantile * 100}th Percentile ({atr_threshold:.2f}). Market too choppy."
    #             _self.log.info(f"MICO: {symbol} failed filter. {reason}")
    #             return {'signal': 'WAIT', 'reason': reason}
    #
    #         reasons.append("✅ Passed ATR Volatility (Low/Moderate) check.")
    #         buy_conditions_met += 1
    #
    #     # --- 6. Run All Other Checks ---
    #     analyzer = TechnicalAnalyzer(df_slice, _self.log)
    #     latest = analyzer.df.iloc[-1]
    #
    #     # Trend Check
    #     if latest['close'] > latest[f'sma_{sma_short}'] and latest[f'sma_{sma_short}'] > latest[f'sma_{sma_long}']:
    #         buy_conditions_met += 1
    #         reasons.append(f"✅ Price > {sma_short}-day SMA & {sma_short} > {sma_long}-day SMA.")
    #     else:
    #         reasons.append(f"❌ Price not in uptrend (SMA {sma_short}/{sma_long}).")
    #
    #     # RSI Check
    #     if latest[f'rsi_{rsi_period}'] < rsi_threshold:
    #         buy_conditions_met += 1
    #         reasons.append(f"✅ RSI ({latest[f'rsi_{rsi_period}']:.1f}) < {rsi_threshold}.")
    #     else:
    #         reasons.append(f"❌ RSI ({latest[f'rsi_{rsi_period}']:.1f}) >= {rsi_threshold}.")
    #
    #     # MACD Check
    #     if latest[f"macd_{macd_fast}_{macd_slow}_{macd_signal}"] > latest[
    #         f"macds_{macd_fast}_{macd_slow}_{macd_signal}"]:
    #         buy_conditions_met += 1
    #         reasons.append("✅ MACD > signal line.")
    #     else:
    #         reasons.append("❌ MACD < signal line.")
    #
    #     # Volume Check
    #     if use_volume_check:
    #         if analyzer.check_volume_breakout():
    #             buy_conditions_met += 1
    #             reasons.append("✅ Volume is strong (above 20-day avg).")
    #         else:
    #             reasons.append("❌ Volume is weak (below 20-day avg).")
    #
    #     # Candlestick Check
    #     if use_candlestick_check:
    #         if analyzer.check_bullish_candlestick():
    #             buy_conditions_met += 1
    #             reasons.append("✅ Bullish candlestick pattern found.")
    #         else:
    #             reasons.append("❌ No bullish candlestick pattern found.")
    #
    #     # Continuation Check
    #     if use_continuation_check:
    #         if analyzer.check_continuation_candlestick():
    #             buy_conditions_met += 1
    #             reasons.append("✅ Indecision/continuation pattern found (Doji/Spinning Top).")
    #         else:
    #             reasons.append("❌ No indecision/continuation pattern found.")
    #
    #     # --- 7. Tally Total Conditions ---
    #     total_conditions = 3 + int(use_volume_check) + int(use_candlestick_check) + \
    #                        int(use_continuation_check) + int(use_multi_timeframe) + int(use_atr_filter)
    #
    #     # --- 8. Final Signal Logic ---
    #     if buy_conditions_met >= min_conditions_to_buy:
    #
    #         # --- START: "NEXT DAY" MODIFICATIONS ---
    #
    #         # 8a. Find the *next* trading day's data (from the full df_raw)
    #         next_day_data = df_raw[df_raw.index > analysis_date_dt]
    #         if next_day_data.empty:
    #             _self.log.warning(f"MICO: BUY signal for {symbol}, but no future data found.")
    #             return {'signal': 'WAIT', 'reason': 'BUY conditions met, but no next-day data.'}
    #
    #         # 8b. Get the *actual* trade info
    #         actual_trade_row = next_day_data.iloc[0]
    #         current_price = actual_trade_row['open']  # Buy at next day's OPEN
    #
    #         # 8c. Get ATR from *decision day* (the 'latest' row) for SL/TP calculation
    #         atr_value = latest[f'atrr_{atr_period}']  # Based on 'latest'
    #
    #         # --- END: "NEXT DAY" MODIFICATIONS ---
    #
    #         # Fundamental Fetch
    #         pe_ratio, ps_ratio, de_ratio = None, None, None
    #         if use_fundamental_filter:
    #             try:
    #                 info = _self.dm.get_fundamental_info(symbol)
    #                 if info:
    #                     pe_ratio = info.get('trailingPE')
    #                     ps_ratio = info.get('priceToSalesTrailing12Months')
    #                     de_ratio = info.get('debtToEquity')
    #             except Exception as e:
    #                 _self.log.warning(f"MICO: Could not fetch fundamentals for {symbol}. Error: {e}", exc_info=True)
    #
    #             if de_ratio is not None and de_ratio > max_debt_equity:
    #                 return {'signal': 'WAIT',
    #                         'reason': f"❌ Failed fundamental check: Debt/Equity ({de_ratio:.2f}) > {max_debt_equity}"}
    #             if pe_ratio is not None and pe_ratio < min_pe_ratio:
    #                 return {'signal': 'WAIT',
    #                         'reason': f"❌ Failed fundamental check: PE Ratio ({pe_ratio:.2f}) < {min_pe_ratio}"}
    #             reasons.append("✅ Passed fundamental checks.")
    #
    #         # S/R Levels
    #         sr_levels = analyzer.find_support_resistance()
    #         nearest_support = sr_levels['support'][-1] if sr_levels['support'] else None
    #
    #         # Stop-Loss Logic
    #         if stop_loss_mode == 'support' and nearest_support:
    #             stop_loss_price = nearest_support * 0.99
    #         elif stop_loss_mode == 'ma':
    #             stop_loss_price = latest[f'sma_{ma_stop_period}']  # SL is still based on decision day's MAs
    #         else:  # Default to ATR
    #             stop_loss_price = current_price - (atr_value * atr_mult_stop)
    #
    #         risk = current_price - stop_loss_price
    #         if risk <= 0:  # Safety check (unchanged)
    #             stop_loss_price = current_price - (atr_value * atr_mult_stop)
    #             risk = current_price - stop_loss_price
    #
    #         # Take-Profit Logic
    #         nearest_resistance = sr_levels['resistance'][0] if sr_levels['resistance'] else None
    #         if nearest_resistance and nearest_resistance > current_price:
    #             profit_target_price = nearest_resistance
    #         else:
    #             profit_target_price = current_price + (risk * atr_mult_profit)
    #
    #         return {
    #             'signal': 'BUY', 'reason': "\n".join(reasons),
    #             'current_price': current_price,  # This is now next day's open
    #             'stop_loss_price': stop_loss_price,
    #             'profit_target_price': profit_target_price,
    #             'debug_rsi': latest[f'rsi_{rsi_period}'],
    #             'PE Ratio': pe_ratio,
    #             'P/S Ratio': ps_ratio,
    #             'Debt/Equity': de_ratio,
    #             'atr_value': atr_value
    #         }
    #
    #     _self.log.debug(
    #         f"MICO: {symbol} did not meet all conditions ({buy_conditions_met}/{total_conditions}). Signal: WAIT.")
    #     return {'signal': 'WAIT', 'reason': "\n".join(reasons)}

    # Place this function inside the MichaAdvisor class, after __init__ or other helper methods.
    def _run_core_buy_checks(self, latest: pd.Series, params: dict, analyzer, reasons: list) -> tuple:
        """
        Runs all mandatory and optional technical buy conditions (Trend, RSI, MACD, etc.).
        Returns (buy_conditions_met, total_conditions).
        """
        buy_conditions_met = 0

        # Extract parameters for clarity
        sma_short = params['sma_short']
        sma_long = params['sma_long']
        rsi_period = params['rsi_period']
        rsi_threshold = params['rsi_threshold']
        macd_fast = params['macd_fast']
        macd_slow = params['macd_slow']
        macd_signal = params['macd_signal']

        # --- FALLING KNIFE (NEGATIVE MOMENTUM) FILTER ---
        # Get the required indicator values (using pandas-ta naming conventions)
        macd_hist_col = f"macdhist_{params['macd_fast']}_{params['macd_slow']}_{params['macd_signal']}"
        daily_return_value = latest.get('daily_return', 0.0)

        # Define the threshold for extreme negative momentum
        MOMENTUM_THRESHOLD = -1.5

        if latest.get(macd_hist_col) is not None and latest[
            macd_hist_col] < MOMENTUM_THRESHOLD and daily_return_value < 0:
            reasons.append(
                f"❌ Failed Falling Knife Filter: MACD Histogram ({latest[macd_hist_col]:.2f}) below {MOMENTUM_THRESHOLD} AND Daily Return is negative. Avoid crash.")
            # CRITICAL: We exit immediately to block the trade
            return 0, 999

        reasons.append("✅ Passed Falling Knife Filter.")

        # Trend Check (Mandatory)
        if latest['close'] > latest[f'sma_{sma_short}'] and latest[f'sma_{sma_short}'] > latest[f'sma_{sma_long}']:
            buy_conditions_met += 1
            reasons.append(f"✅ Price > {sma_short}-day SMA & {sma_short} > {sma_long}-day SMA.")
        else:
            reasons.append(f"❌ Price not in uptrend (SMA {sma_short}/{sma_long}).")

        # RSI Check (Mandatory)
        if latest[f'rsi_{rsi_period}'] < rsi_threshold:
            buy_conditions_met += 1
            reasons.append(f"✅ RSI ({latest[f'rsi_{rsi_period}']:.1f}) < {rsi_threshold}.")
        else:
            reasons.append(f"❌ RSI ({latest[f'rsi_{rsi_period}']:.1f}) >= {rsi_threshold}.")

        # MACD Check (Mandatory)
        if latest[f"macd_{macd_fast}_{macd_slow}_{macd_signal}"] > latest[
            f"macds_{macd_fast}_{macd_slow}_{macd_signal}"]:
            buy_conditions_met += 1
            reasons.append("✅ MACD > signal line.")
        else:
            reasons.append("❌ MACD < signal line.")

        # Volume Check (Optional)
        if params['use_volume_check']:
            if analyzer.check_volume_breakout():
                buy_conditions_met += 1
                reasons.append("✅ Volume is strong (above 20-day avg).")
            else:
                reasons.append("❌ Volume is weak (below 20-day avg).")

        # Candlestick Check (Optional)
        if params['use_candlestick_check']:
            if analyzer.check_bullish_candlestick():
                buy_conditions_met += 1
                reasons.append("✅ Bullish candlestick pattern found.")
            else:
                reasons.append("❌ No bullish candlestick pattern found.")

        # Continuation Check (Optional)
        if params['use_continuation_check']:
            if analyzer.check_continuation_candlestick():
                buy_conditions_met += 1
                reasons.append("✅ Indecision/continuation pattern found (Doji/Spinning Top).")
            else:
                reasons.append("❌ No indecision/continuation pattern found.")

        # --- Tally Total Conditions ---
        total_conditions = 3 + int(params['use_volume_check']) + int(params['use_candlestick_check']) + \
                           int(params['use_continuation_check']) + int(params['use_multi_timeframe']) + int(
            params['use_atr_filter'])

        return buy_conditions_met, total_conditions

    def analyze(self, stock_data, symbol: str, analysis_date, params: dict = None,
                use_market_filter: bool = True) -> dict:

        logging.info(f"MICO: Starting analysis for {symbol} on {analysis_date}.")

        # --- AUTO-LOAD PARAMETERS IF NOT PROVIDED ---
        if not params:
            params = self.get_best_params(symbol)

        # 1. Get and Validate Parameters (using helper)
        validated_params = self._extract_and_validate_params(params)

        # Initialize 'reasons' list here, early in the process ---
        reasons = []

        # 2. Get and Clean Data
        df_raw = clean_raw_data(stock_data)
        analysis_date_dt = pd.to_datetime(analysis_date)

        if df_raw.empty or len(df_raw) < validated_params['sma_long']:
            return {'signal': 'WAIT',
                    'reason': f'Not enough data ({len(df_raw)} < {validated_params["sma_long"]} days).'}

        # Data Slicing (Memory Efficient: .copy() prevents future SettingWithCopyWarning)
        df_slice = df_raw[df_raw.index <= analysis_date_dt].copy()

        # 3. Multi-Timeframe, Indicators, and Pre-Filters (using helper)
        # This function returns an empty DataFrame if Multi-Timeframe or ATR Volatility check fails.
        df_slice = self._run_technical_analysis(df_slice, validated_params, symbol)

        if df_slice.empty:
            return {'signal': 'WAIT', 'reason': 'Failed pre-analysis filter (e.g., Multi-timeframe or ATR Volatility).'}
            # Note: The reason will be logged inside _run_technical_analysis.

        # 4. Earnings and Events Exclusion (from user's original logic)
        if validated_params['use_earnings_filter']:
            # Fetch data manager is needed here to fetch earnings date
            next_earnings_date = self.dm.get_earnings_calendar(symbol)
            if next_earnings_date:
                days_to_earnings = (next_earnings_date - analysis_date_dt.date()).days

                if 0 <= days_to_earnings <= validated_params['earnings_lookahead_days']:
                    reason = f"❌ Failed Earnings check: Earnings are in {days_to_earnings} days (on {next_earnings_date})."
                    self.log.info(f"MICO: {symbol} failed filter. {reason}")
                    return {'signal': 'WAIT', 'reason': reason}

            # # If we pass (or no date is found)
            # validated_params['reasons'].append("✅ Passed Earnings Calendar check.")
            # Append to local 'reasons' list, NOT validated_params
            reasons.append("✅ Passed Earnings Calendar check.")

        # 5. Run Core Buy Checks (using new helper)
        analyzer = TechnicalAnalyzer(df_slice, logging)
        latest = analyzer.df.iloc[-1]
        # reasons = []  # Start collecting reasons for core checks

        # buy_conditions_met, total_conditions = _self._run_core_buy_checks(
        #     latest, validated_params, analyzer, reasons
        # )
        buy_conditions_met, total_conditions = self._run_core_buy_checks(
            latest, validated_params, analyzer, reasons
        )
        # 6. Final Signal Logic (from user's Step 8)
        if buy_conditions_met >= validated_params['min_conditions_to_buy']:

            # --- START: "NEXT DAY" MODIFICATIONS ---
            # Find the *next* trading day's data (from the full df_raw)
            next_day_data = df_raw[df_raw.index > analysis_date_dt]
            if next_day_data.empty:
                self.log.warning(f"MICO: BUY signal for {symbol}, but no future data found.")
                return {'signal': 'WAIT', 'reason': 'BUY conditions met, but no next-day data.'}

            # Get the *actual* trade info
            actual_trade_row = next_day_data.iloc[0]
            current_price = actual_trade_row['open']  # Buy at next day's OPEN
            atr_value = latest[f'atrr_{validated_params["atr_period"]}']  # ATR from decision day
            # --- END: "NEXT DAY" MODIFICATIONS ---

            # Fundamental Fetch (Early Exit Filter for Buy)
            pe_ratio, ps_ratio, de_ratio = None, None, None
            if validated_params['use_fundamental_filter']:
                try:
                    info = self.dm.get_fundamental_info(symbol)
                    if info:
                        pe_ratio = info.get('trailingPE')
                        ps_ratio = info.get('priceToSalesTrailing12Months')
                        de_ratio = info.get('debtToEquity')
                except Exception as e:
                    self.log.warning(f"MICO: Could not fetch fundamentals for {symbol}. Error: {e}", exc_info=True)

                if de_ratio is not None and de_ratio > validated_params['max_debt_equity']:
                    return {'signal': 'WAIT',
                            'reason': f"❌ Failed fundamental check: Debt/Equity ({de_ratio:.2f}) > {validated_params['max_debt_equity']}"}
                if pe_ratio is not None and pe_ratio < validated_params['min_pe_ratio']:
                    return {'signal': 'WAIT',
                            'reason': f"❌ Failed fundamental check: PE Ratio ({pe_ratio:.2f}) < {validated_params['min_pe_ratio']}"}
                reasons.append("✅ Passed fundamental checks.")

            # 7. Calculate Risk Management (using helper)
            stop_loss_price, profit_target_price = self._calculate_risk_management(
                latest, current_price, validated_params, analyzer
            )

            return {
                'signal': 'BUY', 'reason': "\n".join(reasons),
                'current_price': current_price,
                'stop_loss_price': stop_loss_price,
                'profit_target_price': profit_target_price,
                'debug_rsi': latest[f'rsi_{validated_params["rsi_period"]}'],
                'PE Ratio': pe_ratio,
                'P/S Ratio': ps_ratio,
                'Debt/Equity': de_ratio,
                'atr_value': atr_value
            }

        self.log.debug(
            f"MICO: {symbol} did not meet all conditions ({buy_conditions_met}/{total_conditions}). Signal: WAIT.")
        return {'signal': 'WAIT', 'reason': "\n".join(reasons)}