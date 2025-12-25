# trading_models.py

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
    calculates a "composite_score" based on Profit Factor, Win Rate, and
    number of trades.
5.  **Finds Best**: It tracks the parameters that generated the highest
    composite_score.

Outputs:
--------
-   **Streamlit UI**: Displays progress bars, detailed results tables for each
    model, and a final summary of the best-performing model.
-   **JSON File**: Saves the best-performing parameter set for *all* tested
    models into a single `best_params.json` file for future use.

"""

import itertools
import json
import numpy as np
import pandas as pd
import pandas_ta as ta
import streamlit as st
from utils import clean_raw_data
import logging


class MeanReversionAdvisor:
    """
    Buys stocks that are oversold.
    Signal 1: Price touches the lower Bollinger Band.
    Signal 2: RSI is below the oversold threshold.
    """
    def __init__(self, data_manager):
        self.dm = data_manager
        self.log = logging.getLogger(type(self).__name__)

    def create_chart(self, stock_symbol, stock_data, result, analysis_date,
                     show_mico_lines=False, mico_result=None):
        try:
            if stock_data.empty: return None
            # Ensure imports are available for plotting and date handling
            from datetime import timedelta
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go
            import pandas_ta as ta

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.8, 0.2])
            fig.add_trace(
                go.Candlestick(x=stock_data.index, open=stock_data['open'], high=stock_data['high'],
                               low=stock_data['low'],
                               close=stock_data['close'], name='Price'), row=1, col=1)
            for period, color in [(5, 'orange'), (20, 'blue'), (50, 'red')]:
                if len(stock_data) >= period:
                    ma = stock_data['close'].rolling(window=period).mean()
                    fig.add_trace(go.Scatter(x=stock_data.index, y=ma, mode='lines', name=f'MA{period}',
                                             line=dict(color=color, width=1)), row=1, col=1)
            if len(stock_data) >= 20:
                try:
                    bb_df = ta.bbands(close=stock_data['close'], length=20)
                    if bb_df is not None and not bb_df.empty:
                        bb_df.columns = [col.lower() for col in bb_df.columns]
                        bbu_col = next((c for c in bb_df.columns if c.startswith('bbu')), None)
                        bbl_col = next((c for c in bb_df.columns if c.startswith('bbl')), None)
                        if bbu_col and bbl_col:
                            fig.add_trace(
                                go.Scatter(x=stock_data.index, y=bb_df[bbu_col], mode='lines', name='BB Upper',
                                           line=dict(color='gray', dash='dot', width=1), showlegend=False), row=1,
                                col=1)
                            fig.add_trace(
                                go.Scatter(x=stock_data.index, y=bb_df[bbl_col], mode='lines', name='BB Lower',
                                           line=dict(color='gray', dash='dot', width=1), fill='tonexty',
                                           fillcolor='rgba(128,128,128,0.1)', showlegend=False), row=1, col=1)
                except Exception as e:
                    self.log.warning(f"Could not calculate Bollinger Bands: {e}", exc_info=True)
            fig.add_trace(
                go.Bar(x=stock_data.index, y=stock_data['volume'], name='Volume',
                       marker=dict(color='rgba(100,110,120,0.6)')),
                row=2, col=1)
            fig.add_vline(x=analysis_date, line_width=1, line_dash="dash", line_color="white", name="Analysis Date",
                          row=1)
            action = result.get('action', 'WAIT')
            current_price = result.get('current_price', stock_data['close'].iloc[-1] if not stock_data.empty else 0)

            # --- AI ADVISOR MARKERS (CYAN CIRCLE) ---
            if "BUY" in action:
                buy_date = result['buy_date']
                stop_loss = result.get('stop_loss_price')
                profit_target_price = result.get('profit_target_price')

                # Plot the AI Profit Target line
                if profit_target_price:
                    fig.add_hline(y=float(profit_target_price), line_dash="dot", line_color="cyan",
                                  name="AI Profit Target", row=1,
                                  annotation_text=f"AI Target: ${profit_target_price:.2f}",
                                  annotation_position="top right")

                # Plot the "Target Buy" marker (AI)
                if buy_date:
                    fig.add_trace(go.Scatter(
                        x=[buy_date], y=[current_price],
                        mode='markers+text',
                        text=[f"Buy: ${current_price:.2f}"],
                        textposition="middle right",
                        marker=dict(color='cyan', size=12, symbol='circle-open', line=dict(width=2)),
                        name='Target Buy'
                    ), row=1, col=1)

                # Plot the AI Stop-Loss line
                if stop_loss:
                    fig.add_hline(y=float(stop_loss), line_dash="dot", line_color="magenta",
                                  name="AI Stop-Loss", row=1, annotation_text=f"AI Stop: ${stop_loss:.2f}",
                                  annotation_position="bottom right")

            # --- MICO SYSTEM MARKERS AND LINES ---
            if show_mico_lines and mico_result:

                # Add MA200 line for MICO context if not already added by default MAs
                if len(stock_data) >= 200 and 200 not in [5, 20, 50]:
                    sma_200 = stock_data['close'].rolling(window=200).mean()
                    fig.add_trace(go.Scatter(
                        x=stock_data.index, y=sma_200, mode='lines', name='MA200',
                        line=dict(color='purple', width=1)
                    ), row=1, col=1)

                sl_price = mico_result.get('stop_loss_price')
                tp_price = mico_result.get('profit_target_price')
                mico_signal = mico_result.get('signal')
                mico_entry_price = mico_result.get('current_price')

                # 1. Plot MICO Signal Marker (Distinct Green Triangle)
                logging.info(f"MICO DEBUG: Signal={mico_signal}, Entry={mico_entry_price}, Lines={show_mico_lines}")
                if mico_signal == 'BUY' and mico_entry_price:
                    # --- ROBUST DATE LOGIC ---
                    analysis_dt = pd.to_datetime(analysis_date)
                    try:
                        # Find the last date in the data index that is less than or equal to the analysis date.
                        mico_buy_date = stock_data.index[stock_data.index <= analysis_dt][-1]
                    except IndexError:
                        mico_buy_date = analysis_dt
                    # --- END ROBUST DATE LOGIC ---

                    # --- VISIBILITY FIX: Adjust Y position ---
                    # Fetch the low price for the bar we are plotting on
                    # Use .loc to safely access the price using the date
                    try:
                        bar_low = stock_data.loc[mico_buy_date, 'low']
                    except:
                        bar_low = mico_entry_price * 0.99

                    print(f"!!! PLOTTING MICO BUY MARKER at {mico_buy_date} @ ${mico_entry_price:.2f} !!!")
                    logging.info(f"!!! PLOTTING MICO BUY MARKER at {mico_buy_date} @ ${mico_entry_price:.2f} !!!")

                    # Find the price of the low on the decision date
                    # --- Get the LOW for the date bar for better marker placement ---
                    # bar_low = stock_data.loc[mico_buy_date, 'low']

                    # Place the marker slightly below the low of the bar
                    y_position = bar_low * 0.995  # 0.5% below the bar's low price

                    # --- START PLOT CONFIRMATION (Add a simple print to console) ---
                    logging.info(f"!!! PLOTTING MICO BUY MARKER at {mico_buy_date} @ ${mico_entry_price:.2f} !!!")

                    fig.add_trace(go.Scatter(
                        x=[mico_buy_date], y=[y_position],
                        mode='markers+text',
                        text=[f"MICO Buy: ${mico_entry_price:.2f}"],
                        textposition="bottom center",  # Position text differently than AI
                        marker=dict(color='lime', size=14, symbol='triangle-up', line=dict(width=2)),
                        name='MICO Buy Signal'
                    ), row=1, col=1)

                # 2. Plot MICO Stop-Loss Line
                if sl_price:
                    fig.add_hline(
                        y=sl_price, line_width=2, line_dash="dash",
                        line_color="red", name="Mico Stop-Loss",
                        annotation_text="Mico Stop-Loss",
                        annotation_position="bottom left",
                        row=1, col=1
                    )

                # 3. Plot MICO Take-Profit Line
                if tp_price:
                    fig.add_hline(
                        y=tp_price, line_width=2, line_dash="dash",
                        line_color="green", name="Mico Take-Profit",
                        annotation_text="Mico Take-Profit",
                        annotation_position="top left",
                        row=1, col=1
                    )

            # 1. Define the View Window (Existing code)
            zoom_start_date = pd.to_datetime(analysis_date) - timedelta(days=10)
            zoom_end_date = pd.to_datetime(analysis_date) + timedelta(days=120)

            # 2. Calculate Min/Max for the Visible Range
            mask = (stock_data.index >= zoom_start_date) & (stock_data.index <= zoom_end_date)
            visible_data = stock_data.loc[mask]

            y_axis_range = None

            if not visible_data.empty:
                visible_min = visible_data['low'].min()
                visible_max = visible_data['high'].max()

                # Apply your $20 buffer
                y_axis_range = [visible_min - 20, visible_max + 20]

            # 3. Update Layout
            fig.update_layout(title_text=f'{stock_symbol} Price & Volume Analysis', xaxis_rangeslider_visible=False,
                              showlegend=True,
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            # Set X-Axis Zoom
            fig.update_xaxes(range=[zoom_start_date, zoom_end_date], row=1, col=1)
            fig.update_xaxes(range=[zoom_start_date, zoom_end_date], row=2, col=1)

            # 4. Set Y-Axis with Calculated Range
            fig.update_yaxes(title_text="Price (USD)", type="linear", range=y_axis_range, row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            return fig

        except Exception as e:
            logging.error(f"Error creating chart for {stock_symbol}: {e}", exc_info=True)
            return None

    @st.cache_data(ttl=900)
    def analyze(_self, stock_data, symbol: str, analysis_date, params: dict = None,
                use_market_filter: bool = True) -> dict:

        if params is None: params = {}
        # Get correct parameters from optimizer
        bb_length = params.get('bb_length', 20)
        rsi_length = params.get('rsi_length', 14)
        rsi_oversold = params.get('rsi_oversold', 30)

        # df_raw = stock_data
        df_raw = clean_raw_data(stock_data)
        if pd.api.types.is_datetime64_any_dtype(df_raw.index) and df_raw.index.tz is not None:
            df_raw.index = df_raw.index.tz_localize(None)
            logging.debug(f"[{symbol}]: MICO converted data index to tz-naive.")

        df_slice = df_raw[df_raw.index <= pd.to_datetime(analysis_date)]

        if df_slice.empty or len(df_slice) < bb_length + 1: return {'signal': 'WAIT'}

        # Calculate indicators
        df_slice.ta.bbands(length=bb_length, append=True, col_names=('bbl', 'bbm', 'bbu', 'bbb', 'bbp'))
        df_slice.ta.rsi(length=rsi_length, append=True, col_names='rsi_14')
        df_slice.ta.atr(length=14, append=True, col_names='atr_14')
        df_slice.dropna(inplace=True)

        if df_slice.empty: return {'signal': 'WAIT'}

        latest = df_slice.iloc[-1]
        current_price = latest['close']

        # Check for the two mean-reversion signals
        is_touching_bb = current_price <= latest['bbl']
        is_oversold = latest['rsi_14'] < rsi_oversold

        if is_touching_bb or is_oversold:
            reason = f"RSI < {rsi_oversold}" if is_oversold else "Price at Lower BB"
            return {
                'signal': 'BUY', 'current_price': current_price,
                'stop_loss_price': latest['low'] * 0.98, # Simple SL
                'profit_target_price': latest['bbm'], # Target is the mean
                'reason': reason,
                'atr_value': latest['atr_14']
            }
        return {'signal': 'WAIT'}


class BreakoutAdvisor:
    """Buys stocks that break above their 20-day high, indicating strong momentum."""

    def __init__(self, data_manager):
        self.dm = data_manager
        self.log = logging.getLogger(type(self).__name__)

    @st.cache_data(ttl=900)
    def analyze(_self, stock_data, symbol: str, analysis_date, params: dict = None,
                use_market_filter: bool = True) -> dict:
        if params is None: params = {}
        breakout_window = params.get('breakout_window', 20)

        # df_raw = stock_data
        df_raw = clean_raw_data(stock_data)
        if pd.api.types.is_datetime64_any_dtype(df_raw.index) and df_raw.index.tz is not None:
            df_raw.index = df_raw.index.tz_localize(None)
            _self.log.debug(f"[{symbol}]: MICO converted data index to tz-naive.")

        df_slice = df_raw[df_raw.index <= pd.to_datetime(analysis_date)]

        if df_slice.empty or len(df_slice) < breakout_window + 2: return {'signal': 'WAIT'}

        df_slice['high_breakout'] = df_slice['high'].shift(1).rolling(window=breakout_window).max()
        df_slice.ta.atr(length=14, append=True, col_names='atr_14')
        df_slice.dropna(inplace=True)

        if df_slice.empty: return {'signal': 'WAIT'}

        latest = df_slice.iloc[-1]
        current_price = latest['close']

        if latest['high'] > latest['high_breakout']:
            risk = (latest['atr_14'] * 2.5)
            return {
                'signal': 'BUY', 'current_price': current_price,
                'stop_loss_price': current_price - risk,
                'profit_target_price': current_price + (risk * 1.5),
                'reason': f"Price broke above {breakout_window}-day high.",
                'atr_value': latest['atr_14']
            }
        return {'signal': 'WAIT'}


class SuperTrendAdvisor:
    """
    Buys on pullbacks to the SuperTrend line *while* in an uptrend.
    """

    def __init__(self, data_manager):
        self.dm = data_manager
        self.log = logging.getLogger(type(self).__name__)

    @st.cache_data(ttl=900)
    def analyze(_self, stock_data, symbol: str, analysis_date, params: dict = None,
                use_market_filter: bool = True) -> dict:
        if params is None: params = {}
        st_length = params.get('length', 10)
        st_multiplier = params.get('multiplier', 3.0)

        # df_raw = stock_data
        df_raw = clean_raw_data(stock_data)
        if pd.api.types.is_datetime64_any_dtype(df_raw.index) and df_raw.index.tz is not None:
            df_raw.index = df_raw.index.tz_localize(None)
            _self.log.debug(f"[{symbol}]: MICO converted data index to tz-naive.")

        df_slice = df_raw[df_raw.index <= pd.to_datetime(analysis_date)]

        if df_slice.empty or len(df_slice) < st_length + 2: return {'signal': 'WAIT'}

        # Calculate SuperTrend
        df_slice.ta.supertrend(length=st_length, multiplier=st_multiplier, append=True)
        df_slice.ta.atr(length=14, append=True, col_names='atr_14')
        df_slice.dropna(inplace=True)

        if len(df_slice) < 2: return {'signal': 'WAIT'}

        latest = df_slice.iloc[-1]
        current_price = latest['close']

        # Dynamically get column names
        direction_col = f'SUPERTd_{st_length}_{st_multiplier}'
        supertrend_col = f'SUPERT_{st_length}_{st_multiplier}'

        # --- NEW "BUY THE DIP" LOGIC ---
        is_uptrend = latest[direction_col] == 1
        # Price pulled back to touch the SuperTrend line
        is_pullback = latest['low'] <= latest[supertrend_col]
        # And the price is now recovering/closing above it
        is_recovering = current_price > latest[supertrend_col]

        if is_uptrend and is_pullback and is_recovering:
            risk = current_price - latest[supertrend_col]
            if risk <= 0: return {'signal': 'WAIT'}  # Avoid bad data
            return {
                'signal': 'BUY', 'current_price': current_price,
                'stop_loss_price': latest[supertrend_col],  # SL is the ST line
                'profit_target_price': current_price + (risk * 2),
                'reason': "Buy pullback to SuperTrend line.",
                'atr_value': latest['atr_14']
            }
        return {'signal': 'WAIT'}


# In trading_models.py

class MovingAverageCrossoverAdvisor:
    """
    Buys on pullbacks *after* a "Golden Cross" is already active.
    This is a "Buy the Dip in an Uptrend" strategy.
    """
    def __init__(self, data_manager):
        self.dm = data_manager
        self.log = logging.getLogger(type(self).__name__)

    @st.cache_data(ttl=900)
    def analyze(_self, stock_data, symbol: str, analysis_date, params: dict = None,
                use_market_filter: bool = True) -> dict:
        if params is None: params = {}
        short_window = params.get('short_window', 50)
        long_window = params.get('long_window', 200)

        # df_raw = stock_data
        df_raw = clean_raw_data(stock_data)
        if pd.api.types.is_datetime64_any_dtype(df_raw.index) and df_raw.index.tz is not None:
            df_raw.index = df_raw.index.tz_localize(None)
            _self.log.debug(f"[{symbol}]: MICO converted data index to tz-naive.")

        df_slice = df_raw[df_raw.index <= pd.to_datetime(analysis_date)]

        if df_slice.empty or len(df_slice) < long_window: return {'signal': 'WAIT'}

        df_slice[f'sma_{short_window}'] = df_slice['close'].rolling(window=short_window).mean()
        df_slice[f'sma_{long_window}'] = df_slice['close'].rolling(window=long_window).mean()
        df_slice.ta.rsi(length=14, append=True, col_names='rsi_14')
        df_slice.ta.atr(length=14, append=True, col_names='atr_14')
        df_slice.dropna(inplace=True)

        if len(df_slice) < 2: return {'signal': 'WAIT'}

        latest = df_slice.iloc[-1]
        previous = df_slice.iloc[-2]
        current_price = latest['close']

        # --- "BUY THE DIP" LOGIC ---
        is_uptrend = latest[f'sma_{short_window}'] > latest[f'sma_{long_window}']
        is_pullback = previous['rsi_14'] < 40
        is_recovering = latest['rsi_14'] > 40

        if is_uptrend and is_pullback and is_recovering:
            risk = (latest['atr_14'] * 3)
            return {
                'signal': 'BUY', 'current_price': current_price,
                'stop_loss_price': current_price - risk,
                'profit_target_price': current_price + (risk * 2),
                'reason': f"Buy the dip: Golden Cross active and RSI recovered from <40.",
                'atr_value': latest['atr_14']
            }
        return {'signal': 'WAIT'}


class VolumeMomentumAdvisor:
    """Buys when On-Balance Volume (OBV) shows strong momentum, confirming the price trend."""

    def __init__(self, data_manager):
        self.dm = data_manager
        self.log = logging.getLogger(type(self).__name__)

    @st.cache_data(ttl=900)
    def analyze(_self, stock_data, symbol: str, analysis_date, params: dict = None,
                use_market_filter: bool = True) -> dict:
        if params is None: params = {}
        obv_window = params.get('obv_window', 20)

        # df_raw = stock_data
        df_raw = clean_raw_data(stock_data)
        if pd.api.types.is_datetime64_any_dtype(df_raw.index) and df_raw.index.tz is not None:
            df_raw.index = df_raw.index.tz_localize(None)
            _self.log.debug(f"[{symbol}]: MICO converted data index to tz-naive.")

        df_slice = df_raw[df_raw.index <= pd.to_datetime(analysis_date)]

        if df_slice.empty or len(df_slice) < obv_window:
            return {'signal': 'WAIT'}

        df_slice.ta.obv(append=True, col_names='obv')
        df_slice.ta.atr(length=14, append=True, col_names='atr_14')

        # Add a check to ensure the 'obv' column was created before trying to use it.
        if 'obv' not in df_slice.columns:
            return {'signal': 'WAIT'}

        # Now this line will work correctly
        df_slice['obv_sma'] = df_slice['obv'].rolling(window=obv_window).mean()
        df_slice.dropna(inplace=True)

        if len(df_slice) < 2: return {'signal': 'WAIT'}

        latest = df_slice.iloc[-1]
        current_price = latest['close']

        # --- ENTRY RULE ---
        # Buy if OBV is above its moving average and the price is in an uptrend (above its 50-day SMA)
        is_obv_strong = latest['obv'] > latest['obv_sma']
        is_price_trending = current_price > df_slice['close'].rolling(50).mean().iloc[-1]

        if is_obv_strong and is_price_trending:
            stop_loss_price = current_price - (latest['atr_14'] * 2)
            risk = current_price - stop_loss_price
            profit_target_price = current_price + (risk * 2)

            return {
                'signal': 'BUY', 'current_price': current_price,
                'stop_loss_price': stop_loss_price, 'profit_target_price': profit_target_price,
                'reason': f"OBV is above its {obv_window}-day SMA, confirming price momentum.",
                'atr_value': latest['atr_14']
            }
        return {'signal': 'WAIT'}



