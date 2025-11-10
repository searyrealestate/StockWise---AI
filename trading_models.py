# trading_models.py

import pandas as pd
import pandas_ta as ta
import streamlit as st
from utils import clean_raw_data


# In trading_models.py

class MeanReversionAdvisor:
    """
    Buys stocks that are oversold.
    Signal 1: Price touches the lower Bollinger Band.
    Signal 2: RSI is below the oversold threshold.
    """
    def __init__(self, data_manager, logger=None):
        self.dm = data_manager
        self.log = logger if logger else lambda msg, level="INFO": None

    @st.cache_data(ttl=900)
    def analyze(_self, symbol: str, analysis_date, params: dict = None) -> dict:
        if params is None: params = {}
        # Get correct parameters from optimizer
        bb_length = params.get('bb_length', 20)
        rsi_length = params.get('rsi_length', 14)
        rsi_oversold = params.get('rsi_oversold', 30)

        df_raw = _self.dm.get_stock_data(symbol, days_back=bb_length + 50)
        df_slice = df_raw[df_raw.index <= pd.to_datetime(analysis_date)]

        if df_slice.empty or len(df_slice) < bb_length + 1: return {'signal': 'WAIT'}

        # Calculate indicators
        df_slice.ta.bbands(length=bb_length, append=True, col_names=('bbl', 'bbm', 'bbu', 'bbb', 'bbp'))
        df_slice.ta.rsi(length=rsi_length, append=True, col_names='rsi_14')
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
                'reason': reason
            }
        return {'signal': 'WAIT'}


class BreakoutAdvisor:
    """Buys stocks that break above their 20-day high, indicating strong momentum."""

    def __init__(self, data_manager, logger=None):
        self.dm = data_manager
        self.log = logger if logger else lambda msg, level="INFO": None

    @st.cache_data(ttl=900)
    def analyze(_self, symbol: str, analysis_date, params: dict = None) -> dict:
        if params is None: params = {}
        breakout_window = params.get('breakout_window', 20)

        df_raw = _self.dm.get_stock_data(symbol, days_back=breakout_window + 20)
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
                'reason': f"Price broke above {breakout_window}-day high."
            }
        return {'signal': 'WAIT'}


# In trading_models.py

class SuperTrendAdvisor:
    """
    Buys on pullbacks to the SuperTrend line *while* in an uptrend.
    """

    def __init__(self, data_manager, logger=None):
        self.dm = data_manager
        self.log = logger if logger else lambda msg, level="INFO": None

    @st.cache_data(ttl=900)
    def analyze(_self, symbol: str, analysis_date, params: dict = None) -> dict:
        if params is None: params = {}
        st_length = params.get('length', 10)
        st_multiplier = params.get('multiplier', 3.0)

        df_raw = _self.dm.get_stock_data(symbol, days_back=st_length + 50)
        df_slice = df_raw[df_raw.index <= pd.to_datetime(analysis_date)]

        if df_slice.empty or len(df_slice) < st_length + 2: return {'signal': 'WAIT'}

        # Calculate SuperTrend
        df_slice.ta.supertrend(length=st_length, multiplier=st_multiplier, append=True)
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
                'reason': "Buy pullback to SuperTrend line."
            }
        return {'signal': 'WAIT'}


# In trading_models.py

class MovingAverageCrossoverAdvisor:
    """
    Buys on pullbacks *after* a "Golden Cross" is already active.
    This is a "Buy the Dip in an Uptrend" strategy.
    """
    def __init__(self, data_manager, logger=None):
        self.dm = data_manager
        self.log = logger if logger else lambda msg, level="INFO": None

    @st.cache_data(ttl=900)
    def analyze(_self, symbol: str, analysis_date, params: dict = None) -> dict:
        if params is None: params = {}
        short_window = params.get('short_window', 50)
        long_window = params.get('long_window', 200)

        df_raw = _self.dm.get_stock_data(symbol, days_back=long_window + 50)
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
                'reason': f"Buy the dip: Golden Cross active and RSI recovered from <40."
            }
        return {'signal': 'WAIT'}


class VolumeMomentumAdvisor:
    """Buys when On-Balance Volume (OBV) shows strong momentum, confirming the price trend."""

    def __init__(self, data_manager, logger=None):
        self.dm = data_manager
        self.log = logger if logger else lambda msg, level="INFO": None

    @st.cache_data(ttl=900)
    def analyze(_self, symbol: str, analysis_date, params: dict = None) -> dict:
        if params is None: params = {}
        obv_window = params.get('obv_window', 20)

        df_raw = _self.dm.get_stock_data(symbol, days_back=obv_window + 50)
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
                'reason': f"OBV is above its {obv_window}-day SMA, confirming price momentum."
            }
        return {'signal': 'WAIT'}
