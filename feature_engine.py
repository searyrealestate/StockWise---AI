# feature_engine.py

"""
Feature Engineering Engine - Gen-6 Complete
===========================================
This script implements all standard and advanced TradingView indicators
(EMA, SAR, WT, KC, VWAP, Donchian) for use by the Strategy Orchestra.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
import logging
import system_config as cfg
import json
from stockstats import StockDataFrame as SDF
import signal_processor

logger = logging.getLogger("FeatureEngine")


def _body_size(o, c): return (c - o).abs()
def _candle_range(h, l): return (h - l).abs()
def _upper_shadow(o, h, c): return h - pd.concat([o, c], axis=1).max(axis=1)
def _lower_shadow(o, l, c): return pd.concat([o, c], axis=1).min(axis=1) - l
def _is_bull(o, c): return c > o
def _is_bear(o, c): return c < o


def add_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wrapper for RobustFeatureCalculator.add_candlestick_patterns.
    Maintains backward compatibility for module-level imports.
    """
    try:
        df = RobustFeatureCalculator.add_candlestick_patterns(df)
        return df
    except Exception as e:
        logger.error(f"Error in standalone add_candlestick_patterns: {e}")
        return df


def add_wavetrend_stockstats(df: pd.DataFrame) -> pd.DataFrame:
    """
    adding WaveTrend (wt1, wt2) with stockstats.
    open, high, low, close, volume.
    """
    # stockstats changing the DataFrame
    sdf = SDF.retype(df.copy())

    # stockstats returns series with 'wt1' and 'wt2' columns
    df['wt1'] = sdf['wt1']
    df['wt2'] = sdf['wt2']
    return df



# --- MACHINE LEARNING LAYER (GEN-8) ---
def add_lorentzian_ml(df: pd.DataFrame) -> pd.DataFrame:
    """
    Implements a Walk-Forward K-Nearest Neighbors classifier to mimic Lorentzian Classification.
    Uses features: RSI, ADX, CCI, Volume Chg.
    Target: Next Bar Close > Current Close.
    
    Walk-Forward Logic:
    - Train on past 300 bars.
    - Predict next 20 bars.
    - Retrain.
    """
    try:
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.preprocessing import MinMaxScaler
    except ImportError:
        logger.warning("sklearn not found. Skipping ML features.")
        df['ml_signal'] = 0
        return df

    if len(df) < 500:
        logger.warning(f"Not enough data for ML (Need 500+, got {len(df)}).")
        df['ml_signal'] = 0
        return df

    # 1. Feature Preparation
    features = ['RSI_14', 'ADX_14', 'CCI_14_0.015', 'close']
    # Ensure features exist
    for f in features:
        if f not in df.columns:
            if f == 'close': continue
            logger.warning(f"ML Feature missing: {f}")
            df['ml_signal'] = 0
            return df

    # Prepare input matrix X and target y
    df_ml = df.copy()
    
    # Target: 1 if next CLOSE is higher (Bullish), 0 otherwise
    # Shift(-1) allows us to see tomorrow's move today for training.
    # We must be careful ONLY to train on rows where we know the outcome (past).
    df_ml['target'] = (df_ml['close'].shift(-1) > df_ml['close']).astype(int)
    
    # Drop NaN created by shift or indicators
    df_ml = df_ml.dropna()
    
    # We need to predict for the ORIGINAL dataframe indices
    # So we create a Series aligned with original index
    predictions = pd.Series(0, index=df.index, dtype=int)
    
    # Walk-Forward Parameters
    TRAIN_WINDOW = 300  # Lookback for patterns
    PREDICT_WINDOW = 20 # Re-train every month
    
    # Initialize Scaler and Model
    # Metric: Manhattan is closest standard proxy to Lorentzian (Sum of absolute differences)
    knn = KNeighborsClassifier(n_neighbors=9, metric='manhattan', weights='distance')
    scaler = MinMaxScaler()

    # Walk-Forward Loop
    # Start at TRAIN_WINDOW, step by PREDICT_WINDOW
    total_rows = len(df_ml)
    
    # Extract numpy arrays for speed
    X_raw = df_ml[features].values
    y_raw = df_ml['target'].values
    indices = df_ml.index
    
    for i in range(TRAIN_WINDOW, total_rows, PREDICT_WINDOW):
        # 1. Train Window: [i - TRAIN_WINDOW : i]
        start_train = i - TRAIN_WINDOW
        end_train = i
        
        # 2. Predict Window: [i : i + PREDICT_WINDOW]
        start_pred = i
        end_pred = min(i + PREDICT_WINDOW, total_rows)
        
        if start_pred >= total_rows: break
        
        # Train Data
        X_train = X_raw[start_train:end_train]
        y_train = y_raw[start_train:end_train]
        
        # Test Data (To Predict)
        X_test = X_raw[start_pred:end_pred]
        
        # Scale Data (Fit on Train, Transform both)
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Fit & Predict
        knn.fit(X_train_scaled, y_train)
        preds = knn.predict(X_test_scaled)
        
        # Store
        current_indices = indices[start_pred:end_pred]
        predictions.loc[current_indices] = preds

    # Map predictions back to original DF
    df['ml_signal'] = predictions
    
    logger.info("ML Generation Complete.")
    return df

class RobustFeatureCalculator:
    """
    Centralized Feature Engineering Logic.
    Calculates SMAs, RSI, ATR, ADX, Slope, and Relative Strength.
    """

    def __init__(self, params=None):
        self.params = params if params else cfg.STRATEGY_PARAMS

    # 7. Advanced Candlestick Patterns (Gen-7 Addition)
    # Feature Engineering Constants
    SMALL_BODY_RATIO = 0.3
    DOJI_RATIO = 0.1
    LONG_WICK_RATIO = 2.0
    TREND_LOOKBACK = 5

    @staticmethod
    def _is_bull(close, open_p): return close > open_p
    @staticmethod
    def _is_bear(close, open_p): return close < open_p
    @staticmethod
    def _body(open_p, close): return (close - open_p).abs()
    @staticmethod
    def _range(high, low): return (high - low) + 1e-9

    @staticmethod
    def add_advanced_patterns(df):
        """
        Adds complex multi-candle patterns:
        - Rising/Falling Three Methods
        - Bullish/Bearish Three Line Strike
        - Bullish/Bearish Mat Hold
        - Three Stars in the South
        - Advance Block
        """
        # Data aliases
        o = df['open']
        h = df['high']
        l = df['low']
        c = df['close']
        
        # Helpers
        body = (c - o).abs()
        rng = (h - l) + 1e-9
        is_green = c > o
        is_red = c < o
        
        # --- 1. RISING THREE METHODS (Bullish Continuation) ---
        # Long Green, 3 Small Red/Green inside first, Long Green breakout
        c1_long = (body.shift(4) > rng.shift(4) * 0.5) & is_green.shift(4)
        c234_small = (body.shift(3) < rng.shift(4) * 0.4) & \
                     (body.shift(2) < rng.shift(4) * 0.4) & \
                     (body.shift(1) < rng.shift(4) * 0.4)
        c234_within = (h.shift(3) < h.shift(4)) & (l.shift(3) > l.shift(4)) # Strict-ish
        c5_breakout = is_green & (c > c.shift(4)) & (c > h.shift(1))
        
        rising_three = c1_long & c234_small & c5_breakout
        
        # --- 2. FALLING THREE METHODS (Bearish Continuation) ---
        # Long Red, 3 Small Green/Red inside, Long Red breakdown
        c1_long_bear = (body.shift(4) > rng.shift(4) * 0.5) & is_red.shift(4)
        c234_small_bear = (body.shift(3) < rng.shift(4) * 0.4) & \
                          (body.shift(2) < rng.shift(4) * 0.4) & \
                          (body.shift(1) < rng.shift(4) * 0.4)
        c5_breakdown = is_red & (c < c.shift(4)) & (c < l.shift(1))
        
        falling_three = c1_long_bear & c234_small_bear & c5_breakdown
        
        # --- 3. THREE LINE STRIKE (Bullish Reversal/Trap) ---
        # 3 Red candles (lower lows), 4th Green engulfs all 3
        three_reds = is_red.shift(3) & is_red.shift(2) & is_red.shift(1)
        lower_lows = (l.shift(1) < l.shift(2)) & (l.shift(2) < l.shift(3))
        strike_bull = is_green & (o < l.shift(1)) & (c > h.shift(3)) # Engulfs sequence
        
        three_line_strike_bull = three_reds & lower_lows & strike_bull
        
        # --- 4. BEARISH THREE LINE STRIKE (Bearish Reversal/Trap) ---
        three_greens = is_green.shift(3) & is_green.shift(2) & is_green.shift(1)
        higher_highs = (h.shift(1) > h.shift(2)) & (h.shift(2) > h.shift(3))
        strike_bear = is_red & (o > h.shift(1)) & (c < l.shift(3))
        
        three_line_strike_bear = three_greens & higher_highs & strike_bear
        
        # --- 5. BULLISH MAT HOLD (Strong Continuation) ---
        # Similar to Rising Three but small candles hold usually in upper half of C1
        c1_strong = (body.shift(4) > rng.shift(4) * 0.5) & is_green.shift(4)
        gap_up = o.shift(3) > c.shift(4) # Gap on 2nd candle
        c234_hold = (c.shift(3) > o.shift(4)) & (c.shift(2) > o.shift(4)) & (c.shift(1) > o.shift(4))
        c5_cont = is_green & (c > h.shift(1))
        
        mat_hold_bull = c1_strong & gap_up & c234_hold & c5_cont
        
        # --- 6. BEARISH MAT HOLD ---
        c1_strong_bear = (body.shift(4) > rng.shift(4) * 0.5) & is_red.shift(4)
        gap_down = o.shift(3) < c.shift(4)
        c234_hold_bear = (c.shift(3) < o.shift(4)) & (c.shift(2) < o.shift(4)) & (c.shift(1) < o.shift(4))
        c5_cont_bear = is_red & (c < l.shift(1))
        
        mat_hold_bear = c1_strong_bear & gap_down & c234_hold_bear & c5_cont_bear
        
        # --- 7. THREE STARS IN THE SOUTH (Bullish Reversal) ---
        # 3 Red candles, shrinking bodies, consistency
        # C1: Long Red, Long lower shadow
        # C2: Smaller Red, Low > C1 Low
        # C3: Marubozu Red (Small), Low > C2 Low
        c1_setup = is_red.shift(2) & ((l.shift(2) + rng.shift(2)*0.3) < c.shift(2)) # Lower shadow
        c2_setup = is_red.shift(1) & (body.shift(1) < body.shift(2)) & (l.shift(1) > l.shift(2))
        c3_setup = is_red & (body < body.shift(1)) & (l > l.shift(1)) & (l == c) # Marubozu-ish
        
        three_stars_south = c1_setup & c2_setup & c3_setup
        
        # --- 8. ADVANCE BLOCK (Bearish Reversal) ---
        # 3 Green candles, higher closes, BUT shrinking bodies and longer upper wicks (exhaustion)
        three_green_up = is_green.shift(2) & is_green.shift(1) & is_green
        higher_closes = (c > c.shift(1)) & (c.shift(1) > c.shift(2))
        shrinking_bodies = (body < body.shift(1)) & (body.shift(1) < body.shift(2))
        long_upper_wicks = (h - c) > (h.shift(1) - c.shift(1)) # Increasing wick resistance
        
        advance_block = three_green_up & higher_closes & shrinking_bodies & long_upper_wicks
        
        # Assign Outcomes
        df['rising_three_methods'] = rising_three
        df['falling_three_methods'] = falling_three
        df['bullish_three_line_strike'] = three_line_strike_bull
        df['bearish_three_line_strike'] = three_line_strike_bear
        df['bullish_mat_hold'] = mat_hold_bull
        df['bearish_mat_hold'] = mat_hold_bear
        df['three_stars_south'] = three_stars_south
        df['advance_block'] = advance_block
        
        # --- CONTINUATIONS (Calculate before Summary) ---
        df['bullish_continuation'] = rising_three | mat_hold_bull
        df['bearish_continuation'] = falling_three | mat_hold_bear
        
        # --- SUMMARY COLUMNS ---
        
        # Context Filters (Requested Optimization)
        # Use .get() or check existence for safety
        rsi = df['rsi_14'] if 'rsi_14' in df.columns else pd.Series(50, index=df.index)
        ema_21 = df['ema_21'] if 'ema_21' in df.columns else df['close']
        
        # Bullish: Valid if Oversold OR Price below EMA21 (Pullback)
        is_oversold = (rsi < 55) | (df['close'] < ema_21)
        
        # Bearish: Valid if Overbought OR Price above EMA21 (Extension)
        is_overbought = (rsi > 45) | (df['close'] > ema_21)

        # Strong Bullish Reversal
        # Logic: Pattern + Context (unless it's a structural pattern like Mat Hold)
        raw_bullish = (
            df['bullish_three_line_strike'] |
            df['three_stars_south'] |
            (df['morning_star'] if 'morning_star' in df.columns else False)
        )
        
        df['strong_bullish_reversal'] = (
            (raw_bullish & is_oversold) | 
            df['bullish_mat_hold'] | 
            df['bullish_continuation']
        )
        
        # Strong Bearish Reversal
        raw_bearish = (
            df['bearish_three_line_strike'] |
            df['advance_block'] |
            (df['evening_star'] if 'evening_star' in df.columns else False)
        )
        
        df['strong_bearish_reversal'] = (
            (raw_bearish & is_overbought) | 
            df['bearish_continuation']
        )
        
        return df

    @staticmethod
    def add_candlestick_patterns(df):
        """
        Adds 'Smart' candlestick patterns with STRICT Context Awareness.
        Vectorized operations only.
        
        New Features:
        - smart_hammer: 100 if (Lower Wick >= 2*Body) AND Downtrend AND RSI < 40
        - smart_shooting_star: -100 if (Upper Wick >= 2*Body) AND Uptrend AND RSI > 60
        - dragonfly_doji: 100 if Small Body + Long Low Wick + Downtrend
        - gravestone_doji: -100 if Small Body + Long Up Wick + Uptrend
        - candle_confluence: Sum of scores.
        """
        o = df['open']
        h = df['high']
        l = df['low']
        c = df['close']
        
        # Safe Context Helpers
        sma_50 = df['sma_50'] if 'sma_50' in df.columns else c.rolling(50).mean()
        rsi = df['rsi_14'] if 'rsi_14' in df.columns else pd.Series(50, index=df.index)
        
        # Definitions
        body = (c - o).abs()
        range_ = (h - l).replace(0, 0.0001)
        lower_wick = df[['open', 'close']].min(axis=1) - l
        upper_wick = h - df[['open', 'close']].max(axis=1)
        
        in_downtrend = c < sma_50
        in_uptrend = c > sma_50
        
        # 1. Smart Hammer (Bullish +100)
        # Criteria: Long Lower Wick, Downtrend, Oversold (RSI < 40)
        is_hammer_structure = (lower_wick >= 2 * body) & (upper_wick <= body * 0.5)
        smart_hammer_mask = is_hammer_structure & in_downtrend & (rsi < 40)
        df['smart_hammer'] = smart_hammer_mask.astype(int) * 100
        
        # 2. Smart Shooting Star (Bearish -100)
        # Criteria: Long Upper Wick, Uptrend, Overbought (RSI > 60)
        is_star_structure = (upper_wick >= 2 * body) & (lower_wick <= body * 0.5)
        smart_star_mask = is_star_structure & in_uptrend & (rsi > 60)
        df['smart_shooting_star'] = smart_star_mask.astype(int) * -100
        
        # 3. Doji Family
        is_doji_body = body <= (range_ * 0.1)
        
        # Dragonfly (Bullish +100): Doji + Long Lower + Downtrend
        dragonfly_mask = is_doji_body & (lower_wick > range_ * 0.6) & in_downtrend
        df['dragonfly_doji'] = dragonfly_mask.astype(int) * 100
        
        # Gravestone (Bearish -100): Doji + Long Upper + Uptrend
        gravestone_mask = is_doji_body & (upper_wick > range_ * 0.6) & in_uptrend
        df['gravestone_doji'] = gravestone_mask.astype(int) * -100
        
        # 4. Standard Engulfing (Helper for Confluence)
        prev_body = (c.shift(1) - o.shift(1)).abs()
        is_bull_engulf = (c > o) & (c.shift(1) < o.shift(1)) & (c > o.shift(1)) & (o < c.shift(1))
        is_bear_engulf = (c < o) & (c.shift(1) > o.shift(1)) & (c < o.shift(1)) & (o > c.shift(1))
        
        df['bullish_engulfing'] = is_bull_engulf.astype(int) * 100
        df['bearish_engulfing'] = is_bear_engulf.astype(int) * -100

        # --- VSA Features ---
        # 5. Squat Bar (Churning): High Volume + Small Body
        # Logic: Vol > 1.5x Avg, Body < 0.8x Avg
        avg_vol = df['volume'].rolling(20).mean()
        avg_body = body.rolling(20).mean()
        
        df['vsa_squat_bar'] = ((df['volume'] > 1.5 * avg_vol) & (body < 0.8 * avg_body)).astype(int) * 100
        
        # 6. No Demand (Weakness): Up Bar + Low Volume + Uptrend
        df['vsa_no_demand'] = ((c > o) & (df['volume'] < 0.8 * avg_vol) & in_uptrend).astype(int) * -100

        # --- Traps ---
        # 7. Bull Trap: Breakout Attempt Fails
        # Current High > Prev High, but Close < Prev High. On High Volume?
        # Let's use simple logic: Close < Prev High & High > Prev High
        prev_high = h.shift(1)
        df['bull_trap_signal'] = ((h > prev_high) & (c < prev_high)).astype(int) * -100

        # Calculate Confluence Score (Base sum)
        df['candle_confluence'] = (
            df['smart_hammer'] + 
            df['smart_shooting_star'] + 
            df['dragonfly_doji'] + 
            df['gravestone_doji'] +
            df['vsa_squat_bar'] +
            df['vsa_no_demand'] +
            df['bull_trap_signal'] 
        )
        
        return df

    def calculate_features(self, df: pd.DataFrame, context_data: dict = None) -> pd.DataFrame:
        df = df.copy()
        if df.empty: return pd.DataFrame()

        try:
            # --- Handle MultiIndex (Tuple Columns) from YFinance ---
            if isinstance(df.columns, pd.MultiIndex):
                # Flatten: take the first level (e.g., 'Close' from ('Close', 'NVDA'))
                df.columns = df.columns.get_level_values(0)
            
            # Now safe to lowercase strings
            df.columns = [str(col).lower() for col in df.columns]
            df = df.loc[:, ~df.columns.duplicated()]

            # --- Add these lines for Gen-9 AI ---
            # The AI Brain requires percentage changes for momentum inputs
            if 'close' in df.columns:
                df['daily_return'] = df['close'].pct_change().fillna(0)
            if 'volume' in df.columns:
                df['volume_change'] = df['volume'].pct_change().fillna(0)

            s_short = int(self.params.get('sma_short', 20))
            s_long = int(self.params.get('sma_long', 100))

            # --- Gen-7 EMA (9/21-day) for Trend Baseline and Signal Trigger  ---
            df['ema_9'] = ta.ema(df['close'], length=9, append=False)
            df['ema_21'] = ta.ema(df['close'], length=21, append=False)
            df['ema_spread'] = df['ema_9'] - df['ema_21']  # Feature for AI

            # Highest High of last 20 days (shifted 1 day to avoid lookahead)
            df['recent_high'] = df['high'].rolling(window=20).max().shift(1)
            # Lowest Low of last 20 days
            df['recent_low'] = df['low'].rolling(window=20).min().shift(1)

            # Standard SMAs (Kept for backwards compatibility)
            df['sma_short'] = df['close'].rolling(s_short).mean()
            df['sma_long'] = df['close'].rolling(s_long).mean()
            df['sma_50'] = df['sma_short']
            df['sma_200'] = df['sma_long']

            # 3. Slope (Velocity) and ADX (Trend Strength)
            try:
                slope_df = df.ta.slope(length=10)
                if isinstance(slope_df, pd.DataFrame):
                    df['slope_angle'] = slope_df.iloc[:, 0] * (180 / np.pi)
                else:
                    df['slope_angle'] = slope_df * (180 / np.pi)
            except:
                df['slope_angle'] = 0.0

            try:
                # --- 5. Momentum (RSI & ADX) ---
                df['rsi_14'] = df.ta.rsi(length=14)
                adx = df.ta.adx(length=14)
                if adx is not None: df = pd.concat([df, adx], axis=1)
                if 'ADX_14' in df.columns: df.rename(columns={'ADX_14': 'adx'}, inplace=True)
            except:
                pass

            try:
                macd = df.ta.macd(fast=12, slow=26, signal=9)
                if macd is not None:
                    df = pd.concat([df, macd], axis=1)
                    # Rename to standard keys used in Strategy Engine
                    df.rename(columns={
                        'MACD_12_26_9': 'macd',
                        'MACDs_12_26_9': 'macd_signal',
                        'MACDh_12_26_9': 'macd_hist'
                    }, inplace=True)
            except Exception as e:
                logger.warning(f"MACD calculation failed: {e}")

            df = df.loc[:, ~df.columns.duplicated()]

            # 5. Donchian Channels (DC) and Recent High
            df['recent_high'] = df['high'].rolling(50).max().shift(1)

            # --- Donchian Channel (DC) ---
            dc_df = df.ta.donchian(append=False, lower_length=20, upper_length=20)
            if dc_df is not None and not dc_df.empty:
                dc_df.columns = [c.lower() for c in dc_df.columns]
                # Dynamic column finding
                lower_col = next((c for c in dc_df.columns if c.startswith('dcl')), None)
                upper_col = next((c for c in dc_df.columns if c.startswith('dcu')), None)
                if lower_col: df['dc_lower'] = dc_df[lower_col]
                if upper_col: df['dc_upper'] = dc_df[upper_col]
            else:
                df['dc_lower'] = df['low'].rolling(20).min().shift(1)
                df['dc_upper'] = df['high'].rolling(20).max().shift(1)

            # 6. Core Volatility & Momentum (RSI, ATR)
            df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)

            # --- Keltner Channels (KC) ---
            kc_df = df.ta.kc(append=False, length=20, scalar=2.0, atr_length=10)
            if kc_df is not None and not kc_df.empty:
                df['kc_lower'] = kc_df.iloc[:, 0]
                df['kc_middle'] = kc_df.iloc[:, 1]
                df['kc_upper'] = kc_df.iloc[:, 2]
            else:
                df['kc_lower'] = 0.0
                df['kc_middle'] = 0.0
                df['kc_upper'] = 0.0

            # --- WaveTrend (WT) via stockstats (Fallback) ---
            try:
                for c in ['open', 'high', 'low', 'close', 'volume']:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
                df = add_wavetrend_stockstats(df)
            except Exception as e:
                logger.warning(f"WaveTrend via stockstats failed: {e}")

            # --- Parabolic SAR (PSAR) ---
            psar_df = df.ta.psar(append=False)
            if psar_df is not None and not psar_df.empty:
                psar_df.columns = [c.lower() for c in psar_df.columns]
                # Combine the Bullish (l) and Bearish (s) SAR into one column 'psar'
                sar_l_col = next((c for c in psar_df.columns if c.startswith('psarl')), None)
                sar_s_col = next((c for c in psar_df.columns if c.startswith('psars')), None)

                if sar_l_col and sar_s_col:
                    df['psar'] = psar_df[sar_l_col].combine_first(psar_df[sar_s_col])
                else:
                    df['psar'] = 0.0
            else:
                df['psar'] = 0.0

            # Fill SAR NaNs
            df['psar'] = df['psar'].ffill().fillna(df['close'])

            # --- RF/SIGNAL PROCESSING FEATURES (Gen-7 Infrastructure) ---
            try:
                rf_features_df = signal_processor.extract_rf_features(df[['open', 'high', 'low', 'close', 'volume']])
                if not rf_features_df.empty:
                    df = pd.concat([df, rf_features_df], axis=1)
            except Exception as e:
                # logger.error(f"Failed to integrate RF features: {e}")
                pass

            # --- VWAP ---
            df['tp'] = (df['high'] + df['low'] + df['close']) / 3
            df['tp_v'] = df['tp'] * df['volume']
            df['vwap'] = df['tp_v'].cumsum() / df['volume'].cumsum()
            df.drop(columns=['tp', 'tp_v'], inplace=True, errors='ignore')

            # 8. Volume MA & IBS
            vol_series = df['volume']
            if isinstance(vol_series, pd.DataFrame): vol_series = vol_series.iloc[:, 0]
            df['vol_ma'] = vol_series.rolling(20).mean()

            # 9. Custom Indicators
            close_series = df['close']
            if isinstance(close_series, pd.DataFrame): close_series = close_series.iloc[:, 0]
            df['daily_return'] = close_series.pct_change()
            low_s = df['low'] if isinstance(df['low'], pd.Series) else df['low'].iloc[:, 0]
            high_s = df['high'] if isinstance(df['high'], pd.Series) else df['high'].iloc[:, 0]
            df['ibs'] = (close_series - low_s) / ((high_s - low_s) + 1e-9)

            # 10. Context Analysis
            df['rel_strength_qqq'] = 0.0
            df['rel_strength_sector'] = 0.0
            df['sector_daily_return'] = 0.0

            if context_data:
                qqq = context_data.get('qqq', pd.DataFrame())
                if not qqq.empty:
                    qqq = qqq[~qqq.index.duplicated(keep='last')]
                    qqq_ret = qqq['close'].pct_change()
                    df['rel_strength_qqq'] = df['daily_return'] - qqq_ret.reindex(df.index).fillna(0)

                sec = context_data.get('sector', pd.DataFrame())
                if not sec.empty:
                    sec = sec[~sec.index.duplicated(keep='last')]
                    sec_ret = sec['close'].pct_change()
                    aligned_sec = sec_ret.reindex(df.index).fillna(0)
                    df['rel_strength_sector'] = df['daily_return'] - aligned_sec
                    df['sector_daily_return'] = aligned_sec

            # 11. Candlestick Patterns (Gen-7)
            try:
                # Basic
                df = self.add_candlestick_patterns(df)
                # Advanced
                df = self.add_advanced_patterns(df)
            except Exception as e:
                logger.warning(f"Candlestick pattern extraction failed: {e}")

            df.fillna(0, inplace=True)
            cols = pd.Index(df.columns)
            df = df.loc[:, cols.duplicated() == False]

            if not df.empty:
                latest_features = df.iloc[-1].to_dict()
                # logger.debug(f"FEATURES LOGGED {json.dumps(latest_features)}")
            return df

        except Exception as e:
            logger.error(f"Calc Error: {e}", exc_info=True)
            return pd.DataFrame()