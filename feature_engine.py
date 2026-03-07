# feature_engine.py

import pandas as pd
import pandas_ta as ta
import numpy as np
import logging
import warnings
from scipy.signal import argrelextrema
import system_config as cfg

# Silence benign pandas_ta warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pandas_ta")

# Initialize Logger
logger = cfg.LoggerSetup.setup_logger("FeatureEngine")


class FeatureEngine:
    """
    Gen-12 Feature Engine.
    Integrates 85+ Technical Models including Advanced Geometric Pattern Recognition,
    Institutional Trend Templates, and VSA Logic.
    """
    def __init__(self):
        pass

    def calculate_features(self, df, strategy_config=None):
        """
        Orchestrator: Runs calculation blocks based on Strategy Genome.
        """
        if df.empty: return df
        
        # Default to ALL if no specific config
        indicators = strategy_config.get("active_indicators", ["all"]) if strategy_config else ["all"]
        run_all = "all" in indicators

        try:
            # --- BLOCK A: TREND MODELS (Models 1-15) ---
            if run_all or "trend" in indicators:
                df = self.add_trend_block(df)
                logger.debug("Completed Trend Models Block.")
                
            # --- BLOCK A.2: DIGITAL SIGNAL PROCESSING (DSP) ---
            # This is mandatory for the AI Orchestra. It generates the routing data.
            if run_all or "dsp" in indicators:
                df = self.add_dsp_block(df)
                logger.debug("Completed Digital Signal Processing Block.")

            # --- BLOCK B: MOMENTUM MODELS (Models 16-25) ---
            if run_all or "momentum" in indicators:
                df = self.add_momentum_block(df)

            # --- BLOCK C: VOLATILITY MODELS (Models 26-35) ---
            if run_all or "volatility" in indicators:
                df = self.add_volatility_block(df)

            # --- BLOCK D: VOLUME & VSA MODELS (Models 36-45) ---
            if run_all or "volume" in indicators:
                df = self.add_volume_block(df)

            # --- BLOCK E: CANDLESTICK PATTERNS (Models 46-100+) ---
            if run_all or "patterns" in indicators:
                df = self.add_pattern_block(df)

            # --- BLOCK F: GEOMETRIC CHART PATTERNS (Models 101-110) ---
            if run_all or "geometry" in indicators:
                df = self.add_geometry_block(df)

            # === SURGICAL FIX: SAFE NaN HANDLING ===
            # Price columns use forward-fill (last known price is better than 0).
            # Indicator columns use zero-fill (safe for oscillators and booleans).
            # Zero-filling prices corrupts RSI, SMA, ATR and all ratio calculations.
            price_cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
            indicator_cols = [c for c in df.columns if c not in price_cols]
            if price_cols:
                df[price_cols] = df[price_cols].ffill()
            if indicator_cols:
                df[indicator_cols] = df[indicator_cols].fillna(0)

            # --- STANDARDIZE COLUMN NAMES FOR AI AGENT ---
            rename_map = {}
            for col in df.columns:
                if col.startswith('MACD_'): rename_map[col] = 'macd'
                elif col.startswith('MACDh_'): rename_map[col] = 'macd_hist'
                elif col.startswith('MACDs_'): rename_map[col] = 'macd_signal'
                elif col.startswith('STOCHk_'): rename_map[col] = 'stoch_k'
                elif col.startswith('STOCHd_'): rename_map[col] = 'stoch_d'
                elif col.startswith('BBL_'): rename_map[col] = 'bb_lower'
                elif col.startswith('BBM_'): rename_map[col] = 'bb_mid'
                elif col.startswith('BBU_'): rename_map[col] = 'bb_upper'
                elif col.startswith('BBP_'): rename_map[col] = 'bb_width'
                elif col.startswith('KCLe_'): rename_map[col] = 'kc_lower'
                elif col.startswith('KCUe_'): rename_map[col] = 'kc_upper'
            
            if rename_map:
                df.rename(columns=rename_map, inplace=True)
                logger.debug(f"Normalized {len(rename_map)} Pandas-TA columns for AI input.")

        except Exception as e:
            logger.error(f"Critical Feature Calculation Error: {e}")
            
        # Ensure Dynamic Stop-Loss vectors are generated before returning
        df = self._calculate_dynamic_stop_loss(df)
        
        return df


    def add_trend_block(self, df):
        """Calculates Institutional Trend Models with Safety Checks."""
        try:
            # --- SURGICAL FIX: Safe Indicator Extraction ---
            # Prevents 'NoneType' crashes when history < requested period (e.g. 104 weeks vs SMA 150)
            
            # Helper to safely get the series or a 0-filled series if None
            def safe_sma(source, length):
                res = ta.sma(source, length=length)
                if res is None or res.empty:
                    return pd.Series(0.0, index=df.index)
                return res.fillna(0.0) # Ensure no NaNs exist in the valid series

            # [1-5] SMAs
            df['sma_20'] = safe_sma(df['close'], 20)
            df['sma_50'] = safe_sma(df['close'], 50)
            df['sma_100'] = safe_sma(df['close'], 100)
            df['sma_150'] = safe_sma(df['close'], 150)
            df['sma_200'] = safe_sma(df['close'], 200)
            
            # [6-7] EMAs
            res_ema12 = ta.ema(df['close'], length=12)
            df['ema_12'] = res_ema12.fillna(0.0) if res_ema12 is not None else 0.0
            
            res_ema26 = ta.ema(df['close'], length=26)
            df['ema_26'] = res_ema26.fillna(0.0) if res_ema26 is not None else 0.0

            # [8] Parabolic SAR
            psar = ta.psar(df['high'], df['low'], df['close'])
            if psar is not None and not psar.empty:
                df['psar'] = psar.iloc[:, 0].combine_first(psar.iloc[:, 1])
            else:
                df['psar'] = 0.0

            # [9] SuperTrend
            supertrend = ta.supertrend(df['high'], df['low'], df['close'], length=7, multiplier=3.0)
            if supertrend is not None and not supertrend.empty:
                df['supertrend'] = supertrend.iloc[:, 0]
                df['supertrend_direction'] = supertrend.iloc[:, 1]
            else:
                df['supertrend'] = 0.0
                df['supertrend_direction'] = 0.0

            # [10] ADX
            adx = ta.adx(df['high'], df['low'], df['close'])
            if adx is not None and not adx.empty:
                df['adx'] = adx.iloc[:, 0]
            else:
                df['adx'] = 0.0

            # [11-12] Ichimoku (The previous Tuple Fix)
            ichimoku = ta.ichimoku(df['high'], df['low'], df['close'])
            if ichimoku is not None:
                ichi_df = ichimoku[0] if isinstance(ichimoku, tuple) else ichimoku
                if ichi_df is not None and not ichi_df.empty:
                    df['ichimoku_conv'] = ichi_df.iloc[:, 0]
                    df['ichimoku_base'] = ichi_df.iloc[:, 1]
                else:
                    df['ichimoku_conv'] = 0.0
                    df['ichimoku_base'] = 0.0
            else:
                df['ichimoku_conv'] = 0.0
                df['ichimoku_base'] = 0.0

            # [13] Perfect Trend Alignment
            # NOW SAFE: All columns are guaranteed to be floats, even if 0.0.
            condition_stack = (
                (df['close'] > df['sma_50']) &
                (df['sma_50'] > df['sma_100']) &
                (df['sma_100'] > df['sma_150']) &
                (df['sma_150'] > df['sma_200'])
            )
            df['trend_alignment'] = np.where(condition_stack, 1.0, 0.0)

            # [14-15] Crosses
            df['golden_cross'] = ((df['sma_50'] > df['sma_200']) & (df['sma_50'].shift(1) <= df['sma_200'].shift(1)))
            df['death_cross'] = ((df['sma_50'] < df['sma_200']) & (df['sma_50'].shift(1) >= df['sma_200'].shift(1)))

        except Exception as e:
            logger.error(f"Trend Block Failed: {e}", exc_info=True)
        return df

    def add_momentum_block(self, df):
        """Calculates Momentum Oscillators."""
        try:
            # [16] RSI
            df['rsi'] = ta.rsi(df['close'], length=14)
            
            # [17] MACD Line
            # [18] MACD Signal
            # [19] MACD Histogram
            macd = ta.macd(df['close'])
            if macd is not None and not macd.empty:
                df['macd'] = macd.iloc[:, 0]        # MACD Line
                df['macd_hist'] = macd.iloc[:, 1]   # Histogram
                df['macd_signal'] = macd.iloc[:, 2] # Signal Line
            else:
                df['macd'] = 0.0
                df['macd_hist'] = 0.0
                df['macd_signal'] = 0.0
                logger.warning("MACD calculation failed. Check data history.")

            # [20] Stochastic %K
            # [21] Stochastic %D
            stoch = ta.stoch(df['high'], df['low'], df['close'])
            if stoch is not None and not stoch.empty:
                df['stoch_k'] = stoch.iloc[:, 0]
                df['stoch_d'] = stoch.iloc[:, 1]
            else:
                df['stoch_k'] = 0.0
                df['stoch_d'] = 0.0
                logger.warning("Stochastic calculation failed. Check data history.")

            # [22] Williams %R
            df['willr'] = ta.willr(df['high'], df['low'], df['close'])

            # [23] CCI
            df['cci'] = ta.cci(df['high'], df['low'], df['close'])
            
            # [24] ROC (Rate of Change)
            df['roc'] = ta.roc(df['close'], length=10)

        except Exception as e:
            logger.error(f"Momentum Block Failed: {e}")
        return df

    def add_volatility_block(self, df):
        """Calculates Volatility models."""
        try:
            # [25] ATR (Average True Range)
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            
            # [26] Bollinger Upper
            # [27] Bollinger Lower
            # [28] Bollinger Width (Squeeze)
            bb = ta.bbands(df['close'], length=20)
            if bb is not None and not bb.empty:
                # 0: Lower, 1: Mid, 2: Upper, 3: Bandwidth, 4: Percent
                df['bb_lower'] = bb.iloc[:, 0]
                df['bb_mid'] = bb.iloc[:, 1]
                df['bb_upper'] = bb.iloc[:, 2]
                df['bb_width'] = bb.iloc[:, 3] 
            else:
                df['bb_lower'] = 0.0
                df['bb_mid'] = 0.0
                df['bb_upper'] = 0.0
                df['bb_width'] = 0.0
                logger.warning("Bollinger Bands calculation failed. Check data history.")

            # [29] Keltner Upper
            # [30] Keltner Lower
            kc = ta.kc(df['high'], df['low'], df['close'])
            if kc is not None and not kc.empty:
                # 0: Lower, 1: Basis, 2: Upper
                df['kc_lower'] = kc.iloc[:, 0]
                df['kc_upper'] = kc.iloc[:, 2]
            else:
                df['kc_lower'] = 0.0
                df['kc_upper'] = 0.0
                logger.warning("Keltner Channels calculation failed. Check data history.")

            # [Bug 1.6a] Bollinger Squeeze Detection (BB inside KC = energy building)
            if all(c in df.columns for c in ['bb_lower', 'bb_upper', 'kc_lower', 'kc_upper']):
                df['squeeze_on'] = (
                    (df['bb_lower'] > df['kc_lower']) &
                    (df['bb_upper'] < df['kc_upper'])
                ).astype(int)
            else:
                df['squeeze_on'] = 0

            # [Bug 1.6a] Momentum Squeeze: MACD histogram as firing momentum proxy
            if 'macd_hist' in df.columns:
                df['mom_sqz'] = df['macd_hist']
            else:
                df['mom_sqz'] = 0.0

            # [31] Donchian Upper
            # [32] Donchian Lower
            dc = ta.donchian(df['high'], df['low'])
            if dc is not None and not dc.empty:
                df['dc_lower'] = dc.iloc[:, 0]
                df['dc_upper'] = dc.iloc[:, 2] # Typically upper is index 2 in Donchian returns
            else:
                df['dc_lower'] = 0.0
                df['dc_upper'] = 0.0
                logger.warning("Donchian Channels calculation failed. Check data history.")

        except Exception as e:
            logger.error(f"Volatility Block Failed: {e}")
        return df

    def add_volume_block(self, df):
        """Calculates Volume and Liquidity Models."""
        try:
            # [33] Volume SMA 20
            df['vol_avg_20'] = ta.sma(df['volume'], length=20)
            
            # [34] RVOL (Relative Volume)
            df['rvol'] = df['volume'] / df['vol_avg_20'].replace(0, 1)
            
            # [35] OBV (On Balance Volume)
            df['obv'] = ta.obv(df['close'], df['volume'])

            # [36] VWAP (Volume Weighted Avg Price)
            if 'high' in df.columns and 'low' in df.columns:
                df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])

            # [37] VSA Squat Bar (Price Spread vs Volume)
            # High Volume + Small Spread = Manipulation
            df['spread'] = df['high'] - df['low']
            df['spread_avg'] = ta.sma(df['spread'], length=20)
            condition_squat = (df['rvol'] > 1.5) & (df['spread'] < (df['spread_avg'] * 0.8))
            df['vsa_squat_bar'] = condition_squat

            # [38] CMF (Chaikin Money Flow)
            df['cmf'] = ta.cmf(df['high'], df['low'], df['close'], df['volume'])

        except Exception as e:
            logger.error(f"Volume Block Failed: {e}")
        return df

    def add_pattern_block(self, df):
        """Calculates 60+ Candlestick Patterns."""
        try:
            # [46-100+] Runs ALL CDL patterns in pandas_ta
            # Includes: Doji, Engulfing, Harami, Hammer, Shooting Star, etc.
            df.ta.cdl_pattern(name="all", append=True)
        except Exception as e:
            logger.error(f"Pattern Block Failed: {e}")
        return df

    def add_geometry_block(self, df):
        """
        Calculates Geometric Chart Patterns using Pivots.
        Models 101-110.
        """
        try:
            # [39] Gap Up
            df['gap_up'] = df['low'] > df['high'].shift(1)
            # [40] Gap Down
            df['gap_down'] = df['high'] < df['low'].shift(1)

            # [41] Fib 23.6%
            # [42] Fib 38.2%
            # [43] Fib 61.8%
            rolling_high = df['high'].rolling(50).max()
            rolling_low = df['low'].rolling(50).min()
            diff = rolling_high - rolling_low
            df['fib_236'] = rolling_high - (diff * 0.236)
            df['fib_382'] = rolling_high - (diff * 0.382)
            df['fib_618'] = rolling_high - (diff * 0.618)
            df['fib_ext_1272'] = rolling_high + (diff * 0.272)
            df['fib_ext_1618'] = rolling_high + (diff * 0.618)

            # --- Advanced Pivot Detection ---
            n = 5 
            # [44] Local Peaks
            df['peak'] = df.iloc[argrelextrema(df['high'].values, np.greater_equal, order=n)]['high']
            # [45] Local Troughs
            df['trough'] = df.iloc[argrelextrema(df['low'].values, np.less_equal, order=n)]['low']
            
            peaks = df['peak'].fillna(0)
            troughs = df['trough'].fillna(0)

            # [46] Double Top
            is_peak = peaks > 0
            prev_peak_val = peaks.replace(0, np.nan).shift(1).ffill()
            df['double_top'] = is_peak & (abs(df['high'] - prev_peak_val) < (df['high'] * 0.015))

            # [47] Double Bottom
            is_trough = troughs > 0
            prev_trough_val = troughs.replace(0, np.nan).shift(1).ffill()
            df['double_bottom'] = is_trough & (abs(df['low'] - prev_trough_val) < (df['low'] * 0.015))

            # [48] Head and Shoulders (Bearish)
            # P2 (Head) > P1 (Left) AND P2 > P3 (Right)
            p3 = peaks # Current
            p2 = peaks.replace(0, np.nan).shift(1).ffill() 
            p1 = peaks.replace(0, np.nan).shift(2).ffill() 
            df['hs_pattern'] = is_peak & (p2 > p1) & (p2 > p3)

        except Exception as e:
            logger.error(f"Geometry Block Failed: {e}")
        return df

    def add_dsp_block(self, df):
        """
        Digital Signal Processing (DSP) Module.
        This function calculates the mathematical heartbeat of the stock. It figures out 
        if the stock is moving in a clean line (Trend) or thrashing wildly (Chop).
        This data will be used by Agent 1 (The Regime Router).
        """
        try:
            logger.info("Initializing DSP calculation sequence to evaluate Market Regime.")
            
            # 1. Fetch our thresholds from the memory bank (system_config.py)
            slow_lookback = cfg.DSP_CONFIG.get("er_lookback_slow", 20)
            fast_lookback = cfg.DSP_CONFIG.get("er_lookback_fast", 5)
            
            # --- PHASE A: The Slow Efficiency Ratio (The Core Trend) ---
            # We ask the data: Over the last 20 days, what was the absolute distance traveled from point A to point B?
            direction_slow = df['close'].diff(slow_lookback).abs()
            
            # Next, we ask: How much energy did the stock waste bouncing up and down to get there?
            volatility_slow = df['close'].diff().abs().rolling(slow_lookback).sum()
            
            # The Efficiency Ratio is simply the actual distance divided by the wasted energy. 
            # 1.0 = Perfect straight line. 0.0 = Pure noise.
            df['er_slow'] = direction_slow / volatility_slow.replace(0, np.nan)
            logger.debug("Successfully calculated Slow Efficiency Ratio (20-day wave).")
            
            # --- PHASE B: The Fast Efficiency Ratio (The Early Warning System) ---
            # We repeat the exact same calculation, but only looking at the last 5 days.
            # If the 20-day trend is great, but the 5-day trend suddenly collapses, the AI knows a crash is starting.
            direction_fast = df['close'].diff(fast_lookback).abs()
            volatility_fast = df['close'].diff().abs().rolling(fast_lookback).sum()
            df['er_fast'] = direction_fast / volatility_fast.replace(0, np.nan)
            logger.debug("Successfully calculated Fast Efficiency Ratio (5-day wave).")
            
            # --- PHASE C: Sanitization ---
            # Machine learning models explode if we feed them NaN (Not a Number) values. We fill empty slots with 0.
            df['er_slow'] = df['er_slow'].fillna(0)
            df['er_fast'] = df['er_fast'].fillna(0)
            
            logger.info("DSP evaluation complete. Signal-to-Noise arrays appended to the DataFrame.")
            
        except Exception as e:
            # If the math fails, we log the exact error to the debug file to figure out why.
            logger.error("Failed to calculate DSP wave structures. Returning default DataFrame.")
            logger.debug(f"DSP Exception Details: {str(e)}", exc_info=True)
            
        return df

    # def calculate_technical_score(self, df, strategy_config=None):
    #     """Calculates Final Technical Score based on 85+ models."""
    #     if df.empty: return 0.0, []
        
    #     row = df.iloc[-1]
    #     score = 0
    #     reasons = []
        
    #     weights = strategy_config.get("weights", {
    #         "trend": 30, "momentum": 25, "volatility": 15, "volume": 15, "pattern": 15
    #     }) if strategy_config else {"trend": 30, "momentum": 25, "volatility": 15, "volume": 15, "pattern": 15}

    #     # 1. Trend Scoring
    #     if row.get('trend_alignment', 0) == 1.0:
    #         score += weights['trend']
    #         reasons.append("Perfect Trend Alignment (Price>50>100>150>200)")
    #     elif row.get('close', 0) > row.get('sma_200', 0):
    #         score += (weights['trend'] / 2)
    #         reasons.append("Above SMA200")

    #     if row.get('golden_cross'):
    #         score += 20
    #         reasons.append("GOLDEN CROSS")
    #     if row.get('death_cross'):
    #         score -= 100
    #         reasons.append("DEATH CROSS")

    #     # 2. Momentum Scoring
    #     rsi = row.get('rsi', 50)
    #     if 30 < rsi < 70:
    #         score += 5 
    #     elif rsi <= 30:
    #         score += weights['momentum']
    #         reasons.append(f"Oversold (RSI {rsi:.1f})")
    #     elif rsi >= 70:
    #         score -= 10
    #         reasons.append(f"Overbought (RSI {rsi:.1f})")

    #     # 3. Volume Scoring
    #     if row.get('rvol', 1) > 1.5:
    #         score += weights['volume']
    #         reasons.append(f"High Relative Vol (x{row['rvol']:.1f})")
    #     if row.get('vsa_squat_bar'):
    #         score -= 50
    #         reasons.append("VSA Squat Bar (Manipulation)")

    #     # 4. Pattern & Geometry Scoring
    #     for col in df.columns:
    #         # Check all CDL_ columns from pandas_ta
    #         if col.startswith('CDL_') and row[col] != 0:
    #             p_name = col.replace('CDL_', '')
    #             if row[col] > 0:
    #                 score += weights['pattern']
    #                 reasons.append(f"Bullish {p_name}")
    #             else:
    #                 score -= weights['pattern']
    #                 reasons.append(f"Bearish {p_name}")

    #     if row.get('double_bottom'):
    #         score += 15
    #         reasons.append("Double Bottom")
    #     if row.get('gap_up'):
    #         score += 5
    #         reasons.append("Gap Up")
    #     if row.get('hs_pattern'):
    #         score -= 20
    #         reasons.append("Head & Shoulders")

    #     return score, reasons

    def calculate_technical_score(self, df, strategy_config):
        """
        Calculates the technical score by evaluating market patterns and 
        applying noise reduction to prevent double counting of overlapping candles.
        """
        if df is None or df.empty:
            return 0.0, []

        row = df.iloc[-1]
        raw_patterns = []

        # 1. Extract all raw candlestick patterns identified by pandas_ta
        for col in df.columns:
            if col.startswith('CDL_') and row.get(col, 0) != 0:
                raw_patterns.append(col.replace('CDL_', 'CANDLE_'))

        # 2. Extract structural and volume setups
        if row.get('momentum_breakout', False) or row.get('squeeze_on', False):
            raw_patterns.append('MOMENTUM_BREAKOUT')
            
        if row.get('vsa_institutional_buying', False):
            raw_patterns.append('VSA_INSTITUTIONAL_BUYING')
            
        if row.get('oversold_bounce', False):
            raw_patterns.append('OVERSOLD_BOUNCE')

        # 3. Apply Noise Reduction (Group into boolean families)
        patterns = self._reduce_candle_noise(raw_patterns)

        # 4. Calculate Final Technical Score (15 points per unique confirmed setup family)
        # Cap the maximum technical score at 100.0
        score = min(len(patterns) * 15.0, 100.0)

        return score, patterns

    def _reduce_candle_noise(self, raw_patterns):
        """
        [Candle Noise Reduction]
        Groups overlapping candlestick patterns into strictly boolean families 
        to prevent Technical Score inflation (Double Counting).
        """
        indecision_family = ['DOJI', 'SPINNINGTOP', 'HIGHWAVE', 'RICKSHAWMAN', 'LONGLEGGEDDOJI', 'HARAMI', 'INSIDE']
        bullish_family = ['HAMMER', 'ENGULFING_BULL', 'MORNINGSTAR', 'PIERCING', 'DRAGONFLYDOJI', 'TAKURI', 'MATCHINGLOW', 'BELTHOLD']
        bearish_family = ['SHOOTINGSTAR', 'ENGULFING_BEAR', 'EVENINGSTAR', 'DARKCLOUDCOVER', 'GRAVESTONEDOJI', 'HIKKAKE']
        
        reduced_patterns = set()
        has_indecision = has_bullish = has_bearish = False
        
        for p in raw_patterns:
            p_upper = p.upper()
            if any(ind in p_upper for ind in indecision_family):
                has_indecision = True
            elif any(bull in p_upper for bull in bullish_family):
                has_bullish = True
            elif any(bear in p_upper for bear in bearish_family):
                has_bearish = True
            else:
                # Retain structural patterns (e.g., MOMENTUM_BREAKOUT)
                reduced_patterns.add(p)
                
        if has_indecision: reduced_patterns.add('CANDLE_INDECISION')
        if has_bullish: reduced_patterns.add('CANDLE_BULLISH_REVERSAL')
        if has_bearish: reduced_patterns.add('CANDLE_BEARISH_REVERSAL')
        
        return list(reduced_patterns)

    def _calculate_dynamic_stop_loss(self, df):
        """
        Gen-13: Calculates dynamic stop loss using ATR (Average True Range).
        Bulletproof implementation that bypasses pandas_ta naming inconsistencies.
        """
        try:
            # 1. Force explicit calculation and bind it directly to the 'atr' column
            if 'atr' not in df.columns:
                df['atr'] = df.ta.atr(length=14)
            
            # 2. Calculate the Dynamic Stop Loss (Entry - 2.5 * ATR)
            # We are now 100% guaranteed that df['atr'] exists
            df['dynamic_stop_loss'] = df['close'] - (2.5 * df['atr'])
            
            return df
        except Exception as e:
            # Log the error gracefully without crashing the pipeline
            import logging
            logging.getLogger("FeatureEngine").error(f"Dynamic Stop Loss calculation failed: {e}")
            return df
