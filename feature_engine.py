# feature_engine.py

"""
Feature Engineering Engine
==========================

This script defines the `RobustFeatureCalculator` class, which serves as the
centralized logic for generating technical indicators and contextual features
for the trading system.

It is designed to be robust against common data issues (duplicate columns,
missing values) and flexible enough to handle dynamic parameters.

Key Functionality:
------------------
-   **Technical Indicators**: Calculates a suite of standard indicators including
    Simple Moving Averages (SMAs), Relative Strength Index (RSI), Average True Range (ATR),
    Average Directional Index (ADX), and Donchian Channels.
-   **Advanced Metrics**:
    -   **Slope (Velocity)**: Calculates the linear regression angle of the price
        to determine trend intensity.
    -   **IBS (Internal Bar Strength)**: Computes the position of the close relative
        to the high-low range.
-   **Contextual Analysis**: If provided with benchmark data (QQQ, Sector ETFs),
    it calculates Relative Strength metrics to compare the stock's performance against
    the broader market and its specific sector.
-   **Data Sanitization**: Automatically handles column duplication, casing issues,
    and fills missing values to ensure downstream models receive clean data.

Usage:
------
    calculator = RobustFeatureCalculator(params={'sma_short': 20, 'sma_long': 100})
    features_df = calculator.calculate_features(stock_df, context_data={'qqq': qqq_df})
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
import logging
import system_config as cfg

logger = logging.getLogger("FeatureEngine")


class RobustFeatureCalculator:
    """
    Centralized Feature Engineering Logic.
    Calculates SMAs, RSI, ATR, ADX, Slope, and Relative Strength.
    """

    def __init__(self, params=None):
        self.params = params if params else cfg.STRATEGY_PARAMS

    def calculate_features(self, df: pd.DataFrame, context_data: dict = None) -> pd.DataFrame:
        df = df.copy()
        if df.empty: return pd.DataFrame()

        try:
            df.columns = [col.lower() for col in df.columns]

            # 1. Remove Duplicates (Critical Fix)
            df = df.loc[:, ~df.columns.duplicated()]

            # 2. Dynamic SMAs
            s_short = int(self.params.get('sma_short', 20))
            s_long = int(self.params.get('sma_long', 100))

            df['sma_short'] = df['close'].rolling(s_short).mean()
            df['sma_long'] = df['close'].rolling(s_long).mean()

            # Map for backward compatibility
            df['sma_50'] = df['sma_short']
            df['sma_200'] = df['sma_long']

            # 3. Slope (Velocity)
            try:
                slope_df = df.ta.slope(length=10)
                if isinstance(slope_df, pd.DataFrame):
                    df['slope_angle'] = slope_df.iloc[:, 0] * (180 / np.pi)
                else:
                    df['slope_angle'] = slope_df * (180 / np.pi)
            except:
                df['slope_angle'] = 0.0

            # 4. ADX (Trend Strength)
            try:
                adx = df.ta.adx(length=14)
                if adx is not None: df = pd.concat([df, adx], axis=1)
            except:
                pass

            # Re-clean after concat
            df = df.loc[:, ~df.columns.duplicated()]

            # 5. Donchian Channels (Breakout)
            df['recent_high'] = df['high'].rolling(50).max().shift(1)

            # 6. Volatility & Momentum
            df.ta.atr(length=14, append=True, col_names='atr_14')
            df.ta.rsi(length=14, append=True, col_names='rsi_14')

            # 7. Volume MA
            vol_series = df['volume']
            if isinstance(vol_series, pd.DataFrame): vol_series = vol_series.iloc[:, 0]
            df['vol_ma'] = vol_series.rolling(20).mean()

            # 8. Custom Indicators
            close_series = df['close']
            if isinstance(close_series, pd.DataFrame): close_series = close_series.iloc[:, 0]

            df['daily_return'] = close_series.pct_change()

            low_s = df['low'] if isinstance(df['low'], pd.Series) else df['low'].iloc[:, 0]
            high_s = df['high'] if isinstance(df['high'], pd.Series) else df['high'].iloc[:, 0]
            df['ibs'] = (close_series - low_s) / ((high_s - low_s) + 1e-9)

            # 9. Context Analysis (Sector & Market)
            df['rel_strength_qqq'] = 0.0
            df['rel_strength_sector'] = 0.0
            df['sector_daily_return'] = 0.0

            if context_data:
                # QQQ
                qqq = context_data.get('qqq', pd.DataFrame())
                if not qqq.empty:
                    qqq = qqq[~qqq.index.duplicated(keep='last')]
                    qqq_ret = qqq['close'].pct_change()
                    df['rel_strength_qqq'] = df['daily_return'] - qqq_ret.reindex(df.index).fillna(0)

                # SECTOR
                sec = context_data.get('sector', pd.DataFrame())
                if not sec.empty:
                    sec = sec[~sec.index.duplicated(keep='last')]
                    sec_ret = sec['close'].pct_change()
                    aligned_sec = sec_ret.reindex(df.index).fillna(0)
                    df['rel_strength_sector'] = df['daily_return'] - aligned_sec
                    df['sector_daily_return'] = aligned_sec

            df.fillna(0, inplace=True)
            return df

        except Exception as e:
            logger.error(f"Calc Error: {e}", exc_info=True)
            return pd.DataFrame()