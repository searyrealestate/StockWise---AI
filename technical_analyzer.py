# technical_analyzer.py

import pandas as pd
import pandas_ta as ta
import numpy as np
from scipy.signal import find_peaks


class TechnicalAnalyzer:
    """
    A dedicated class to find complex technical patterns, S/R levels,
    and candlestick patterns on a given DataFrame.
    """

    def __init__(self, df: pd.DataFrame, logger=None):
        self.df = df.copy()
        self.log = logger if logger else lambda msg, level="INFO": None

        # --- Pre-calculate all patterns on init ---
        self.find_all_candlestick_patterns()

    def find_support_resistance(self, lookback_days=120, cluster_pct=0.02):
        """
        Finds Support and Resistance zones by finding price pivots and
        clustering them.

        Args:
            lookback_days (int): How many days back to look for pivots.
            cluster_pct (float): How close (in %) levels must be to form a zone.

        Returns:
            dict: {'support': [list], 'resistance': [list]}
        """
        data = self.df.tail(lookback_days)
        if data.empty:
            return {'support': [], 'resistance': []}

        # 1. Find pivot highs (resistance) and lows (support)
        # We use a 5-day window to define a local pivot
        peak_indices, _ = find_peaks(data['high'], distance=5)
        trough_indices, _ = find_peaks(-data['low'], distance=5)

        resistance_levels = data.iloc[peak_indices]['high']
        support_levels = data.iloc[trough_indices]['low']

        # 2. Cluster the levels to form zones
        support_zones = self._cluster_levels(support_levels, cluster_pct)
        resistance_zones = self._cluster_levels(resistance_levels, cluster_pct)

        return {'support': support_zones, 'resistance': resistance_zones}

    def _cluster_levels(self, levels: pd.Series, cluster_pct: float):
        """Helper to group price levels into zones."""
        if levels.empty:
            return []

        levels_sorted = levels.sort_values().values
        clusters = []
        current_cluster = [levels_sorted[0]]

        for level in levels_sorted[1:]:
            if level <= (current_cluster[-1] * (1 + cluster_pct)):
                current_cluster.append(level)
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [level]

        if current_cluster:
            clusters.append(np.mean(current_cluster))

        return clusters

    def check_volume_breakout(self, window=20, multiplier=1.5):
        """
        Checks if the latest volume is 'strong' (i.e., above its
        moving average by a certain multiplier).
        """
        if self.df.empty or len(self.df) < window:
            return False

        self.df['volume_sma_20'] = self.df['volume'].rolling(window).mean()
        latest = self.df.iloc[-1]

        return latest['volume'] > (latest['volume_sma_20'] * multiplier)

    def find_all_candlestick_patterns(self):
        """
        Uses pandas-ta to scan for ALL known candlestick patterns.
        This creates many new columns, e.g., 'cdl_hammer'.
        """
        try:
            # This single command runs over 70 pattern recognizers
            self.df.ta.cdl_pattern(name="all", append=True)
        except Exception as e:
            self.log(f"Candlestick pattern analysis failed: {e}", "WARN")

    def check_bullish_candlestick(self):
        """
        Checks the latest row for any common bullish reversal patterns
        (e.g., Hammer, Bullish Engulfing, Morning Star).

        A value of 100 indicates a confirmed pattern.
        """
        if self.df.empty:
            return False

        latest = self.df.iloc[-1]

        # List of bullish pattern columns created by pandas-ta
        bullish_patterns = [
            'cdl_hammer', 'cdl_invertedhammer', 'cdl_bullishengulfing',
            'cdl_morningstar', 'cdl_piercing'
        ]

        for pattern_col in bullish_patterns:
            if pattern_col in latest and latest[pattern_col] == 100:
                return True

        return False

    def check_continuation_candlestick(self):
        """
        Checks the latest row for continuation/indecision patterns
        (e.g., Doji, Spinning Top).

        A value > 0 indicates a pattern.
        """
        if self.df.empty:
            return False

        latest = self.df.iloc[-1]

        # List of continuation/indecision pattern columns
        continuation_patterns = [
            'cdl_doji', 'cdl_spinningtop'
        ]

        for pattern_col in continuation_patterns:
            # These patterns are not 100/-100, they are just > 0
            if pattern_col in latest and latest[pattern_col] > 0:
                return True

        return False