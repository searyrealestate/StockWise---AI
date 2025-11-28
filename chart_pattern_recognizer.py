# chart_pattern_recognizer.py

"""
Chart Pattern Recognizer
========================

This module implements rule-based functions to detect common chart patterns
like the "Cup & Handle" and "Head & Shoulders".

Note: Algorithmic pattern detection is notoriously difficult and can be
unreliable. These functions are simplified implementations.
"""

import pandas as pd
import numpy as np
import logging
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)


def detect_cup_and_handle(data: pd.DataFrame, min_length=60, max_length=250, min_depth_pct=0.15,
                          handle_pct=0.1) -> dict:
    """
    Attempts to detect a Cup and Handle pattern.

    :param data: DataFrame with 'high', 'low', 'close'
    :param min_length: Min days for the cup formation
    :param max_length: Max days for the cup formation
    :param min_depth_pct: Minimum depth of the cup as % of the left lip
    :param handle_pct: Max % of cup depth for the handle to pull back
    :return: A dictionary with signal and breakout level
    """
    result = {'signal': 'NONE', 'breakout_level': None}
    try:
        # Use high prices for finding the rim
        highs = data['high']
        if len(highs) < max_length:
            max_length = len(highs) - 1
            if max_length < min_length:
                return result

        # 1. Find the right lip (most recent significant peak)
        peaks, _ = find_peaks(highs, distance=5, prominence=highs.mean() * 0.05)
        if len(peaks) < 2:
            return result

        right_lip_idx = peaks[-1]
        right_lip_price = highs.iloc[right_lip_idx]

        # 2. Find the left lip
        # Look backwards from the right lip for a similar-level peak
        potential_left_lips = peaks[:-1][
            (highs.iloc[peaks[:-1]] >= right_lip_price * 0.95) &
            (highs.iloc[peaks[:-1]] <= right_lip_price * 1.05)
            ]
        if len(potential_left_lips) == 0:
            return result

        left_lip_idx = potential_left_lips[-1]
        left_lip_price = highs.iloc[left_lip_idx]

        cup_length = right_lip_idx - left_lip_idx
        if not (min_length <= cup_length <= max_length):
            return result

        # 3. Find the cup bottom
        cup_data = data.iloc[left_lip_idx:right_lip_idx]
        cup_bottom_price = cup_data['low'].min()
        cup_depth = left_lip_price - cup_bottom_price

        if cup_depth / left_lip_price < min_depth_pct:
            return result  # Cup is not deep enough

        # 4. Find the handle
        handle_data = data.iloc[right_lip_idx:]
        if len(handle_data) < 5:  # Handle needs at least 5 days
            return result

        handle_low = handle_data['low'].min()
        handle_pullback = right_lip_price - handle_low

        # Handle shouldn't be too deep
        if handle_pullback > (cup_depth * handle_pct):
            return result

        breakout_level = max(left_lip_price, right_lip_price)
        current_price = data['close'].iloc[-1]

        if current_price >= breakout_level:
            result['signal'] = 'BUY_SIGNAL'
        elif current_price >= breakout_level * 0.97:  # Within 3%
            result['signal'] = 'NEAR_BREAKOUT'

        result['breakout_level'] = breakout_level

        if result['signal'] != 'NONE':
            logger.info(f"Cup & Handle detected for {data['symbol'].iloc[0]}: {result}")

        return result

    except Exception as e:
        logger.error(f"Error in Cup & Handle detection: {e}", exc_info=True)
        return result


def detect_head_and_shoulders(data: pd.DataFrame, lookback=120) -> dict:
    """
    Attempts to detect a Head and Shoulders (bearish) pattern.
    """
    result = {'signal': 'NONE', 'breakdown_level': None}
    try:
        prices = data['close'].iloc[-lookback:]
        if len(prices) < 60: return result  # Need min data

        # Find all peaks and troughs
        peaks, _ = find_peaks(prices, distance=10)
        troughs, _ = find_peaks(-prices, distance=10)

        if len(peaks) < 3 or len(troughs) < 2:
            return result

        # 1. Identify Head (highest peak)
        head_idx = peaks[np.argmax(prices.iloc[peaks])]
        head_price = prices.iloc[head_idx]

        # 2. Identify Left Shoulder (a peak before the head)
        left_shoulders = peaks[peaks < head_idx]
        if len(left_shoulders) == 0: return result
        left_shoulder_idx = left_shoulders[np.argmax(prices.iloc[left_shoulders])]

        # 3. Identify Right Shoulder (a peak after the head)
        right_shoulders = peaks[peaks > head_idx]
        if len(right_shoulders) == 0: return result
        right_shoulder_idx = right_shoulders[np.argmax(prices.iloc[right_shoulders])]

        # Basic H&S validation
        if not (prices.iloc[left_shoulder_idx] < head_price and prices.iloc[right_shoulder_idx] < head_price):
            return result  # Head is not the highest

        # 4. Identify Neckline (troughs between shoulders/head)
        left_trough = troughs[troughs > left_shoulder_idx & troughs < head_idx]
        right_trough = troughs[troughs > head_idx & troughs < right_shoulder_idx]

        if len(left_trough) == 0 or len(right_trough) == 0:
            return result

        # Simplification: Use the lower of the two troughs as the neckline
        neckline_price = min(prices.iloc[left_trough[0]], prices.iloc[right_trough[0]])

        current_price = prices.iloc[-1]
        result['breakdown_level'] = neckline_price

        if current_price < neckline_price:
            result['signal'] = 'SELL_SIGNAL'
        elif current_price <= neckline_price * 1.03:  # Within 3%
            result['signal'] = 'NEAR_BREAKDOWN'

        if result['signal'] != 'NONE':
            logger.info(f"Head & Shoulders detected for {data['symbol'].iloc[0]}: {result}")

        return result

    except Exception as e:
        logger.error(f"Error in Head & Shoulders detection: {e}", exc_info=True)
        return result