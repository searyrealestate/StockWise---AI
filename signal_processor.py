import pandas as pd
import numpy as np
import logging
import pandas_ta as ta

logger = logging.getLogger("SignalProcessor")


def extract_rf_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates initial RF and Signal Processing proxy features for Gen-7 infrastructure.
    Outputs: DataFrame with new features: kalman_smooth, wavelet_std_ratio, lyapunov_local.
    """
    df = df.copy()
    n = len(df)

    # Needs a minimum history for the rolling calculations
    if n < 200:
        # Return a placeholder DataFrame aligned to the input index
        return pd.DataFrame({'kalman_smooth': 0.0, 'wavelet_std_ratio': 0.0, 'lyapunov_local': 0.0}, index=df.index)

    # --- 1. Kalman Filter Proxy (kalman_smooth) ---
    # Concept: Low-lag trend deviation. Deviation from a fast EMA acts as a Kalman proxy.
    try:
        df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
        # The feature is the current price deviation from the smoothed price
        df['kalman_smooth'] = df['close'] - df['ema_10']
    except Exception as e:
        logger.warning(f"Kalman Proxy Feature Error: {e}")
        df['kalman_smooth'] = 0.0

    # 2. Wavelet Features Proxy (wavelet_std_ratio)
    # Concept: Volatility Clustering/Noise Ratio. High ratio signals noisy market conditions.
    try:
        df.ta.true_range(append=True, col_names='tr')
        df['atr_20'] = df['tr'].ewm(span=20, adjust=False).mean()
        df['atr_ema_20'] = df['atr_20'].ewm(span=20, adjust=False).mean()

        # High ratio suggests the market is entering a choppy/noisy regime
        df['wavelet_std_ratio'] = (df['atr_20'] / df['atr_ema_20'])

        # Clean up NaNs/Infs
        df['wavelet_std_ratio'] = df['wavelet_std_ratio'].replace([np.inf, -np.inf], 0).fillna(0)

    except Exception as e:
        logger.warning(f"Wavelet Proxy Feature Error: {e}")
        df['wavelet_std_ratio'] = 0.0

    # 3. Chaos/Complexity Metrics Proxy (lyapunov_local)
    # Concept: Local Lyapunov Exponent (predictability). Proxy using 5-bar momentum.
    try:
        df['lyapunov_local'] = df['close'].diff(5)
    except Exception as e:
        logger.warning(f"Chaos Proxy Feature Error: {e}")
        df['lyapunov_local'] = 0.0

    # Final cleanup and alignment
    new_features = df[['kalman_smooth', 'wavelet_std_ratio', 'lyapunov_local']].ffill().bfill().fillna(0)
    return new_features