"""
📊 StockWise NASDAQ Pipeline - Gen-3 Comprehensive Version

This script handles the comprehensive fetching, processing, and saving of NASDAQ
stock data into Parquet files, optimized for the Gen-3 specialist model training.

Key Gen-3 Upgrades:
- Calculates new statistical features (Z-Score, BB_Width, Correlation).
- Implements robust, global volatility clustering for specialist model routing.
- Uses the advanced Triple Barrier Method for intelligent, risk-aware labeling.
- Optimizes data source management to initialize connections only once.

Usage:
    python Create_parquet_file_NASDAQ.py --max-stocks 2000
    python Create_parquet_file_NASDAQ.py --profit-mode=1per
"""

import os
import random
import sys
import argparse
import logging
import traceback
import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm
from datetime import timedelta, datetime
import pandas_ta as ta
from data_source_manager import DataSourceManager

try:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', stream=sys.stdout)
    logger = logging.getLogger(__name__)
    logger.info("✅ Successfully imported DataSourceManager.")
except ImportError as e:
    print(f"❌ Critical Error: Could not import DataSourceManager. "
          f"Please ensure 'data_source_manager.py' is in the correct directory. {e}")
    sys.exit(1)

# --- Directory Definitions ---
BASE_DIR = os.getcwd()
LOG_DIR = os.path.join(BASE_DIR, "logs")
TRAIN_DIR = os.path.join(BASE_DIR, "models/NASDAQ-training set")
TEST_DIR = os.path.join(BASE_DIR, "models/NASDAQ-testing set")
TRAIN_FEATURES_DIR = os.path.join(TRAIN_DIR, "features")
TEST_FEATURES_DIR = os.path.join(TEST_DIR, "features")

# Create directories if they don't exist
os.makedirs(TRAIN_FEATURES_DIR, exist_ok=True)
os.makedirs(TEST_FEATURES_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
# --- End Directory Definitions ---


def calculate_global_volatility_thresholds(tickers, ibkr_manager):
    """Calculates 33rd and 66th percentiles of 90-day volatility across the dataset."""
    logger.info(f"🌀 Analyzing volatility across all {len(tickers)} training stocks...")
    all_volatilities = []
    for symbol in tqdm(tickers, desc="Analyzing volatility"):
        df_raw = ibkr_manager.get_stock_data(symbol, days_back=5*365)
        df = clean_raw_data(df_raw)

        # Ensure there's enough data for a 90-day rolling window
        if df is not None and not df.empty and len(df) > 90:
            if 'close' in df.columns:
                volatility_90d = df['close'].pct_change().rolling(window=90).std()
                all_volatilities.append(volatility_90d)

        if not df.empty and 'close' in df.columns:
            volatility_90d = df['close'].pct_change().rolling(window=90).std()
            all_volatilities.append(volatility_90d)

    if not all_volatilities:
        logger.error("❌ Could not calculate any volatility data. Using default thresholds.")
        return 0.02, 0.04

    combined_vol = pd.concat(all_volatilities).dropna()
    low_thresh, high_thresh = combined_vol.quantile([0.33, 0.66])
    logger.info(f"✅ Global Volatility Thresholds Calculated: Low < {low_thresh:.4f}, High > {high_thresh:.4f}")
    return low_thresh, high_thresh


def load_vix_data(data_manager):
    """Fetches and cleans VIX data."""
    try:
        vix_raw = data_manager.get_stock_data("^VIX")
        vix_clean = clean_raw_data(vix_raw)
        return vix_clean['close'] if not vix_clean.empty else pd.Series()
    except Exception as e:
        logger.error(f"❌ Could not download VIX data. Feature will be disabled. Error: {e}")
        return pd.Series()


def load_tlt_data(data_manager):
    """Fetches and cleans TLT (bond ETF) data."""
    try:
        tlt_raw = data_manager.get_stock_data("TLT")
        tlt_clean = clean_raw_data(tlt_raw)
        return tlt_clean['close'] if not tlt_clean.empty else pd.Series()
    except Exception as e:
        logger.error(f"❌ Could not download TLT data. Feature will be disabled. Error: {e}")
        return pd.Series()


def load_qqq_data(data_manager):
    """Fetches and cleans QQQ data to be used for correlation calculations."""
    try:
        qqq_raw = data_manager.get_stock_data("QQQ")
        qqq_clean = clean_raw_data(qqq_raw)
        return qqq_clean['close'] if not qqq_clean.empty else pd.Series()
    except Exception as e:
        logger.error(f"❌ Could not download QQQ data. Correlation feature will be disabled. Error: {e}")
        return pd.Series()


# In Create_parquet_file_NASDAQ.py

def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """A single, robust function to clean raw data immediately after fetching."""
    if df is None or df.empty:
        return pd.DataFrame()

    # This handles the case of multi-ticker downloads from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(0)

    df.columns = [col.lower() for col in df.columns]

    # Ensure standard OHLCV columns are numeric, coercing errors
    standard_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in standard_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Only drop NaNs from the columns that actually exist in the DataFrame.
    existing_cols = [col for col in standard_cols if col in df.columns]
    df.dropna(subset=existing_cols, inplace=True)
    return df


def load_nasdaq_tickers(max_stocks=None):
    """Loads NASDAQ ticker symbols from a CSV file."""
    try:
        df = pd.read_csv("nasdaq_stocks.csv")
        df.columns = [col.strip().title() for col in df.columns]
        if 'Symbol' not in df.columns:
            logger.error("❌ 'Symbol' column not found in nasdaq_stocks.csv")
            return []

        # Filter out symbols with '^' or '.' which are often indices or warrants
        tickers = df[~df['Symbol'].str.contains(r'\^|\.', na=True)]['Symbol'].dropna().tolist()
        logger.info(f"✅ Loaded {len(tickers)} symbols from CSV after robust filtering.")
        if max_stocks:
            logger.info(f"📊 Limiting to {max_stocks} stocks from {len(tickers)} available.")
            return tickers[:max_stocks]
        return tickers
    except FileNotFoundError:
        logger.error("❌ 'nasdaq_stocks.csv' not found. Please ensure the file is in the root directory.")
        return []


def get_symbols_from_csv(file_path):
    """Loads and robustly filters NASDAQ symbols from a local CSV file."""
    if not os.path.exists(file_path):
        logger.error(f"❌ CSV file not found at: {file_path}")
        return []
    try:
        df = pd.read_csv(file_path)
        if 'Symbol' not in df.columns:
            logger.error(f"❌ CSV file '{file_path}' does not contain a 'Symbol' column.")
            return []
        df['Symbol'] = df['Symbol'].fillna('').astype(str).str.strip()
        problematic_patterns = r'[RUWXPZ]$|\.|\^|/|-|\$|\s'
        nasdaq_symbols = df[
            (df['Symbol'].str.len() > 0) &
            (~df['Symbol'].str.contains(problematic_patterns, regex=True)) &
            (df['Symbol'].str.isalpha()) &
            (df['Symbol'].str.len() <= 5)
        ]['Symbol'].tolist()
        logger.info(f"✅ Loaded {len(nasdaq_symbols)} symbols from CSV after robust filtering.")
        return nasdaq_symbols
    except Exception as e:
        logger.error(f"❌ Error loading or parsing CSV file '{file_path}': {e}")
        return []


def generate_comprehensive_nasdaq_ticker_lists(train_file, test_file, max_stocks_limit=None):
    """Generates and splits NASDAQ tickers into training and testing lists."""
    logger.info("🚀 COMPREHENSIVE MODE: Loading NASDAQ symbols from CSV.")
    # csv_file_path = os.path.join(BASE_DIR, "nasdaq_full_list.csv")
    all_nasdaq_symbols = load_nasdaq_tickers(max_stocks_limit)

    if not all_nasdaq_symbols:
        logger.critical("❌ No NASDAQ symbols loaded. Cannot proceed.")
        sys.exit(1)

    if max_stocks_limit and len(all_nasdaq_symbols) > max_stocks_limit:
        logger.info(f"📊 Limiting to {max_stocks_limit} stocks from {len(all_nasdaq_symbols)} available.")
        all_nasdaq_symbols = all_nasdaq_symbols[:max_stocks_limit]

    random.shuffle(all_nasdaq_symbols)
    split_point = int(len(all_nasdaq_symbols) * 0.7)
    train_tickers = all_nasdaq_symbols[:split_point]
    test_tickers = all_nasdaq_symbols[split_point:]

    with open(train_file, 'w') as f:
        f.write('\n'.join(train_tickers))
    with open(test_file, 'w') as f:
        f.write('\n'.join(test_tickers))

    logger.info(f"✅ Generated training list ({len(train_tickers)} stocks) and testing list ({len(test_tickers)} stocks).")
    return train_tickers, test_tickers


def get_dominant_cycle(data: pd.Series, min_period=3, max_period=100) -> float:
    """Uses FFT to find the dominant cycle period in a time series."""
    data = data.dropna()
    if len(data) < min_period: return 0.0
    detrended = data - np.poly1d(np.polyfit(np.arange(len(data)), data, 1))(np.arange(len(data)))
    fft_result = np.fft.fft(detrended.values)
    frequencies = np.fft.fftfreq(len(detrended))
    power = np.abs(fft_result) ** 2
    positive_freq_mask = frequencies > 0
    if not np.any(positive_freq_mask): return 0.0
    periods = 1 / frequencies[positive_freq_mask]
    period_mask = (periods >= min_period) & (periods <= max_period)
    if not np.any(period_mask): return 0.0
    dominant_idx = np.argmax(power[positive_freq_mask][period_mask])
    return periods[period_mask][dominant_idx]


def apply_triple_barrier(
        close_prices: pd.Series,
        high_prices: pd.Series,
        low_prices: pd.Series,
        atr: pd.Series,
        profit_take_mult: float,
        stop_loss_mult: float,
        time_limit_days: int,
        profit_mode: str = 'dynamic',
        net_profit_target: float = 0.03
) -> pd.Series:
    """Implements the Triple Barrier Method for labeling financial time series data."""
    logger.info(f"Applying Triple Barrier Method in '{profit_mode}' mode...")
    outcomes = pd.Series(index=close_prices.index, dtype=np.int8, data=0)

    for i in tqdm(range(len(close_prices) - time_limit_days), desc="Labeling events", leave=False, ascii=True):
        entry_price = close_prices.iloc[i]
        current_atr = atr.iloc[i]
        if pd.isna(current_atr) or current_atr == 0:
            continue

        # --- Logic switch for calculating the upper (profit take) barrier ---
        if profit_mode == 'fixed_net':
            # Calculate the required gross profit to achieve the net target
            tax_rate = 0.25
            commission_rate = 0.004
            required_gross_profit = (net_profit_target + commission_rate) / (1 - tax_rate)
            upper_barrier = entry_price * (1 + required_gross_profit)
        else:  # 'dynamic' mode
            # This is your original ATR-based system
            upper_barrier = entry_price + (current_atr * profit_take_mult)

        lower_barrier = entry_price - (current_atr * stop_loss_mult)

        for j in range(1, time_limit_days + 1):
            future_high = high_prices.iloc[i + j]
            future_low = low_prices.iloc[i + j]

            if future_high >= upper_barrier:
                outcomes.iloc[i] = 1
                break
            if future_low <= lower_barrier:
                outcomes.iloc[i] = -1
                break
    logger.info("✅ Triple Barrier labeling complete.")
    return outcomes


def calculate_global_volatility_thresholds(tickers: list, ibkr_manager) -> tuple:
    """Performs a pre-calculation across all training tickers to find global volatility thresholds."""
    logger.info("🌀 Calculating global volatility thresholds across the training dataset...")
    all_volatilities = []
    sample_tickers = tickers
    logger.info(f"Analyzing volatility across all {len(sample_tickers)} training stocks...")

    for symbol in tqdm(sample_tickers, desc="Analyzing volatility", ascii=True):
        try:
            df_raw = ibkr_manager.get_stock_data(symbol, days_back=5*365)
            df = clean_raw_data(df_raw)
            if df is not None and not df.empty and len(df) > 90:

                # Check again after normalization, in case data was unusable
                if not df.empty and 'close' in df.columns:
                    volatility_90d = df['close'].pct_change().rolling(window=90).std()
                    all_volatilities.append(volatility_90d)
        except Exception as e:
            # CORRECTED: Log the full traceback to help with debugging
            logger.error(f"❌ Failed to calculate volatility for {symbol}. Error: {e}")
            logger.debug(traceback.format_exc())
            continue # Continue to the next stock, but report the error.

    if not all_volatilities:
        logger.error("❌ Could not calculate any volatilities. Falling back to default thresholds.")
        return 0.015, 0.03

    combined_vol = pd.concat(all_volatilities).dropna()

    if combined_vol.empty:
        logger.error("❌ Combined volatility series is empty. Falling back to default thresholds.")
        return 0.015, 0.03

    low_thresh = combined_vol.quantile(0.33)
    high_thresh = combined_vol.quantile(0.66)
    logger.info(f"✅ Global Volatility Thresholds Calculated: Low < {low_thresh:.4f}, High > {high_thresh:.4f}")
    return low_thresh, high_thresh


# def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Normalizes a multi-indexed DataFrame with a single ticker column
#     to a simple single-level index.
#     """
#     if isinstance(df.columns, pd.MultiIndex):
#         if df.columns.nlevels == 2 and 'Ticker' in df.columns.names:
#             # Assuming a structure like [('Close', 'AAPL'), ('Volume', 'AAPL')]
#             # Drop the Ticker level from the columns
#             df.columns = df.columns.droplevel('Ticker')
#             return df
#     return df


def add_technical_indicators_and_features(
        df: pd.DataFrame,
        vol_thresholds: tuple,
        qqq_close: pd.Series,
        profit_mode: str,
        net_profit_target: float,
        vix_close: pd.Series,
        tlt_close: pd.Series
) -> pd.DataFrame:

    df = df.copy()
    if df.empty or len(df) < 50:
        return pd.DataFrame()

    logger.debug(f"--- Feature Engineering Started ---")
    logger.debug(f"Initial columns: {df.columns.tolist()}")
    logger.debug(f"Data shape: {df.shape}")

    # --- Standard Indicators (from pandas-ta) ---
    df.ta.bbands(length=20, append=True, col_names=("bb_lower", "bb_middle", "bb_upper", "bb_width", "bb_position"))
    df.ta.atr(length=14, append=True, col_names="atr_14")
    df.ta.rsi(length=14, append=True, col_names="rsi_14")
    df.ta.rsi(length=28, append=True, col_names="rsi_28")
    df.ta.macd(append=True, col_names=("macd", "macd_histogram", "macd_signal"))
    df.ta.adx(length=14, append=True, col_names=("adx", "adx_pos", "adx_neg", "adxr_temp"))
    df.drop(columns=["adxr_temp"], inplace=True)
    df.ta.mom(length=5, append=True, col_names="momentum_5")
    df.ta.obv(append=True)

    # --- New & Calculated Features (Request #5) ---
    df['daily_return'] = df['close'].pct_change()
    df['volume_ma_20'] = df['volume'].rolling(20).mean()
    df['volatility_20d'] = df['daily_return'].rolling(20).std()
    df['z_score_20'] = (df['close'] - df['bb_middle']) / df['close'].rolling(20).std()
    df['kama_10'] = calculate_kama(df['close'], window=10)
    df['stoch_k'], df['stoch_d'] = calculate_stochastic(df['high'], df['low'], df['close'])
    df['dominant_cycle'] = df['close'].rolling(window=252, min_periods=90).apply(get_dominant_cycle, raw=False)
    df.ta.cmf(append=True, col_names="cmf")

    # Correlation
    if not qqq_close.empty:
        aligned_qqq = qqq_close.reindex(df.index, method='ffill')
        df['correlation_50d_qqq'] = df['close'].rolling(50).corr(aligned_qqq)
    else:
        df['correlation_50d_qqq'] = 0.0

    if not tlt_close.empty:
        aligned_tlt = tlt_close.reindex(df.index, method='ffill')
        df['corr_tlt'] = df['close'].rolling(50).corr(aligned_tlt)
    else:
        df['corr_tlt'] = 0.0

    if not vix_close.empty:
        aligned_vix = vix_close.reindex(df.index, method='ffill')
        df['vix_close'] = aligned_vix
    else:
        df['vix_close'] = 0.0

    # Volatility Clustering
    low_thresh, high_thresh = vol_thresholds
    df['volatility_90d'] = df['daily_return'].rolling(90).std()
    df['volatility_cluster'] = pd.cut(df['volatility_90d'], bins=[-np.inf, low_thresh, high_thresh, np.inf],
                                      labels=['low', 'mid', 'high'])

    logger.debug(f"DEBUG: Columns after all feature calculations: {df.columns.tolist()}")

    # --- Advanced Target Labeling (Request #2) ---
    tb_labels = apply_triple_barrier(
        close_prices=df['close'], high_prices=df['high'], low_prices=df['low'], atr=df['atr_14'],
        profit_take_mult=2.0, stop_loss_mult=2.5, time_limit_days=15,
        profit_mode=profit_mode, net_profit_target=net_profit_target
    )
    df['target_entry'] = (tb_labels == 1).astype(int)
    df['target_cut_loss'] = (tb_labels == -1).astype(int)

    # Smarter Profit Take Label
    overbought_condition = df['rsi_14'] > 75
    df['target_profit_take'] = ((tb_labels == 1) & overbought_condition).astype(int)

    # New Trailing Stop Label
    df['target_trailing_stop'] = 0
    profitable_mask = (df['high'] > df['close'].shift(1) * (1 + 0.5 * df['atr_14'] / df['close'].shift(1)))
    trailing_stop_hit = (
                df['low'] < df['high'].rolling(5).max() * (1 - 1.5 * df['atr_14'] / df['high'].rolling(5).max()))
    df.loc[profitable_mask & trailing_stop_hit, 'target_trailing_stop'] = 1

    # --- Final Cleanup ---
    df = df.bfill().ffill()
    df.dropna(inplace=True)
    df.columns = [col.lower() for col in df.columns]

    expected_columns = [
        'open', 'high', 'low', 'close', 'volume', 'volume_ma_20', 'rsi_14', 'momentum_5', 'macd', 'macd_signal',
        'macd_histogram', 'bb_position', 'volatility_20d', 'atr_14', 'adx', 'adx_pos', 'adx_neg', 'obv', 'rsi_28',
        'z_score_20', 'bb_width', 'correlation_50d_qqq', 'vix_close','corr_tlt', 'cmf',
        'bb_upper', 'bb_lower', 'bb_middle', 'daily_return',
        'kama_10', 'stoch_k', 'stoch_d', 'dominant_cycle',
        'volatility_cluster', 'target_entry', 'target_profit_take', 'target_cut_loss', 'target_trailing_stop'
    ]
    existing_cols = [col for col in expected_columns if col in df.columns]
    return df[existing_cols]


def extend_date_range_for_features(start_date: pd.Timestamp, lookback_days: int) -> pd.Timestamp:
    """
    Given a requested start date and a lookback window (e.g., 39 days),
    returns the extended start date to fetch data earlier.
    """
    return start_date - timedelta(days=lookback_days)


def process_ticker_list(
        tickers: list,
        output_dir: str, # This is now the FINAL, correct directory (e.g., .../features/2per_profit)
        vol_thresholds: tuple,
        data_source_manager,
        qqq_data: pd.Series,
        profit_mode: str,
        net_profit_target: float,
        vix_data: pd.Series,
        tlt_data: pd.Series
):
    """
    Process a list of tickers by extracting features and saving them as parquet files.
    """
    # FIXED: The correct strategy-specific path is now passed directly into this function.
    # No need to construct it again here.
    logger.info(f"Saving processed files into: {output_dir}")

    processed_count = 0
    skipped_tickers = {}

    for symbol in tqdm(tickers, desc=f"Processing stocks for '{profit_mode}'"):
        try:
            # Fetch raw stock data
            df_raw = data_source_manager.get_stock_data(symbol)
            df = clean_raw_data(df_raw)

            if df.empty or len(df) < 252:  # Require at least a year of data
                skipped_tickers[symbol] = "Insufficient data"
                continue

            # Generate features
            featured_df = add_technical_indicators_and_features(
                df.copy(), vol_thresholds, qqq_data, profit_mode, net_profit_target, vix_data, tlt_data)

            # Single, correct save location.
            if not featured_df.empty:
                output_path = os.path.join(output_dir, f"{symbol}_features.parquet")
                featured_df.to_parquet(output_path)
                processed_count += 1
            else:
                skipped_tickers[symbol] = "DataFrame became empty after feature engineering"
                logger.warning(f"Skipping {symbol}: feature engineering returned empty")

        except Exception as e:
            skipped_tickers[symbol] = f"General processing error: {e}"
            logger.error(f"Failed to process {symbol}: {e}")
            logger.debug(traceback.format_exc()) # Added for better debugging

    logger.info(f"Processed {processed_count} stocks successfully for this run.")
    if skipped_tickers:
        logger.warning(f"Skipped {len(skipped_tickers)} stocks: {list(skipped_tickers.keys())}")


def calculate_kama(close, window=10, pow1=2, pow2=30):
    """Calculates Kaufman's Adaptive Moving Average (KAMA) manually."""
    diff = abs(close.diff(1))

    # --- FIX #1: Prevent division by zero ---
    rolling_sum_diff = diff.rolling(window).sum()
    # Replace zeros with a small number to avoid NaN, then calculate er
    er = abs(close.diff(window)) / rolling_sum_diff.replace(0, np.nan)
    er.fillna(0, inplace=True)  # Fill any resulting NaNs with 0

    sc = (er * (2 / (pow1 + 1) - 2 / (pow2 + 1)) + 2 / (pow2 + 1)) ** 2

    # --- FIX #2: Create a float array to handle NaN values correctly ---
    kama = np.zeros_like(close, dtype=float)

    kama[:window] = close.iloc[:window]
    for i in range(window, len(close)):
        # Check if smoothing constant is a valid number before calculation
        if not np.isnan(sc.iloc[i]):
            kama[i] = kama[i - 1] + sc.iloc[i] * (close.iloc[i] - kama[i - 1])
        else:
            kama[i] = kama[i - 1]  # If sc is NaN, carry over the last value

    return pd.Series(kama, index=close.index)


# Manually implement Stochastic Oscillator
def calculate_stochastic(high, low, close, window=14, smooth_window=3):
    """Calculates the Stochastic Oscillator (%K and %D) manually."""
    lowest_low = low.rolling(window=window).min()
    highest_high = high.rolling(window=window).max()
    percent_k = 100 * ((close - lowest_low) / (highest_high - lowest_low + 1e-10))
    percent_d = percent_k.rolling(window=smooth_window).mean()
    return percent_k, percent_d


def main():
    """Main function to orchestrate the data processing pipeline for multiple strategies."""
    parser = argparse.ArgumentParser(description='StockWise NASDAQ Data Processor - Gen 3')
    parser.add_argument('--max-stocks', type=int, help='Maximum number of stocks to process')
    parser.add_argument('--small-test', action='store_true', help='Run on a small, hardcoded test set.')
    # NEW: Add arguments to control the strategy from the command line
    parser.add_argument('--profit-mode', type=str, default='all',
                        choices=['all', 'dynamic', '1per', '2per', '3per', '4per'],
                        help='Which profit-taking strategy to generate data for.')
    parser.add_argument('--net-profit-target', type=float,
                        help='Single net profit target for fixed_net mode (e.g., 0.03).')
    args = parser.parse_args()

    datamanager = DataSourceManager()

    logger.info("Attempting to connect to IBKR TWS...")
    if not datamanager.connect_to_ibkr():
        logger.warning("Could not connect to IBKR TWS.")
        logger.info("Proceeding with yfinance as the data source.")
        datamanager.use_ibkr = False  # Explicitly disable IBKR for the rest of the run
    else:
        logger.info("✅ Successfully connected to IBKR TWS.")

    # --- Define and create the base output directories here ---
    base_output_dir = "models"
    train_dir = os.path.join(base_output_dir, "NASDAQ-training set", "features")
    test_dir = os.path.join(base_output_dir, "NASDAQ-testing set", "features")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Fetch QQQ Data ONCE at the start using the robust data manager
    logger.info("📅 Fetching QQQ data for correlation calculations...")
    qqq_close = load_qqq_data(datamanager)

    # Load VIX data
    logger.info("📅 Fetching VIX data for market volatility context...")
    vix_close = load_vix_data(datamanager)

    logger.info("📅 Fetching TLT data for Correlation to Bonds...")
    tlt_close = load_tlt_data(datamanager)

    # Get ticker lists
    if args.small_test:
        train_tickers, test_tickers = ['AAPL', 'MSFT', 'GOOGL'], ['AMZN', 'NVDA']
    else:
        train_file = os.path.join(LOG_DIR, "nasdaq_train_comprehensive.txt")
        test_file = os.path.join(LOG_DIR, "nasdaq_test_comprehensive.txt")
        train_tickers, test_tickers = generate_comprehensive_nasdaq_ticker_lists(train_file, test_file, args.max_stocks)

    global_vol_thresholds = calculate_global_volatility_thresholds(train_tickers, datamanager)

    # FIXED: Define strategies to run based on explicit command-line arguments
    strategies = []
    if args.profit_mode == 'all':
        strategies.append({'mode': 'dynamic', 'target': 0})
        strategies.extend([
            {'mode': 'fixed_net', 'target': 0.01},
            {'mode': 'fixed_net', 'target': 0.02},
            {'mode': 'fixed_net', 'target': 0.03},
            {'mode': 'fixed_net', 'target': 0.04}
        ])
    elif args.profit_mode == 'dynamic':
        strategies.append({'mode': 'dynamic', 'target': 0})
    elif args.profit_mode == '1per':
        strategies.append({'mode': 'fixed_net', 'target': 0.01})
    elif args.profit_mode == '2per':
        strategies.append({'mode': 'fixed_net', 'target': 0.02})
    elif args.profit_mode == '3per':
        strategies.append({'mode': 'fixed_net', 'target': 0.03})
    elif args.profit_mode == '4per':
        strategies.append({'mode': 'fixed_net', 'target': 0.04})

    # Loop through the defined strategies
    for strategy in strategies:
        mode = strategy['mode']
        target = strategy['target']

        if mode == 'dynamic':
            subdir_name = "dynamic_profit"
        else:  # fixed_net
            subdir_name = f"{int(target * 100)}per_profit"

        logger.info(f"\n{'=' * 80}\n🚀 GENERATING DATA FOR STRATEGY: {subdir_name.upper()}\n{'=' * 80}")

        train_output_dir = os.path.join(TRAIN_FEATURES_DIR, subdir_name)
        test_output_dir = os.path.join(TEST_FEATURES_DIR, subdir_name)
        os.makedirs(train_output_dir, exist_ok=True)
        os.makedirs(test_output_dir, exist_ok=True)

        # Pass the correct strategy-specific output directory to the processing function.
        if train_tickers:
            process_ticker_list(train_tickers, train_output_dir, global_vol_thresholds, datamanager, qqq_close,
                                mode, target,vix_close, tlt_close)
        if test_tickers:
            process_ticker_list(test_tickers, test_output_dir, global_vol_thresholds, datamanager, qqq_close,
                                mode, target,vix_close, tlt_close)

    datamanager.disconnect()
    logger.info("\n🎉 Disconnecting from IBKR.")
    logger.info("✅ All data generation pipelines have finished successfully.")


if __name__ == "__main__":
    main()