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
    python Create_parquet_file_NASDAQ.py --profit-mode=2per
    python Create_parquet_file_NASDAQ.py --ticker-file logs/rerun_skipped.txt # Rerun skipped stocks
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
# --- Import for multithreading ---
import concurrent.futures
import queue
import threading
import numba

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
        return qqq_clean['close'] if not qqq_clean.empty and 'close' in qqq_clean.columns else pd.Series()
    except Exception as e:
        logger.error(f"❌ Could not download QQQ data. Correlation feature will be disabled. Error: {e}")
        return pd.Series()


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """A single, robust function to clean raw data immediately after fetching."""
    if df is None or df.empty:
        return pd.DataFrame()
    # Correctly handle MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.columns = [col.lower() for col in df.columns]
    standard_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in standard_cols):
        return pd.DataFrame()
    for col in standard_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=standard_cols, inplace=True)
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


def generate_comprehensive_nasdaq_ticker_lists(train_file, test_file, max_stocks_limit=None):
    """Generates and splits NASDAQ tickers into training and testing lists."""
    logger.info("🚀 COMPREHENSIVE MODE: Loading NASDAQ symbols from CSV.")
    # csv_file_path = os.path.join(BASE_DIR, "nasdaq_full_list.csv")
    all_nasdaq_symbols = load_nasdaq_tickers(max_stocks_limit)

    if not all_nasdaq_symbols:
        logger.critical("❌ No NASDAQ symbols loaded. Cannot proceed.")
        sys.exit(1)

    random.shuffle(all_nasdaq_symbols)
    split_point = int(len(all_nasdaq_symbols) * 0.7)
    train_tickers, test_tickers = all_nasdaq_symbols[:split_point], all_nasdaq_symbols[split_point:]
    with open(train_file, 'w') as f: f.write('\n'.join(train_tickers))
    with open(test_file, 'w') as f: f.write('\n'.join(test_tickers))
    logger.info(
        f"✅ Generated training list ({len(train_tickers)} stocks) and testing list ({len(test_tickers)} stocks).")
    return train_tickers, test_tickers


def get_dominant_cycle(data: pd.Series, min_period=3, max_period=100) -> float:
    """Uses FFT to find the dominant cycle period in a time series."""
    data = data.dropna()
    if len(data) < min_period: return 0.0
    detrended = data - np.poly1d(np.polyfit(np.arange(len(data)), data, 1))(np.arange(len(data)))
    fft_result, freqs = np.fft.fft(detrended.values), np.fft.fftfreq(len(detrended))
    power = np.abs(fft_result) ** 2
    pos_mask = freqs > 0
    if not np.any(pos_mask): return 0.0
    periods = 1 / freqs[pos_mask]
    period_mask = (periods >= min_period) & (periods <= max_period)
    if not np.any(period_mask) or np.sum(period_mask) == 0: return 0.0
    dominant_idx = np.argmax(power[pos_mask][period_mask])
    return float(periods[period_mask][dominant_idx])


@numba.jit(nopython=True)
def _calculate_labels_numba(close_prices, high_prices, low_prices, atr,
                            profit_take_mult, stop_loss_mult, time_limit_bars,
                            profit_mode_is_fixed, net_profit_target):
    """
    A Numba-accelerated JIT function to calculate triple-barrier outcomes.
    This function works with raw NumPy arrays for maximum speed.
    """
    n = len(close_prices)
    outcomes = np.zeros(n, dtype=np.int8)

    for i in range(n - time_limit_bars):
        entry_price = close_prices[i]
        current_atr = atr[i]

        if np.isnan(current_atr) or current_atr == 0:
            continue

        if profit_mode_is_fixed:
            upper_barrier = entry_price * (1 + (net_profit_target + 0.004) / 0.75)
        else:
            upper_barrier = entry_price + (current_atr * profit_take_mult)

        lower_barrier = entry_price - (current_atr * stop_loss_mult)

        for j in range(1, time_limit_bars + 1):
            if high_prices[i + j] >= upper_barrier:
                outcomes[i] = 1
                break
            if low_prices[i + j] <= lower_barrier:
                outcomes[i] = -1
                break

    return outcomes


# @numba.jit(nopython=True)
# def apply_triple_barrier(
#         close_prices: pd.Series,
#         high_prices: pd.Series,
#         low_prices: pd.Series,
#         atr: pd.Series,
#         profit_take_mult: float,
#         stop_loss_mult: float,
#         time_limit_bars: int,
#         profit_mode: str = 'dynamic',
#         net_profit_target: float = 0.03
# ) -> pd.Series:
#     """Implements the Triple Barrier Method for labeling financial time series data."""
#     logger.info(f"Applying Triple Barrier Method in '{profit_mode}' mode...")
#     outcomes = pd.Series(index=close_prices.index, dtype=np.int8, data=0)
#     import sys  # Make sure this is imported at the top of the file
#     for i in tqdm(range(len(close_prices) - time_limit_bars), desc="Labeling events", leave=False, ascii=True,
#                   file=sys.stdout):
#     # for i in tqdm(range(len(close_prices) - time_limit_bars), desc="Labeling events", leave=False, ascii=True):
#         entry_price = close_prices.iloc[i]
#         current_atr = atr.iloc[i]
#         if pd.isna(current_atr) or current_atr == 0: continue
#         upper_barrier = (entry_price * (1 + (net_profit_target + 0.004) / 0.75)) if profit_mode == 'fixed_net' else (entry_price + (current_atr * profit_take_mult))
#         lower_barrier = entry_price - (current_atr * stop_loss_mult)
#         for j in range(1, time_limit_bars + 1):
#             if high_prices.iloc[i + j] >= upper_barrier: outcomes.iloc[i] = 1; break
#             if low_prices.iloc[i + j] <= lower_barrier: outcomes.iloc[i] = -1; break
#     logger.info("✅ Triple Barrier labeling complete.")
#     return outcomes
def apply_triple_barrier(
        close_prices: pd.Series,
        high_prices: pd.Series,
        low_prices: pd.Series,
        atr: pd.Series,
        profit_take_mult: float,
        stop_loss_mult: float,
        time_limit_bars: int,
        profit_mode: str = 'dynamic',
        net_profit_target: float = 0.03
) -> pd.Series:
    """
    A wrapper for the Numba-accelerated Triple Barrier Method. This function
    handles the conversion between pandas Series and NumPy arrays.
    """
    # The logger and tqdm are kept here for high-level feedback, but the core loop is now in C/machine code.
    logger.info(f"Applying Numba-accelerated Triple Barrier Method in '{profit_mode}' mode...")

    # Convert pandas Series to NumPy arrays for Numba
    close_np = close_prices.to_numpy()
    high_np = high_prices.to_numpy()
    low_np = low_prices.to_numpy()
    atr_np = atr.to_numpy()

    # Call the fast, compiled function
    outcomes_np = _calculate_labels_numba(
        close_np, high_np, low_np, atr_np,
        profit_take_mult, stop_loss_mult, time_limit_bars,
        profit_mode == 'fixed_net', net_profit_target
    )

    logger.info("✅ Triple Barrier labeling complete.")

    # Return the result as a pandas Series with the correct index
    return pd.Series(outcomes_np, index=close_prices.index)


def calculate_kama(close, window=10, pow1=2, pow2=30):
    """Calculates Kaufman's Adaptive Moving Average (KAMA) manually."""

    diff = abs(close.diff(1))
    rolling_sum_diff = diff.rolling(window).sum()
    er = abs(close.diff(window)) / rolling_sum_diff.replace(0, np.nan)
    er.fillna(0, inplace=True)
    sc = (er * (2 / (pow1 + 1) - 2 / (pow2 + 1)) + 2 / (pow2 + 1)) ** 2
    kama = np.zeros_like(close, dtype=float)
    kama[:window] = close.iloc[:window].values
    for i in range(window, len(close)):
        kama[i] = kama[i - 1] + sc.iloc[i] * (close.iloc[i] - kama[i - 1]) if not np.isnan(sc.iloc[i]) else kama[
            i - 1]
    return pd.Series(kama, index=close.index)


def calculate_stochastic(high, low, close, window=14, smooth_window=3):
    """Calculates the Stochastic Oscillator (%K and %D) manually."""
    lowest_low = low.rolling(window=window).min()
    highest_high = high.rolling(window=window).max()
    percent_k = 100 * ((close - lowest_low) / (highest_high - lowest_low + 1e-10))
    percent_d = percent_k.rolling(window=smooth_window).mean()
    return percent_k, percent_d


def add_technical_indicators_and_features(df, vol_thresholds, qqq_close, profit_mode, net_profit_target, vix_close,
                                          tlt_close):
    df = df.copy()
    if df.empty or len(df) < 252: return pd.DataFrame()
    df.ta.bbands(length=20, append=True, col_names=("bb_lower", "bb_middle", "bb_upper", "bb_width", "bb_position"))
    df.ta.atr(length=14, append=True, col_names="atr_14")
    df.ta.rsi(length=14, append=True, col_names="rsi_14")
    df.ta.rsi(length=28, append=True, col_names="rsi_28")
    df.ta.macd(append=True, col_names=("macd", "macd_histogram", "macd_signal"))
    df.ta.adx(length=14, append=True, col_names=("adx", "adx_pos", "adx_neg", "adxr_temp"))
    df.ta.mom(length=5, append=True, col_names="momentum_5")
    df.ta.obv(append=True)
    df.ta.cmf(append=True, col_names="cmf")
    df['daily_return'] = df['close'].pct_change()
    df['volume_ma_20'] = df['volume'].rolling(20).mean()
    df['volatility_20d'] = df['daily_return'].rolling(20).std()
    df['z_score_20'] = (df['close'] - df['bb_middle']) / df['close'].rolling(20).std()
    df['kama_10'] = calculate_kama(df['close'], window=10)
    df['stoch_k'], df['stoch_d'] = calculate_stochastic(df['high'], df['low'], df['close'])
    df['dominant_cycle'] = df['close'].rolling(window=252, min_periods=90).apply(get_dominant_cycle, raw=False)
    df.columns = [col.lower() for col in df.columns]
    if not qqq_close.empty:
        df['correlation_50d_qqq'] = df['close'].rolling(50).corr(qqq_close.reindex(df.index, method='ffill'))
    else:
        df['correlation_50d_qqq'] = 0.0
    if not tlt_close.empty:
        df['corr_tlt'] = df['close'].rolling(50).corr(tlt_close.reindex(df.index, method='ffill'))
    else:
        df['corr_tlt'] = 0.0
    if not vix_close.empty:
        df['vix_close'] = vix_close.reindex(df.index, method='ffill')
    else:
        df['vix_close'] = 0.0
    df['volatility_90d'] = df['daily_return'].rolling(90).std()
    df['volatility_cluster'] = pd.cut(df['volatility_90d'],
                                      bins=[-np.inf, vol_thresholds[0], vol_thresholds[1], np.inf],
                                      labels=['low', 'mid', 'high'])

    # --- Adjusted time limit for 15-min data ---
    time_limit_bars = 390  # 15 days * 26 bars per day
    tb_labels = apply_triple_barrier(
        close_prices=df['close'], high_prices=df['high'], low_prices=df['low'], atr=df['atr_14'],
        profit_take_mult=2.0, stop_loss_mult=2.5, time_limit_bars=time_limit_bars,
        profit_mode=profit_mode, net_profit_target=net_profit_target
    )
    df['target_entry'] = np.where(tb_labels == 1, 1, 0)
    df['target_cut_loss'] = np.where(tb_labels == -1, 1, 0)
    overbought_condition = df['rsi_14'] > 75
    df['target_profit_take'] = np.where((tb_labels == 1) & (overbought_condition), 1, 0)
    df['target_trailing_stop'] = 0
    profitable_mask = (df['high'] > df['close'].shift(1) * (1 + 0.5 * df['atr_14'] / df['close'].shift(1)))
    trailing_stop_hit = (
                df['low'] < df['high'].rolling(5).max() * (1 - 1.5 * df['atr_14'] / df['high'].rolling(5).max()))
    df.loc[profitable_mask & trailing_stop_hit, 'target_trailing_stop'] = 1
    df = df.bfill().ffill().dropna()
    expected_columns = ['open', 'high', 'low', 'close', 'volume', 'volume_ma_20', 'rsi_14', 'momentum_5', 'macd',
                        'macd_signal', 'macd_histogram', 'bb_position', 'volatility_20d', 'atr_14', 'adx', 'adx_pos',
                        'adx_neg', 'obv', 'rsi_28', 'z_score_20', 'bb_width', 'correlation_50d_qqq', 'vix_close',
                        'corr_tlt', 'cmf', 'bb_upper', 'bb_lower', 'bb_middle', 'daily_return', 'kama_10', 'stoch_k',
                        'stoch_d', 'dominant_cycle', 'volatility_cluster', 'target_entry', 'target_profit_take',
                        'target_cut_loss', 'target_trailing_stop']
    existing_cols = [col for col in expected_columns if col in df.columns]
    return df[existing_cols]


# --- Parallelized functions ---
# def fetch_volatility_for_stock(args):
#     """Helper to fetch data for a single stock for volatility calculation."""
#     symbol, data_manager = args
#     try:
#         end_date = datetime.now()
#         start_date = end_date - timedelta(days=2 * 365)
#         df_intraday = data_manager.get_stock_data(symbol, days_back=(end_date - start_date).days, interval="15 mins")
#         df = clean_raw_data(df_intraday)
#         if df is not None and not df.empty and len(df) > 90:
#             df_daily = df.resample('D').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
#             if not df_daily.empty and 'close' in df_daily.columns and len(df_daily) > 90:
#                 return df_daily['close'].pct_change().rolling(window=90).std()
#     except Exception:
#         pass
#     return None
def fetch_volatility_for_stock(symbol):
    """Helper to fetch data for a single stock for volatility calculation.
    Creates its own data manager for thread safety."""
    data_manager = None
    try:
        # Each thread gets its own connection with a unique ID
        data_manager = DataSourceManager(use_ibkr=True, host='127.0.0.1', port=7497)
        if not data_manager.connect_to_ibkr():
            data_manager.use_ibkr = False  # Fallback if connection fails

        df_raw = data_manager.get_stock_data(symbol, days_back=3 * 365, interval="1 day")
        df = clean_raw_data(df_raw)

        if df is not None and not df.empty and 'close' in df.columns and len(df) > 90:
            return df['close'].pct_change().rolling(window=90).std()
    except Exception:
        pass  # Return None on any failure
    finally:
        if data_manager and data_manager.isConnected():
            data_manager.disconnect()
    return None


def calculate_global_volatility_thresholds(tickers, datamanager):
    """
    Analyzes a sample of tickers to find global volatility thresholds.
    This version correctly handles the cleaned data.
    """
    logger.info(f"🌀 Analyzing volatility across a sample of {len(tickers)} stocks...")
    all_volatilities = []
    for ticker in tqdm(tickers, desc="Analyzing volatility"):
        try:
            raw_df = datamanager.get_stock_data(ticker)
            df = clean_raw_data(raw_df) # Use the corrected clean function
            if not df.empty and 'close' in df.columns:
                df['daily_return'] = df['close'].pct_change()
                volatility = df['daily_return'].rolling(90).std() * np.sqrt(252)
                all_volatilities.append(volatility.dropna())
        except Exception as e:
            logger.warning(f"Could not process volatility for {ticker}: {e}")
            continue
    if not all_volatilities:
        logger.error("❌ Could not calculate volatility for any stocks. Using default thresholds.")
        return {'low_thresh': 0.20, 'high_thresh': 0.40}
    combined_vol = pd.concat(all_volatilities)
    low_thresh = combined_vol.quantile(0.33)
    high_thresh = combined_vol.quantile(0.66)
    logger.info(f"✅ Global Volatility Thresholds Calculated: Low < {low_thresh:.3f}, Mid < {high_thresh:.3f}")
    return (low_thresh, high_thresh)

# def process_ticker_list(
#         tickers: list,
#         vol_thresholds: tuple,
#         qqq_data: pd.Series,
#         vix_data: pd.Series,
#         tlt_data: pd.Series,
#         strategies: list,
#         train_tickers: list,
#         test_tickers: list,
#         use_ibkr_flag: bool
# )-> dict:
#     """Manages the parallel processing of a list of tickers."""
#     logger.info(f"Submitting {len(tickers)} tickers for parallel processing...")
#     MAX_WORKERS = 10
#     tasks = [(s, vol_thresholds, qqq_data, vix_data, tlt_data, strategies, train_tickers,
#               test_tickers, use_ibkr_flag) for s in tickers]
#
#     processed_count, skipped_tickers = 0, {}
#     with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#         results = list(tqdm(executor.map(process_single_stock, tasks), total=len(tasks),
#                             desc="Processing All Stocks"))
#
#     for symbol, status in results:
#         if status == "Success":
#             processed_count += 1
#         else:
#             skipped_tickers[symbol] = status
#
#     logger.info(f"Processed {processed_count} stocks successfully.")
#     if skipped_tickers:
#         logger.warning(f"Skipped {len(skipped_tickers)} stocks: {list(skipped_tickers.keys())}")
#
#         return skipped_tickers

# REPLACE the old process_single_stock function with these two new ones.

def verify_downloaded_data(df: pd.DataFrame, symbol: str, logger) -> bool:
    """
    Performs a series of sanity checks on the downloaded DataFrame.
    Returns True if data is valid, False otherwise.
    """
    # Check 1: Ensure the DataFrame is not empty
    if df.empty:
        logger.warning(f"[{symbol}] Verification FAILED: No data was downloaded.")
        return False

    # Check 2: Verify essential columns exist
    required_cols = {'open', 'high', 'low', 'close', 'volume'}
    if not required_cols.issubset(df.columns):
        logger.warning(f"[{symbol}] Verification FAILED: Missing required columns. Found: {list(df.columns)}")
        return False

    # Check 3: Check for any NaN (Not a Number) values in critical columns
    if df[['open', 'high', 'low', 'close', 'volume']].isnull().values.any():
        logger.warning(f"[{symbol}] Verification FAILED: Data contains NaN values in critical columns.")
        return False

    # Check 4: Check for invalid OHLC data (e.g., Low > High)
    if (df['low'] > df['high']).any():
        logger.warning(f"[{symbol}] Verification FAILED: Found rows where Low price is greater than High price.")
        return False

    # Check 5: Check for significant gaps in the time series (more than 7 days)
    # This helps detect if the download was partial.
    gaps = df.index.to_series().diff().dt.days.gt(7).sum()
    if gaps > 0:
        logger.warning(f"[{symbol}] Verification WARNING: Found {gaps} significant time gap(s) > 7 days.")
        # This is a warning, but you could change `return True` to `return False` to be stricter

    logger.info(f"[{symbol}] ✅ Data verification passed successfully.")
    return True


def data_worker(q, vol_thresholds, qqq_data, vix_data, tlt_data, strategies, train_tickers, test_tickers, use_ibkr_flag,
                ibkr_host, ibkr_port, pbar, skipped_tickers_lock, skipped_tickers):
    """
    This is the worker function for each thread. It continuously fetches a symbol
    from the queue and processes it until the queue is empty.
    """
    data_manager = DataSourceManager(use_ibkr=use_ibkr_flag, host=ibkr_host, port=ibkr_port, allow_fallback=False)
    if use_ibkr_flag and not data_manager.connect_to_ibkr():
        data_manager.use_ibkr = False

    while not q.empty():
        try:
            symbol = q.get_nowait()
            status = "Unknown Error"

            try:
                # Step A: Download the data from IBKR ONLY
                df_raw = data_manager.get_stock_data(symbol, days_back=3 * 365, interval="1 day")
                df = clean_raw_data(df_raw)

                # Step B: Verify the downloaded data
                if not verify_downloaded_data(df, symbol, logger):
                    status = "Data Verification Failed"
                elif len(df) < 252:
                    status = "Insufficient data after verification"
                else:
                    # Step C: If verification passes, proceed with processing
                    for strategy in strategies:
                        mode = strategy['mode']
                        target = strategy['target']
                        subdir_name = f"{int(target * 100)}per_profit" if mode == 'fixed_net' else "dynamic_profit"

                        if symbol in train_tickers:
                            output_dir = os.path.join(TRAIN_FEATURES_DIR, subdir_name)
                        elif symbol in test_tickers:
                            output_dir = os.path.join(TEST_FEATURES_DIR, subdir_name)
                        else:
                            continue

                        featured_df = add_technical_indicators_and_features(df.copy(), vol_thresholds, qqq_data, mode,
                                                                            target, vix_data, tlt_data)

                        if not featured_df.empty:
                            os.makedirs(output_dir, exist_ok=True)
                            output_path = os.path.join(output_dir, f"{symbol}_daily_context.parquet")
                            featured_df.to_parquet(output_path)
                    status = "Success"

            except Exception as e:
                logger.error(f"Failed to process {symbol}: {e}\n{traceback.format_exc()}")
                status = f"General processing error: {e}"

            if status != "Success":
                with skipped_tickers_lock:
                    skipped_tickers[symbol] = status

            pbar.update(1)
        except queue.Empty:
            continue
        finally:
            q.task_done()

    if data_manager and data_manager.isConnected():
        data_manager.disconnect()


# REPLACE this entire function

def process_ticker_list_producer_consumer(tickers: list, vol_thresholds: tuple, datamanager: DataSourceManager,
                                          qqq_data: pd.Series, vix_data: pd.Series, tlt_data: pd.Series,
                                          strategies: list, train_tickers: list, test_tickers: list) -> dict:
    """
    Orchestrator using the producer-consumer model with corrected arguments.
    """
    logger.info(f"Starting producer-consumer processing for {len(tickers)} tickers...")

    task_queue = queue.Queue()
    for s in tickers:
        task_queue.put(s)

    skipped_tickers = {}
    skipped_tickers_lock = threading.Lock()

    pbar = tqdm(total=len(tickers), desc="Processing All Stocks")

    threads = []
    MAX_WORKERS = 5
    for _ in range(MAX_WORKERS):
        # --- THIS IS THE FULLY CORRECTED ARGUMENT LIST ---
        thread = threading.Thread(target=data_worker, args=(
            task_queue,
            vol_thresholds,
            qqq_data,
            vix_data,
            tlt_data,
            strategies,
            train_tickers,
            test_tickers,
            datamanager.use_ibkr,
            datamanager.host,
            datamanager.port,
            pbar,
            skipped_tickers_lock,
            skipped_tickers
        ))
        thread.start()
        threads.append(thread)

    task_queue.join()

    for thread in threads:
        thread.join()

    pbar.close()

    logger.info(f"Processed {len(tickers) - len(skipped_tickers)} stocks successfully.")
    if skipped_tickers:
        logger.warning(f"Skipped {len(skipped_tickers)} stocks: {list(skipped_tickers.keys())}")

    return skipped_tickers

def process_single_stock(args):
    """Processes a single stock for the main data generation task."""
    # Unpack arguments
    (symbol, vol_thresholds, qqq_data, vix_data, tlt_data, strategies_to_run, train_tickers,
     test_tickers, use_ibkr_flag) = args

    data_manager = None  # Initialize to ensure finally block works
    try:
        # --- Instantiate a new manager for each worker process ---
        data_manager = DataSourceManager(use_ibkr=use_ibkr_flag, host='127.0.0.1', port=7497)
        if not data_manager.connect_to_ibkr():
            # Fallback to yfinance if the connection fails
            data_manager.use_ibkr = False

        # --- Download Data ONCE ---
        # Use 15 mins interval for the main data processing
        # df_raw = data_manager.get_stock_data(symbol, days_back=2 * 365, interval="15 mins")
        df_raw = data_manager.get_stock_data(symbol, days_back=3 * 365, interval="1 day")
        df = clean_raw_data(df_raw)

        # if df.empty or len(df) < 252 * 26:  # Require at least a year of 15-min data
        if df.empty or len(df) < 252:  # Require at least a year of DAILY data
            return symbol, "Insufficient data"

        # --- Loop through strategies and process the downloaded data ---
        for strategy in strategies_to_run:
            mode = strategy['mode']
            target = strategy['target']

            subdir_name = f"{int(target * 100)}per_profit" if mode == 'fixed_net' else "dynamic_profit"

            if symbol in train_tickers:
                output_dir = os.path.join(TRAIN_FEATURES_DIR, subdir_name)
            elif symbol in test_tickers:
                output_dir = os.path.join(TEST_FEATURES_DIR, subdir_name)
            else:
                continue  # Skip if symbol not in either list

            featured_df = add_technical_indicators_and_features(df.copy(), vol_thresholds, qqq_data,
                                                                mode, target, vix_data, tlt_data)

            if not featured_df.empty:
                os.makedirs(output_dir, exist_ok=True)  # Ensure subdir exists
                # output_path = os.path.join(output_dir, f"{symbol}_features.parquet")
                output_path = os.path.join(output_dir, f"{symbol}_daily_context.parquet")  # New filename
                featured_df.to_parquet(output_path)

        return symbol, "Success"

    except Exception as e:
        logger.error(f"Failed to process {symbol}: {e}\n{traceback.format_exc()}")
        return symbol, f"General processing error: {e}"

    finally:
        if data_manager and data_manager.isConnected():
            data_manager.disconnect()


def get_historical_intraday_data(datamanager, ticker, start_date_str, end_date_str, interval="15m"):
    """
    Fetches historical intraday data by looping through API-limited chunks.
    yfinance is used here as it's more robust for bulk historical downloads than the TWS API.
    """
    all_data = []
    # yfinance has a 730-day limit for intraday, so we fetch in 2-year chunks.
    date_ranges = pd.date_range(start=start_date_str, end=end_date_str, freq='729D')
    logger.info(f"Starting historical intraday download for {ticker}...")

    for i in range(len(date_ranges)):
        chunk_start = date_ranges[i]
        chunk_end = date_ranges[i + 1] if i + 1 < len(date_ranges) else pd.to_datetime(end_date_str)
        try:
            df_chunk = yf.download(tickers=ticker, start=chunk_start, end=chunk_end, interval=interval,
                                   auto_adjust=True, progress=False)
            if not df_chunk.empty:
                all_data.append(df_chunk)
        except Exception as e:
            logger.warning(f"Could not download chunk for {ticker}. Error: {e}")

    if not all_data:
        return pd.DataFrame()
    return pd.concat(all_data)


def apply_volatility_clustering(df: pd.DataFrame, global_vol_thresholds: dict) -> pd.DataFrame:
    """Applies volatility labels based on pre-calculated global thresholds."""
    df['volatility_90d'] = df['daily_return'].rolling(90).std() * np.sqrt(252)  # Annualized
    low_thresh = global_vol_thresholds.get('low_thresh', 0.20)
    high_thresh = global_vol_thresholds.get('high_thresh', 0.40)
    df['volatility_cluster'] = pd.cut(df['volatility_90d'],
                                      bins=[-np.inf, low_thresh, high_thresh, np.inf],
                                      labels=['low', 'mid', 'high'])
    return df


def apply_triple_barrier_method(df: pd.DataFrame, mode='dynamic', target_pct=0.02) -> pd.DataFrame:
    """
    Placeholder for the Triple Barrier Method logic.
    You will need to integrate your specific labeling logic from your old
    'process_ticker_list' function here.
    """
    logger.warning("Applying placeholder Triple Barrier Method. Integrate your labeling logic here.")
    df['target_entry'] = 0
    df['target_profit_take'] = 0
    df['target_cut_loss'] = 0
    return df


def process_single_ticker(args_tuple):
    """
    Worker function for a single stock. Returns a tuple: (ticker, status_string).
    """
    ticker, strategies_to_run, global_vol_thresholds, train_tickers, test_tickers = args_tuple
    datamanager = None
    try:
        datamanager = DataSourceManager(use_ibkr=True)
        if not datamanager.connect_to_ibkr():
            datamanager.use_ibkr = False

        qqq_close = load_qqq_data(datamanager)
        vix_close = load_vix_data(datamanager)
        tlt_close = load_tlt_data(datamanager)
        raw_df = datamanager.get_stock_data(ticker)
        clean_df = clean_raw_data(raw_df)

        if clean_df.empty or len(clean_df) < 252:
            return (ticker, f"Skipped: Insufficient clean data ({len(clean_df)} bars).")

        featured_df = add_technical_indicators_and_features(clean_df, qqq_close, vix_close, tlt_close)
        df_with_clusters = apply_volatility_clustering(featured_df, global_vol_thresholds)

        for strategy in strategies_to_run:
            mode, target = strategy['mode'], strategy['target']
            subdir_name = f"{int(target * 100)}per_profit" if mode == 'fixed_net' else "dynamic_profit"
            output_dir = TRAIN_FEATURES_DIR if ticker in train_tickers else TEST_FEATURES_DIR
            final_output_path = os.path.join(output_dir, subdir_name, f"{ticker}.parquet")
            df_with_labels = apply_triple_barrier_method(df_with_clusters.copy(), mode, target)
            df_with_labels.to_parquet(final_output_path, index=True)
        return (ticker, "OK")
    except Exception:
        return (ticker, f"ERROR: {traceback.format_exc()}")
    finally:
        if datamanager and datamanager.isConnected():
            datamanager.disconnect()


def main():
    """Main function to orchestrate the data processing pipeline for multiple strategies."""
    parser = argparse.ArgumentParser(description='StockWise NASDAQ Data Processor')
    parser.add_argument('--max-stocks', type=int, help='Maximum number of stocks to process')
    parser.add_argument('--small-test', action='store_true', help='Run on a small, hardcoded test set.')
    parser.add_argument('--profit-mode', type=str, default='all',
                        choices=['all', 'dynamic', '1per', '2per', '3per', '4per'])
    # Inside main() function, with the other parser.add_argument lines...
    parser.add_argument('--ticker-file', type=str,
                        help='Path to a text file containing a comma-separated list of tickers to process.')

    parser.add_argument('--net-profit-target', type=float,
                        help='Single net profit target for fixed_net mode (e.g., 0.03).')
    args = parser.parse_args()

    # --- File Logging Setup ---
    log_file_path = os.path.join(LOG_DIR, f"data_pipeline_run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(file_handler)

    datamanager = None
    try:
        datamanager = DataSourceManager(allow_fallback=False)
        logger.info("Attempting to connect to IBKR TWS...")
        if not datamanager.connect_to_ibkr():
            logger.warning("Could not connect to IBKR TWS. Proceeding with yfinance as the data source.")
            datamanager.use_ibkr = False
        else:
            logger.info("✅ Successfully connected to IBKR TWS.")

        qqq_close = load_qqq_data(datamanager)
        vix_close = load_vix_data(datamanager)
        tlt_close = load_tlt_data(datamanager)

        # --- NEW: Logic to load tickers from file or generate them ---
        if args.ticker_file:
            logger.info(f"💾 Loading tickers from provided file: {args.ticker_file}")
            try:
                with open(args.ticker_file, 'r') as f:
                    content = f.read()
                    # Split by comma and strip whitespace from each ticker
                    all_tickers = [ticker.strip().replace("'", "") for ticker in content.split(',')]
                # For file-based runs, we'll treat them all as a single group
                train_tickers, test_tickers = all_tickers, []
                logger.info(f"✅ Loaded {len(all_tickers)} tickers for processing.")
            except FileNotFoundError:
                logger.critical(f"❌ Ticker file not found at '{args.ticker_file}'. Exiting.")
                sys.exit(1)
        elif args.small_test:
            # train_tickers, test_tickers = ['AAPL', 'MSFT', 'GOOGL'], ['AMZN', 'NVDA']
            train_tickers, test_tickers = ['MSTR', 'WTF', 'RUM'], ['INDI', 'QHDG']
        else:
            train_file = os.path.join(LOG_DIR, "nasdaq_train_comprehensive.txt")
            test_file = os.path.join(LOG_DIR, "nasdaq_test_comprehensive.txt")
            train_tickers, test_tickers = generate_comprehensive_nasdaq_ticker_lists(train_file, test_file,
                                                                                     args.max_stocks)

        # --- Call the parallelized volatility function ---
        global_vol_thresholds = calculate_global_volatility_thresholds(
            random.sample(train_tickers, min(100, len(train_tickers))), datamanager
        )

        all_strategies = [
            {'mode': 'dynamic', 'target': 0},
            {'mode': 'fixed_net', 'target': 0.01},
            {'mode': 'fixed_net', 'target': 0.02},
            {'mode': 'fixed_net', 'target': 0.03},
            {'mode': 'fixed_net', 'target': 0.04},
        ]
        mode_map = {
            'dynamic': all_strategies[0], '1per': all_strategies[1], '2per': all_strategies[2],
            '3per': all_strategies[3], '4per': all_strategies[4]
        }

        strategies_to_run = []
        if args.profit_mode == 'all':
            strategies_to_run = all_strategies
        elif args.profit_mode in mode_map:
            strategies_to_run.append(mode_map[args.profit_mode])

        if not strategies_to_run:
            logger.error("No valid strategies selected to run. Exiting.")
            sys.exit(1)

        # Create all subdirectories first
        for strategy in strategies_to_run:
            mode, target = strategy['mode'], strategy['target']
            subdir = f"{int(target * 100)}per_profit" if mode == 'fixed_net' else "dynamic_profit"
            os.makedirs(os.path.join(TRAIN_FEATURES_DIR, subdir), exist_ok=True)
            os.makedirs(os.path.join(TEST_FEATURES_DIR, subdir), exist_ok=True)

        logger.info("\n" + "=" * 80 + "\n🚀 GENERATING DATA FOR ALL STRATEGIES\n" + "=" * 80)

        # Process the entire training and test sets in one go
        all_tickers_to_process = train_tickers + test_tickers

        skipped_tickers = process_ticker_list_producer_consumer(
            tickers=all_tickers_to_process,
            vol_thresholds= global_vol_thresholds,
            datamanager=datamanager,
            qqq_data=qqq_close,
            vix_data=vix_close,
            tlt_data=tlt_close,
            strategies=strategies_to_run,
            train_tickers=train_tickers,
            test_tickers=test_tickers
        )

        # --- Save the skipped tickers to a log file ---
        if skipped_tickers:
            # Save the detailed log file for debugging
            skipped_log_path = os.path.join(LOG_DIR, "skipped_stocks.log")
            with open(skipped_log_path, 'w') as f:
                for symbol, reason in skipped_tickers.items():
                    f.write(f"{symbol},{reason}\n")
            logger.info(f"📝 Detailed list of {len(skipped_tickers)} skipped stocks saved to '{skipped_log_path}'")

            # Save the simple, comma-separated file for easy re-running
            rerun_file_path = os.path.join(LOG_DIR, "rerun_skipped.txt")
            with open(rerun_file_path, 'w') as f:
                f.write(','.join(skipped_tickers.keys()))
            logger.info(f"✅ A clean list for re-running has been saved to '{rerun_file_path}'")
            logger.info(f"script log file has been saved to: '{log_file_path}'")

    except Exception as e:
        logger.critical(f"A critical error occurred in the main process: {e}\n{traceback.format_exc()}")
    finally:
        if 'datamanager' in locals() and datamanager.isConnected():
            datamanager.disconnect()
        logger.info("\n🎉 Data generation process finished.")


# def main():
#     """Main function to orchestrate the data processing pipeline for multiple strategies."""
#     parser = argparse.ArgumentParser(description='StockWise NASDAQ Data Processor')
#     parser.add_argument('--max-stocks', type=int, help='Maximum number of stocks to process')
#     parser.add_argument('--small-test', action='store_true', help='Run on a small, hardcoded test set.')
#     parser.add_argument('--profit-mode', type=str, default='all',
#                         choices=['all', 'dynamic', '1per', '2per', '3per', '4per'])
#     # Inside main() function, with the other parser.add_argument lines...
#     parser.add_argument('--ticker-file', type=str,
#                         help='Path to a text file containing a comma-separated list of tickers to process.')
#
#     parser.add_argument('--net-profit-target', type=float,
#                         help='Single net profit target for fixed_net mode (e.g., 0.03).')
#     args = parser.parse_args()
#
#     # --- File Logging Setup ---
#     log_file_path = os.path.join(LOG_DIR, f"data_pipeline_run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
#     file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
#     file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
#     logger.addHandler(file_handler)
#
#     datamanager = None
#     try:
#         datamanager = DataSourceManager()
#         logger.info("Attempting to connect to IBKR TWS...")
#         if not datamanager.connect_to_ibkr():
#             logger.warning("Could not connect to IBKR TWS. Proceeding with yfinance as the data source.")
#             datamanager.use_ibkr = False
#         else:
#             logger.info("✅ Successfully connected to IBKR TWS.")
#
#         qqq_close = load_qqq_data(datamanager)
#         vix_close = load_vix_data(datamanager)
#         tlt_close = load_tlt_data(datamanager)
#
#         # --- NEW: Logic to load tickers from file or generate them ---
#         if args.ticker_file:
#             logger.info(f"💾 Loading tickers from provided file: {args.ticker_file}")
#             try:
#                 with open(args.ticker_file, 'r') as f:
#                     content = f.read()
#                     # Split by comma and strip whitespace from each ticker
#                     all_tickers = [ticker.strip().replace("'", "") for ticker in content.split(',')]
#                 # For file-based runs, we'll treat them all as a single group
#                 train_tickers, test_tickers = all_tickers, []
#                 logger.info(f"✅ Loaded {len(all_tickers)} tickers for processing.")
#             except FileNotFoundError:
#                 logger.critical(f"❌ Ticker file not found at '{args.ticker_file}'. Exiting.")
#                 sys.exit(1)
#         elif args.small_test:
#             # train_tickers, test_tickers = ['AAPL', 'MSFT', 'GOOGL'], ['AMZN', 'NVDA']
#             train_tickers, test_tickers = ['MSTR', 'WTF', 'RUM'], ['INDI', 'QHDG']
#         else:
#             train_file = os.path.join(LOG_DIR, "nasdaq_train_comprehensive.txt")
#             test_file = os.path.join(LOG_DIR, "nasdaq_test_comprehensive.txt")
#             train_tickers, test_tickers = generate_comprehensive_nasdaq_ticker_lists(train_file, test_file,
#                                                                                      args.max_stocks)
#
#         # --- Call the parallelized volatility function ---
#         global_vol_thresholds = calculate_global_volatility_thresholds(
#             random.sample(train_tickers, min(100, len(train_tickers)))
#         )
#
#         all_strategies = [
#             {'mode': 'dynamic', 'target': 0},
#             {'mode': 'fixed_net', 'target': 0.01},
#             {'mode': 'fixed_net', 'target': 0.02},
#             {'mode': 'fixed_net', 'target': 0.03},
#             {'mode': 'fixed_net', 'target': 0.04},
#         ]
#         mode_map = {
#             'dynamic': all_strategies[0], '1per': all_strategies[1], '2per': all_strategies[2],
#             '3per': all_strategies[3], '4per': all_strategies[4]
#         }
#
#         strategies_to_run = []
#         if args.profit_mode == 'all':
#             strategies_to_run = all_strategies
#         elif args.profit_mode in mode_map:
#             strategies_to_run.append(mode_map[args.profit_mode])
#
#         if not strategies_to_run:
#             logger.error("No valid strategies selected to run. Exiting.")
#             sys.exit(1)
#
#         # Create all subdirectories first
#         for strategy in strategies_to_run:
#             mode, target = strategy['mode'], strategy['target']
#             subdir = f"{int(target * 100)}per_profit" if mode == 'fixed_net' else "dynamic_profit"
#             os.makedirs(os.path.join(TRAIN_FEATURES_DIR, subdir), exist_ok=True)
#             os.makedirs(os.path.join(TEST_FEATURES_DIR, subdir), exist_ok=True)
#
#         logger.info("\n" + "=" * 80 + "\n🚀 GENERATING DATA FOR ALL STRATEGIES\n" + "=" * 80)
#
#         # Process the entire training and test sets in one go
#         all_tickers_to_process = train_tickers + test_tickers
#         # --- Capture the skipped_tickers dictionary ---
#         skipped_tickers = process_ticker_list(all_tickers_to_process, global_vol_thresholds, datamanager, qqq_close,
#                                               vix_close, tlt_close, strategies_to_run, train_tickers, test_tickers)
#
#         # --- Save the skipped tickers to a log file ---
#         if skipped_tickers:
#             # Save the detailed log file for debugging
#             skipped_log_path = os.path.join(LOG_DIR, "skipped_stocks.log")
#             with open(skipped_log_path, 'w') as f:
#                 for symbol, reason in skipped_tickers.items():
#                     f.write(f"{symbol},{reason}\n")
#             logger.info(f"📝 Detailed list of {len(skipped_tickers)} skipped stocks saved to '{skipped_log_path}'")
#
#             # Save the simple, comma-separated file for easy re-running
#             rerun_file_path = os.path.join(LOG_DIR, "rerun_skipped.txt")
#             with open(rerun_file_path, 'w') as f:
#                 f.write(','.join(skipped_tickers.keys()))
#             logger.info(f"✅ A clean list for re-running has been saved to '{rerun_file_path}'")
#
#
#     except Exception as e:
#         logger.critical(f"A critical error occurred in the main process: {e}\n{traceback.format_exc()}")
#     finally:
#         if 'datamanager' in locals() and datamanager.isConnected():
#             datamanager.disconnect()
#         logger.info("\n🎉 Data generation process finished.")


if __name__ == "__main__":
    main()