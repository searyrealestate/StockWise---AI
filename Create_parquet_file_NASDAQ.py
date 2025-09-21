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
import ibkr_connection_manager


try:
    from ibkr_connection_manager import ProfessionalIBKRManager
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', stream=sys.stdout)
    logger = logging.getLogger(__name__)
    logger.info("✅ Successfully imported DataSourceManager.")
except ImportError as e:
    print(f"❌ Critical Error: Could not import DataSourceManager. Please ensure 'data_source_manager.py' is in the correct directory. {e}")
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
    csv_file_path = os.path.join(BASE_DIR, "nasdaq_full_list.csv")
    all_nasdaq_symbols = get_symbols_from_csv(csv_file_path)

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
            df = ibkr_manager.get_stock_data(symbol, days_back=5*365)
            if df is not None and not df.empty and len(df) > 90:
                # Add this line to normalize the DataFrame columns
                df = normalize_dataframe_columns(df)

                # Check again after normalization, in case data was unusable
                if df is not None and not df.empty and 'Close' in df.columns:
                    volatility_90d = df['Close'].pct_change().rolling(window=90).std()
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


# Manually implement KAMA with a corrected and more robust method
def calculate_kama(close, window=10):
    """Calculates Kaufman's Adaptive Moving Average (KAMA) manually."""
    close_delta = abs(close.diff())
    er_num = abs(close.diff(window))
    er_den = abs(close_delta).rolling(window=window).sum()
    er = er_num / (er_den + 1e-10)
    er.fillna(0, inplace=True)
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    kama = close.copy()

    for i in range(window, len(close)):
        sc_ = (er.iloc[i] * (fast_sc - slow_sc) + slow_sc) if er.iloc[i] > 0 else slow_sc
        kama.iloc[i] = kama.iloc[i - 1] + sc_ * (close.iloc[i] - kama.iloc[i - 1])

    return kama


# Manually implement Stochastic Oscillator
def calculate_stochastic(high, low, close, window=14, smooth_window=3):
    """Calculates the Stochastic Oscillator (%K and %D) manually."""
    lowest_low = low.rolling(window=window).min()
    highest_high = high.rolling(window=window).max()
    percent_k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    percent_d = percent_k.rolling(window=smooth_window).mean()
    return percent_k, percent_d


def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes a multi-indexed DataFrame with a single ticker column
    to a simple single-level index.
    """
    if isinstance(df.columns, pd.MultiIndex):
        if df.columns.nlevels == 2 and 'Ticker' in df.columns.names:
            # Assuming a structure like [('Close', 'AAPL'), ('Volume', 'AAPL')]
            # Drop the Ticker level from the columns
            df.columns = df.columns.droplevel('Ticker')
            return df
    return df


# In Create_parquet_file_NASDAQ.py

def add_technical_indicators_and_features(
        df: pd.DataFrame,
        vol_thresholds: tuple,
        qqq_close: pd.Series,
        profit_mode: str,
        net_profit_target: float,
        debug: bool = False
) -> pd.DataFrame:
    df = df.copy()

    if df.empty or len(df) < 90:
        return pd.DataFrame()

    if debug:
        print("\n--- STARTING DEBUG FOR add_technical_indicators_and_features ---")
        print(f"Initial columns: {df.columns.tolist()}")

    df.ta.bbands(length=20, append=True, col_names=("BB_Lower", "BB_Middle", "BB_Upper", "BB_Width", "BB_Position"))
    df.ta.atr(length=14, append=True, col_names="ATR_14")
    df.ta.rsi(length=14, append=True, col_names="RSI_14")
    df.ta.rsi(length=28, append=True, col_names="RSI_28")
    df.ta.macd(append=True, col_names=("MACD", "MACD_Histogram", "MACD_Signal"))

    # --- THIS IS THE FIX ---
    # MODIFIED: Provide 4 names to satisfy the library's expectation for ADX.
    df.ta.adx(length=14, append=True, col_names=("ADX", "ADX_pos", "ADX_neg", "ADXR_temp"))
    # MODIFIED: Immediately drop the temporary 4th column which we don't need.
    df.drop(columns=["ADXR_temp"], inplace=True)
    # --- END FIX ---

    df.ta.mom(length=5, append=True, col_names="Momentum_5")
    df.ta.obv(append=True)

    if debug:
        print(f"Columns after ALL indicators: {df.columns.tolist()}")
        print(f"--- FINISHED DEBUG ---")

    df['Daily_Return'] = df['Close'].pct_change()
    df['Volume_MA_20'] = df['Volume'].rolling(20).mean()
    df['Volatility_20D'] = df['Daily_Return'].rolling(20).std()
    df['Z_Score_20'] = (df['Close'] - df['BB_Middle']) / df['Close'].rolling(20).std()

    if not qqq_close.empty:
        aligned_qqq = qqq_close.reindex(df.index, method='ffill')
        df['Correlation_50D_QQQ'] = df['Close'].rolling(50).corr(aligned_qqq)
    else:
        df['Correlation_50D_QQQ'] = 0.0

    low_thresh, high_thresh = vol_thresholds
    df['Volatility_90D'] = df['Daily_Return'].rolling(90).std()
    df['Volatility_Cluster'] = pd.cut(df['Volatility_90D'], bins=[-np.inf, low_thresh, high_thresh, np.inf],
                                      labels=['low', 'mid', 'high'])

    tb_labels = apply_triple_barrier(
        close_prices=df['Close'], high_prices=df['High'], low_prices=df['Low'], atr=df['ATR_14'],
        profit_take_mult=2.0, stop_loss_mult=2.5, time_limit_days=15,
        profit_mode=profit_mode, net_profit_target=net_profit_target
    )
    # --- FIX: Implement correct, distinct labeling ---
    # Target_Entry is 1 if the trade is expected to be profitable (hits upper barrier first)
    df['target_entry'] = (tb_labels == 1).astype(int)

    # Target_Profit_Take should ALSO be 1 if it hits the upper barrier.
    # In this simple model, the entry and take-profit signals are the same.
    # A more advanced version would detect reversals, but this is a correct starting point.
    df['target_profit_take'] = (tb_labels == 1).astype(int)

    # Target_Cut_Loss is 1 if the trade hits the lower barrier first.
    # This is mutually exclusive with the other two.
    df['target_cut_loss'] = (tb_labels == -1).astype(int)

    df = df.bfill().ffill()
    df.dropna(inplace=True)

    # --- Standardize all columns to lowercase before saving ---
    df.columns = [col.lower() for col in df.columns]

    expected_columns = [
        'open', 'high', 'low', 'close', 'volume', 'volume_ma_20', 'rsi_14', 'momentum_5', 'macd', 'macd_signal',
        'macd_histogram',
        'bb_position', 'volatility_20d', 'atr_14', 'adx', 'adx_pos', 'adx_neg', 'obv', 'rsi_28',
        'z_score_20', 'bb_width', 'correlation_50d_qqq', 'bb_upper', 'bb_lower', 'bb_middle', 'daily_return',
        'volatility_cluster', 'target_entry', 'target_profit_take', 'target_cut_loss'
    ]
    existing_cols = [col for col in expected_columns if col in df.columns]
    return df[existing_cols]


def extend_date_range_for_features(start_date: pd.Timestamp, lookback_days: int) -> pd.Timestamp:
    """
    Given a requested start date and a lookback window (e.g., 39 days),
    returns the extended start date to fetch data earlier.
    """
    return start_date - timedelta(days=lookback_days)


def get_enhanced_stock_data(symbol: str, user_start_date: pd.Timestamp, user_end_date: pd.Timestamp, lookback_days: int = 39):
    """
    Fetch stock data extended backward for feature calculation, compute features,
    then trim to user requested date range.
    """
    extended_start_date = extend_date_range_for_features(user_start_date, lookback_days)

    # Example: using yfinance for demo, replace with your data source as needed
    df = yf.download(symbol, start=extended_start_date.strftime('%Y-%m-%d'), end=user_end_date.strftime('%Y-%m-%d'),
                     progress=False, auto_adjust=True)

    if df.empty or len(df) < lookback_days:
        return pd.DataFrame()  # or handle insufficient data

    # Compute your features on df including lookback period
    # Assuming you have a function like add_technical_indicators_and_features
    vol_thresholds = (0.015, 0.03)   # Example, calculate or pass your thresholds

    featured_df = add_technical_indicators_and_features(df, vol_thresholds)

    # Trim back to user requested date range (drop lookback rows before user_start_date)
    featured_df = featured_df.loc[featured_df.index >= user_start_date]

    return featured_df


def process_ticker_list(tickers, output_dir, vol_thresholds, data_source_manager, qqq_data, profit_mode, net_profit_target):
    """
    Process a list of tickers by extracting features and saving them as parquet files.

    Parameters:
    - tickers: List of stock symbols to process.
    - output_dir: Directory path where processed files will be saved.
    - vol_thresholds: Tuple with volatility thresholds for feature calculations.
    - data_source_manager: An instance providing stock data fetching capabilities.
    """
    processed_count = 0
    skipped_count = 0
    skipped_tickers_log = []

    for symbol in tqdm(tickers, desc="Processing stocks"):
        try:
            # Fetch raw stock data
            df = data_source_manager.get_stock_data(symbol)

            # Normalize the DataFrame columns immediately after fetching
            df = normalize_dataframe_columns(df)

            # Check if data exists and is not empty
            if df is None or df.empty:
                skipped_count += 1
                skipped_tickers_log.append(f"{symbol}: No data retrieved or empty dataframe.")
                continue

            # Generate features using the feature engineering function
            featured_df = add_technical_indicators_and_features(
                df.copy(), vol_thresholds, qqq_data, profit_mode, net_profit_target
            )

            # Skip if feature extraction resulted in empty dataframe
            if featured_df.empty:
                skipped_count += 1
                skipped_tickers_log.append(f"{symbol}: Feature dataframe empty after processing.")
                continue

            # Save the feature dataframe to parquet file
            output_path = os.path.join(output_dir, f"{symbol}_features.parquet")
            featured_df.to_parquet(output_path)

            processed_count += 1

        except Exception as e:
            skipped_count += 1
            skipped_tickers_log.append(f"{symbol}: Exception occurred: {e}")
            logger.error(f"Failed to process {symbol}: {e}")

    # Logging summary of processing
    logger.info(f"Processed {processed_count} stocks successfully; skipped {skipped_count}.")

    # Save detailed log of skipped tickers if any
    if skipped_tickers_log:
        log_filename = f"skipped_tickers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_filepath = os.path.join("logs", log_filename)
        with open(log_filepath, "w", encoding="utf-8") as log_file:
            log_file.write("\n".join(skipped_tickers_log))
        logger.info(f"Detailed skipped tickers log saved at: {log_filepath}")


def main():
    """Main function to orchestrate the data processing pipeline for multiple strategies."""
    parser = argparse.ArgumentParser(description='StockWise NASDAQ Data Processor - Gen 3')
    parser.add_argument('--max-stocks', type=int, help='Maximum number of stocks to process')
    parser.add_argument('--small-test', action='store_true', help='Run on a small, hardcoded test set.')
    # NEW: Add arguments to control the strategy from the command line
    parser.add_argument('--profit-mode', type=str, default='all', choices=['dynamic', 'fixed_net', 'all'],
                        help='Profit target mode to generate data for.')
    parser.add_argument('--net-profit-target', type=float,
                        help='Single net profit target for fixed_net mode (e.g., 0.03).')
    args = parser.parse_args()

    # In main()
    logger.info("🔌 Initializing Professional IBKR Manager for data collection...")
    ibkr_manager = ProfessionalIBKRManager(debug=False)  # Set debug=True for more verbose output
    if not ibkr_manager.connect_with_fallback():
        logger.critical("❌ Failed to connect to IBKR. Please ensure TWS or Gateway is running.")
        sys.exit(1)  # Exit if we can't connect

    # NEW: Fetch QQQ Data ONCE at the start
    logger.info("📅 Fetching QQQ data for correlation calculations...")
    try:
        qqq_df = yf.download("QQQ", period="max", progress=False, auto_adjust=True)
        qqq_close = qqq_df['Close']
    except Exception as e:
        logger.error(f"❌ Could not download QQQ data. Correlation feature will be disabled. Error: {e}")
        qqq_close = pd.Series()

    # Get ticker lists
    if args.small_test:
        train_tickers, test_tickers = ['AAPL', 'MSFT', 'GOOGL'], ['AMZN', 'NVDA']
    else:
        train_file = os.path.join(LOG_DIR, "nasdaq_train_comprehensive.txt")
        test_file = os.path.join(LOG_DIR, "nasdaq_test_comprehensive.txt")
        train_tickers, test_tickers = generate_comprehensive_nasdaq_ticker_lists(train_file, test_file, args.max_stocks)

    global_vol_thresholds = calculate_global_volatility_thresholds(train_tickers, ibkr_manager)

    # NEW: Define strategies to run
    strategies = []
    if args.profit_mode in ['dynamic', 'all']:
        strategies.append({'mode': 'dynamic', 'target': 0})  # Target is unused in dynamic mode
    if args.profit_mode in ['fixed_net', 'all']:
        if args.net_profit_target:  # If a single target is specified
            strategies.append({'mode': 'fixed_net', 'target': args.net_profit_target})
        else:  # Default to the 2, 3, 4 percent set
            for target in [0.02, 0.03, 0.04]:
                strategies.append({'mode': 'fixed_net', 'target': target})

    # NEW: Loop through the defined strategies
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

        if train_tickers:
            process_ticker_list(train_tickers, train_output_dir, global_vol_thresholds, ibkr_manager, qqq_close,
                                mode, target)
        if test_tickers:
            process_ticker_list(test_tickers, test_output_dir, global_vol_thresholds, ibkr_manager, qqq_close,
                                mode, target)

    logger.info("\n🎉 Disconnecting from IBKR.")
    ibkr_manager.disconnect()
    logger.info("🎉 All data generation pipelines finished successfully.")


if __name__ == "__main__":
    main()