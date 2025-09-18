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
import ta
import yfinance as yf
from tqdm import tqdm
from ta.momentum import StochasticOscillator, RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from ta.volatility import AverageTrueRange
from ta.trend import ADXIndicator
from datetime import timedelta, datetime


# --- Logging Setup ---
try:
    from data_source_manager import DataSourceManager
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', stream=sys.stdout)
    logger = logging.getLogger(__name__)
    logger.info("✅ Successfully imported DataSourceManager.")
except ImportError as e:
    print(f"❌ Critical Error: Could not import DataSourceManager. Please ensure 'data_source_manager.py' is in the correct directory. {e}")
    sys.exit(1)
# --- End Logging Setup ---

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
    time_limit_days: int
) -> pd.Series:
    """Implements the Triple Barrier Method for labeling financial time series data."""
    logger.info("Applying Triple Barrier Method for intelligent labeling...")
    outcomes = pd.Series(index=close_prices.index, dtype=np.int8, data=0)

    for i in tqdm(range(len(close_prices) - time_limit_days), desc="Labeling events", leave=False, ascii=True):
        entry_price = close_prices.iloc[i]
        current_atr = atr.iloc[i]
        if pd.isna(current_atr) or current_atr == 0:
            continue

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


def calculate_global_volatility_thresholds(tickers: list, data_source_manager_instance) -> tuple:
    """Performs a pre-calculation across all training tickers to find global volatility thresholds."""
    logger.info("🌀 Calculating global volatility thresholds across the training dataset...")
    all_volatilities = []
    sample_tickers = tickers
    logger.info(f"Analyzing volatility across all {len(sample_tickers)} training stocks...")

    for symbol in tqdm(sample_tickers, desc="Analyzing volatility", ascii=True):
        try:
            df = data_source_manager_instance.get_stock_data(symbol)
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


def add_technical_indicators_and_features(df: pd.DataFrame, vol_thresholds: tuple) -> pd.DataFrame:
    df = df.copy()
    df = normalize_dataframe_columns(df)  # Add this line here
    print("Names of df columns - after df.copy()", df.columns)

    # Check minimal length required to calculate any rolling features (20 here)
    # if df.empty or len(df) < 20:
    #     logger.warning(f"Insufficient data length ({len(df)} rows) for feature engineering.")
    #     return pd.DataFrame()

    # Calculate daily return early
    df['Daily_Return'] = df['Close'].pct_change()

    # Calculate Bollinger Bands (20-day window)
    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_Middle'] = bb.bollinger_mavg()
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()

    # Fill NaNs in these columns to prevent data loss
    df[['Daily_Return', 'BB_Middle', 'BB_Upper', 'BB_Lower']] = \
        df[['Daily_Return', 'BB_Middle', 'BB_Upper', 'BB_Lower']].bfill().ffill()

    # Calculate Z-Score (make sure to use consistent naming)
    rolling_std_20 = df['Close'].rolling(20).std()
    df['Z_Score_20'] = (df['Close'] - df['BB_Middle']) / rolling_std_20

    # Calculate Bollinger Band Width
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']

    # Calculate BB Position relative to bands
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

    # ATR (Average True Range)
    atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    df['ATR_14'] = atr.average_true_range()

    # Volume 20-day Moving Average
    df['Volume_MA_20'] = df['Volume'].rolling(20).mean()  # עדכן לשם הנכון

    # Momentum over 5 days
    df['Momentum_5'] = df['Close'] - df['Close'].shift(5)

    # ADX indicators
    adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    df['ADX'] = adx.adx()
    df['ADX_pos'] = adx.adx_pos()
    df['ADX_neg'] = adx.adx_neg()

    # On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    # RSI 14 and RSI 28
    rsi_14 = RSIIndicator(close=df['Close'], window=14)
    df['RSI_14'] = rsi_14.rsi()
    df['RSI_14_Smoothed'] = df['RSI_14'].ewm(span=5, adjust=False).mean()

    rsi_28 = RSIIndicator(close=df['Close'], window=28)
    df['RSI_28'] = rsi_28.rsi()

    # Stochastic Oscillator
    stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
    df['Stoch_%K'] = stoch.stoch()
    df['Stoch_%D'] = stoch.stoch_signal()

    # MACD and signals
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Histogram'] = macd.macd_diff()

    # KAMA (using your existing function)
    df['KAMA_10'] = calculate_kama(df['Close'], window=10)

    # Detrended Price Oscillator (DPO)
    dpo_period = 20
    dpo_sma = df['Close'].rolling(window=dpo_period).mean().shift(int(dpo_period / 2) + 1)
    df['DPO_20'] = df['Close'] - dpo_sma

    # Correlation with QQQ ETF (50-day window)
    try:
        qqq_df = yf.download("QQQ", start=df.index.min(), end=df.index.max(), progress=False, auto_adjust=True)
        qqq_close = qqq_df['Close'].reindex(df.index).ffill()
    except Exception as e:
        logger.warning(f"Could not download QQQ data: {e}, setting correlation to 0")
        qqq_close = pd.Series(0, index=df.index)

    df['Correlation_50D_QQQ'] = df['Close'].rolling(50).corr(qqq_close)

    # Smoothed Close price (EWMA)
    df['Smoothed_Close_5D'] = df['Close'].ewm(span=5, adjust=False).mean()

    # Volatility 20-day std dev
    df['Volatility_20D'] = df['Close'].pct_change().rolling(20).std()

    # Volatility Clustering based on thresholds
    low_thresh, high_thresh = vol_thresholds
    df['Volatility_90D'] = df['Close'].pct_change().rolling(90).std()
    df['Volatility_Cluster'] = 'mid'
    df.loc[df['Volatility_90D'] < low_thresh, 'Volatility_Cluster'] = 'low'
    df.loc[df['Volatility_90D'] > high_thresh, 'Volatility_Cluster'] = 'high'

    # Triple Barrier labeling (your function)
    tb_labels = apply_triple_barrier(df['Close'], df['High'], df['Low'], df['ATR_14'], 2.0, 2.5, 15)
    df['Target_Entry'] = (tb_labels == 1).astype(int)
    df['Target_Profit_Take'] = (tb_labels == 1).astype(int)
    df['Target_Cut_Loss'] = (tb_labels == -1).astype(int)

    # Future return based target labels (deprecated, keep for backward compatibility)
    df['Future_Return'] = df['Close'].pct_change(5).shift(-5)
    df['Target'] = np.select([df['Future_Return'] > 0.02, df['Future_Return'] < -0.02], [1, -1], default=0)

    # Dominant Cycle Period (use your function)
    df['Dominant_Cycle_126D'] = df['Close'].rolling(126).apply(lambda x: get_dominant_cycle(x),
                                                               raw=False)

    # Fill remaining NaNs extensively to avoid loss of data rows
    df = df.bfill().ffill()  #

    print("[DEBUG] Columns after feature calculations:")
    print(df.columns.tolist())
    print("[DEBUG] Sample rows after feature calculations:")
    print(df.head(3))

    # Ensure the expected columns are present
    expected_columns = [
        # Core data
        'Open', 'High', 'Low', 'Close', 'Volume', 'Datetime',
        # Gen-2 Features (retained)
        'Volume_MA_20', 'RSI_14', 'Momentum_5', 'MACD', 'MACD_Signal', 'MACD_Histogram',
        'BB_Position', 'Volatility_20D', 'ATR_14', 'ADX', 'ADX_pos', 'ADX_neg',
        'OBV', 'RSI_28', 'Dominant_Cycle_126D',
        # New Gen-3 Features
        'Z_Score_20', 'BB_Width', 'Correlation_50D_QQQ', 'Smoothed_Close_5D',
        'RSI_14_Smoothed', 'BB_Upper', 'BB_Lower', 'BB_Middle', 'Daily_Return',
        # Gen-3 and Gen-2 Targets/Labels
        'Volatility_Cluster',
        'Target_Entry', 'Target_Profit_Take', 'Target_Cut_Loss', 'Target'
    ]

    # Ensure the expected columns are present
    # df.columns = [c.capitalize() for c in df.columns]  # Capitalize to match the expected schema
    # Ensure the expected columns are present
    actual_columns = [col for col in expected_columns if col in df.columns]
    return df[actual_columns]


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


def process_ticker_list(tickers, output_dir, vol_thresholds, data_source_manager):
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

            # Check if data exists and is not empty
            if df is None or df.empty:
                skipped_count += 1
                skipped_tickers_log.append(f"{symbol}: No data retrieved or empty dataframe.")
                continue

            # Generate features using the feature engineering function
            featured_df = add_technical_indicators_and_features(df.copy(), vol_thresholds)

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


# def main():
#     """Main function to orchestrate the data processing pipeline."""
#     parser = argparse.ArgumentParser(description='StockWise NASDAQ Data Processor - Gen 3')
#     parser.add_argument('--max-stocks', type=int, help='Maximum number of stocks to process')
#     args = parser.parse_args()
#
#     logger.info("✨ Initializing DataSourceManager for the entire session...")
#     data_source_manager = DataSourceManager()
#     logger.info("✅ DataSourceManager initialized.")
#
#     train_file = os.path.join(LOG_DIR, "nasdaq_train_comprehensive.txt")
#     test_file = os.path.join(LOG_DIR, "nasdaq_test_comprehensive.txt")
#
#     # --- CHANGE THIS SECTION FOR A SMALL, QUICK TEST ---
#     # Overwrite the generated lists with a very small, fixed set of tickers
#     train_tickers = ['AAPL', 'MSFT', 'GOOGL']
#     test_tickers = ['AMZN', 'NVDA']
#     logger.info(f"Using a small test set: Training {train_tickers}, Testing {test_tickers}")
#     # --- END OF TEST CHANGE ---
#
#     train_tickers, test_tickers = generate_comprehensive_nasdaq_ticker_lists(train_file, test_file, args.max_stocks)
#
#     global_vol_thresholds = calculate_global_volatility_thresholds(train_tickers, data_source_manager)
#
#     if train_tickers:
#         logger.info(f"\n🚀 Processing {len(train_tickers)} training tickers...")
#         process_ticker_list(train_tickers, TRAIN_FEATURES_DIR, global_vol_thresholds, data_source_manager)
#
#     if test_tickers:
#         logger.info(f"\n🚀 Processing {len(test_tickers)} testing tickers...")
#         process_ticker_list(test_tickers, TEST_FEATURES_DIR, global_vol_thresholds, data_source_manager)
#
#     # No need to disconnect explicitly from yfinance in this version
#     logger.info("\n🎉 Gen-3 feature file generation pipeline finished.")

def main():
    """Main function to orchestrate the data processing pipeline."""
    parser = argparse.ArgumentParser(description='StockWise NASDAQ Data Processor - Gen 3')
    parser.add_argument('--max-stocks', type=int, help='Maximum number of stocks to process')
    # Add a new boolean argument for the small test mode
    parser.add_argument('--small-test', action='store_true', help='Run on a small, hardcoded test set.')
    args = parser.parse_args()

    logger.info("✨ Initializing DataSourceManager for the entire session...")
    data_source_manager = DataSourceManager()
    logger.info("✅ DataSourceManager initialized.")

    train_file = os.path.join(LOG_DIR, "nasdaq_train_comprehensive.txt")
    test_file = os.path.join(LOG_DIR, "nasdaq_test_comprehensive.txt")

    # --- NEW LOGIC: Use an if/else block to select the ticker list ---
    if args.small_test:
        train_tickers = ['AAPL', 'MSFT', 'GOOGL']
        test_tickers = ['AMZN', 'NVDA']
        logger.info(f"Using a small test set: Training {train_tickers}, Testing {test_tickers}")
    else:
        # This is the original logic for running a full or limited run
        train_tickers, test_tickers = generate_comprehensive_nasdaq_ticker_lists(train_file, test_file, args.max_stocks)
    # --- END OF NEW LOGIC ---

    global_vol_thresholds = calculate_global_volatility_thresholds(train_tickers, data_source_manager)

    if train_tickers:
        logger.info(f"\n🚀 Processing {len(train_tickers)} training tickers...")
        process_ticker_list(train_tickers, TRAIN_FEATURES_DIR, global_vol_thresholds, data_source_manager)

    if test_tickers:
        logger.info(f"\n🚀 Processing {len(test_tickers)} testing tickers...")
        process_ticker_list(test_tickers, TEST_FEATURES_DIR, global_vol_thresholds, data_source_manager)

    logger.info("\n🎉 Gen-3 feature file generation pipeline finished.")


if __name__ == "__main__":
    main()