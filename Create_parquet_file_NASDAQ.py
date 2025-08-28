"""
📊 StockWise NASDAQ Pipeline - COMPREHENSIVE VERSION
───────────────────────────────────────────────────

This script handles the comprehensive fetching, processing, and saving of NASDAQ
stock data into Parquet files, suitable for model training and evaluation.
It now integrates with DataSourceManager for robust data retrieval (IBKR with yfinance fallback).

Usage:
    python Create_parquet_file_NASDAQ.py                    # Process all available stocks (default: 1000 max)
    python Create_parquet_file_NASDAQ.py --debug            # Enable debug mode with detailed output
    python Create_parquet_file_NASDAQ.py --stock AAPL       # Process only a specific stock (for debugging)
    python Create_parquet_file_NASDAQ.py --max-stocks 500   # Limit the number of stocks to process
    python Create_parquet_file_NASDAQ.py --quick            # Quick mode - use a smaller, predefined list of quality stocks
    python Create_parquet_file_NASDAQ.py --use-ibkr         # Attempt to use IBKR for data retrieval (requires TWS/Gateway)
    python Create_parquet_file_NASDAQ.py --ibkr-host 127.0.0.1 --ibkr-port 7497 # Specify IBKR host/port
"""

import os
import glob
import gc
import joblib
import pandas as pd
import yfinance as yf
from xgboost import XGBClassifier
from datetime import datetime, timedelta
import requests
import ta
from io import StringIO
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np
import argparse
import time
import logging
import sys
import traceback
from tqdm import tqdm
import random


# --- Logging Setup ---
try:
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
    if sys.stderr.encoding != 'utf-8':
        sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)
except Exception as e:
    print(f"Warning: Could not reconfigure console encoding to UTF-8. Emojis may appear as '?' or cause errors: {e}",
          file=sys.__stderr__)

# Define directories early for logging setup
BASE_DIR = os.getcwd()
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True) # Ensure the logs directory exists

main_log_file_path = os.path.join(LOG_DIR, f"create_parquet_main_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO, # Default to INFO, will be changed by argparse --debug
    format='%(asctime)s | %(levelname)s | %(message)s', # Standard format
    handlers=[
        logging.StreamHandler(sys.stdout), # Output to console
        logging.FileHandler(main_log_file_path, encoding='utf-8') # Output to a file
    ]
)
logger = logging.getLogger(__name__) # Get a logger for this module

logging.getLogger('yfinance').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)
# --- End Logging Setup ---







# Define directories
TRAIN_DIR = os.path.join(BASE_DIR, "models/NASDAQ-training set")
TEST_DIR = os.path.join(BASE_DIR, "models/NASDAQ-testing set")
TRAIN_FEATURES_DIR = os.path.join(TRAIN_DIR, "features")
TRAIN_MODELS_DIR = os.path.join(TRAIN_DIR, "models")
TEST_FEATURES_DIR = os.path.join(TEST_DIR, "features")
TEST_MODELS_DIR = os.path.join(TEST_DIR, "models")


# Create directories if they don't exist
os.makedirs(TRAIN_FEATURES_DIR, exist_ok=True)
os.makedirs(TRAIN_MODELS_DIR, exist_ok=True)
os.makedirs(TEST_FEATURES_DIR, exist_ok=True)
os.makedirs(TEST_MODELS_DIR, exist_ok=True)


# --- Import DataSourceManager ---
try:
    from data_source_manager import DataSourceManager
    logger.info("✅ Successfully imported DataSourceManager.")
except ImportError as e:
    logger.error(f"❌ Error importing DataSourceManager: {e}")
    logger.error("Please ensure 'data_source_manager.py' is in the same directory "
                 "or in your Python path and check for any syntax errors.")
    sys.exit(1)
# --- End Import ---


def requests_retry_session(
    retries=3,
    backoff_factor=0.3,
    status_forcelist=(500, 502, 504),
    session=None,
):
    """Helper for robust web requests with retries."""
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def get_symbols_from_csv(file_path):
    """
    Loads NASDAQ symbols from a local CSV file.
    Assumes the CSV has a 'Symbol' column.
    """
    if not os.path.exists(file_path):
        logger.error(f"❌ CSV file not found at: {file_path}")
        return []
    try:
        df = pd.read_csv(file_path)
        if 'Symbol' not in df.columns:
            logger.error(f"❌ CSV file '{file_path}' does not contain a 'Symbol' column.")
            return []

        # ⭐ CRITICAL FIX: Robustly convert 'Symbol' column to string, handle non-alphanumeric/problematic patterns
        df['Symbol'] = df['Symbol'].fillna('').astype(str).str.strip()

        # Filter out common non-stock symbols or problematic patterns for yfinance
        # These patterns often indicate warrants (W), rights (R), units (U), preferred shares (P), etc.
        # or symbols with special characters/numbers that YF struggles with for historical data.
        problematic_patterns = [
            r'[RUWXPZ]$', # Ends with R, U, W, X, P, Z (common for warrants, rights, units, special classes)
            r'\.',        # Contains a dot (e.g., BRK.B, may be handled by YF but can cause issues for others)
            r'^[0-9]',    # Starts with a number
            r'/',         # Contains a slash
            r'-',         # Contains a hyphen
            r'^\$',       # Starts with $
            r'\s'         # Contains whitespace
        ]

        # Create a regex pattern to match any of the problematic patterns
        combined_pattern = '|'.join(problematic_patterns)

        # Filter out symbols matching any problematic pattern AND ensure it's alphanumeric and not too long
        # Keep only symbols that are composed purely of alphabetic characters
        nasdaq_symbols = df[
            (df['Symbol'].str.len() > 0) &  # Not empty
            (~df['Symbol'].str.contains(combined_pattern, regex=True)) &  # Does not contain problematic patterns
            (df['Symbol'].str.isalpha()) &  # Strictly alphabetic characters (e.g., no numbers, no special chars)
            (df['Symbol'].str.len() <= 5)  # Keep length reasonable for common tickers
            ]['Symbol'].tolist()

        logger.info(f"✅ Loaded {len(nasdaq_symbols)} symbols from CSV: {file_path} after robust filtering.")
        return nasdaq_symbols
    except Exception as e:
        logger.error(f"❌ Error loading or parsing CSV file '{file_path}': {e}. Full traceback: {traceback.format_exc()}")
        return []


# Removed get_nasdaq_ftp_symbols and get_finviz_symbols as they are no longer needed
# if using the CSV as the primary source. Keeping them commented out for reference.
# def get_nasdaq_ftp_symbols(): ...
# def get_finviz_symbols(num_pages=10): ...

def get_yfinance_nasdaq_symbols_fallback(min_market_cap=150_000_000, max_stocks_to_check=500):
    """Fallback to Yahoo Finance for large cap NASDAQ stocks (slower)."""
    logger.info("Falling back to Yahoo Finance to get active NASDAQ symbols (may take time)...")
    try:
        common_nasdaq_symbols = ["AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "NVDA", "TSLA", "META", "NFLX", "ADBE"]

        extended_list_path = os.path.join(BASE_DIR, "nasdaq_quality_stocks.txt")
        if os.path.exists(extended_list_path):
            with open(extended_list_path, 'r') as f:
                extended_symbols = [line.strip() for line in f if line.strip()]
            common_nasdaq_symbols.extend(extended_symbols)
            common_nasdaq_symbols = list(set(common_nasdaq_symbols))
            logger.info(f"Loaded {len(extended_symbols)} symbols from nasdaq_quality_stocks.txt.")
        else:
            logger.warning("nasdaq_quality_stocks.txt not found. Using a small default list for yfinance fallback.")


        valid_symbols = []

        random.shuffle(common_nasdaq_symbols)

        for i, symbol in enumerate(common_nasdaq_symbols):
            if i >= max_stocks_to_check:
                logger.info(f"Reached max_stocks_to_check ({max_stocks_to_check}) for Yahoo Finance fallback.")
                break
            try:
                # Ensure symbol is a string before passing to yfinance
                if not isinstance(symbol, str) or not symbol.isalnum(): # .isalnum() checks for alphanumeric (letters and numbers)
                    logger.debug(f"Skipping invalid symbol in fallback: {symbol}")
                    continue

                ticker = yf.Ticker(symbol)
                info = ticker.info
                market_cap = info.get('marketCap')
                if market_cap and market_cap >= min_market_cap:
                    valid_symbols.append(symbol)
                    if len(valid_symbols) % 50 == 0:
                        logger.debug(f"Found {len(valid_symbols)} large-cap symbols so far.")
                time.sleep(0.1)
            except Exception:
                pass
        logger.info(f"Yahoo Finance Fallback: Found {len(valid_symbols)} large-cap NASDAQ symbols.")
        return valid_symbols
    except Exception as e:
        logger.error(f"Error in Yahoo Finance fallback: {e}")
        return []


def generate_comprehensive_nasdaq_ticker_lists(train_file, test_file):
    """
    Generates comprehensive lists of NASDAQ tickers for training and testing.
    Now loads symbols directly from nasdaq_full_list.csv.
    """
    logger.info("🚀 COMPREHENSIVE MODE: Loading NASDAQ symbols from CSV.")
    logger.info("=" * 70)

    # ⭐ CRITICAL CHANGE: Load symbols directly from the CSV file
    # Assuming nasdaq_full_list.csv is in the BASE_DIR (project root)
    csv_file_path = os.path.join(BASE_DIR, "nasdaq_full_list.csv")
    all_nasdaq_symbols = get_symbols_from_csv(csv_file_path)

    if not all_nasdaq_symbols:
        logger.critical("❌ No NASDAQ symbols loaded from CSV. Cannot proceed. Exiting.")
        sys.exit(1) # Exit if no symbols are loaded from the primary source

    # Apply max_stocks limit if specified
    if max_stocks and len(all_nasdaq_symbols) > max_stocks:
        all_nasdaq_symbols = all_nasdaq_symbols[:max_stocks]
        logger.info(f"📊 Limiting to {max_stocks} stocks (from {len(all_nasdaq_symbols)} available)")

    logger.info("=" * 70)
    logger.info(f"🎯 TOTAL NASDAQ SYMBOLS FOUND: {len(all_nasdaq_symbols)}")
    logger.info("=" * 70)

    logger.debug(f"Final list of symbols before splitting: {all_nasdaq_symbols[:10]}... ({len(all_nasdaq_symbols)} total)")

    split_point = int(len(all_nasdaq_symbols) * 0.7)
    train_tickers = all_nasdaq_symbols[:split_point]
    test_tickers = all_nasdaq_symbols[split_point:]

    if len(test_tickers) < 5 and len(all_nasdaq_symbols) >= 10:
        train_tickers = all_nasdaq_symbols[:-5]
        test_tickers = all_nasdaq_symbols[-5:]
    elif len(all_nasdaq_symbols) < 10:
        train_tickers = all_nasdaq_symbols
        test_tickers = []

    logger.debug(f"Train tickers count: {len(train_tickers)}, Test tickers count: {len(test_tickers)}")

    with open(train_file, 'w') as f:
        for ticker in train_tickers:
            f.write(f"{ticker}\n")
    with open(test_file, 'w') as f:
        for ticker in test_tickers:
            f.write(f"{ticker}\n")

    logger.info(f"✅ Generated training list ({len(train_tickers)} stocks) and testing list ({len(test_tickers)} stocks).")
    return train_tickers, test_tickers


def get_dominant_cycle(data: pd.Series, min_period=3, max_period=100) -> float:
    """
    Uses FFT to find the dominant cycle period in a time series on a rolling window.
    """
    if data.isnull().all() or len(data) < min_period:
        return 0.0

    # Detrend the data to focus on cycles rather than the overall trend
    detrended = data - np.poly1d(np.polyfit(np.arange(len(data)), data, 1))(np.arange(len(data)))

    # Apply the Fast Fourier Transform
    fft_result = np.fft.fft(detrended.values)
    frequencies = np.fft.fftfreq(len(detrended))

    # Find the power (magnitude squared) of each frequency
    power = np.abs(fft_result) ** 2

    # Focus only on positive frequencies (the second half is a mirror image)
    positive_freq_mask = frequencies > 0
    periods = 1 / frequencies[positive_freq_mask]

    # Filter for periods within a reasonable range for trading (e.g., 3 to 100 days)
    period_mask = (periods >= min_period) & (periods <= max_period)

    if not np.any(period_mask):
        return 0.0  # No cycles found in the desired range

    # Find the period with the highest power
    dominant_idx = power[positive_freq_mask][period_mask].argmax()
    dominant_period = periods[period_mask][dominant_idx]

    return dominant_period


# Replace your existing function with this upgraded version
def add_technical_indicators_and_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a comprehensive and expanded set of technical indicators and features.
    """
    df = df.copy()

    if df.empty or len(df) < 50:  # Increased minimum length for new indicators
        logger.warning("Input DataFrame is empty or too short (< 50 rows), cannot add all indicators.")
        return pd.DataFrame()

    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col not in df.columns:
            logger.error(f"Missing required column for indicators: {col}")
            return pd.DataFrame()
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=['Close', 'Volume'], inplace=True)

    # --- Standard Features ---
    df['Volume_MA_20'] = ta.trend.sma_indicator(df['Volume'], window=20)
    df['RSI_14'] = ta.momentum.rsi(df['Close'], window=14)
    df['Momentum_5'] = df['Close'].diff(5)
    macd = ta.trend.MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Histogram'] = macd.macd_diff()
    bb = ta.volatility.BollingerBands(close=df["Close"], window=20, window_dev=2)
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()
    df['BB_Middle'] = bb.bollinger_mavg()
    df['BB_Position'] = bb.bollinger_pband()  # This calculates %B
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility_20D'] = df['Daily_Return'].rolling(window=20).std()

    # --- NEW Advanced Features ---
    # Volatility Indicator
    df['ATR_14'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'],
                                                  window=14).average_true_range()

    # Trend Strength Indicator
    adx_indicator = ta.trend.ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    df['ADX'] = adx_indicator.adx()
    df['ADX_pos'] = adx_indicator.adx_pos()
    df['ADX_neg'] = adx_indicator.adx_neg()

    # Volume-Based Indicator
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()

    # Multiple Time Windows
    df['RSI_28'] = ta.momentum.rsi(df['Close'], window=28)  # Longer-term RSI

    # --- NEW FFT Feature ---
    # Calculate dominant cycle on a rolling 126-day window (approx 6 months)
    df['Dominant_Cycle_126D'] = df['Close'].rolling(window=126, min_periods=50).apply(get_dominant_cycle, raw=False)

    # --- Target Definition ---
    future_window = 5
    gain_threshold = 0.01
    df['Future_Close'] = df['Close'].shift(-future_window)
    df['Target'] = ((df['Future_Close'] / df['Close']) - 1 > gain_threshold).astype(int)

    # 1. Create a smoothed version of the Close price
    df['Smoothed_Close_5D'] = df['Close'].rolling(window=5).mean()

    # 2. Calculate a new RSI based on the smoothed price instead of the raw price
    df['RSI_14_Smoothed'] = ta.momentum.rsi(df['Smoothed_Close_5D'], window=14)

    initial_rows = len(df)
    df.dropna(inplace=True)
    final_rows = len(df)
    logger.debug(f"Dropped {initial_rows - final_rows} rows due to NaN values after indicator calculation.")

    # --- IMPORTANT: Update the final list of columns ---
    expected_features_for_model = [
        'Volume_MA_20', 'RSI_14', 'Momentum_5', 'MACD', 'MACD_Signal',
        'MACD_Histogram', 'BB_Upper', 'BB_Lower', 'BB_Middle',
        'BB_Position', 'Daily_Return', 'Volatility_20D',
        # Add the new features to the list
        'ATR_14', 'ADX', 'ADX_pos', 'ADX_neg', 'OBV', 'RSI_28', 'Dominant_Cycle_126D', "Smoothed_Close_5D",
        "RSI_14_Smoothed"
    ]

    missing_features = [f for f in expected_features_for_model if f not in df.columns]
    if missing_features:
        logger.error(f"❌ After indicator calculation, missing expected features: {missing_features}")
        return pd.DataFrame()

    final_cols = expected_features_for_model + ['Target', 'Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
    df = df[final_cols]

    logger.debug(f"DataFrame after feature engineering. Shape: {df.shape}")
    return df


def train_model(df, symbol, model_output_dir):
    """
    Trains an ML model for a given symbol and saves it.
    This function saves a dummy model for the pipeline to continue.
    """
    if df.empty:
        logger.warning(f"No data to train model for {symbol}. Skipping model training.")
        return None, None

    os.makedirs(model_output_dir, exist_ok=True)
    model_path = os.path.join(model_output_dir, f"{symbol}_model.pkl")

    try:
        from sklearn.dummy import DummyClassifier
        dummy_model = DummyClassifier(strategy="most_frequent")

        features = [
            'Volume_MA_20', 'RSI_14', 'Momentum_5', 'MACD', 'MACD_Signal',
            'MACD_Histogram', 'BB_Upper', 'BB_Lower', 'BB_Middle',
            'BB_Position', 'Daily_Return', 'Volatility_20D'
        ]

        actual_features = [f for f in features if f in df.columns]
        if len(actual_features) != len(features):
            logger.error(f"Missing expected features for training {symbol}: {list(set(features) - set(actual_features))}. Cannot train model.")
            return None, None

        X = df[actual_features]
        y = df['Target']

        if y.nunique() < 2:
            logger.warning(f"Target column for {symbol} has only {y.nunique()} unique classes. Skipping model training as it requires at least 2 classes for classification.")
            return None, None

        if not X.empty and not y.empty:
            dummy_model.fit(X, y)
            joblib.dump(dummy_model, model_path)
            logger.debug(f"✅ Saved dummy model for {symbol} to {model_path}")
            return dummy_model, df
        else:
            logger.warning(f"No sufficient data to train model for {symbol} after feature selection. Skipping training.")
            return None, None

    except ImportError:
        logger.warning("Scikit-learn (or DummyClassifier) not found. Skipping model training.")
        return None, None
    except Exception as e:
        logger.error(f"Error during dummy model training for {symbol}: {e}")
        return None, None


def process_ticker_list(tickers, output_base_dir, train=True, data_source_manager_instance=None, debug_stock=None):
    # ...:
    """Processes a list of tickers to download data, add features, and save."""
    processed_count = 0
    skipped_tickers_log = []

    if not data_source_manager_instance:
        logger.critical("Critical Error: DataSourceManager instance not provided to process_ticker_list. Cannot proceed.")
        sys.exit(1)

    output_features_dir = os.path.join(output_base_dir, "features")
    output_models_dir = os.path.join(output_base_dir, "models")
    os.makedirs(output_features_dir, exist_ok=True)
    os.makedirs(output_models_dir, exist_ok=True)


    if debug_stock:
        original_tickers = list(tickers)
        tickers = [t for t in tickers if t == debug_stock]
        if not tickers:
            logger.critical(f"❌ Debug stock '{debug_stock}' not found in the generated comprehensive ticker list. Exiting.")
            sys.exit(1)

    logger.debug(f"Output features directory: {output_features_dir}")
    logger.debug(f"Output models directory: {output_models_dir}")
    logger.debug(f"Training mode: {train}")

    with tqdm(total=len(tickers), desc="Processing stocks") as pbar:
        for i, symbol in enumerate(tickers):
            logger.debug(f"=== PROCESSING {symbol} ({i + 1}/{len(tickers)}) ===")

            try:
                logger.debug(f"Downloading data for {symbol} using DataSourceManager...")

                df = data_source_manager_instance.get_stock_data(symbol)

                if df is None or df.empty or 'Close' not in df.columns:
                    error_msg = f"❌ Critical: Failed to retrieve sufficient data for {symbol} from any source after all attempts."
                    logger.critical(error_msg)
                    skipped_tickers_log.append(f"{symbol}: Data retrieval failed after all attempts.")
                    if debug_stock and symbol == debug_stock:
                        sys.exit(1)
                    else:
                        pbar.update(1)
                        continue

                if 'Datetime' not in df.columns:
                    df['Datetime'] = df.index
                df['Datetime'] = pd.to_datetime(df['Datetime'])


                logger.debug(f"Adding technical indicators for {symbol}...")
                df = add_technical_indicators_and_features(df)

                if df.empty:
                    error_msg = f"❌ Critical: Empty DataFrame after feature engineering for {symbol}."
                    logger.critical(error_msg)
                    skipped_tickers_log.append(f"{symbol}: Empty DataFrame after feature engineering.")
                    if debug_stock and symbol == debug_stock:
                        sys.exit(1)
                    else:
                        pbar.update(1)
                        continue

                if train:
                    model, trained_df = train_model(df, symbol, output_models_dir)
                    if model is None:
                        error_msg = f"Skipping model training for {symbol} due to data issues (e.g., single class in target)."
                        logger.warning(error_msg)
                        skipped_tickers_log.append(f"{symbol}: Model training failed.")
                        if debug_stock and symbol == debug_stock:
                            sys.exit(1)
                        else:
                            pbar.update(1)
                            continue
                    df = trained_df

                file_name = f"{symbol}_features_{datetime.now().strftime('%Y%m%d')}.parquet"
                file_path = os.path.join(output_features_dir, file_name)
                df.to_parquet(file_path, index=False)
                logger.debug(f"✅ Saved features for {symbol} to {file_path}")
                processed_count += 1

            except Exception as e:
                error_msg = f"{symbol} skipped due to unexpected error during processing: {e}"
                logger.error(f"Error processing {symbol}: {e}")
                skipped_tickers_log.append(f"{symbol}: {traceback.format_exc()}")
                if debug_stock and symbol == debug_stock:
                    logger.critical(f"❌ {error_msg}. Exiting script because it's the debug stock.")
                    sys.exit(1)
                if debug:
                    traceback.print_exc()

            pbar.update(1)

    logger.debug("\n=== PROCESSING SUMMARY ===")
    logger.info(f"Successful: {processed_count}/{len(tickers)}")
    logger.warning(f"Skipped: {len(skipped_tickers_log)}")

    if skipped_tickers_log:
        skipped_log_file = os.path.join(LOG_DIR,
                                        f"skipped_tickers_verbose_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        with open(skipped_log_file, 'w', encoding='utf-8') as f:
            for entry in skipped_tickers_log:
                f.write(entry + '\n' + '-' * 60 + '\n')
        logger.info(f"📝 Verbose skipped tickers saved to: {skipped_log_file}")

    logger.info(f"📊 Completed processing for the current batch. Processed {processed_count} stocks.")


def main(ticker_list_file, train, data_source_manager_instance, debug_stock=None):
    """Main function to orchestrate the data processing pipeline."""
    target_base_dir = TRAIN_DIR if train else TEST_DIR
    logger.info(f"✨ Starting data processing for {ticker_list_file} to {target_base_dir}")

    with open(ticker_list_file, 'r') as f:
        tickers = [line.strip() for line in f if line.strip()]

    if not tickers:
        logger.warning(f"No tickers found in {ticker_list_file}. Exiting data processing for this file.")
        return

    logger.info(f"Loaded {len(tickers)} tickers from {ticker_list_file}")

    process_ticker_list(tickers, output_base_dir=target_base_dir, train=train,
                        data_source_manager_instance=data_source_manager_instance, debug_stock=debug_stock)


if __name__ == "__main__":
    """🚀 Main execution block when the script is run directly."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='StockWise NASDAQ Data Processor - Comprehensive')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with detailed output')
    parser.add_argument('--stock', type=str, help='Process only specific stock (for debugging)')
    parser.add_argument('--max-stocks', type=int, default=1000,
                        help='Maximum number of stocks to process')  # Default set to 1000
    parser.add_argument('--quick', action='store_true',
                        help='Enable quick mode, using a predefined small list of quality stocks')
    parser.add_argument('--use-ibkr', action='store_true', default=True,
                        help='Attempt to use IBKR for data retrieval (requires TWS/Gateway)')
    parser.add_argument('--ibkr-host', type=str, default="127.0.0.1", help='IBKR TWS/Gateway host')
    parser.add_argument('--ibkr-port', type=int, default=7497, help='IBKR TWS/Gateway port')
    parser.add_argument('--yfinance-retries', type=int, default=10, help='Number of retries for yfinance data fetching')
    parser.add_argument('--yfinance-retry-delay', type=int, default=1, help='Delay in seconds between yfinance retries')

    args = parser.parse_args()

    debug = args.debug
    debug_stock = args.stock
    max_stocks = args.max_stocks
    quick_mode = args.quick
    use_ibkr = args.use_ibkr
    ibkr_host = args.ibkr_host
    ibkr_port = args.ibkr_port
    yfinance_max_retries = args.yfinance_retries
    yfinance_retry_delay = args.yfinance_retry_delay

    if debug:
        logger.setLevel(logging.DEBUG)

    if debug:
        logger.info("🐛 DEBUG MODE ENABLED")
        if debug_stock:
            logger.info(f"🎯 Debug stock filter: {debug_stock}")
        if use_ibkr:
            logger.info(f"🌐 Attempting IBKR connection on {ibkr_host}:{ibkr_port}")
        logger.info("=" * 60)

    if quick_mode:
        logger.info("⚡ QUICK MODE: Using quality fallback stocks only")

    logger.info(
        f"COMPREHENSIVE: Fetching ALL NASDAQ stocks. Debug={'ON' if debug else 'OFF'}, Quick={'ON' if quick_mode else 'OFF'}, Use IBKR={'ON' if use_ibkr else 'OFF'}")

    logger.info("✨ Initializing DataSourceManager for data retrieval...")
    data_source_manager = DataSourceManager(
        debug=debug,
        use_ibkr=use_ibkr,
        ibkr_host=ibkr_host,
        ibkr_port=ibkr_port,
        yfinance_max_retries=yfinance_max_retries,
        yfinance_retry_delay=yfinance_retry_delay
    )
    logger.info("✅ DataSourceManager initialized.")

    logger.info("\n🚀 GENERATING COMPREHENSIVE NASDAQ TICKER LISTS")
    train_tickers_file = os.path.join(LOG_DIR, "nasdaq_train_comprehensive.txt")
    test_tickers_file = os.path.join(LOG_DIR, "nasdaq_test_comprehensive.txt")

    train_tickers, test_tickers = generate_comprehensive_nasdaq_ticker_lists(
        train_file=train_tickers_file,
        test_file=test_tickers_file
    )

    if train_tickers and (not debug_stock or debug_stock in train_tickers):
        logger.info(f"\n🚀 Processing {len(train_tickers)} training tickers...")
        main(train_tickers_file, train=True, data_source_manager_instance=data_source_manager, debug_stock=args.stock)

    if test_tickers and (not debug_stock or debug_stock in test_tickers):
        logger.info(f"\n🚀 Processing {len(test_tickers)} testing tickers...")
        main(test_tickers_file, train=False, data_source_manager_instance=data_source_manager, debug_stock=args.stock)

    logger.info("\n✅ COMPREHENSIVE PROCESSING COMPLETE!")
    total_processed_stocks = len(train_tickers) + len(test_tickers)
    logger.info(f"📊 Total stocks processed for feature generation: {total_processed_stocks}")
    logger.info("🎉 Feature file generation pipeline finished.")

    data_source_manager.disconnect()
    logger.info("✅ DataSourceManager disconnected.")
