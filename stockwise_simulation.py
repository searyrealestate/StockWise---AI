"""
StockWise AI Trading Advisor - Gen-3 Streamlit Application
=========================================================

This script creates a sophisticated, interactive web application using Streamlit
that serves as the front-end for the Gen-3 multi-agent AI trading system.

It allows users to select a stock, a date, and a specific AI trading agent to
receive a real-time trading recommendation based on a complex, event-driven
state machine.

Key Architectural Features:
---------------------------
-   **Multi-Agent System**: Users can choose from several AI agents (e.g.,
    'Dynamic Profit', '2% Net Profit'), each with its own suite of nine
    specialist models trained for different market volatility regimes and
    trading actions.
-   **State Machine Logic**: The application maintains a state for each stock
    (i.e., whether a position is currently "open"). This allows it to provide
    context-aware recommendations:
    -   If no position is open, it uses the "entry" models to look for a BUY signal.
    -   If a position is open, it uses the "profit-take" and "cut-loss" models
        to look for a SELL or CUT LOSS signal.
-   **Live Feature Engineering**: It calculates all required Gen-3 technical
    indicators on the fly using the `pandas-ta` library, ensuring consistency
    with the model training pipeline.
-   **Market Regime Filter**: As a top-level risk management rule, the system
    first checks the broader market trend (SPY vs. its 200-day SMA) and will
    avoid issuing BUY signals during a market downtrend.

UI and Outputs:
---------------
-   **Interactive Sidebar**: Allows for the selection of the AI agent, stock
    symbol, and analysis date.
-   **Clear Recommendations**: Provides a single, clear recommendation (BUY, SELL,
    HOLD, WAIT, CUT LOSS) with the confidence level and the specific agent
    that made the decision.
-   **Detailed Financial Metrics**: For a BUY signal, it displays the current price,
    a dynamic ATR-based stop-loss, a profit target, and a hypothetical net
    profit calculation.
-   **Interactive Charting**: Generates a Plotly candlestick chart showing the
    price action, moving averages, Bollinger Bands, and key trade levels like
    the entry point, stop-loss, and profit target.

"""

from google.cloud import storage
from google.oauth2 import service_account
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import joblib
import json
import os
import glob
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
import sys
import traceback
from plotly.subplots import make_subplots
from data_source_manager import DataSourceManager
import screener
import urllib.request
import shap
import matplotlib.pyplot as plt
from utils import clean_raw_data
from mico_system import MichaAdvisor
import results_analyzer
from trading_models import (MeanReversionAdvisor, BreakoutAdvisor, SuperTrendAdvisor,
                            MovingAverageCrossoverAdvisor, VolumeMomentumAdvisor)
import mico_optimizer
from io import BytesIO


# --- Page Configuration ---
st.set_page_config(
    page_title="StockWise AI Trading Advisor",
    page_icon="🏢",
    layout="wide"
)


@st.cache_data
def get_sp500_tickers():
    """Scrapes and caches the list of S&P 500 tickers from Wikipedia."""
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

        # Create a request with a browser User-Agent header to avoid 403 error.
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        req = urllib.request.Request(url, headers=headers)

        with urllib.request.urlopen(req) as response:
            table = pd.read_html(response)

        sp500_df = table[0]
        tickers = sp500_df['Symbol'].tolist()
        # Clean up tickers for yfinance compatibility (e.g., 'BRK.B' -> 'BRK-B')
        tickers = [ticker.replace('.', '-') for ticker in tickers]
        st.session_state.sp500_tickers = tickers
        return tickers
    except Exception as e:
        st.error(f"Failed to fetch S&P 500 list: {e}")
        return []


@st.cache_data
def get_nasdaq100_tickers():
    """Scrapes and caches the list of NASDAQ 100 tickers from Wikipedia."""
    try:
        url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
        # Create a request with a browser User-Agent header to avoid 403 error.
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        req = urllib.request.Request(url, headers=headers)

        # Manually open the request and pass the response object to pandas.
        with urllib.request.urlopen(req) as response:
            table = pd.read_html(response)
        # The table with tickers is the 4th one on this Wikipedia page
        nasdaq100_df = table[4]
        tickers = nasdaq100_df['Ticker'].tolist()
        st.session_state.nasdaq100_tickers = tickers
        return tickers
    except Exception as e:
        st.error(f"Failed to fetch NASDAQ 100 list: {e}")
        return []


@st.cache_data
def load_nasdaq_tickers(max_stocks=None):
    """Loads NASDAQ ticker symbols from a CSV file."""
    csv_path = "nasdaq_stocks.csv"
    try:
        df = pd.read_csv(csv_path)
        # Standardize column names for robustness
        df.columns = [col.strip().title() for col in df.columns]
        if 'Symbol' not in df.columns:
            # FIX: Use st.error for UI feedback instead of the missing logger
            st.error(f"FATAL: '{csv_path}' is missing the required 'Symbol' column.")
            return []

        # Filter out non-stock symbols that can cause errors
        tickers = df[~df['Symbol'].str.contains(r'\^|\.', na=True)]['Symbol'].dropna().tolist()
        return tickers
    except FileNotFoundError:
        # FIX: Use st.error for UI feedback instead of the missing logger
        st.error(f"FATAL: '{csv_path}' not found. Please ensure it is in the project's root directory.")
        return []


# def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
#     """A single, robust function to clean raw data immediately after fetching."""
#     if df is None or df.empty:
#         return pd.DataFrame()
#
#     # This handles the case of multi-ticker downloads from yfinance
#     if isinstance(df.columns, pd.MultiIndex):
#         # The ticker symbol is level 1 of the MultiIndex. Drop it, keep level 0.
#         df.columns = df.columns.droplevel(1)
#
#     df.columns = [col.lower() for col in df.columns]
#
#     # Ensure standard OHLCV columns are numeric, coercing errors
#     standard_cols = ['open', 'high', 'low', 'close', 'volume']
#     for col in standard_cols:
#         if col in df.columns:
#             df[col] = pd.to_numeric(df[col], errors='coerce')
#
#     # Only drop NaNs from the columns that actually exist in the DataFrame.
#     existing_cols = [col for col in standard_cols if col in df.columns]
#     if existing_cols:
#         df.dropna(subset=existing_cols, inplace=True)
#
#     return df


def calculate_kama(close, window=10, pow1=2, pow2=30):
    """Calculates Kaufman's Adaptive Moving Average (KAMA) manually."""
    diff = abs(close.diff(1))
    rolling_sum_diff = diff.rolling(window).sum()
    er = abs(close.diff(window)) / rolling_sum_diff.replace(0, np.nan)
    er.fillna(0, inplace=True)
    sc = (er * (2 / (pow1 + 1) - 2 / (pow2 + 1)) + 2 / (pow2 + 1)) ** 2
    kama = np.zeros_like(close, dtype=float)
    kama[:window] = close.iloc[:window]
    for i in range(window, len(close)):
        if not np.isnan(sc.iloc[i]):
            kama[i] = kama[i - 1] + sc.iloc[i] * (close.iloc[i] - kama[i - 1])
        else:
            kama[i] = kama[i - 1]
    return pd.Series(kama, index=close.index)


def calculate_stochastic(high, low, close, window=14, smooth_window=3):
    """Calculates the Stochastic Oscillator (%K and %D) manually."""
    lowest_low = low.rolling(window=window).min()
    highest_high = high.rolling(window=window).max()
    percent_k = 100 * ((close - lowest_low) / (highest_high - lowest_low + 1e-10))
    percent_d = percent_k.rolling(window=smooth_window).mean()
    return percent_k, percent_d


# --- Feature Engineering Pipeline (for Gen-3 Model) ---
class FeatureCalculator:
    """A dedicated class to handle all feature calculations for the Gen-3 model."""
    def __init__(self, data_manager):
        self.data_manager = data_manager

    def get_dominant_cycle(self, data, min_period=3, max_period=100) -> float:
        data = pd.Series(data).dropna()
        if len(data) < min_period: return 0.0
        detrended = data - np.poly1d(np.polyfit(np.arange(len(data)), data.values, 1))(np.arange(len(data)))
        fft_result = np.fft.fft(detrended.values)
        frequencies = np.fft.fftfreq(len(detrended))
        power = np.abs(fft_result) ** 2
        positive_freq_mask = frequencies > 0
        if not np.any(positive_freq_mask): return 0.0
        periods = 1 / frequencies[positive_freq_mask]
        period_mask = (periods >= min_period) & (periods <= max_period)
        if not np.any(period_mask): return 0.0
        dominant_idx = power[positive_freq_mask][period_mask].argmax()
        return periods[period_mask][dominant_idx]

    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if df.empty or len(df) < 90:
            return pd.DataFrame()

        # --- DEBUG PRINTS ---
        # print("\n--- STARTING DEBUG FOR FeatureCalculator ---")
        # print(f"1. Initial columns: {df.columns.tolist()}")

        # MODIFIED: Standardize column names to lowercase for pandas-ta compatibility
        df.columns = [col.lower() for col in df.columns]
        # print(f"2. Columns after converting to lowercase: {df.columns.tolist()}")

        try:
            df.ta.bbands(length=20, append=True,
                         col_names=("bb_lower", "bb_middle", "bb_upper", "bb_width", "bb_position"))
            df.ta.atr(length=14, append=True, col_names="atr_14")
            df.ta.rsi(length=14, append=True, col_names="rsi_14")
            df.ta.rsi(length=28, append=True, col_names="rsi_28")
            df.ta.macd(append=True, col_names=("macd", "macd_histogram", "macd_signal"))
            # df.ta.adx(length=14, append=True, col_names=("adx", "adx_pos", "adx_neg"))
            df.ta.adx(length=14, append=True, col_names=("adx", "adx_pos", "adx_neg", "adxr_temp"))
            df.drop(columns=["adxr_temp"], inplace=True, errors='ignore')
            # CORRECT: Use a consistent lowercase name for the new column
            df.ta.mom(length=5, append=True, col_names="momentum_5")
            df.ta.obv(append=True)
            df.ta.cmf(append=True, col_names="cmf")

            # Add a final lowercase conversion to handle default names like 'OBV'
            df.columns = [col.lower() for col in df.columns]

        except Exception as e:
            st.error(f"Error during pandas-ta calculations: {e}")
            return pd.DataFrame()

        # Consistently use lowercase column names ('close', 'volume', etc.)
        df['daily_return'] = df['close'].pct_change()
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        df['volatility_20d'] = df['daily_return'].rolling(20).std()
        df['z_score_20'] = (df['close'] - df['bb_middle']) / df['close'].rolling(20).std()

        # Add calculations for KAMA, Stochastic, and Dominant Cycle
        df['kama_10'] = calculate_kama(df['close'], window=10)
        df['stoch_k'], df['stoch_d'] = calculate_stochastic(df['high'], df['low'], df['close'])
        df['dominant_cycle'] = df['close'].rolling(window=252, min_periods=90).apply(self.get_dominant_cycle,
                                                                                         raw=False)
        try:
            # --- START OF ENHANCED DEBUG BLOCK ---
            # print("DEBUG: Attempting to download QQQ data...")
            qqq_data = yf.download("QQQ", start=df.index.min() - timedelta(days=70), end=df.index.max(), progress=False,
                                   auto_adjust=True)

            # print(f"DEBUG: Type of qqq_data is: {type(qqq_data)}")
            # print(f"DEBUG: Value of qqq_data is: {qqq_data}")

            # Add a check to ensure yfinance returned a valid DataFrame
            if isinstance(qqq_data, pd.DataFrame) and not qqq_data.empty:
                # print("DEBUG: qqq_data is a valid DataFrame. Entering IF block.")
                if isinstance(qqq_data.columns, pd.MultiIndex):
                    # The ticker symbol is level 1 of the MultiIndex. Drop it, keep level 0.
                    # print("DEBUG: qqq_data has a MultiIndex. Dropping level 1 and keeping level 0.")
                    qqq_data.columns = qqq_data.columns.droplevel(1)

                qqq_data.columns = [col.lower() for col in qqq_data.columns]
                qqq_close = qqq_data['close'].reindex(df.index, method='ffill')
                df['correlation_50d_qqq'] = df['close'].rolling(50).corr(qqq_close)
                # print("DEBUG: IF block completed successfully.")
            else:
                # print("DEBUG: qqq_data is NOT a valid DataFrame. Entering ELSE block.")
                print("WARNING: Could not download or process QQQ data. Correlation feature will be zero.")
                df['correlation_50d_qqq'] = 0.0
        except Exception as e:
            print(f"--- An exception occurred during QQQ data processing: {e} ---")
            # print(f"DEBUG: Type of qqq_data at time of exception was: {type(qqq_data)}")
            df['correlation_50d_qqq'] = 0.0

        try:
            # VIX Data
            vix_raw = self.data_manager.get_stock_data('^VIX')
            vix_clean = clean_raw_data(vix_raw)
            if not vix_clean.empty:
                aligned_vix = vix_clean['close'].reindex(df.index, method='ffill')
                df['vix_close'] = aligned_vix
            else:
                df['vix_close'] = 0.0

            # TLT Data
            tlt_raw = self.data_manager.get_stock_data('TLT')
            tlt_clean = clean_raw_data(tlt_raw)
            if not tlt_clean.empty:
                aligned_tlt = tlt_clean['close'].reindex(df.index, method='ffill')
                df['corr_tlt'] = df['close'].rolling(50).corr(aligned_tlt)
            else:
                df['corr_tlt'] = 0.0
        except Exception:
            df['vix_close'] = 0.0
            df['corr_tlt'] = 0.0

        # Mocking the cluster labels for live prediction
        df['volatility_90d'] = df['daily_return'].rolling(90).std()
        # These values are based on the defaults in the training script.
        low_thresh, high_thresh = 0.015, 0.030
        df['volatility_cluster'] = pd.cut(df['volatility_90d'], bins=[-np.inf, low_thresh, high_thresh, np.inf],
                                          labels=['low', 'mid', 'high'])

        # # Rename the core columns back to TitleCase before returning the DataFrame
        # df.rename(columns={
        #     'open': 'Open', 'high': 'High', 'low': 'Low',
        #     'close': 'Close', 'volume': 'Volume'
        # }, inplace=True)

        df.bfill(inplace=True)
        df.ffill(inplace=True)
        return df


# --- Main Application Class (with Gen-3 Architecture) ---
class ProfessionalStockAdvisor:
    def __init__(self, model_dir: str, data_source_manager=None, debug=False, testing_mode=False, download_log=False):
        self.log_entries = []
        self.debug = debug
        self.model_dir = model_dir
        self.download_log = download_log
        self.testing_mode = testing_mode

        if data_source_manager:
            self.data_source_manager = data_source_manager
        elif self.testing_mode:
            self.data_source_manager = None
        else:
            self.data_source_manager = DataSourceManager(use_ibkr=True)

        # Pass the SINGLE data manager instance to the FeatureCalculator.
        self.calculator = FeatureCalculator(data_manager=self.data_source_manager)

        # --- GEN-3: Load the entire suite of specialist models ---
        self.models, self.feature_names = self._load_gen3_models()
        self.tax = 0.25
        self.broker_fee = 0.004
        self.position = {}
        self.model_version_info = f"Gen-3: {os.path.basename(model_dir)}"
        if self.download_log: self.log_file = self.setup_log_file()
        self.log("Application Initialized.", "INFO")

    # In stockwise_simulation.py
    # INSIDE the ProfessionalStockAdvisor class

    @st.cache_resource(ttl=3600)  # Cache for 1 hour
    def _download_and_load_models(_self, bucket_name="stockwise-gen3-models-public"):
        """
        Securely downloads models from GCS using Streamlit Secrets
        and loads them into memory.
        """
        _self.log("Attempting to load models...")
        models = {}
        feature_names = {}

        try:
            # 1. Load the secret JSON key from Streamlit Secrets
            creds_json = st.secrets["gcs_service_account"]
            credentials = service_account.Credentials.from_service_account_info(creds_json)
            storage_client = storage.Client(credentials=credentials)
            bucket = storage_client.bucket(bucket_name)

            # 2. Find all model files
            model_files_blob = list(bucket.list_blobs(prefix=f"{_self.model_dir}/"))
            if not model_files_blob:
                _self.log(f"No models found in GCS at gs://{bucket_name}/{_self.model_dir}", "ERROR")
                st.error(f"GCS Error: No models found in bucket for path '{_self.model_dir}'.")
                return None, None

            # 3. Download and load each model
            for blob in model_files_blob:
                if blob.name.endswith(".pkl"):
                    model_name = os.path.basename(blob.name).replace(".pkl", "")
                    features_path = blob.name.replace(".pkl", "_features.json")

                    # Load model
                    model_bytes = blob.download_as_bytes()
                    models[model_name] = joblib.load(BytesIO(model_bytes))

                    # Load features
                    features_blob = bucket.blob(features_path)
                    if features_blob.exists():
                        features_bytes = features_blob.download_as_bytes()
                        feature_names[model_name] = json.loads(features_bytes.decode('utf-8'))
                    else:
                        _self.log(f"Missing feature file: {features_path}", "WARNING")

            if not models:
                _self.log("Model loading failed. No .pkl files found.", "ERROR")
                return None, None

            _self.log(f"✅ Successfully loaded {len(models)} specialist models from GCS.", "INFO")
            return models, feature_names

        except Exception as e:
            _self.log(f"❌ FATAL: Failed to download/load models: {e}", "ERROR")
            st.error(f"FATAL: Could not load models. Check secrets. {e}")
            return None, None

    # This replaces your OLD _load_gen3_models
    def _load_gen3_models(self):
        # This function is now just a wrapper
        # NOTE: Make sure your bucket name here is correct
        return self._download_and_load_models(bucket_name="stockwise-gen3-models-public")

    # def _load_gen3_models(self):
    #     """
    #     Loads the entire suite of specialist models for Gen-3.
    #     """
    #     models = {}
    #     feature_names = {}
    #     try:
    #         model_files = glob.glob(os.path.join(self.model_dir, "*.pkl"))
    #         if not model_files:
    #             self.log(f"No models found in {self.model_dir}. Please run the model trainer.", "ERROR")
    #             return None, None
    #
    #         for model_path in model_files:
    #             model_name = os.path.basename(model_path).replace(".pkl", "")
    #             features_path = model_path.replace(".pkl", "_features.json")
    #             models[model_name] = joblib.load(model_path)
    #             with open(features_path, 'r') as f:
    #                 feature_names[model_name] = json.load(f)
    #
    #         self.log(f"✅ Successfully loaded {len(models)} specialist models for Gen-3.", "INFO")
    #         return models, feature_names
    #     except Exception as e:
    #         self.log(f"Error loading Gen-3 models: {e}", "ERROR")
    #         return None, None

    # log, setup_log_file, validate_symbol_professional methods are correct and do not need changes...
    def log(self, message, level="INFO"):
        timestamp = datetime.now().strftime('%H:%M:%S')
        entry = f"[{timestamp}] [{level}] {message}"
        if not hasattr(self, 'log_entries'):
            self.log_entries = []
        self.log_entries.append(entry)
        if self.download_log and hasattr(self, 'log_file') and self.log_file:
            try:
                with open(self.log_file, "a", encoding='utf-8') as f:
                    f.write(entry + "\n")
            except Exception as e:
                st.error(f"Failed to write to log file: {e}")

    def setup_log_file(self):
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        return os.path.join(log_dir, f"stockwise_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    def validate_symbol_professional(self, symbol):
        self.log(f"Using yfinance for validation of {symbol}", "INFO")
        try:
            ticker = yf.Ticker(symbol)
            if 'regularMarketPrice' in ticker.info and ticker.info['regularMarketPrice'] is not None:
                return True
            if 'currentPrice' in ticker.info and ticker.info['currentPrice'] is not None:
                return True
            return False
        except Exception:
            return False

    def get_market_health_index(self, analysis_date):
        """
        Calculates a Market Health Index (0-4) based on SPY, VIX, and trend indicators.
        A score of 3 or higher is considered a 'risk-on' environment.
        """
        health_score = 0
        reasons = []
        try:
            spy_data_raw = self.data_source_manager.get_stock_data("SPY", days_back=300)
            spy_data = clean_raw_data(spy_data_raw)
            spy_data_slice = spy_data[spy_data.index <= pd.to_datetime(analysis_date)]
            if len(spy_data_slice) < 200: return 0, ["Not enough SPY data."]

            spy_data_slice.ta.sma(length=50, append=True, col_names='sma_50')
            spy_data_slice.ta.sma(length=200, append=True, col_names='sma_200')
            spy_data_slice.ta.rsi(length=14, append=True, col_names='rsi_14')
            latest_spy = spy_data_slice.iloc[-1]

            vix_data_raw = self.data_source_manager.get_stock_data("^VIX", days_back=5)
            vix_data = clean_raw_data(vix_data_raw)
            vix_slice = vix_data[vix_data.index <= pd.to_datetime(analysis_date)]
            latest_vix = vix_slice.iloc[-1] if not vix_slice.empty else None
            # latest_vix = vix_data[vix_data.index <= pd.to_datetime(analysis_date)].iloc[
            #     -1] if not vix_data.empty else None

            # --- Rule 1: Price vs. 50-day SMA ---
            if latest_spy['close'] > latest_spy['sma_50']:
                health_score += 1
                reasons.append("✅ SPY > 50-day SMA")
            else:
                reasons.append("❌ SPY < 50-day SMA")

            # --- Rule 2: 50-day SMA vs. 200-day SMA ---
            if latest_spy['sma_50'] > latest_spy['sma_200']:
                health_score += 1
                reasons.append("✅ 50-day SMA > 200-day SMA")
            else:
                reasons.append("❌ 50-day SMA < 200-day SMA")

            # --- Rule 3: VIX Level ---
            if latest_vix is not None and latest_vix['close'] < 30:
                health_score += 1
                reasons.append(f"✅ VIX < 30 ({latest_vix['close']:.2f})")
            else:
                reason_str = f"❌ VIX > 30 ({latest_vix['close']:.2f})" if latest_vix is not None else \
                    "❌ VIX data unavailable"
                reasons.append(reason_str)

            # --- Rule 4: RSI Momentum ---
            if latest_spy['rsi_14'] > 50:
                health_score += 1
                reasons.append(f"✅ SPY RSI > 50 ({latest_spy['rsi_14']:.2f})")
            else:
                reasons.append(f"❌ SPY RSI < 50 ({latest_spy['rsi_14']:.2f})")

            return health_score, reasons
        except Exception as e:
            self.log(f"Error in market health check: {e}", "ERROR")
            return 0, [f"Error: {e}"]

    def run_analysis(self, ticker_symbol, analysis_date, use_market_filter=True):
        debug = False
        try:
            if debug: st.write("--- `run_analysis` started. ---")

            full_stock_data = self.data_source_manager.get_stock_data(ticker_symbol)
            if full_stock_data is None or full_stock_data.empty:
                return pd.DataFrame(), {'action': "WAIT", 'reason': f"No data found for symbol {ticker_symbol}."}
            full_stock_data = clean_raw_data(full_stock_data)

            data_up_to_date = full_stock_data[full_stock_data.index <= pd.to_datetime(analysis_date)]
            if data_up_to_date.empty:
                return full_stock_data, {'action': "WAIT", 'reason': "No data available for this date.",
                                         'current_price': 0, 'agent': "System"}

            price_on_date = data_up_to_date.iloc[-1]['close']
            if use_market_filter:
                health_score, health_reasons = self.get_market_health_index(analysis_date)
                market_health_results = {'health_score': health_score, 'health_reasons': health_reasons}

                if debug: st.write(f"--- Market Health Check complete. Score: {health_score}/4 ---")

                if health_score < 3:
                    if debug: st.write("--- Market Health FAILED. Returning WAIT/AVOID. ---")
                    return full_stock_data, {**market_health_results, 'action': "WAIT / AVOID", 'confidence': 99.9,
                                             'current_price': price_on_date,
                                             'reason': f"Market Health Index: {health_score}/4. Conditions not met.",
                                             'buy_date': None, 'agent': "Market Regime Agent"}
            else:
                # If the filter is disabled, create a placeholder dictionary and log it
                market_health_results = {'health_score': 'N/A',
                                         'health_reasons': ["Market Health Filter was manually disabled."]}
                self.log("Market Health Filter was disabled by user.", "WARNING")

            if debug: st.write("--- Market Health PASSED. Proceeding to feature engineering. ---")

            featured_data = self.calculator.calculate_all_features(data_up_to_date)
            if featured_data.empty:
                return full_stock_data, {**market_health_results, 'action': "WAIT",
                                         'reason': "Insufficient data for analysis.", 'current_price': price_on_date,
                                         'agent': "System"}

            latest_row = featured_data.iloc[-1]
            cluster = latest_row['volatility_cluster']
            all_features_dict = latest_row.to_dict()

            def get_shap_explanation(model, features_df):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(features_df)
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    return shap_values[1], explainer.expected_value[1]
                else:
                    return shap_values, explainer.expected_value

            if self.position.get(ticker_symbol):  # State: Position is OPEN
                action = "HOLD"  # Default action
                confidence = 0
                agent_name = f"{cluster.capitalize()}-Volatility Hold Agent"

                # Define model names
                profit_model_name = f"profit_take_model_{cluster}_vol"
                loss_model_name = f"cut_loss_model_{cluster}_vol"

                # Ensure models are loaded
                if profit_model_name in self.models and loss_model_name in self.models:
                    profit_model = self.models[profit_model_name]
                    loss_model = self.models[loss_model_name]

                    # Prepare features for both models (assuming they use the same feature set)
                    features = latest_row[self.feature_names[loss_model_name]].astype(float).to_frame().T

                    # --- Priority System: Check for Cut-Loss signal FIRST ---
                    if loss_model.predict(features)[0] == 1:
                        action = "CUT LOSS"
                        confidence = loss_model.predict_proba(features)[0][1] * 100
                        agent_name = f"{cluster.capitalize()}-Volatility Cut-Loss Agent"
                        # Since we are exiting, clear the position
                        del self.position[ticker_symbol]

                    # --- If no cut-loss, check for a Profit-Take signal ---
                    elif profit_model.predict(features)[0] == 1:
                        action = "SELL"  # Changed from "PROFIT TAKE" for consistency
                        confidence = profit_model.predict_proba(features)[0][1] * 100
                        agent_name = f"{cluster.capitalize()}-Volatility Profit-Take Agent"
                        # Since we are exiting, clear the position
                        del self.position[ticker_symbol]
                else:
                    self.log(f"Missing profit or loss models for cluster '{cluster}'. Defaulting to HOLD.", "WARNING")

                action_result = {
                    'action': action,
                    'confidence': confidence,
                    'current_price': float(latest_row['close']),
                    'agent': agent_name,
                    'buy_date': self.position.get(ticker_symbol, {}).get('entry_date')  # Pass along original buy date
                }

                return full_stock_data, {**market_health_results, **action_result, 'all_features': all_features_dict}

            else:  # State: No Position
                entry_model_name = f"entry_model_{cluster}_vol"
                entry_model = self.models.get(entry_model_name)

                if not entry_model:
                    return full_stock_data, {**market_health_results, 'action': "WAIT", 'reason': "Missing Models.",
                                             'current_price': price_on_date, 'agent': "System",
                                             'all_features': all_features_dict}

                features = latest_row[self.feature_names[entry_model_name]].astype(float).to_frame().T
                entry_pred = entry_model.predict(features)[0]
                entry_prob = entry_model.predict_proba(features)[0]
                result = {}

                if entry_pred == 1:
                    stop_loss_price = latest_row['close'] - (latest_row['atr_14'] * 2.5)
                    shap_values_for_buy, base_value_for_buy = get_shap_explanation(entry_model, features)
                    result = {
                        'action': "BUY", 'confidence': entry_prob[1] * 100, 'current_price': float(latest_row['close']),
                        'buy_date': latest_row.name.date(),
                        'agent': f"{cluster.capitalize()}-Volatility Entry Agent", 'stop_loss_price': stop_loss_price,
                        'shap_values': shap_values_for_buy[0], 'shap_base_value': base_value_for_buy,
                        'feature_names': features.columns.tolist(), 'feature_values': features.iloc[0].tolist()
                    }
                    self.position[ticker_symbol] = {'entry_price': latest_row['close'],
                                                    'stop_loss_price': stop_loss_price}
                else:
                    result = {'action': "WAIT", 'confidence': entry_prob[0] * 100,
                              'current_price': float(latest_row['close']),
                              'agent': f"{cluster.capitalize()}-Volatility Entry Agent"}

                final_result = {**market_health_results, **result, 'all_features': all_features_dict}
                if debug:
                    st.write("--- Final result dictionary being returned: ---")
                    st.json(final_result)
                return full_stock_data, final_result

        except Exception as e:
            st.code(traceback.format_exc())
            return None, None

    def analyze(self, symbol, analysis_date, params: dict = None):
        """
        An adapter method to make the AI Advisor compatible with the modular screener.
        It calls the main analysis function and returns just the results dictionary.
        """
        # The run_analysis function returns two items: (stock_data, result_dict)
        # We only need the second item for the screener.
        _, result = self.run_analysis(symbol, analysis_date, use_market_filter=True)
        return result


    def calculate_dynamic_profit_target(self, confidence):
        # This function is now deprecated in favor of model-driven 'SELL' signals.
        # It is kept for legacy UI compatibility if needed.
        if confidence > 90:
            return 8.0
        elif confidence > 75:
            return 6.5
        elif confidence > 60:
            return 5.0
        else:
            return 3.5

    def apply_israeli_fees_and_tax(self, gross_profit_dollars, num_shares):
        per_share_fee = 0.008 * num_shares
        minimum_fee = 2.50
        single_transaction_fee = max(per_share_fee, minimum_fee)
        total_fees_dollars = single_transaction_fee * 2
        profit_after_fees_dollars = gross_profit_dollars - total_fees_dollars
        tax_dollars = (profit_after_fees_dollars * self.tax) if profit_after_fees_dollars > 0 else 0
        net_profit_dollars = profit_after_fees_dollars - tax_dollars
        total_deducted_dollars = total_fees_dollars + tax_dollars
        return net_profit_dollars, total_deducted_dollars

    def create_chart(self, stock_symbol, stock_data, result, analysis_date):
        if stock_data.empty: return None
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.8, 0.2])
        fig.add_trace(
            go.Candlestick(x=stock_data.index, open=stock_data['open'], high=stock_data['high'], low=stock_data['low'],
                           close=stock_data['close'], name='Price'), row=1, col=1)
        for period, color in [(5, 'orange'), (20, 'blue'), (50, 'red')]:
            if len(stock_data) >= period:
                ma = stock_data['close'].rolling(window=period).mean()
                fig.add_trace(go.Scatter(x=stock_data.index, y=ma, mode='lines', name=f'MA{period}',
                                         line=dict(color=color, width=1)), row=1, col=1)
        if len(stock_data) >= 20:
            try:
                # Create a temporary DataFrame for the calculation
                bb_df = ta.bbands(close=stock_data['close'], length=20)
                fig.add_trace(go.Scatter(x=stock_data.index, y=bb_df['BBU_20_2.0'], mode='lines', name='BB Upper',
                                         line=dict(color='gray', dash='dot', width=1), showlegend=False), row=1, col=1)
                fig.add_trace(go.Scatter(x=stock_data.index, y=bb_df['BBL_20_2.0'], mode='lines', name='BB Lower',
                                         line=dict(color='gray', dash='dot', width=1), fill='tonexty',
                                         fillcolor='rgba(128,128,128,0.1)', showlegend=False), row=1, col=1)
            except Exception as e:
                self.log(f"Could not calculate Bollinger Bands: {e}", "WARNING")
        fig.add_trace(
            go.Bar(x=stock_data.index, y=stock_data['volume'], name='Volume', marker=dict(color='rgba(100,110,120,0.6)')),
            row=2, col=1)
        fig.add_vline(x=analysis_date, line_width=1, line_dash="dash", line_color="white", name="Analysis Date",
                      row=1)
        action = result['action']
        # current_price = result['current_price']
        current_price = result.get('current_price', stock_data['close'].iloc[-1] if not stock_data.empty else 0)

        if "BUY" in action:
            buy_date = result['buy_date']
            stop_loss = result.get('stop_loss_price')
            confidence = result.get('confidence', 0)
            profit_target_pct = self.calculate_dynamic_profit_target(confidence)
            profit_target_price = current_price * (1 + profit_target_pct / 100)

            fig.add_hline(y=float(profit_target_price), line_dash="dash", line_color="lightgreen",
                          name="Profit Target", row=1, annotation_text=f"Profit Target: ${profit_target_price:.2f}",
                          annotation_position="top right")
            fig.add_trace(go.Scatter(
                x=[buy_date], y=[current_price], mode='markers',
                marker=dict(color='cyan', size=12, symbol='circle-open', line=dict(width=2)), name='Target Buy'
            ), row=1, col=1)
            if stop_loss:
                fig.add_hline(y=float(stop_loss), line_dash="dash", line_color="red",
                              name="Stop-Loss Price", row=1, annotation_text=f"Stop-Loss: ${stop_loss:.2f}",
                              annotation_position="bottom right")

        zoom_start_date = pd.to_datetime(analysis_date) - timedelta(days=10)
        zoom_end_date = pd.to_datetime(analysis_date) + timedelta(days=120)
        fig.update_layout(title_text=f'{stock_symbol} Price & Volume Analysis', xaxis_rangeslider_visible=False,
                          showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig.update_xaxes(range=[zoom_start_date, zoom_end_date], row=1, col=1)
        fig.update_xaxes(range=[zoom_start_date, zoom_end_date], row=2, col=1)
        fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        return fig


def display_analysis_results(ai_result, mico_result, stock_data, stock_symbol, analysis_date, advisor):
    """
    Renders the entire multi-stage analysis UI in a consistent format.
    This function is responsible for all display logic.
    """

    debug = False

    if debug:
        st.write("--- Data received by `display_analysis_results`: ---")
        st.json(ai_result)
    # --- 1. TOP-LEVEL METRICS ---
    st.subheader("System Recommendations")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🧠 AI Advisor (Gen-3)")
        if ai_result:
            action = ai_result.get('action', 'N/A')
            confidence = ai_result.get('confidence', 0)
            current_price = ai_result.get('current_price', 0)

            if "BUY" in action:
                st.success(f"**Signal: {action}**")
            else:
                st.warning(f"**Signal: {action}**")

            profit_target_price, net_profit = 0, 0

            # Calculate profit metrics only if it's a BUY signal
            if "BUY" in action and current_price > 0:
                est_profit_pct = advisor.calculate_dynamic_profit_target(confidence)
                profit_target_price = current_price * (1 + est_profit_pct / 100)
                hypothetical_shares = 1000 / current_price
                gross_profit = (profit_target_price - current_price) * hypothetical_shares
                net_profit, _ = advisor.apply_israeli_fees_and_tax(gross_profit, hypothetical_shares)

            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Confidence", f"{confidence:.1f}%" if confidence > 0 else "N/A")
            m_col2.metric("Profit Target", f"${profit_target_price:.2f}" if profit_target_price > 0 else "-")
            m_col3.metric("Hypothetical Net Profit", f"${net_profit:.2f}" if net_profit > 0 else "-")
            st.caption(f"Agent: {ai_result.get('agent', 'N/A')}")

        else:
            st.info("AI Advisor was not run.")

    with col2:
        st.markdown("#### 📜 Micha System (Rule-Based)")
        if mico_result:
            if mico_result['signal'] == 'BUY':
                st.success("**Signal: BUY**")
            else:
                st.warning(f"**Signal: {mico_result['signal']}**")
            with st.expander("Show Micha Rule Analysis"):
                st.markdown(f"_{mico_result['reason']}_")
        else:
            st.info("Micha System was not run.")

    st.markdown("---")
    st.subheader("1. Market Health Analysis (SPY)")
    if ai_result:
        health_score = ai_result.get('health_score')
        health_reasons = ai_result.get('health_reasons', [])
        st.metric("Market Health Index", f"{health_score}/4" if health_score is not None else "N/A")
        if health_reasons:
            for reason in health_reasons:
                st.markdown(reason)
    else:
        st.info("Market Health was not analyzed (AI Advisor disabled).")
    st.markdown("---")

    # --- 3. PRICE CHART ---
    st.subheader("2. Price Chart")
    fig = advisor.create_chart(stock_symbol, stock_data, ai_result, analysis_date)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Could not generate price chart.")
    st.markdown("---")

    # --- 4. SUMMARY & PARAMETERS ---
    st.subheader("3. Analysis Summary")
    agent_name = ai_result.get('agent', 'N/A')

    if "Market Regime Agent" in agent_name:
        st.error(
            f"**Reason for Signal:** Analysis halted by the `{agent_name}`. Market conditions are unfavorable for new BUY signals.")
    else:
        st.info(f"**Agent Hand-off:** Based on volatility, the system selected the `{agent_name}`.")

    if 'shap_values' in ai_result:
        with st.expander("Show Key Factors in AI Decision (SHAP Analysis)"):
            explanation = shap.Explanation(
                values=ai_result['shap_values'],
                base_values=ai_result['shap_base_value'],
                data=ai_result['feature_values'],
                feature_names=ai_result['feature_names']
            )
            fig, ax = plt.subplots()
            shap.waterfall_plot(explanation, max_display=10, show=False)
            st.pyplot(fig)

    if 'all_features' in ai_result:
        with st.expander("Show All Technical Parameters Used"):
            features_df = pd.DataFrame.from_dict(ai_result['all_features'], orient='index', columns=['Value'])
            features_df.index.name = 'Feature'
            features_df['Value'] = features_df['Value'].astype(str)
            st.dataframe(features_df.style.format(precision=4), use_container_width=True)


# def create_enhanced_interface():
#     # --- Connection Status Indicator ---
#     _, col2 = st.columns([4, 1])
#
#     status_placeholder = col2.empty()
#
#     st.title("🏢 StockWise AI Trading Advisor")
#
#     AGENT_CONFIGS = {
#         'Dynamic Profit (Recommended)': "models/NASDAQ-gen3-dynamic",
#         '1% Net Profit': "models/NASDAQ-gen3-1pct",
#         '2% Net Profit': "models/NASDAQ-gen3-2pct",
#         '3% Net Profit': "models/NASDAQ-gen3-3pct",
#         '4% Net Profit': "models/NASDAQ-gen3-4pct"
#     }
#
#     # --- Sidebar UI ---
#     st.sidebar.header("🎯 Trading Analysis")
#     st.sidebar.markdown("**Select Systems to Run:**")
#     # run_ai = st.sidebar.checkbox("Run AI Advisor (Gen-3)", value=True)
#     # run_mico = st.sidebar.checkbox("Run Micha System (Rule-Based)", value=True)
#     run_ai, run_mico = True, True
#
#     # Add checkboxes for the new models
#     with st.sidebar.expander("Advanced Models", expanded=True):
#         run_mean_reversion = st.sidebar.checkbox("Run Mean Reversion Model", value=True)
#         run_breakout = st.sidebar.checkbox("Run Breakout Model", value=True)
#         run_supertrend = st.sidebar.checkbox("Run SuperTrend Model", value=True)
#         run_ma_crossover = st.sidebar.checkbox("Run MA Crossover Model", value=False)
#         run_volume_momentum = st.sidebar.checkbox("Run Volume Momentum Model", value=False)
#
#
#
#     selected_agent_name = st.sidebar.selectbox("🧠 Select AI Agent", options=list(AGENT_CONFIGS.keys()))
#     investment_amount = st.sidebar.number_input(
#         "Hypothetical Investment per Trade ($)",
#         min_value=100,
#         value=1000,
#         step=50,
#         help="Set the amount to use for calculating hypothetical net profit in the screener."
#     )
#     stock_symbol = st.sidebar.text_input("📊 Stock Symbol", value="NVDA").upper().strip()
#
#     if 'analysis_date_input' not in st.session_state:
#         st.session_state.analysis_date_input = datetime.now().date()
#
#     def set_date_to_today():
#         st.session_state.analysis_date_input = datetime.now().date()
#
#     # --- Sidebar Date Input with "Today" Button ---
#     col1, col2 = st.sidebar.columns([2, 1])
#     with col1:
#         # The date_input widget itself doesn't need to change
#         analysis_date = st.date_input("📅 Analysis Date", key='analysis_date_input')
#     with col2:
#         st.write("")
#         st.write("")
#         # Call the function using on_click. No 'if' block or 'st.rerun()' is needed.
#         st.button("Today", on_click=set_date_to_today, use_container_width=True)
#
#     analyze_btn = st.sidebar.button("🚀 Run Professional Analysis", type="primary", use_container_width=True)
#     use_market_filter = st.sidebar.checkbox("Enable Market Health Filter (SPY)", value=True)
#
#     st.sidebar.markdown("---")
#     st.sidebar.header("📈 Market Screener")
#
#     # Add a dedicated date input for the screener
#     if 'screener_date_input' not in st.session_state:
#         st.session_state.screener_date_input = datetime.now().date()
#
#     # Add the debug mode checkbox
#     debug_mode = st.sidebar.checkbox("Enable Screener Debug Mode", value=False)
#     use_optimized = st.sidebar.checkbox("✅ Use Optimized Parameters", value=False,
#                                         help="Requires running the Optimizer first to generate best_params.json")
#     # scan_btn = st.sidebar.button("Scan Universe for Opportunities", use_container_width=True)
#     if debug_mode:
#         def set_screener_date_to_today():
#             st.session_state.screener_date_input = datetime.now().date()
#
#         col1_scr, col2_scr = st.sidebar.columns([2, 1])
#         with col1_scr:
#             screener_analysis_date = col1_scr.date_input("🗓️ Screener Date", key='screener_date_input')
#         with col2_scr:
#             st.write("")
#             st.write("")
#             col2_scr.button("Today", on_click=set_screener_date_to_today, use_container_width=True, key='screener_today_btn')
#     else:
#         screener_analysis_date = analysis_date
#
#     # Reordered the screener options logically
#     universe_options = {
#         "Full NASDAQ (from file)": load_nasdaq_tickers,
#         "S&P 500": get_sp500_tickers,
#         "NASDAQ 100": get_nasdaq100_tickers
#     }
#     selected_universe_name = st.sidebar.selectbox(
#         "Select universe to scan:",
#         options=list(universe_options.keys()),
#         key="universe_selector"
#     )
#
#     # Defined the "Scan" button only ONCE, after all its options
#     scan_btn = st.sidebar.button("Scan Universe for Opportunities", use_container_width=True)
#
#     # st.sidebar.markdown("---")
#     # st.sidebar.header("⚙️ Strategy Optimizer")
#
#     # opt_symbol = st.sidebar.text_input("Stock to Optimize", value="AAPL").upper().strip()
#
#     # Define your advisors for the optimizer dropdown
#     optimizer_advisors = {
#         # Added all optimizable models to this dictionary
#         "MichaAdvisor": st.session_state.mico_advisor,
#         "SuperTrendAdvisor": st.session_state.supertrend_advisor,
#         "MeanReversionAdvisor": st.session_state.mean_reversion_advisor,
#         "BreakoutAdvisor": st.session_state.breakout_advisor,
#         "MovingAverageCrossoverAdvisor": st.session_state.ma_crossover_advisor,
#         "VolumeMomentumAdvisor": st.session_state.volume_momentum_advisor,
#     }
#
#     # opt_advisor_name = st.sidebar.selectbox("Model to Optimize", options=optimizer_advisors.keys())
#     #
#     # col1_opt, col2_opt = st.sidebar.columns(2)
#     # opt_start_date = col1_opt.date_input("Opt. Start Date", datetime.now().date() - timedelta(days=365))
#     # opt_end_date = col2_opt.date_input("Opt. End Date", datetime.now().date())
#     #
#     # optimize_btn = st.sidebar.button("Find Best Parameters", use_container_width=True)
#
#     # --- Advisor Loading Logic ---
#     selected_model_dir = AGENT_CONFIGS[selected_agent_name]
#     if st.session_state.advisor.model_dir != selected_model_dir:
#         with st.spinner(f"Loading '{selected_agent_name}' agent..."):
#             st.session_state.advisor = ProfessionalStockAdvisor(
#                 model_dir=selected_model_dir,
#                 data_source_manager=st.session_state.data_manager
#             )
#     advisor = st.session_state.advisor
#     st.markdown(f"### Now using `{selected_agent_name}` Agent")
#     st.markdown("---")
#
#     # --- Main Analysis & Display Logic ---
#     if analyze_btn:
#         if not run_ai and not run_mico:
#             st.warning("Please select the AI Advisor system to run.")
#             return
#         if not stock_symbol:
#             st.warning("Please enter a stock symbol.")
#             return
#
#         ai_result, mico_result, stock_data = None, None, None
#         with st.spinner(f"Running full analysis for {stock_symbol}..."):
#             if run_ai:
#                 stock_data, ai_result = advisor.run_analysis(stock_symbol, analysis_date, use_market_filter)
#             if run_mico:
#                 mico_result = st.session_state.mico_advisor.analyze(stock_symbol, analysis_date)
#                 # If AI didn't run, we still need the stock_data for the chart
#                 if stock_data is None:
#                     raw_data = advisor.data_source_manager.get_stock_data(stock_symbol)
#                     stock_data = clean_raw_data(raw_data)
#
#         if not ai_result and not mico_result:
#             st.error("Analysis failed. Could not retrieve data or run models.")
#             status_placeholder.markdown("<p style='text-align: right;'>🔴 <strong>DATA FAILED</strong></p>",
#                                         unsafe_allow_html=True)
#         else:
#             if st.session_state.data_manager.use_ibkr and st.session_state.data_manager.isConnected():
#                 status_placeholder.markdown("<p style='text-align: right;'>🟢 <strong>IBKR</strong></p>",
#                                             unsafe_allow_html=True)
#             else:
#                 status_placeholder.markdown("<p style='text-align: right;'>🟡 <strong>YFINANCE</strong></p>",
#                                             unsafe_allow_html=True)
#
#             display_analysis_results(ai_result, mico_result, stock_data, stock_symbol, analysis_date, advisor)
#
#     if scan_btn:
#         st.session_state.analysis_run = False
#
#         # Clear previous results before starting a new scan
#         if 'screener_results' in st.session_state:
#             del st.session_state['screener_results']
#
#         active_advisors = {}
#         if run_ai: active_advisors["AI"] = st.session_state.advisor
#         if run_mico: active_advisors["MICO"] = st.session_state.mico_advisor
#         if run_mean_reversion: active_advisors["MeanReversion"] = st.session_state.mean_reversion_advisor
#         if run_breakout: active_advisors["Breakout"] = st.session_state.breakout_advisor
#         if run_supertrend: active_advisors["SuperTrend"] = st.session_state.supertrend_advisor
#         if run_ma_crossover: active_advisors["MACrossover"] = st.session_state.ma_crossover_advisor
#         if run_volume_momentum: active_advisors["VolumeMomentum"] = st.session_state.volume_momentum_advisor
#
#         if not active_advisors:
#             st.warning("Please select at least one system to run for the screener.")
#             return
#
#         # Load the selected stock universe by calling the chosen function
#         load_function = universe_options[selected_universe_name]
#         stock_universe = load_function()
#
#         if not stock_universe:
#             st.error(f"Could not load the '{selected_universe_name}' stock list.")
#         else:
#             st.info(f"Scanning {len(stock_universe)} symbols from the '{selected_universe_name}' universe...")
#             recommended_trades_df = screener.run_unified_screener(
#                 active_advisors=active_advisors,
#                 stock_universe=stock_universe,
#                 analysis_date=screener_analysis_date,
#                 investment_amount=investment_amount,
#                 debug_mode=debug_mode,
#                 use_optimized_params=use_optimized
#             )
#             if recommended_trades_df is not None and not recommended_trades_df.empty:
#                 st.session_state.screener_results = recommended_trades_df
#
#         # The redundant MICO screener call has been removed.
#         if st.session_state.data_manager.use_ibkr and st.session_state.data_manager.isConnected():
#             # Display the final connection status after the scan is complete
#             status_placeholder.markdown("<p style='text-align: right;'>🟢 <strong>IBKR</strong></p>",
#                                         unsafe_allow_html=True)
#         else:
#             status_placeholder.markdown("<p style='text-align: right;'>🟡 <strong>YFINANCE</strong></p>",
#                                         unsafe_allow_html=True)
#
#     if not analyze_btn and not scan_btn:
#         st.info("Select an action from the sidebar to begin.")
#
#     # This section shows the analysis button ONLY if results exist
#     if 'screener_results' in st.session_state:
#         st.markdown("---")
#         st.success(f"Screener found {len(st.session_state.screener_results)} trade opportunities.")
#
#         analyze_screener_btn = st.button("🔬 Analyze Screener Results", type="primary", use_container_width=True)
#         # If the button is clicked, set the state to True
#         if analyze_screener_btn:
#             st.session_state.analysis_run = True
#
#         # Display the backtest if the state is True
#         if st.session_state.get('analysis_run', False):
#             with st.spinner("Running backtest simulation on results..."):
#                 results_analyzer.run_backtest(
#                     trades_df=st.session_state.screener_results,
#                     data_manager=st.session_state.data_manager
#                 )

def create_enhanced_interface():
    # --- Connection Status Indicator ---
    _, col2 = st.columns([4, 1])
    status_placeholder = col2.empty()

    st.title("🏢 StockWise AI Trading Advisor")

    AGENT_CONFIGS = {
        'Dynamic Profit (Recommended)': "models/NASDAQ-gen3-dynamic",
        '1% Net Profit': "models/NASDAQ-gen3-1pct",
        '2% Net Profit': "models/NASDAQ-gen3-2pct",
        '3% Net Profit': "models/NASDAQ-gen3-3pct",
        '4% Net Profit': "models/NASDAQ-gen3-4pct"
    }

    # ==============================================================================
    # --- SIDEBAR UI (Restored to your layout) ---
    # ==============================================================================

    st.sidebar.header("🎯 Trading Analysis")
    st.sidebar.markdown("**Select Systems to Run:**")

    with st.sidebar.expander("Select Models to Run", expanded=True):
        run_ai = st.sidebar.checkbox("Run AI Advisor (Gen-3)", value=True)
        run_mico = st.sidebar.checkbox("Run Micha System (Rule-Based)", value=True)
        run_mean_reversion = st.sidebar.checkbox("Run Mean Reversion Model", value=True)
        run_breakout = st.sidebar.checkbox("Run Breakout Model", value=True)
        run_supertrend = st.sidebar.checkbox("Run SuperTrend Model", value=True)
        run_ma_crossover = st.sidebar.checkbox("Run MA Crossover Model", value=True)
        run_volume_momentum = st.sidebar.checkbox("Run Volume Momentum Model", value=True)

    selected_agent_name = st.sidebar.selectbox("🧠 Select AI Agent", options=list(AGENT_CONFIGS.keys()))
    investment_amount = st.sidebar.number_input(
        "Hypothetical Investment per Trade ($)",
        min_value=100, value=1000, step=50,
        help="Set the amount to use for calculating hypothetical net profit in the screener."
    )
    stock_symbol = st.sidebar.text_input("📊 Stock Symbol", value="NVDA").upper().strip()

    if 'analysis_date_input' not in st.session_state:
        st.session_state.analysis_date_input = datetime.now().date()

    def set_date_to_today():
        st.session_state.analysis_date_input = datetime.now().date()

    col1, col2 = st.sidebar.columns([2, 1])
    with col1:
        analysis_date = st.date_input("📅 Analysis Date", key='analysis_date_input')
    with col2:
        st.write("")
        st.write("")
        st.button("Today", on_click=set_date_to_today, use_container_width=True, key='analysis_today_btn')

    analyze_btn = st.sidebar.button("🚀 Run Professional Analysis", type="primary", use_container_width=True)
    use_market_filter = st.sidebar.checkbox("Enable Market Health Filter (SPY)", value=True)

    st.sidebar.markdown("---")
    st.sidebar.header("📈 Market Screener")

    debug_mode = st.sidebar.checkbox("Enable Screener Debug Mode", value=False)
    use_optimized = st.sidebar.checkbox("✅ Use Optimized Parameters", value=True,
                                        help="Requires running the Optimizer first to generate best_params.json")

    # Your original date logic for the screener
    if debug_mode:
        if 'screener_date_input' not in st.session_state:
            st.session_state.screener_date_input = datetime.now().date()

        def set_screener_date_to_today():
            st.session_state.screener_date_input = datetime.now().date()

        col1_scr, col2_scr = st.sidebar.columns([2, 1])
        with col1_scr:
            screener_analysis_date = col1_scr.date_input("🗓️ Screener Date", key='screener_date_input')
        with col2_scr:
            st.write("")
            st.write("")
            col2_scr.button("Today", on_click=set_screener_date_to_today, use_container_width=True,
                            key='screener_today_btn')
    else:
        screener_analysis_date = analysis_date  # Default to the main analysis date

    universe_options = {
        "Full NASDAQ (from file)": load_nasdaq_tickers,
        "S&P 500": get_sp500_tickers,
        "NASDAQ 100": get_nasdaq100_tickers
    }
    selected_universe_name = st.sidebar.selectbox(
        "Select universe to scan:",
        options=list(universe_options.keys()),
        key="universe_selector"
    )
    scan_btn = st.sidebar.button("Scan Universe for Opportunities", use_container_width=True)

    st.sidebar.markdown("---")

    # --- Strategy Optimizer Section (Restored and Implemented) ---
    st.sidebar.header("⚙️ Strategy Optimizer")
    st.sidebar.info("Run this to find & save the best parameters for a model.")

    # We no longer need the single-model selectbox
    opt_symbol = st.sidebar.text_input("Stock to Optimize On", value="SPY").upper().strip()
    col1_opt, col2_opt = st.sidebar.columns(2)
    opt_start_date = col1_opt.date_input("Opt. Start Date", datetime.now().date() - timedelta(days=365))
    opt_end_date = col2_opt.date_input("Opt. End Date", datetime.now().date())

    # Button text is updated for clarity
    optimize_btn = st.sidebar.button("Run Full System Calibration", use_container_width=True)

    # ==============================================================================
    # --- LOGIC BLOCKS (Restructured for clarity) ---
    # ==============================================================================

    # --- Advisor Loading Logic ---
    selected_model_dir = AGENT_CONFIGS[selected_agent_name]
    if st.session_state.advisor.model_dir != selected_model_dir:
        with st.spinner(f"Loading '{selected_agent_name}' agent..."):
            st.session_state.advisor = ProfessionalStockAdvisor(
                model_dir=selected_model_dir,
                data_source_manager=st.session_state.data_manager
            )
    advisor = st.session_state.advisor
    st.markdown(f"### Now using `{selected_agent_name}` Agent")
    st.markdown("---")

    # --- Single Stock Analysis Logic ---
    if analyze_btn:
        if not run_ai and not run_mico:
            st.warning("Please select at least one system to run.")
            return
        if not stock_symbol:
            st.warning("Please enter a stock symbol.")
            return

        ai_result, mico_result, stock_data = None, None, None
        with st.spinner(f"Running full analysis for {stock_symbol}..."):
            stock_data, ai_result = advisor.run_analysis(stock_symbol, analysis_date, use_market_filter)
            mico_result = st.session_state.mico_advisor.analyze(stock_symbol, analysis_date, params={})
            display_analysis_results(ai_result, mico_result, stock_data, stock_symbol, analysis_date, advisor)

    # --- Optimizer Logic (Now separate and functional) ---
    if optimize_btn:
        with st.spinner(f"Running full system calibration on {opt_symbol}... This may take several minutes."):

            # 1. Define all optimizable advisors
            all_optimizer_advisors = {
                "MichaAdvisor": st.session_state.mico_advisor,
                "SuperTrendAdvisor": st.session_state.supertrend_advisor,
                "MeanReversionAdvisor": st.session_state.mean_reversion_advisor,
                "BreakoutAdvisor": st.session_state.breakout_advisor,
                "MovingAverageCrossoverAdvisor": st.session_state.ma_crossover_advisor,
                "VolumeMomentumAdvisor": st.session_state.volume_momentum_advisor,
            }
            # 2. Define all parameter grids
            all_parameter_grids = {
                "MichaAdvisor": {
                    'sma_short': [20, 50],
                    'sma_long': [100, 200],
                    'rsi_period': [14],
                    'rsi_threshold': [65, 70, 75],
                    'atr_mult_stop': [1.5, 2.0, 2.5],
                    'atr_mult_profit': [1.5, 2.0, 3.0]
                },
                "SuperTrendAdvisor": {'length': [7, 10, 14], 'multiplier': [1.5, 2.0, 2.5, 3.0]},
                "MeanReversionAdvisor": {'bb_length': [20, 30], 'rsi_oversold': [25, 30, 35]},
                "BreakoutAdvisor": {'breakout_window': [20, 30, 50]},
                "MovingAverageCrossoverAdvisor": {'short_window': [20, 50], 'long_window': [100, 200]},
                "VolumeMomentumAdvisor": {'obv_window': [20, 30, 50]}
            }

            # 3. Call the new master function
            mico_optimizer.run_full_optimization(
                optimizer_advisors=all_optimizer_advisors,
                parameter_grids=all_parameter_grids,
                symbol=opt_symbol,
                start_date=opt_start_date,
                end_date=opt_end_date
            )
    # --- Screener Logic (Separate and functional) ---
    if scan_btn:
        st.session_state.analysis_run = False
        if 'screener_results' in st.session_state: del st.session_state['screener_results']

        active_advisors = {}
        if run_ai: active_advisors["AI"] = st.session_state.advisor
        if run_mico: active_advisors["MICO"] = st.session_state.mico_advisor
        if run_mean_reversion: active_advisors["MeanReversion"] = st.session_state.mean_reversion_advisor
        if run_breakout: active_advisors["Breakout"] = st.session_state.breakout_advisor
        if run_supertrend: active_advisors["SuperTrend"] = st.session_state.supertrend_advisor
        if run_ma_crossover: active_advisors["MACrossover"] = st.session_state.ma_crossover_advisor
        if run_volume_momentum: active_advisors["VolumeMomentum"] = st.session_state.volume_momentum_advisor

        if not active_advisors:
            st.warning("Please select at least one system to run for the screener.")
        else:
            load_function = universe_options[selected_universe_name]
            stock_universe = load_function()
            if not stock_universe:
                st.error(f"Could not load the '{selected_universe_name}' stock list.")
            else:
                st.info(f"Scanning {len(stock_universe)} symbols from the '{selected_universe_name}' universe...")
                recommended_trades_df = screener.run_unified_screener(
                    active_advisors=active_advisors, stock_universe=stock_universe,
                    analysis_date=screener_analysis_date, investment_amount=investment_amount,
                    debug_mode=debug_mode, use_optimized_params=use_optimized
                )
                if recommended_trades_df is not None and not recommended_trades_df.empty:
                    st.session_state.screener_results = recommended_trades_df

        if st.session_state.data_manager.use_ibkr and st.session_state.data_manager.isConnected():
            status_placeholder.markdown("<p style='text-align: right;'>🟢 <strong>IBKR</strong></p>",
                                        unsafe_allow_html=True)
        else:
            status_placeholder.markdown("<p style='text-align: right;'>🟡 <strong>YFINANCE</strong></p>",
                                        unsafe_allow_html=True)

    # --- Backtest Results Display Logic (Unchanged) ---
    if 'screener_results' in st.session_state:
        st.markdown("---")
        st.success(f"Screener found {len(st.session_state.screener_results)} trade opportunities.")
        analyze_screener_btn = st.button("🔬 Analyze Screener Results", type="primary", use_container_width=True)
        if analyze_screener_btn: st.session_state.analysis_run = True
        if st.session_state.get('analysis_run', False):
            with st.spinner("Running backtest simulation on results..."):
                results_analyzer.run_backtest(
                    trades_df=st.session_state.screener_results,
                    data_manager=st.session_state.data_manager,
                    investment_amount=investment_amount
                )

    if not analyze_btn and not scan_btn and not optimize_btn:
        st.info("Select an action from the sidebar to begin.")


# --- Main Execution ---
if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="StockWise AI Advisor")

    # Define the default agent to load on the very first run
    DEFAULT_AGENT_MODEL_DIR = "models/NASDAQ-gen3-dynamic"

    # Initialize the DataSourceManager ONCE and store it in the session state
    if 'data_manager' not in st.session_state:
        st.session_state.data_manager = DataSourceManager(use_ibkr=False)
        st.session_state.data_manager.connect_to_ibkr()

    # Initialize the advisor in the session state ONCE if it doesn't exist
    if 'advisor' not in st.session_state:
        st.session_state.advisor = ProfessionalStockAdvisor(
            model_dir=DEFAULT_AGENT_MODEL_DIR,
            data_source_manager=st.session_state.data_manager
        )

    # Initialize the Micha Stock advisor in the session state ONCE if it doesn't exist
    if 'mico_advisor' not in st.session_state:
        # Pass the same data manager to the MicoAdvisor
        st.session_state.mico_advisor = MichaAdvisor(data_manager=st.session_state.data_manager)

    # Initialize the new trading models and store them in the session state
    if 'mean_reversion_advisor' not in st.session_state:
        st.session_state.mean_reversion_advisor = MeanReversionAdvisor(data_manager=st.session_state.data_manager)

    if 'breakout_advisor' not in st.session_state:
        st.session_state.breakout_advisor = BreakoutAdvisor(data_manager=st.session_state.data_manager)

    if 'supertrend_advisor' not in st.session_state:
        st.session_state.supertrend_advisor = SuperTrendAdvisor(data_manager=st.session_state.data_manager)

    if 'ma_crossover_advisor' not in st.session_state:
        st.session_state.ma_crossover_advisor = MovingAverageCrossoverAdvisor(
            data_manager=st.session_state.data_manager)
    if 'volume_momentum_advisor' not in st.session_state:
        st.session_state.volume_momentum_advisor = VolumeMomentumAdvisor(data_manager=st.session_state.data_manager)

    # Check if models were loaded successfully before running the UI
    if st.session_state.advisor.models:
        create_enhanced_interface()
    else:
        st.error(f"FATAL: Default models could not be loaded from '{DEFAULT_AGENT_MODEL_DIR}'.")