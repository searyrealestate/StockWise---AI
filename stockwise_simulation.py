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


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """A single, robust function to clean raw data immediately after fetching."""
    if df is None or df.empty:
        return pd.DataFrame()

    # This handles the case of multi-ticker downloads from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        # The ticker symbol is level 1 of the MultiIndex. Drop it, keep level 0.
        df.columns = df.columns.droplevel(1)

    df.columns = [col.lower() for col in df.columns]

    # Ensure standard OHLCV columns are numeric, coercing errors
    standard_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in standard_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Only drop NaNs from the columns that actually exist in the DataFrame.
    existing_cols = [col for col in standard_cols if col in df.columns]
    if existing_cols:
        df.dropna(subset=existing_cols, inplace=True)

    return df


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

    def _load_gen3_models(self):
        """
        Loads the entire suite of specialist models for Gen-3.
        """
        models = {}
        feature_names = {}
        try:
            model_files = glob.glob(os.path.join(self.model_dir, "*.pkl"))
            if not model_files:
                self.log(f"No models found in {self.model_dir}. Please run the model trainer.", "ERROR")
                return None, None

            for model_path in model_files:
                model_name = os.path.basename(model_path).replace(".pkl", "")
                features_path = model_path.replace(".pkl", "_features.json")
                models[model_name] = joblib.load(model_path)
                with open(features_path, 'r') as f:
                    feature_names[model_name] = json.load(f)

            self.log(f"✅ Successfully loaded {len(models)} specialist models for Gen-3.", "INFO")
            return models, feature_names
        except Exception as e:
            self.log(f"Error loading Gen-3 models: {e}", "ERROR")
            return None, None

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

    def is_market_in_uptrend(self, analysis_date, days=200):
        """
        The Conductor: Checks if the general market (SPY) was in an uptrend
        on a specific historical date.
        """
        try:
            # 1. Fetch the full history for SPY
            spy_data = self.data_source_manager.get_stock_data("SPY")
            if spy_data is None or spy_data.empty:
                self.log("Could not download SPY data for market trend analysis.", "WARNING")
                return False

            spy_data = clean_raw_data(spy_data)

            # 2. Slice the data to only include historical data up to the analysis date
            spy_data_up_to_date = spy_data[spy_data.index <= pd.to_datetime(analysis_date)]

            if len(spy_data_up_to_date) < days:
                self.log(f"Not enough SPY data ({len(spy_data_up_to_date)} bars) for {days}-day MA.", "WARNING")
                return False

            # 3. Calculate the SMA on the historically-sliced data
            spy_data_up_to_date[f'sma_{days}'] = ta.trend.sma_indicator(spy_data_up_to_date['close'], window=days)

            latest_row = spy_data_up_to_date.iloc[-1]
            latest_price = latest_row['close']
            moving_average = latest_row[f'sma_{days}']

            if pd.isna(moving_average):
                self.log("Could not calculate SMA for SPY (result is NaN).", "WARNING")
                return False

            return latest_price > moving_average
        except Exception as e:
            self.log(f"Error during market trend analysis: {e}", "WARNING")
            return False

    # --- GEN-3: The core state machine logic for prediction ---
    def run_analysis(self, ticker_symbol, analysis_date):
        try:
            full_stock_data = self.data_source_manager.get_stock_data(ticker_symbol)
            if full_stock_data is None or full_stock_data.empty:
                return None, None
            full_stock_data = clean_raw_data(full_stock_data)

            # Get the data for the specific date being analyzed
            data_up_to_date = full_stock_data[full_stock_data.index <= pd.to_datetime(analysis_date)]
            if data_up_to_date.empty:
                return full_stock_data, {'action': "WAIT", 'reason': "No data available for this date.",
                                         'current_price': 0, 'agent': "System"}

            price_on_date = data_up_to_date.iloc[-1]['close']

            # Step 1: The Conductor. Assess overall market health.
            # Pass the analysis_date to the market trend function
            if not self.is_market_in_uptrend(analysis_date):
                return full_stock_data, {'action': "WAIT / AVOID", 'confidence': 99.9, 'current_price': price_on_date,
                                         'reason': "Market Downtrend", 'buy_date': None, 'agent': "Market Regime Agent"}

            # Step 2: Feature Engineering
            featured_data = self.calculator.calculate_all_features(data_up_to_date)
            if featured_data.empty:
                return full_stock_data, {'action': "WAIT", 'reason': "Insufficient data for analysis.",
                                         'current_price': price_on_date, 'agent': "System"}

            latest_row = featured_data.iloc[-1]
            cluster = latest_row['volatility_cluster']

            # Step 3: State Check (Is there an open position?)
            if self.position.get(ticker_symbol):
                current_position = self.position[ticker_symbol]
                if latest_row['close'] <= current_position['stop_loss_price']:
                    del self.position[ticker_symbol]
                    return full_stock_data, {'action': "CUT LOSS", 'reason': "Stop-loss hit.",
                                             'current_price': price_on_date, 'agent': "Risk Manager"}

                profit_model_name = f"profit_take_model_{cluster}_vol"
                loss_model_name = f"cut_loss_model_{cluster}_vol"
                profit_model = self.models.get(profit_model_name)
                loss_model = self.models.get(loss_model_name)

                if not profit_model or not loss_model:
                    return full_stock_data, {'action': "HOLD", 'reason': "Missing Models.",
                                             'current_price': price_on_date, 'agent': "System"}

                features = latest_row[self.feature_names[profit_model_name]].astype(float).to_frame().T
                profit_pred, loss_pred = profit_model.predict(features)[0], loss_model.predict(features)[0]

                if loss_pred == 1:
                    del self.position[ticker_symbol]
                    return full_stock_data, {'action': "CUT LOSS",
                                             'confidence': loss_model.predict_proba(features)[0][1],
                                             'current_price': price_on_date,
                                             'agent': f"{cluster.capitalize()}-Volatility Risk Agent"}
                elif profit_pred == 1:
                    del self.position[ticker_symbol]
                    return full_stock_data, {'action': "SELL", 'confidence': profit_model.predict_proba(features)[0][1],
                                             'current_price': price_on_date,
                                             'agent': f"{cluster.capitalize()}-Volatility Profit Agent"}
                else:
                    return full_stock_data, {'action': "HOLD", 'reason': "No exit signal.",
                                             'current_price': price_on_date, 'agent': "System"}

            else:  # State: No Position
                entry_model_name = f"entry_model_{cluster}_vol"
                entry_model = self.models.get(entry_model_name)

                if not entry_model:
                    return full_stock_data, {'action': "WAIT", 'reason': "Missing Models.",
                                             'current_price': price_on_date, 'agent': "System"}

                features = latest_row[self.feature_names[entry_model_name]].astype(float).to_frame().T
                entry_pred = entry_model.predict(features)[0]
                entry_prob = entry_model.predict_proba(features)[0]

                if entry_pred == 1:
                    stop_loss_price = latest_row['close'] - (latest_row['atr_14'] * 2.5)
                    self.position[ticker_symbol] = {'entry_price': latest_row['close'],
                                                    'stop_loss_price': stop_loss_price}
                    return full_stock_data, {'action': "BUY", 'confidence': entry_prob[1] * 100,
                                             'current_price': float(latest_row['close']),
                                             'buy_date': latest_row.name.date(),
                                             'agent': f"{cluster.capitalize()}-Volatility Entry Agent",
                                             'stop_loss_price': stop_loss_price}
                else:
                    return full_stock_data, {'action': "WAIT", 'confidence': entry_prob[0] * 100,
                                             'current_price': float(latest_row['close']),
                                             'agent': f"{cluster.capitalize()}-Volatility Entry Agent"}

        except Exception as e:
            st.code(traceback.format_exc())
            return None, None

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
            go.Bar(x=stock_data.index, y=stock_data['volume'], name='Volume', marker_color='rgba(100,110,120,0.6)'),
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


def create_enhanced_interface():
    st.title("🏢 StockWise AI Trading Advisor")

    AGENT_CONFIGS = {
        'Dynamic Profit (Recommended)': "models/NASDAQ-gen3-dynamic",
        '2% Net Profit': "models/NASDAQ-gen3-2pct",
        '3% Net Profit': "models/NASDAQ-gen3-3pct",
        '4% Net Profit': "models/NASDAQ-gen3-4pct"
    }

    # --- FIX: Define ALL sidebar inputs at the top in the correct order ---
    st.sidebar.header("🎯 Trading Analysis")
    selected_agent_name = st.sidebar.selectbox("🧠 Select AI Agent", options=list(AGENT_CONFIGS.keys()))
    stock_symbol = st.sidebar.text_input("📊 Stock Symbol", value="NVDA").upper().strip()
    analysis_date = st.sidebar.date_input("📅 Analysis Date", value=datetime.now().date())
    analyze_btn = st.sidebar.button("🚀 Run Professional Analysis", type="primary", use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.header("📈 Market Screener")
    # --- Add a dropdown to select the stock universe ---
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

    # Load the correct advisor based on the agent selection
    selected_model_dir = AGENT_CONFIGS[selected_agent_name]
    if st.session_state.advisor.model_dir != selected_model_dir:
        with st.spinner(f"Loading '{selected_agent_name}' agent..."):
            st.session_state.advisor = ProfessionalStockAdvisor(model_dir=selected_model_dir)
    advisor = st.session_state.advisor

    st.markdown(f"### Now using `{selected_agent_name}` Agent")
    st.markdown("---")

    # --- Restore the logic for the scan button ---
    if scan_btn:
        # Dynamically load the selected stock list
        load_function = universe_options[selected_universe_name]
        stock_universe = load_function()

        if not stock_universe:
            # Error messages are handled within the loading functions
            return

        st.subheader(f"BULL SCAN | Top BUY Opportunities in: {selected_universe_name}")
        st.info(f"Scanning {len(stock_universe)} stocks for 'BUY' signals on {analysis_date.strftime('%Y-%m-%d')}...")

        opportunities_df = screener.find_buy_opportunities(advisor, stock_universe, analysis_date)

        if not opportunities_df.empty:
            st.success(f"Scan complete! Found {len(opportunities_df)} potential opportunities.")

    # Logic for the single-stock analysis button
    if analyze_btn:
        if not stock_symbol:
            st.warning("Please enter a stock symbol.")
            return

        with st.spinner(f"Running analysis for {stock_symbol}..."):
            stock_data, result = advisor.run_analysis(stock_symbol, analysis_date)

        if not result:
            st.error("Analysis failed.");
            return

        action = result['action']
        confidence = result.get('confidence', 0)
        current_price = result.get('current_price', 0)
        agent = result.get('agent', "Unknown Agent")

        col1, col2, col3 = st.columns([3, 2, 2])
        with col1:
            if "BUY" in action:
                st.success(f"🟢 **RECOMMENDATION: {action}**")
            elif "SELL" in action or "CUT LOSS" in action:
                st.error(f"🔴 **RECOMMENDATION: {action}**")
            else:
                st.warning(f"🟡 **RECOMMENDATION: {action}**")
        with col2:
            st.info(f"🧠 **Agent**: {agent}")
        with col3:
            st.metric("Model Confidence", f"{confidence:.1f}%")

        st.subheader("💰 Price Information & Analysis")
        price_col, target_buy_col, profit_col, stop_col = st.columns(4)
        price_col.metric("Current Price", f"${current_price:.2f}")

        if "BUY" in action:
            buy_price = result.get('current_price')
            target_buy_col.metric("🎯 Target Buy", f"${buy_price:.2f}")
            stop_loss_price = result.get('stop_loss_price')
            stop_col.metric("🔴 Stop-Loss", f"${stop_loss_price:.2f}")
            profit_target_pct = advisor.calculate_dynamic_profit_target(confidence)
            profit_target_price = buy_price * (1 + profit_target_pct / 100)
            profit_col.metric("✅ Profit Target", f"${profit_target_price:.2f}", f"+{profit_target_pct:.1f}%")

        st.subheader("📊 Price Chart")
        fig = advisor.create_chart(stock_symbol, stock_data, result, analysis_date)
        if fig: st.plotly_chart(fig, use_container_width=True)

        st.subheader("📝 Action Summary")
        if "BUY" in action and result.get('buy_date'):
            hypothetical_investment = 1000
            buy_price = result.get('current_price', 1)
            profit_target_pct = advisor.calculate_dynamic_profit_target(confidence)
            profit_target_price = buy_price * (1 + profit_target_pct / 100)
            hypothetical_shares = hypothetical_investment / buy_price
            gross_profit_dollars = (profit_target_price - buy_price) * hypothetical_shares
            net_profit_dollars, total_deducted = advisor.apply_israeli_fees_and_tax(gross_profit_dollars,
                                                                                    hypothetical_shares)
            net_profit_percentage = (
                                                net_profit_dollars / hypothetical_investment) * 100 if hypothetical_investment > 0 else 0

            # Corrected the corrupted and malformed summary text.
            summary_text = (
                f"- **Action:** The model recommends **BUYING** {stock_symbol} at **${current_price:.2f}**.\n"
                f"- **Agent:** The decision was made by the `{agent}`.\n"
                f"- **Risk Management:** A dynamic stop-loss is suggested at **${result.get('stop_loss_price'):.2f}**.\n"
                f"- **Profit Scenario:** Based on a hypothetical **${hypothetical_investment:,}** investment, "
                f"if the Profit Target of **${profit_target_price:.2f}** is reached, the estimated "
                f"**Net Profit** (after fees & taxes) would be approximately **${net_profit_dollars:.2f}** "
                f"(a **{net_profit_percentage:.2f}%** net return)."
            )
            st.success(summary_text)

        elif "SELL" in action or "CUT LOSS" in action:
            st.error(f"- **Action:** The model recommends **{action}** the position in {stock_symbol}.\n"
                     f"- **Agent:** The exit signal was triggered by the `{agent}`.")
        else:
            st.warning(f"- **Action:** The model recommends to **WAIT or AVOID** buying {stock_symbol}.\n"
                       f"- **Agent:** The decision was made by the `{agent}`.")
    else:
        st.info("Select an action from the sidebar.")

    # Logic for the single-stock analysis button
    # stock_symbol = st.sidebar.text_input("📊 Stock Symbol", value="NVDA").upper().strip()
    # analysis_date = st.sidebar.date_input("📅 Analysis Date", value=datetime.now().date())

    # Define the button only ONCE
    # analyze_btn = st.sidebar.button("🚀 Run Professional Analysis", type="primary", use_container_width=True)

    # if not analyze_btn:
    #     st.info("Enter a stock symbol and date in the sidebar, then click 'Run Analysis' to begin.")
    #     return
    #
    # if not stock_symbol:
    #     st.warning("Please enter a stock symbol.")
    #     return
    #
    # if not advisor.models:
    #     st.error("AI models could not be loaded. Please check the logs.")
    #     return
    #
    # with st.spinner(f"Running analysis for {stock_symbol}..."):
    #     stock_data, result = advisor.run_analysis(stock_symbol, analysis_date)
    #
    # if not result:
    #     st.error("Analysis failed. Please check the debug logs for more information.")
    #     return
    #
    # # --- Display Successful Results ---
    # action = result['action']
    # confidence = result.get('confidence', 0)
    # current_price = result.get('current_price', 0)
    # agent = result.get('agent', "Unknown Agent")
    #
    # col1, col2, col3 = st.columns([3, 2, 2])
    # with col1:
    #     if "BUY" in action:
    #         st.success(f"🟢 **RECOMMENDATION: {action}**")
    #     elif "SELL" in action or "CUT LOSS" in action:
    #         st.error(f"🔴 **RECOMMENDATION: {action}**")
    #     else:
    #         st.warning(f"🟡 **RECOMMENDATION: {action}**")
    # with col2:
    #     st.info(f"🧠 **Agent**: {agent}")
    # with col3:
    #     st.metric("Model Confidence", f"{confidence:.1f}%")
    #
    # st.subheader("💰 Price Information & Analysis")
    # price_col, target_buy_col, profit_col, stop_col = st.columns(4)
    # price_col.metric("Current Price", f"${current_price:.2f}")
    #
    # if "BUY" in action:
    #     buy_price = result.get('current_price')
    #     target_buy_col.metric("🎯 Target Buy", f"${buy_price:.2f}")
    #     stop_loss_price = result.get('stop_loss_price')
    #     stop_col.metric("🔴 Stop-Loss", f"${stop_loss_price:.2f}")
    #
    #     # Calculate and display the profit target price
    #     profit_target_pct = advisor.calculate_dynamic_profit_target(confidence)
    #     profit_target_price = buy_price * (1 + profit_target_pct / 100)
    #     profit_col.metric("✅ Profit Target", f"${profit_target_price:.2f}", f"+{profit_target_pct:.1f}%")
    #
    # st.subheader("📊 Price Chart")
    # fig = advisor.create_chart(stock_symbol, stock_data, result, analysis_date)
    # if fig:
    #     st.plotly_chart(fig, use_container_width=True)
    #
    # st.subheader("📝 Action Summary")
    # if "BUY" in action and result.get('buy_date'):
    #     # Calculate hypothetical net profit for the summary
    #     hypothetical_investment = 1000  # Assume a $1,000 investment
    #     hypothetical_shares = hypothetical_investment / buy_price
    #     gross_profit_dollars = (profit_target_price - buy_price) * hypothetical_shares
    #     net_profit_dollars, total_deducted = advisor.apply_israeli_fees_and_tax(gross_profit_dollars,
    #                                                                             hypothetical_shares)
    #
    #     summary_text = (
    #         f"- **Action:** The model recommends **BUYING** {stock_symbol} at **${current_price:.2f}**.\n"
    #         f"- **Agent:** The decision was made by the `{agent}`.\n"
    #         f"- **Risk Management:** A dynamic stop-loss is suggested at **${result.get('stop_loss_price'):.2f}**.\n"
    #         f"- **Profit Scenario:** Based on a hypothetical **${hypothetical_investment:,}** investment, "
    #         f"if the Profit Target of **${profit_target_price:.2f}** is reached, the estimated "
    #         f"**Net Profit** (after fees & taxes) would be approximately **${net_profit_dollars:.2f} quel to "
    #         f"{profit_target_pct:.1f}%**."
    #     )
    #     st.success(summary_text)
    #
    # elif "SELL" in action or "CUT LOSS" in action:
    #     st.error(f"""
    #         - **Action:** The model recommends **{action}** the position in {stock_symbol}.
    #         - **Agent:** The exit signal was triggered by the `{agent}`.
    #         """)
    # else:
    #     st.warning(f"""
    #         - **Action:** The model recommends to **WAIT or AVOID** buying {stock_symbol}.
    #         - **Agent:** The decision was made by the `{agent}`.
    #         """)


# --- Main Execution ---
if __name__ == "__main__":
    # Define the default agent to load on the very first run
    DEFAULT_AGENT_MODEL_DIR = "models/NASDAQ-gen3-dynamic"

    # Initialize the advisor in the session state ONCE if it doesn't exist
    if 'advisor' not in st.session_state:
        st.session_state.advisor = ProfessionalStockAdvisor(model_dir=DEFAULT_AGENT_MODEL_DIR)

    # Check if models were loaded successfully before running the UI
    if st.session_state.advisor.models:
        create_enhanced_interface()
    else:
        st.error(f"FATAL: Default models could not be loaded from '{DEFAULT_AGENT_MODEL_DIR}'.")