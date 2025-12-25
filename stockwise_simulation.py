# stockwise_simulation.py
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
import stockwise_scanner
from logging_setup import setup_json_logging
from google.cloud import storage
from google.oauth2 import service_account
import streamlit as st
import streamlit_authenticator as stauth
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
import yaml
from yaml.loader import SafeLoader
# from realtime_data_feed import RealTimeDataFeed
from risk_manager import RiskManager
import chart_pattern_recognizer
from mico_ai_module import AI_ParamPredictor
from technical_analyzer import TechnicalAnalyzer
import system_config as cfg
from feature_engine import RobustFeatureCalculator
from strategy_engine import MarketRegimeDetector, StrategyOrchestra
from notification_manager import NotificationManager


# --- GLOBAL IMPORT FOR ADAPTIVE LEARNER ---
try:
    from adaptive_learning_engine import AdaptiveLearner
except ImportError:
    AdaptiveLearner = None

# --- Page Configuration ---
st.set_page_config(
    page_title="StockWise AI Trading Advisor",
    page_icon="🏢",
    layout="wide"
)

# todo: add function that delete old LOGFILE lines after X Mb

logger = logging.getLogger(__name__)


def calculate_net_pnl_raw(gross_profit_dollars, num_shares, entry_price, exit_price):
    """
    Calculates Net PnL using IBKR Tiered Pricing Logic.
    Logic: Fee = (Shares * Rate), subject to Min($0.35) and Max(1% of Trade Value).
    """

    def get_leg_fee(price, shares):
        trade_value = price * shares
        # 1. Base: Greater of Min Fee or Per Share Rate
        base_fee = max(cfg.MINIMUM_FEE, shares * cfg.FEE_PER_SHARE)
        # 2. Cap: Lesser of Base Fee or 1% of Value
        final_fee = min(base_fee, trade_value * cfg.MAX_FEE_PCT)
        return final_fee

    # 1. Calculate Commissions
    entry_fee = get_leg_fee(entry_price, num_shares)
    exit_fee = get_leg_fee(exit_price, num_shares)
    total_fees = entry_fee + exit_fee

    # 2. Net PnL
    profit_after_fees = gross_profit_dollars - total_fees

    # 3. Tax (Only on positive profit)
    tax = (profit_after_fees * cfg.TAX_RATE) if profit_after_fees > 0 else 0

    net_profit = profit_after_fees - tax

    return net_profit, total_fees + tax


def setup_logging_from_st():
    """
    Called by Streamlit to set the global log level.
    Reads the value from st.session_state AFTER it has been updated.
    """
    # Read the key from the checkbox
    is_debug = st.session_state.get('debug_logging_enabled', False)
    level = "DEBUG" if is_debug else "INFO"

    log_level = logging.DEBUG if is_debug else logging.INFO
    setup_json_logging(log_level)
    logger = logging.getLogger(__name__)
    logger.info(f"Log level set to {level}")


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
        sp500_df = table[1]

        # --- Flatten the MultiIndex header first ---
        if isinstance(sp500_df.columns, pd.MultiIndex):
            sp500_df.columns = sp500_df.columns.get_level_values(-1)

        # NOW, standardize the flattened string columns (adding str() for safety)
        sp500_df.columns = [str(col).strip().title() for col in sp500_df.columns]

        # Check for the correct column name
        if 'Symbol' in sp500_df.columns:
            tickers = sp500_df['Symbol'].tolist()
        elif 'Ticker' in sp500_df.columns:
            tickers = sp500_df['Ticker'].tolist()
        else:
            logger.error("Could not find 'Symbol' or 'Ticker' column in S&P 500 list.", exc_info=True)
            raise KeyError("Could not find 'Symbol' or 'Ticker' column in S&P 500 list.")

        # Clean up tickers for yfinance compatibility (e.g., 'BRK.B' -> 'BRK-B')
        # tickers = [ticker.replace('.', '-') for ticker in tickers]
        st.session_state.sp500_tickers = tickers
        return tickers
    except Exception as e:
        logger.error(f"Failed to fetch S&P 500 list: {e}", exc_info=True)
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
        # 1. Flatten the MultiIndex header, in case it exists
        if isinstance(nasdaq100_df.columns, pd.MultiIndex):
            nasdaq100_df.columns = nasdaq100_df.columns.get_level_values(-1)

        # 2. Standardize column names (convert to string, strip, and title case)
        nasdaq100_df.columns = [str(col).strip().title() for col in nasdaq100_df.columns]

        # 3. Find the correct ticker column
        if 'Ticker' in nasdaq100_df.columns:
            tickers = nasdaq100_df['Ticker'].tolist()
        elif 'Symbol' in nasdaq100_df.columns:
            tickers = nasdaq100_df['Symbol'].tolist()
        else:
            logger.error("Could not find 'Ticker' or 'Symbol' column in NASDAQ 100 list.", exc_info=True)
            raise KeyError("Could not find 'Ticker' or 'Symbol' column in NASDAQ 100 list.")

        st.session_state.nasdaq100_tickers = tickers
        return tickers

    except Exception as e:
        logger.error(f"Failed to fetch NASDAQ 100 list: {e}", exc_info=True)
        st.error(f"Failed to fetch NASDAQ 100 list: {e}")
        return []


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_contextual_data(_data_manager):
    """Downloads QQQ, VIX, and TLT data once."""
    try:
        qqq_data = yf.download("QQQ", period="max", progress=False, auto_adjust=True)
        if isinstance(qqq_data.columns, pd.MultiIndex):
            qqq_data.columns = qqq_data.columns.droplevel(1)
        qqq_data.columns = [col.lower() for col in qqq_data.columns]

        vix_raw = _data_manager.get_stock_data('^VIX')
        vix_clean = clean_raw_data(vix_raw)

        tlt_raw = _data_manager.get_stock_data('TLT')
        tlt_clean = clean_raw_data(tlt_raw)

        return {
            'qqq': qqq_data,
            'vix': vix_clean,
            'tlt': tlt_clean
        }
    except Exception as e:
        logger.error(f"Failed to load contextual data: {e}", exc_info=True)
    return {}


@st.cache_data
def load_nasdaq_tickers(max_stocks=None):
    """Loads NASDAQ ticker symbols from a CSV file."""
    csv_path = "nasdaq_stocks.csv"
    try:
        df = pd.read_csv(csv_path)
        # Standardize column names for robustness
        df.columns = [col.strip().title() for col in df.columns]
        if 'Symbol' not in df.columns:
            # --- Use the module-level logger ---
            logger.error(f"FATAL: '{csv_path}' is missing the required 'Symbol' column.")
            st.error(f"FATAL: '{csv_path}' is missing the required 'Symbol' column.")
            return []
        # This new filter ONLY removes indices (like ^NDX) but KEEPS stocks with a '.'
        tickers = df[~df['Symbol'].str.contains(r'\^', na=True)]['Symbol'].dropna().tolist()
        return tickers
    except FileNotFoundError as e:
        # --- Use the module-level logger ---
        logger.error(f"FATAL: 'nasdaq_stocks.csv' not found. {e}", exc_info=True)
        st.error(f"FATAL: 'nasdaq_stocks.csv' not found.")
        return []
    except Exception as e:
        # ---  Use the module-level logger ---
        logger.error(f"Error loading nasdaq_tickers: {e}", exc_info=True)
        st.error(f"Error loading nasdaq_tickers: {e}")
        return []


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

    def __init__(self, data_manager, contextual_data, is_cloud):
        self.qqq_data = contextual_data.get('qqq', pd.DataFrame())
        self.vix_data = contextual_data.get('vix', pd.DataFrame())
        self.tlt_data = contextual_data.get('tlt', pd.DataFrame())
        self.is_cloud = is_cloud
        # 2. ADD this line to store the logger
        self.log = logger if logger else lambda msg, level="INFO": None

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

        self.log.info(f"Initial columns: {df.columns.tolist()}")

        try:
            # Standardize column names to lowercase
            df.columns = [col.lower() for col in df.columns]

            # --- 1. Pandas-TA Features ---

            # These are single-column or correctly-named indicators
            df.ta.bbands(length=20, append=True,
                         col_names=('bb_lower', 'bb_middle', 'bb_upper', 'bb_width', 'bb_position'))
            df.ta.atr(length=14, append=True, col_names='atr_14')
            df.ta.rsi(length=14, append=True, col_names='rsi_14')
            df.ta.rsi(length=28, append=True, col_names='rsi_28')
            df.ta.mom(length=5, append=True, col_names='momentum_5')
            df.ta.obv(append=True, col_names='obv')
            df.ta.cmf(append=True, col_names='cmf')

            # --- Run multi-column indicators with default names ---
            # The 'col_names' argument is ignored by pandas-ta for these
            df.ta.macd(append=True)
            df.ta.adx(length=14, append=True)

            # Add a final lowercase conversion
            df.columns = [col.lower() for col in df.columns]

            # --- Explicitly rename the columns pandas-ta created ---
            # Find the default names (now lowercase) and rename them to what the model expects
            rename_map = {
                'macd_12_26_9': 'macd',
                'macdh_12_26_9': 'macd_histogram',
                'macds_12_26_9': 'macd_signal',
                'adx_14': 'adx',
                'dmp_14': 'adx_pos',
                'dmn_14': 'adx_neg'
            }
            # Check which columns exist before renaming to avoid errors
            actual_rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
            df.rename(columns=actual_rename_map, inplace=True)
            self.log.debug(f"FeatureCalculator: Renamed columns: {list(actual_rename_map.keys())}")

            # --- 2. Manual Features (NOW PROTECTED) ---
            df['daily_return'] = df['close'].pct_change()
            df['volume_ma_20'] = df['volume'].rolling(20).mean()
            df['volatility_20d'] = df['daily_return'].rolling(20).std()

            # --- 3. NEW: "Inside the Candle" & Micro-Structure Features ---

            # A. IBS (Internal Bar Strength): (Close - Low) / (High - Low)
            # High values (>0.8) mean strong intraday buying (closing near high)
            # We use 1e-9 to avoid division by zero errors on flat days
            df['ibs'] = (df['close'] - df['low']) / ((df['high'] - df['low']) + 1e-9)

            # B. Wick Rejection (Lower Wick Size relative to total range)
            # (Min(Open, Close) - Low) / (High - Low)
            # High values indicate bears tried to push down but failed.
            df['lower_wick_ratio'] = (df[['open', 'close']].min(axis=1) - df['low']) / ((df['high'] - df['low']) + 1e-9)

            # --- 4. Contextual Features  ---
            try:
                if not self.qqq_data.empty:
                    # Align QQQ data
                    qqq_aligned = self.qqq_data['close'].reindex(df.index, method='ffill')

                    # Handle missing QQQ data (NaNs) by filling with 0 change
                    qqq_aligned = qqq_aligned.bfill().ffill()

                    # Existing Correlation
                    df['correlation_50d_qqq'] = df['close'].rolling(50).corr(qqq_aligned)

                    # Relative Strength (Daily Alpha)
                    # Did the stock beat the market today?
                    qqq_daily_return = qqq_aligned.pct_change().fillna(0)
                    df['rel_strength_qqq'] = df['daily_return'] - qqq_daily_return
                else:
                    df['correlation_50d_qqq'] = 0.0
                    df['rel_strength_qqq'] = 0.0  # Default neutral
            except Exception as e:
                    self.log.warning(f"QQQ Data Processing Error: {e}")
                    df['correlation_50d_qqq'] = 0.0
                    df['rel_strength_qqq'] = 0.0

            # Add safety check for 'bb_middle'
            if 'bb_middle' in df.columns:
                df['z_score_20'] = (df['close'] - df['bb_middle']) / df['close'].rolling(20).std()
            else:
                df['z_score_20'] = 0
                self.log.warning("Could not calculate z_score_20: 'bb_middle' column missing.")

            df['kama_10'] = calculate_kama(df['close'], window=10)
            df['stoch_k'], df['stoch_d'] = calculate_stochastic(df['high'], df['low'], df['close'])
            df['dominant_cycle'] = df['close'].rolling(window=252, min_periods=90).apply(self.get_dominant_cycle,
                                                                                         raw=False)

            # --- 3. Technical Analyzer Features (for AI Learning) ---
            analyzer = TechnicalAnalyzer(df, self.log)

            # Note: We assign the binary results directly to the feature columns.
            # We use .shift(1) because the check_ methods use the latest row's data
            # which is the analysis date. We want the features to be available
            # *on* that analysis date, not the next day.

            # Calculate patterns and set the binary flags (0 or 1)
            # The analyzer is designed to be stateless for these checks

            df['bullish_candlestick'] = df.apply(
                lambda row: 1 if analyzer.check_bullish_candlestick() else 0, axis=1).shift(1).fillna(0)

            df['continuation_candlestick'] = df.apply(
                lambda row: 1 if analyzer.check_continuation_candlestick() else 0,axis=1).shift(1).fillna(0)

            df['volume_breakout_flag'] = df.apply(
                lambda row:1 if analyzer.check_volume_breakout() else 0, axis=1).shift(1).fillna(0)

            self.log.debug(
                f"Technical flags added: Bullish={df['bullish_candlestick'].iloc[-1]}, "
                f"Volume={df['volume_breakout_flag'].iloc[-1]}")

            # --- 4. Contextual (External) Features (NOW PROTECTED) ---
            try:
                if not self.qqq_data.empty:
                    qqq_close = self.qqq_data['close'].reindex(df.index, method='ffill')
                    df['correlation_50d_qqq'] = df['close'].rolling(50).corr(qqq_close)
                else:
                    self.log.warning("Pre-loaded QQQ data is empty.")
                    df['correlation_50d_qqq'] = 0.0
            except Exception as e:
                self.log.error(f"An exception occurred during QQQ data processing: {e}", exc_info=True)
                df['correlation_50d_qqq'] = 0.0

            try:
                # VIX Data
                if not self.vix_data.empty:
                    aligned_vix = self.vix_data['close'].reindex(df.index, method='ffill')
                    df['vix_close'] = aligned_vix
                else:
                    self.log.warning("Pre-loaded VIX data is empty.")
                    df['vix_close'] = 0.0
            except Exception as e:
                self.log.error(f"VIX data processing error: {e}", exc_info=True)
                df['vix_close'] = 0.0

            try:
                # TLT Data
                if not self.tlt_data.empty:
                    aligned_tlt = self.tlt_data['close'].reindex(df.index, method='ffill')
                    df['corr_tlt'] = df['close'].rolling(50).corr(aligned_tlt)
                else:
                    self.log.warning("Pre-loaded TLT data is empty.")
                    df['corr_tlt'] = 0.0
            except Exception as e:
                self.log.error(f"TLT data processing error: {e}", exc_info=True)
                df['corr_tlt'] = 0.0

            # --- 4. Final Processing (NOW PROTECTED) ---
            df['volatility_90d'] = df['daily_return'].rolling(90).std()
            low_thresh, high_thresh = 0.015, 0.030
            df['volatility_cluster'] = pd.cut(df['volatility_90d'], bins=[-np.inf, low_thresh, high_thresh, np.inf],
                                              labels=['low', 'mid', 'high'])

            # Fill remaining NaNs from rolling windows using Backfill first
            df.bfill(inplace=True)
            df.ffill(inplace=True)

            return df

        except Exception as e:
            self.log.error(f"Feature calculation FAILED. Error: {e}. Skipping this stock.", exc_info=True)
            st.exception(e)  # This prints the full error traceback
            return pd.DataFrame()  # Return an empty frame so the screener can continue


# --- Main Application Class (with Gen-3 Architecture) ---
class ProfessionalStockAdvisor:
    def __init__(self, model_dir: str, data_source_manager=None, debug=False, testing_mode=False, download_log=False,
                 calculator=None):
        self.log = logging.getLogger(type(self).__name__)
        self.log.info("Application Initializing...")
        self.debug = debug
        self.model_dir = model_dir
        self.download_log = download_log
        self.testing_mode = testing_mode
        self.calculator = RobustFeatureCalculator(params=cfg.STRATEGY_PARAMS)

        if data_source_manager:
            self.log.info("Using provided data source manager.")
            self.data_source_manager = data_source_manager
        else:
            self.log.info("Initializing data source manager.")
            self.data_source_manager = DataSourceManager(use_ibkr=False)

        self.models, self.feature_names = self._load_gen3_models_v2()

        # Initialize Adaptive Learner ---
        try:
            from adaptive_learning_engine import AdaptiveLearner
            self.learner = AdaptiveLearner()
        except ImportError:
            self.log.warning("AdaptiveLearner not found. Using default static threshold.")
            self.learner = None

        self.tax = 0.25
        self.broker_fee = 0.004
        # self.position = {}
        self.model_version_info = f"Gen-3: {os.path.basename(model_dir)}"

        # --- Initialize Adaptive Learner (Corrected Logic) ---
        # Use the global variable defined at the top of the file
        if AdaptiveLearner:
            self.learner = AdaptiveLearner()
        else:
            self.log.warning("AdaptiveLearner module not found. Using default static logic.")
            self.learner = None

        # --- The Weighted Scoring Engine (Moved from Debugger to Core) ---
        def calculate_composite_score(self, features, ai_action):
            score = 0
            reasons = []

            # Extract Features
            ibs = features.get('ibs', 0.5)
            rel_strength = features.get('rel_strength_qqq', 0)
            macd_hist = features.get('macd_histogram', 0)
            daily_ret = features.get('daily_return', 0)
            has_bullish_candle = features.get('bullish_candlestick', 0) == 1
            has_vol_breakout = features.get('volume_breakout_flag', 0) == 1
            is_green_candle = daily_ret > 0

            # 1. Base Score
            if ai_action == 'BUY':
                score += 40
                reasons.append("AI Model Signal (+40)")

            # 2. Super Technicals
            if rel_strength > 0.03:
                score += 35;
                reasons.append("Institutional Alpha (+35)")
            elif rel_strength > 0.01:
                score += 20;
                reasons.append("Strong Alpha (+20)")
            elif rel_strength > 0:
                score += 10;
                reasons.append("Positive Alpha (+10)")

            if ibs > 0.95:
                score += 20;
                reasons.append("Dead High Close (+20)")
            elif ibs > 0.8:
                score += 15;
                reasons.append("Strong Close (+15)")

            if has_vol_breakout:
                if is_green_candle:
                    score += 15;
                    reasons.append("Buying Volume (+15)")
                else:
                    score -= 15;
                    reasons.append("Selling Volume (-15)")

            if has_bullish_candle:
                score += 10;
                reasons.append("Bullish Pattern (+10)")

            if macd_hist > 0:
                score += 10;
                reasons.append("Pos Momentum (+10)")

            # 3. Penalties
            if macd_hist < -1.5:
                score -= 40;
                reasons.append("Falling Knife Risk (-40)")
            if daily_ret < -0.03:
                score -= 50;
                reasons.append("Crash Protection (-50)")

            return score, reasons

    def adjust_trailing_stop_ratchet(self, entry_price, current_price, current_stop_loss, atr_value):
        """
        Implements the 'Ratchet' logic: Adjusts the stop-loss up based on the
        current price, but never moves the stop down. Uses the ATR Trailing Multiplier
        from system_config.py.
        """
        # 1. Calculate the new potential stop level
        trailing_distance = atr_value * cfg.TRAILLING_STOP_ATR
        new_potential_stop = current_price - trailing_distance

        # 2. The Ratchet: The new stop is the maximum of the *current* stop and the *new potential* stop.
        # This locks in profits by preventing the stop from moving down.
        # It ensures the stop is at least at the current_stop_loss (which includes the original entry break-even point).
        ratcheted_stop = max(current_stop_loss, new_potential_stop)

        # 3. Safety Check: Ensure the stop is not above the current price (should be slightly below)
        final_stop = min(ratcheted_stop, current_price * 0.999)

        return final_stop

    # Smart Stop Calculation ---
    def calculate_smart_stop(self, df, entry_date, lookback=10):
        """Finds Swing Low in the last N days before entry."""
        try:
            history = df[df.index < pd.to_datetime(entry_date)].tail(lookback)
            if history.empty: return None
            return history['low'].min() * 0.995
        except:
            return None

    # Weighted Scoring Logic ---
    def calculate_composite_score(self, features, ai_action):
        score = 0
        reasons = []

        # Extract Features
        ibs = features.get('ibs', 0.5)
        rel_strength = features.get('rel_strength_qqq', 0)
        macd_hist = features.get('macd_histogram', 0)
        daily_ret = features.get('daily_return', 0)
        has_bullish_candle = features.get('bullish_candlestick', 0) == 1
        has_vol_breakout = features.get('volume_breakout_flag', 0) == 1
        is_green_candle = daily_ret > 0

        # 1. Base Score
        if ai_action == 'BUY':
            score += 40;
            reasons.append("AI Signal")

        # 2. Super Technicals
        if rel_strength > 0.03:
            score += 35; reasons.append("Inst. Alpha")
        elif rel_strength > 0.01:
            score += 20; reasons.append("Strong Alpha")
        elif rel_strength > 0:
            score += 10; reasons.append("Pos Alpha")

        if ibs > 0.95:
            score += 20; reasons.append("Dead High Close")
        elif ibs > 0.8:
            score += 15; reasons.append("Strong Close")

        if has_vol_breakout:
            if is_green_candle:
                score += 15; reasons.append("Buy Vol")
            else:
                score -= 15; reasons.append("Sell Vol")

        if has_bullish_candle: score += 10; reasons.append("Bull Pattern")
        if macd_hist > 0: score += 10; reasons.append("Pos Mom")

        # # 3. Penalties
        # if macd_hist < -1.5: score -= 40; reasons.append("Falling Knife")
        # if daily_ret < -0.03: score -= 50; reasons.append("Crash Prot")

        return score, reasons

    @st.cache_resource(ttl=3600)
    def _download_and_load_models(_self, bucket_name="stockwise-gen3-models-public"):
        _self.log.info("Attempting to load models...")
        models = {}
        feature_names = {}
        try:
            creds_json = st.secrets["gcs_service_account"]
            credentials = service_account.Credentials.from_service_account_info(creds_json)
            storage_client = storage.Client(credentials=credentials)
            bucket = storage_client.bucket(bucket_name)
            model_files_blob = list(bucket.list_blobs(prefix=f"{_self.model_dir}/"))
            if not model_files_blob:
                _self.log.error(f"No models found in GCS at gs://{bucket_name}/{_self.model_dir}.")
                st.error(f"GCS Error: No models found in bucket for path '{_self.model_dir}'.")
                return None, None
            for blob in model_files_blob:
                if blob.name.endswith(".pkl"):
                    model_name = os.path.basename(blob.name).replace(".pkl", "")
                    features_path = blob.name.replace(".pkl", "_features.json")
                    model_bytes = blob.download_as_bytes()
                    models[model_name] = joblib.load(BytesIO(model_bytes))
                    features_blob = bucket.blob(features_path)
                    if features_blob.exists():
                        features_bytes = features_blob.download_as_bytes()
                        feature_names[model_name] = json.loads(features_bytes.decode('utf-8'))
                    else:
                        _self.log.warning(f"Missing feature file: {features_path}")
            if not models:
                _self.log.error("Model loading failed. No .pkl files found.")
                return None, None
            _self.log.info(f"✅ Successfully loaded {len(models)} specialist models from GCS.")
            return models, feature_names
        except Exception as e:
            _self.log.error(f"❌ FATAL: Failed to download/load models: {e}", exc_info=True)
            st.error(f"FATAL: Could not load models. Check secrets. {e}")
            return None, None

    def _load_models_from_disk(self):
        models = {}
        feature_names = {}
        try:
            model_files = glob.glob(os.path.join(self.model_dir, "*.pkl"))
            if not model_files:
                self.log.error(f"No local models found in {self.model_dir}.")
                return models, feature_names
            for model_path in model_files:
                model_name = os.path.basename(model_path).replace(".pkl", "")
                features_path = model_path.replace(".pkl", "_features.json")
                models[model_name] = joblib.load(model_path)
                with open(features_path, 'r') as f:
                    feature_names[model_name] = json.load(f)
            self.log.info(f"✅ Successfully loaded {len(models)} models from local disk.")
            return models, feature_names
        except Exception as e:
            self.log.error(f"Error loading local models: {e}", exc_info=True)
            return models, feature_names

    @st.cache_resource(ttl=3600)
    def _load_gen3_models_v2(_self):
        """
        Smart model loader:
        Tries to load from GCS (for Streamlit Cloud).
        If it fails, it falls back to loading from the local disk.
        """
        try:
            if 'IS_CLOUD' not in st.session_state or not st.session_state.IS_CLOUD:
                # This is a local run. Raise an exception to jump to the 'except' block.
                raise Exception("Local run detected (IS_CLOUD=False). Skipping GCS.")

            _self.log.info("Attempting to load models from GCS (Cloud Mode)...")
            creds_json = st.secrets["gcs_service_account"]
            credentials = service_account.Credentials.from_service_account_info(creds_json)
            storage_client = storage.Client(credentials=credentials)

            # *** bucket name ***
            bucket_name = "stockwise-gen3-models-public"
            bucket = storage_client.bucket(bucket_name)
            models = {}
            feature_names = {}
            gcs_path = f"StockWise/{_self.model_dir}/"
            blobs = list(bucket.list_blobs(prefix=gcs_path))
            if not blobs:
                _self.log.error(f"No models found in GCS at gs://{bucket.name}/{_self.model_dir}/")
                return models, feature_names
            for blob in blobs:
                if blob.name.endswith(".pkl"):
                    model_name = os.path.basename(blob.name).replace(".pkl", "")
                    features_path = blob.name.replace(".pkl", "_features.json")

                    model_bytes = blob.download_as_bytes()
                    models[model_name] = joblib.load(BytesIO(model_bytes))

                    features_blob = bucket.blob(features_path)
                    if features_blob.exists():
                        features_bytes = features_blob.download_as_bytes()
                        feature_names[model_name] = json.loads(features_bytes.decode('utf-8'))
                    else:
                        _self.log.warning(f"Missing feature file: {features_path}")
            _self.log.info(f"✅ Successfully loaded {len(models)} models from GCS.")
            return models, feature_names

        except Exception as e:
            _self.log.warning(f"GCS load failed (Error: {e}). Loading from disk...", exc_info=True)
            return _self._load_models_from_disk()

    def validate_symbol_professional(self, symbol):
        self.log.info(f"Using yfinance for validation of {symbol}")
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
            spy_data_raw = self.data_source_manager.get_stock_data("SPY", days_back=300, end_date=analysis_date)
            spy_data = clean_raw_data(spy_data_raw)
            spy_data_slice = spy_data[spy_data.index <= pd.to_datetime(analysis_date)]
            if len(spy_data_slice) < 200: return 0, ["Not enough SPY data."]

            spy_data_slice.ta.sma(length=50, append=True, col_names='sma_50')
            spy_data_slice.ta.sma(length=200, append=True, col_names='sma_200')
            spy_data_slice.ta.rsi(length=14, append=True, col_names='rsi_14')
            latest_spy = spy_data_slice.iloc[-1]
            vix_data_raw = self.data_source_manager.get_stock_data("^VIX", days_back=5, end_date=analysis_date)
            vix_data = clean_raw_data(vix_data_raw)
            vix_slice = vix_data[vix_data.index <= pd.to_datetime(analysis_date)]
            latest_vix = vix_slice.iloc[-1] if not vix_slice.empty else None
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
            self.log.error(f"Error in market health check: {e}", "ERROR", exc_info=True)
            return 0, [f"Error: {e}"]

    # def run_analysis(self, full_stock_data, ticker_symbol, analysis_date, use_market_filter=True):
    #
    #     debug = False
    #     try:
    #         # --- CLEAR POSITION STATE FOR SINGLE-RUN ANALYSIS ---
    #         # This ensures the function always starts by checking for a new entry,
    #         # instead of assuming a lingering position from the previous button click.
    #         # self.position = {}
    #         if debug:
    #             st.write("--- `run_analysis` started. ---")
    #             logger.info(f"--- `run_analysis` started. ---")
    #             logger.info(f"function name: {sys._getframe().f_code.co_name}")
    #
    #         # DATA IS NOW PASSED IN - NO NEED TO RE-FETCH
    #         if full_stock_data is None or full_stock_data.empty:
    #             logger.error(f"No data provided for symbol {ticker_symbol}.")
    #             return pd.DataFrame(), {'action': "WAIT", 'reason': f"No data provided for symbol {ticker_symbol}."}
    #
    #         # We still clean it, as this function is also used by the single-stock analyzer
    #         full_stock_data = clean_raw_data(full_stock_data)
    #
    #         # --- DEBUG 1: Log incoming data ---
    #         logger.info(
    #             f"[{ticker_symbol}]: AI Advisor received data shape: {full_stock_data.shape}. "
    #             f"Date range: {full_stock_data.index.min()} to {full_stock_data.index.max()}")
    #
    #         if pd.api.types.is_datetime64_any_dtype(full_stock_data.index) and full_stock_data.index.tz is not None:
    #             full_stock_data.index = full_stock_data.index.tz_localize(None)
    #             self.log.debug(f"[{ticker_symbol}]: Converted data index to tz-naive.")
    #
    #         data_up_to_date = full_stock_data[full_stock_data.index <= pd.to_datetime(analysis_date)]
    #         if data_up_to_date.empty:
    #             return full_stock_data, {'action': "WAIT", 'reason': "No data available for this date.",
    #                                      'current_price': 0, 'agent': "System"}
    #
    #         # --- DEBUG 2: Log sliced data shape ---
    #         self.log.debug(
    #             f"[{ticker_symbol}]: Data sliced to {analysis_date}. Shape is now: {data_up_to_date.shape}.")
    #
    #         price_on_date = data_up_to_date.iloc[-1]['close']
    #         if use_market_filter:
    #             health_score, health_reasons = self.get_market_health_index(analysis_date)
    #             market_health_results = {'health_score': health_score, 'health_reasons': health_reasons}
    #             # --- DEBUG 3: Log Market Health Filter (THE LIKELY CULPRIT) ---
    #             self.log.debug(
    #                 f"[{ticker_symbol}]: Market filter is ON. Score: {health_score}/4 on {analysis_date}. Reasons: {health_reasons}")
    #
    #             if debug: st.write(f"--- Market Health Check complete. Score: {health_score}/4 ---")
    #             if health_score < 3:
    #                 if debug: st.write("--- Market Health FAILED. Returning WAIT/AVOID. ---")
    #                 return full_stock_data, {**market_health_results, 'action': "WAIT / AVOID", 'confidence': 99.9,
    #                                          'current_price': price_on_date,
    #                                          'reason': f"Market Health Index: {health_score}/4. Conditions not met.",
    #                                          'buy_date': None, 'agent': "Market Regime Agent"}
    #         else:
    #             market_health_results = {'health_score': 'N/A',
    #                                      'health_reasons': ["Market Health Filter was manually disabled."]}
    #             self.log.warning("Market Health Filter was disabled by user.")
    #         if debug: st.write("--- Market Health PASSED. Proceeding to feature engineering. ---")
    #         featured_data = self.calculator.calculate_all_features(data_up_to_date)
    #
    #         # --- DEBUG 4: Log feature data shape ---
    #         self.log.debug(
    #             f"[{ticker_symbol}]: Feature calculation complete. "
    #             f"Final data shape (after dropna): {featured_data.shape}")
    #
    #         if featured_data.empty:
    #             self.log.error(f"[{ticker_symbol}]: Feature calculation returned empty DataFrame.")
    #             return full_stock_data, {**market_health_results, 'action': "WAIT",
    #                                      'reason': "Insufficient data for analysis.", 'current_price': price_on_date,
    #                                      'agent': "System"}
    #         latest_row = featured_data.iloc[-1]
    #         all_features_dict = latest_row.to_dict()
    #         cluster = latest_row['volatility_cluster']
    #
    #         # --- NEW DEBUG LOGS TO FIND THE KEYERROR ---
    #         entry_model_name = f"entry_model_{cluster}_vol"
    #
    #         # Log all columns that are ACTUALLY in the DataFrame
    #         logger.debug(
    #             f"[{ticker_symbol}]: All columns available (from FeatureCalculator): {latest_row.index.tolist()}")
    #
    #         # Log all features the model is REQUESTING
    #         if entry_model_name in self.feature_names:
    #             logger.debug(
    #                 f"[{ticker_symbol}]: Model '{entry_model_name}' is REQUESTING features: {self.feature_names[entry_model_name]}")
    #         else:
    #             logger.error(f"[{ticker_symbol}]: Model name '{entry_model_name}' not found in self.feature_names!")
    #
    #         # def get_shap_explanation(model, features_df):
    #         #     explainer = shap.TreeExplainer(model)
    #         #     shap_values = explainer.shap_values(features_df)
    #         #     if isinstance(shap_values, list) and len(shap_values) == 2:
    #         #         logger.info(f"SHAP values for {ticker_symbol}: {shap_values[1]}")
    #         #         return shap_values[1], explainer.expected_value[1]
    #         #     else:
    #         #         logger.info(f"SHAP values for {ticker_symbol}: {shap_values}")
    #         #         return shap_values, explainer.expected_value
    #
    #         # if self.position.get(ticker_symbol):  # State: Position is OPEN
    #         #     # --- THIS 'HOLD' LOGIC IS UNCHANGED ---
    #         #     action = "HOLD"
    #         #     confidence = 0
    #         #     agent_name = f"{cluster.capitalize()}-Volatility Hold Agent"
    #         #     profit_model_name = f"profit_take_model_{cluster}_vol"
    #         #     loss_model_name = f"cut_loss_model_{cluster}_vol"
    #         #     logger.info(f"[{ticker_symbol}]: Position is OPEN. Model names: {profit_model_name}, "
    #         #                 f"{loss_model_name}, Action: {action}")
    #         #     if profit_model_name in self.models and loss_model_name in self.models:
    #         #         profit_model = self.models[profit_model_name]
    #         #         loss_model = self.models[loss_model_name]
    #         #         features = latest_row[self.feature_names[loss_model_name]].astype(float).to_frame().T
    #         #         if loss_model.predict(features)[0] == 1:
    #         #             action = "CUT LOSS"
    #         #             confidence = loss_model.predict_proba(features)[0][1] * 100
    #         #             agent_name = f"{cluster.capitalize()}-Volatility Cut-Loss Agent"
    #         #             logger.info(f"[{ticker_symbol}]: Cut Loss Triggered. Action: {action}")
    #         #             del self.position[ticker_symbol]
    #         #         elif profit_model.predict(features)[0] == 1:
    #         #             action = "SELL"
    #         #             confidence = profit_model.predict_proba(features)[0][1] * 100
    #         #             agent_name = f"{cluster.capitalize()}-Volatility Profit-Take Agent"
    #         #             logger.info(f"[{ticker_symbol}]: Profit Take Triggered. Action: {action}")
    #         #             del self.position[ticker_symbol]
    #         #     else:
    #         #         self.log.warning(f"Missing profit or loss models for cluster '{cluster}'. Defaulting to HOLD.")
    #         #     action_result = {
    #         #         'action': action,
    #         #         'confidence': confidence,
    #         #         'current_price': float(latest_row['close']),
    #         #         'agent': agent_name,
    #         #         'buy_date': self.position.get(ticker_symbol, {}).get('entry_date')
    #         #     }
    #         #     logger.info(f"[{ticker_symbol}]: Action Result: {action_result}")
    #         #     return full_stock_data, {**market_health_results, **action_result, 'all_features': all_features_dict}
    #         #     # --- END OF 'HOLD' LOGIC ---
    #         # else:  # State: No Position
    #         #     entry_model_name = f"entry_model_{cluster}_vol"
    #         #     entry_model = self.models.get(entry_model_name)
    #         #     if not entry_model:
    #         #         logger.warning(f"Missing entry model for cluster '{cluster}'. Defaulting to WAIT.")
    #         #         return full_stock_data, {**market_health_results, 'action': "WAIT", 'reason': "Missing Models.",
    #         #                                  'current_price': price_on_date, 'agent': "System",
    #         #                                  'all_features': all_features_dict}
    #         #     features = latest_row[self.feature_names[entry_model_name]].astype(float).to_frame().T
    #         #     entry_pred = entry_model.predict(features)[0]
    #         #     entry_prob = entry_model.predict_proba(features)[0]
    #         #     result = {}
    #         #
    #         #     # --- DEBUG 5: Log AI Model's final decision ---
    #         #     logger.debug(
    #         #         f"[{ticker_symbol}]: AI Advisor selected. Cluster: '{cluster}'. Model: '{entry_model_name}'. "
    #         #         f"Prediction: {entry_pred} (1=BUY, 0=WAIT). Confidence: {entry_prob[entry_pred] * 100:.2f}%")
    #         #
    #         #     if entry_pred == 1:
    #         #         # --- START: NEW "NEXT DAY" & "TAKE PROFIT" LOGIC ---
    #         #
    #         #         # 1. Get the decision day's ATR for calculations
    #         #         decision_atr = latest_row['atr_14']
    #         #
    #         #         # 2. Find the next trading day from the *full* data
    #         #         analysis_date_dt = pd.to_datetime(analysis_date)
    #         #         next_day_data = full_stock_data[full_stock_data.index > analysis_date_dt]
    #         #
    #         #         if next_day_data.empty:
    #         #             logger.warning(
    #         #                 f"BUY signal for {ticker_symbol} on {analysis_date}, but no future data found to execute trade.")
    #         #             return full_stock_data, {**market_health_results, 'action': 'WAIT',
    #         #                                      'reason': 'Buy signal but no next-day data.',
    #         #                                      'all_features': all_features_dict}
    #         #
    #         #         # 3. Get the *actual* trade info from the next day
    #         #         actual_trade_row = next_day_data.iloc[0]
    #         #         buy_price = actual_trade_row['open']  # Buy at next day's OPEN
    #         #         buy_date = actual_trade_row.name.date()  # The *actual* trade date
    #         #
    #         #         # 4. Calculate Risk amount (using your 2.5 ATR multiplier)
    #         #         risk_per_share_dollars = decision_atr * 2.5
    #         #
    #         #         # 5. Calculate SL and TP based on the *actual* buy_price and risk
    #         #         stop_loss_price = buy_price - risk_per_share_dollars
    #         #
    #         #         # 6. ADDED: Calculate Profit Target (2:1 Reward:Risk Ratio)
    #         #         # You can change the '2.0' to 1.5, 2.5, etc.
    #         #         profit_target_price = buy_price + (risk_per_share_dollars * 2.0)
    #         #         shap_values_for_buy, base_value_for_buy = get_shap_explanation(entry_model, features)
    #         #         # 7. Build the result dictionary with the new values
    #         #         result = {
    #         #             'action': "BUY", 'confidence': entry_prob[1] * 100,
    #         #             'current_price': float(buy_price),
    #         #             'buy_date': buy_date,
    #         #             'agent': f"{cluster.capitalize()}-Volatility Entry Agent",
    #         #             'stop_loss_price': stop_loss_price,
    #         #             'profit_target_price': profit_target_price,
    #         #             'shap_values': shap_values_for_buy[0], 'shap_base_value': base_value_for_buy,
    #         #             'feature_names': features.columns.tolist(), 'feature_values': features.iloc[0].tolist()
    #         #         }
    #         #         logger.info(f"[{ticker_symbol}]: Action Result: {result}")
    #         #         # 8. Update position tracker with new values
    #         #         self.position[ticker_symbol] = {'entry_price': buy_price,
    #         #                                         'stop_loss_price': stop_loss_price}
    #         #     else:
    #         #         result = {'action': "WAIT", 'confidence': entry_prob[0] * 100,
    #         #                   'current_price': float(latest_row['close']),
    #         #                   'agent': f"{cluster.capitalize()}-Volatility Entry Agent"}
    #         #         logger.info(f"[{ticker_symbol}]: Action Result: {result}")
    #         #
    #         #     final_result = {**market_health_results, **result, 'all_features': all_features_dict}
    #         #     if debug:
    #         #         st.write("--- Final result dictionary being returned: ---")
    #         #         logger.info(f"Final result dictionary being returned: {final_result}")
    #         #         st.json(final_result)
    #         #     return full_stock_data, final_result
    #     except Exception as e:
    #         self.log.error(f"Error in run_analysis for {ticker_symbol}", exc_info=True)
    #         st.code(traceback.format_exc())
    #         return None, None

    def run_analysis(self, full_stock_data, ticker_symbol, analysis_date, use_market_filter=True):

        # --- HELPER FUNCTION DEFINED AT TOP OF SCOPE ---
        def get_shap_explanation(model, features_df):
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(features_df)
                # Handle binary classification SHAP output (list of 2 vs single array)
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    return shap_values[1], explainer.expected_value[1]
                else:
                    return shap_values, explainer.expected_value
            except Exception as e:
                self.log.warning(f"SHAP explanation failed: {e}")
                return None, 0

        try:
            self.log.info(f"--- `run_analysis` started for {ticker_symbol} on {analysis_date} ---")

            # 1. Data Validation
            if full_stock_data is None or full_stock_data.empty:
                self.log.error(f"❌ No data provided for symbol {ticker_symbol}.")
                return pd.DataFrame(), {'action': "WAIT", 'reason': f"No data provided for symbol {ticker_symbol}."}

            full_stock_data = clean_raw_data(full_stock_data)

            # --- DEBUG: Log Incoming Data ---
            self.log.info(
                f"[{ticker_symbol}]: Data Shape: {full_stock_data.shape}. Range: {full_stock_data.index.min()} -> {full_stock_data.index.max()}")

            if pd.api.types.is_datetime64_any_dtype(full_stock_data.index) and full_stock_data.index.tz is not None:
                full_stock_data.index = full_stock_data.index.tz_localize(None)

            # Slice Data (Prevent Lookahead)
            analysis_dt = pd.to_datetime(analysis_date)
            data_up_to_date = full_stock_data[full_stock_data.index <= analysis_dt]

            if data_up_to_date.empty:
                self.log.warning(
                    f"❌ Data slice empty for {analysis_date}. Max date in data: {full_stock_data.index.max()}")
                return full_stock_data, {'action': "WAIT", 'reason': "No data available for this date.",
                                         'current_price': 0, 'agent': "System"}

            price_on_date = data_up_to_date.iloc[-1]['close']

            # 2. Market Filter
            market_health_results = {'health_score': 'N/A', 'health_reasons': []}
            if use_market_filter:
                health_score, health_reasons = self.get_market_health_index(analysis_date)
                market_health_results = {'health_score': health_score, 'health_reasons': health_reasons}
                self.log.info(f"--- Market Health: {health_score}/4 ---")

                if health_score < 3:
                    self.log.info(f"⛔ Market Filter Triggered (Score {health_score}). Action: WAIT.")
                    return full_stock_data, {**market_health_results, 'action': "WAIT",
                                             'confidence': 99.9, 'reason': "Market Health Fail",
                                             'agent': "Market Filter"}

            # 3. Feature Engineering
            self.log.info(f"🔄 Calculating Features for {len(data_up_to_date)} rows...")
            featured_data = self.calculator.calculate_all_features(data_up_to_date)

            # --- CRITICAL DEBUG: CHECK FEATURES HERE ---
            if featured_data.empty:
                self.log.error(f"❌ Feature Calculation Returned EMPTY DataFrame!")
                return full_stock_data, {'action': "WAIT", 'reason': "Feature Calc Fail"}

            latest_row = featured_data.iloc[-1]
            all_features_dict = latest_row.to_dict()

            # Print the keys to see if 'ibs' or 'rel_strength' are missing
            self.log.info(f"✅ Features Calculated. Keys found: {list(all_features_dict.keys())}")
            self.log.info(
                f"🔍 DEBUG VALUES: IBS={all_features_dict.get('ibs')}, RelStrength={all_features_dict.get('rel_strength_qqq')}")

            cluster = latest_row.get('volatility_cluster', 'mid')

            # 4. AI Model Prediction
            entry_model_name = f"entry_model_{cluster}_vol"
            entry_model = self.models.get(entry_model_name)
            ai_action = "WAIT"
            ai_conf = 0
            shap_values = None
            base_value = 0

            if entry_model:
                try:
                    # Filter features to match model expectation
                    required_feats = self.feature_names.get(entry_model_name, [])
                    if not required_feats:
                        self.log.warning(f"⚠️ No feature list found for model {entry_model_name}. Using all available.")
                        model_features = latest_row.to_frame().T
                    else:
                        # Ensure all required features exist, fill missing with 0
                        valid_feats = {k: latest_row.get(k, 0) for k in required_feats}
                        model_features = pd.DataFrame([valid_feats])

                    entry_pred = entry_model.predict(model_features)[0]
                    entry_prob = entry_model.predict_proba(model_features)[0]

                    if entry_pred == 1:
                        ai_action = "BUY"
                        ai_conf = entry_prob[1] * 100
                        # Calculate SHAP only on BUY to save time
                        shap_values, base_value = get_shap_explanation(entry_model, model_features)
                        self.log.info(f"🤖 AI Model ({entry_model_name}) Signal: BUY ({ai_conf:.1f}%)")
                    else:
                        self.log.info(f"🤖 AI Model ({entry_model_name}) Signal: WAIT")
                except Exception as e:
                    self.log.warning(f"⚠️ Model prediction failed: {e}")
            else:
                self.log.warning(f"⚠️ Model {entry_model_name} not found. Skipping AI prediction.")

            # 5. Composite Scoring
            final_score, score_reasons = self.calculate_composite_score(all_features_dict, ai_action)
            self.log.info(f"📊 Final Score: {final_score} | Reasons: {score_reasons}")

            # Get Threshold
            current_threshold = self.learner.get_threshold() if self.learner else 50
            final_action = "WAIT"
            decision_type = "BLOCKED"

            if final_score >= 60:
                final_action = "BUY"
                decision_type = "STRONG_BUY"
            elif final_score >= current_threshold:
                final_action = "BUY"
                decision_type = "WEAK_BUY"

            # 6. Result Packaging
            result = {
                'action': final_action,
                'decision_type': decision_type,
                'score': final_score,
                'threshold_used': current_threshold,
                'confidence': ai_conf,
                'agent': f"{cluster.capitalize()}-Vol Agent",
                'reason': f"Score: {final_score} (Req: {current_threshold}) | {', '.join(score_reasons)}",
                'current_price': price_on_date,
                'all_features': all_features_dict,
                'shap_values': shap_values[0] if shap_values is not None else None,
                'shap_base_value': base_value,
                'feature_names': list(model_features.columns) if entry_model else [],
                'feature_values': model_features.iloc[0].tolist() if entry_model else [],
                **market_health_results
            }

            if final_action == "BUY":
                # Smart Stop Logic
                smart_stop = self.calculate_smart_stop(full_stock_data, analysis_date)
                atr = latest_row.get('atr_14', 0)

                if smart_stop and smart_stop < price_on_date:
                    stop_loss = smart_stop
                else:
                    stop_loss = price_on_date - (atr * 3.0)

                risk = price_on_date - stop_loss
                profit_target = price_on_date + (risk * 2.0)

                result['stop_loss_price'] = stop_loss
                result['profit_target_price'] = profit_target
                result['buy_date'] = analysis_dt.date()

                self.log.info(f"🚀 BUY SIGNAL GENERATED: Entry {price_on_date:.2f}, Stop {stop_loss:.2f}")

            return full_stock_data, result

        except Exception as e:
            self.log.error(f"🔥 CRITICAL ERROR in run_analysis: {e}", exc_info=True)
            return full_stock_data, {'action': "WAIT", 'reason': f"Error: {e}"}

    def analyze(self, stock_data, symbol, analysis_date, params: dict = None, use_market_filter: bool = True):
        """
        An adapter method to make the AI Advisor compatible with the modular screener.
        """
        _, result = self.run_analysis(stock_data, symbol, analysis_date, use_market_filter=use_market_filter)
        logger.info(f"Result for {symbol} on {analysis_date}: {result}")
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

    # stockwise_simulation.py

    # Inside ProfessionalStockAdvisor class
    def apply_israeli_fees_and_tax(self, gross_profit_dollars, num_shares):
        """
        Calculates Net Profit using the brokerage's tiered/minimum fee structure
        for USD stocks and centrally applies the 25% tax rate.

        The fee for USD stocks is max(num_shares * $0.01, $2.50) per transaction.
        """

        # 1. Commission Calculation (USD Shares): max(Shares * $0.01, $2.50)
        FEE_PER_SHARE = cfg.FEE_PER_SHARE
        MINIMUM_FEE = cfg.MINIMUM_FEE

        per_share_fee_calc = FEE_PER_SHARE * num_shares

        # Single transaction fee (Buy OR Sell)
        single_transaction_fee = max(per_share_fee_calc, MINIMUM_FEE)

        # Total Fees: Buy transaction + Sell transaction
        total_fees_dollars = single_transaction_fee * 2

        profit_after_fees_dollars = gross_profit_dollars - total_fees_dollars

        # 2. Tax: Use centralized TAX_RATE (25%) on positive profits
        # We ensure cfg.TAX_RATE is used for architectural consistency.
        tax_dollars = (profit_after_fees_dollars * cfg.TAX_RATE) if profit_after_fees_dollars > 0 else 0

        net_profit_dollars = profit_after_fees_dollars - tax_dollars
        total_deducted_dollars = total_fees_dollars + tax_dollars

        return net_profit_dollars, total_deducted_dollars

    def create_chart(self, stock_symbol, stock_data, result, analysis_date,
                     show_mico_lines=False, mico_result=None):
        try:
            if stock_data.empty: return None
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.8, 0.2])
            fig.add_trace(
                go.Candlestick(x=stock_data.index, open=stock_data['open'], high=stock_data['high'],
                               low=stock_data['low'],
                               close=stock_data['close'], name='Price'), row=1, col=1)
            for period, color in [(5, 'orange'), (20, 'blue'), (50, 'red')]:
                if len(stock_data) >= period:
                    ma = stock_data['close'].rolling(window=period).mean()
                    fig.add_trace(go.Scatter(x=stock_data.index, y=ma, mode='lines', name=f'MA{period}',
                                             line=dict(color=color, width=1)), row=1, col=1)
            if len(stock_data) >= 20:
                try:
                    bb_df = ta.bbands(close=stock_data['close'], length=20)
                    if bb_df is not None and not bb_df.empty:
                        # 1. Force lowercase to normalize
                        bb_df.columns = [col.lower() for col in bb_df.columns]

                        # 2. Dynamic Search: Find the columns regardless of the exact name (2.0 vs 2)
                        # We look for columns starting with 'bbu' (upper) and 'bbl' (lower)
                        bbu_col = next((c for c in bb_df.columns if c.startswith('bbu')), None)
                        bbl_col = next((c for c in bb_df.columns if c.startswith('bbl')), None)

                        # 3. Plot using the found column names
                        if bbu_col and bbl_col:
                            fig.add_trace(
                                go.Scatter(x=stock_data.index, y=bb_df[bbu_col], mode='lines', name='BB Upper',
                                           line=dict(color='gray', dash='dot', width=1), showlegend=False), row=1,
                                col=1)
                            fig.add_trace(
                                go.Scatter(x=stock_data.index, y=bb_df[bbl_col], mode='lines', name='BB Lower',
                                           line=dict(color='gray', dash='dot', width=1), fill='tonexty',
                                           fillcolor='rgba(128,128,128,0.1)', showlegend=False), row=1, col=1)
                except Exception as e:
                    self.log.warning(f"Could not calculate Bollinger Bands: {e}", exc_info=True)
            fig.add_trace(
                go.Bar(x=stock_data.index, y=stock_data['volume'], name='Volume',
                       marker=dict(color='rgba(100,110,120,0.6)')),
                row=2, col=1)
            fig.add_vline(x=analysis_date, line_width=1, line_dash="dash", line_color="white", name="Analysis Date",
                          row=1)
            action = result.get('action', 'WAIT')
            current_price = result.get('current_price', stock_data['close'].iloc[-1] if not stock_data.empty else 0)
            if "BUY" in action:
                buy_date = result['buy_date']
                stop_loss = result.get('stop_loss_price')

                # --- Read the pre-calculated price from the analysis results ---
                profit_target_price = result.get('profit_target_price')

                # --- Only plot the line if the profit target exists ---
                if profit_target_price:
                    fig.add_hline(y=float(profit_target_price), line_dash="dot", line_color="cyan",
                                  name="AI Profit Target", row=1,
                                  annotation_text=f"AI Target: ${profit_target_price:.2f}",
                                  annotation_position="top right")
                # Plot the "Target Buy" marker
                if buy_date:
                    fig.add_trace(go.Scatter(
                        x=[buy_date], y=[current_price],
                        mode='markers+text',
                        text=[f"Buy: ${current_price:.2f}"],  # Your fix from before
                        textposition="middle right",
                        marker=dict(color='cyan', size=12, symbol='circle-open', line=dict(width=2)),
                        name='Target Buy'
                    ), row=1, col=1)

                # Plot the Stop-Loss line
                if stop_loss:
                    fig.add_hline(y=float(stop_loss), line_dash="dot", line_color="magenta",
                                  name="AI Stop-Loss", row=1, annotation_text=f"AI Stop: ${stop_loss:.2f}",
                                  annotation_position="bottom right")
            if show_mico_lines:
                if len(stock_data) >= 200:
                    sma_200 = stock_data['close'].rolling(window=200).mean()
                    fig.add_trace(go.Scatter(
                        x=stock_data.index, y=sma_200, mode='lines', name='MA200',
                        line=dict(color='purple', width=1)
                    ), row=1, col=1)
                if mico_result and mico_result.get('signal') == 'BUY':
                    sl_price = mico_result.get('stop_loss_price')
                    tp_price = mico_result.get('profit_target_price')
                    if sl_price:
                        fig.add_hline(
                            y=sl_price, line_width=2, line_dash="dash",
                            line_color="red", name="Mico Stop-Loss",
                            annotation_text="Mico Stop-Loss",
                            annotation_position="bottom left",
                            row=1, col=1
                        )
                    if tp_price:
                        fig.add_hline(
                            y=tp_price, line_width=2, line_dash="dash",
                            line_color="green", name="Mico Take-Profit",
                            annotation_text="Mico Take-Profit",
                            annotation_position="top left",
                            row=1, col=1
                        )
            # 1. Define the View Window (Existing code)
            zoom_start_date = pd.to_datetime(analysis_date) - timedelta(days=10)
            zoom_end_date = pd.to_datetime(analysis_date) + timedelta(days=120)

            # 2. --- Calculate Min/Max for the Visible Range ---
            # We slice the data to finding the high/low ONLY within the zoom window
            mask = (stock_data.index >= zoom_start_date) & (stock_data.index <= zoom_end_date)
            visible_data = stock_data.loc[mask]

            y_axis_range = None  # Default to auto if logic fails

            if not visible_data.empty:
                # Find min/max in the visible window
                visible_min = visible_data['low'].min()
                visible_max = visible_data['high'].max()

                # Apply your $40 buffer
                y_axis_range = [visible_min - 20, visible_max + 20]

            # 3. Update Layout
            fig.update_layout(title_text=f'{stock_symbol} Price & Volume Analysis', xaxis_rangeslider_visible=False,
                              showlegend=True,
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            # Set X-Axis Zoom
            fig.update_xaxes(range=[zoom_start_date, zoom_end_date], row=1, col=1)
            fig.update_xaxes(range=[zoom_start_date, zoom_end_date], row=2, col=1)

            # 4. Set Y-Axis with Calculated Range ---
            # We switch to "linear" type because fixed $ buffers don't work well on log scales
            fig.update_yaxes(title_text="Price (USD)", type="linear",range=y_axis_range, row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            return fig

        except Exception as e:
            self.log.error(f"Error creating chart for {stock_symbol}: {e}", exc_info=True)
            st.error("Failed to create chart.")
            return None


def render_live_dashboard():
    """
    Renders the "no-cost" live dashboard tab.
    """
    st.subheader("Watchlist Live View")
    watchlist_symbols = st.session_state.get('watchlist', "AAPL\nNVDA\nMSFT\nGOOG")
    symbols_list = [s.strip().upper() for s in watchlist_symbols.split('\n') if s.strip()]
    if not symbols_list:
        st.info("Add symbols to the Watchlist in the sidebar to begin.")
        return

    # --- 3. Live Dashboard UI ---
    refresh_button = st.button("🔄 Refresh Live Data")
    st.markdown("---")
    cols = st.columns(min(len(symbols_list), 4))

    for i, symbol in enumerate(symbols_list):
        col = cols[i % 4]
        position = st.session_state.positions.get(symbol)
        if position:
            border_css = "border: 2px solid #00A36C; border-radius: 5px; padding: 10px;"
        else:
            border_css = "border: 1px solid #888; border-radius: 5px; padding: 10px;"

        with col:
            st.markdown(f"<div style='{border_css}'>", unsafe_allow_html=True)
            st.subheader(symbol)
            placeholder = st.empty()
            if refresh_button:
                with st.spinner(f"Loading {symbol}..."):
                    live_data = st.session_state.data_manager.get_stock_data(symbol, days_back=200)
                    if live_data.empty:
                        placeholder.error("No data.")
                        continue

                    # Add indicators needed by RiskManager
                    live_data.ta.atr(length=14, append=True, col_names='atr_14')
                    live_data['sma_150'] = live_data['close'].rolling(150).mean()
                    live_data.bfill(inplace=True)

                    latest_bar = live_data.iloc[-1]
                    current_price = latest_bar['close']
                    placeholder.metric("Current Price", f"${current_price:.2f}")

                    if position:
                        st.info(f"Entry: ${position['entry_price']:.2f}")
                        signal, updated_pos_data = st.session_state.risk_manager.manage_open_position(
                            current_day_data=latest_bar,
                            position_data=position
                        )
                        if signal == "EXIT_SIGNAL":
                            alert_msg = f"🔔 SELL ALERT ({symbol}): Stop-loss hit!"
                            st.error(alert_msg)
                            st.session_state.alerts.append(alert_msg)
                            del st.session_state.positions[symbol]
                        else:
                            st.success(f"Holding. New SL: ${updated_pos_data['current_stop_loss']:.2f}")
                            st.session_state.positions[symbol] = updated_pos_data
            else:
                if position:
                    placeholder.info(f"Position Open. Entry: ${position['entry_price']:.2f}")
                else:
                    placeholder.metric("Current Price", "N/A (Refresh)")
            st.markdown("</div>", unsafe_allow_html=True)


def display_analysis_results(ai_result, mico_result_A, mico_result_B, stock_data, stock_symbol, analysis_date, advisor,
                             show_mico_lines,
                             investment_amount=cfg.INVESTMENT_AMOUNT):
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
    col1, col2, col3 = st.columns(3)
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

            profit_target_price, net_profit, net_profit_per_share = 0, 0, 0
            if "BUY" in action and current_price > 0:
                est_profit_pct = advisor.calculate_dynamic_profit_target(confidence)
                profit_target_price = current_price * (1 + est_profit_pct / 100)
                # todo: add the option to buy part of the stock
                hypothetical_shares = investment_amount / current_price
                gross_profit = (profit_target_price - current_price) * hypothetical_shares
                net_profit, _ = advisor.apply_israeli_fees_and_tax(gross_profit, hypothetical_shares)
                net_profit_per_share = (net_profit / investment_amount) * 100 # Calculate Profit %
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Confidence", f"{confidence:.1f}%" if confidence > 0 else "N/A")
            m_col2.metric("Profit Target", f"${profit_target_price:.2f}" if profit_target_price > 0 else "-")
            m_col3.metric("Hypothetical Net Profit", f"${net_profit:.2f}" if net_profit > 0 else "-")
            stop_loss_price = ai_result.get('stop_loss_price', 0)
            m_col4, m_col5, m_col6 = st.columns(3)
            m_col4.metric("Entry Price", f"${current_price:.2f}" if current_price > 0 else "-")
            m_col5.metric("Stop-Loss", f"${stop_loss_price:.2f}" if stop_loss_price > 0 else "-")
            m_col6.metric("Net Profit [%]", f"{net_profit_per_share:.2f}%" if net_profit_per_share > 0 else "-")
            st.caption(f"Agent: {ai_result.get('agent', 'N/A')}")

            st.markdown("1. Market Health Analysis (SPY)")
            if ai_result:
                health_score = ai_result.get('health_score')
                health_reasons = ai_result.get('health_reasons', [])
                st.metric("Market Health Index", f"{health_score}/4" if health_score is not None else "N/A")
                if health_reasons:
                    for reason in health_reasons:
                        st.markdown(reason)
            else:
                st.info("Market Health was not analyzed (AI Advisor disabled).")

        else:
            st.info("AI Advisor was not run.")

    # 2. MICO RULE-BASED (A)
    with col2:
        st.markdown("#### 📜 MICO Rule-Based (A)")
        _display_mico_metrics(mico_result_A, advisor, investment_amount)

    # 3. MICO AI-ADAPTIVE (B)
    with col3:
        st.markdown("#### ✨ MICO AI-Adaptive (B)")
        _display_mico_metrics(mico_result_B, advisor, investment_amount, is_adaptive=True)

    st.markdown("---")

    # --- 3. PRICE CHART ---
    st.subheader("2. Price Chart")
    fig = advisor.create_chart(
        stock_symbol,
        stock_data,
        ai_result,
        analysis_date,
        show_mico_lines=show_mico_lines,
        mico_result=mico_result_A  # Plot the lines/markers from the MICO A result (which holds A or B data)
    )
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

    if 'shap_values' in ai_result and ai_result['shap_values'] is not None:
        with st.expander("Show Key Factors in AI Decision (SHAP Analysis)"):

            # --- CRITICAL FIX: Ensure SHAP values are indexed correctly for the plot ---
            shap_values = ai_result['shap_values']

            # Defensive indexing for the waterfall plot's expected single output
            if isinstance(shap_values, np.ndarray) and shap_values.ndim > 1:
                # If it's a matrix (e.g., [1, N]), index to the first row/output
                shap_values = shap_values[0]
            elif isinstance(shap_values, list) and len(shap_values) > 1:
                # If it's a list of outputs (multi-class), index to the BUY output (index 1)
                shap_values = shap_values[1]
            # NOTE: If shap_values is a list/array of length 1, it passes unchanged.

            # --- FINAL ROBUSTNESS CHECK: Ensure the array has elements ---
            # If the result of the indexing is an empty array or a scalar (which SHAP dislikes), exit gracefully.
            if np.size(shap_values) < 1:
                st.warning("SHAP explanation data is empty or malformed (shape ()). Cannot plot.")
                # We stop processing the SHAP section here
                # NOTE: The rest of the function (all_features expander) continues normally.

            else:
                # After indexing, proceed with plotting
                try:
                    explanation = shap.Explanation(
                        values=shap_values,
                        base_values=ai_result['shap_base_value'],
                        data=ai_result['feature_values'],
                        feature_names=ai_result['feature_names']
                    )

                    # Final plot check is simplified as we handled indexing above
                    if isinstance(explanation.values, np.ndarray) and explanation.values.ndim > 1:
                        st.error(
                            f"Final SHAP values array shape is incorrect: {explanation.values.shape}. Skipping plot.")
                    else:
                        fig, ax = plt.subplots()
                        shap.waterfall_plot(explanation, max_display=10, show=False)
                        st.pyplot(fig)
                except Exception as e:
                    # Catch any remaining SHAP library internal errors
                    st.error(f"Error generating SHAP plot: {e}")
                    logger.error(f"Error generating SHAP plot: {e}", exc_info=True)

    if 'all_features' in ai_result:
        with st.expander("Show All Technical Parameters Used"):
            features_df = pd.DataFrame.from_dict(ai_result['all_features'], orient='index', columns=['Value'])
            features_df.index.name = 'Feature'
            features_df['Value'] = features_df['Value'].astype(str)
            st.dataframe(features_df.style.format(precision=4), use_container_width=True)


# =============================================================================
# --- MAIN UI FUNCTION ---
# =============================================================================
#     # todo: create a scheduler that will run the screener every month with 5000 stocks

def create_enhanced_interface(IS_CLOUD=False):
    # --- Layout Setup ---
    st.title("🏢 StockWise AI Trading Advisor")

    # Status Indicator (Top Right)
    _, col_status = st.columns([4, 1])
    status_placeholder = col_status.empty()

    # Helper functions for status
    def set_status_indicator():
        # Use the actual data_source attribute set during initialization
        current_source = st.session_state.data_manager.data_source

        if current_source == 'ibkr' and st.session_state.data_manager.isConnected():
            status_placeholder.markdown("<div style='text-align:right; color:#00cc00;'>● IBKR (Live)</div>",
                                        unsafe_allow_html=True)
        elif current_source == 'alpaca':
            status_placeholder.markdown("<div style='text-align:right; color:#ffcc00;'>● ALPACA (Paper)</div>",
                                        unsafe_allow_html=True)
        else:
            # This covers YFinance, or failed IBKR when Alpaca keys are missing.
            status_placeholder.markdown("<div style='text-align:right; color:#00aaff;'>● YFINANCE (Free)</div>",
                                        unsafe_allow_html=True)

    # Initialize Status
    set_status_indicator()

    # ==============================================================================
    # --- 1. SIDEBAR: GLOBAL CONFIGURATION ---
    # ==============================================================================
    st.sidebar.header("⚙️ Global Config")

    # Portfolio Settings (Restored Original Values)
    # col_inv, col_risk = st.sidebar.columns(2)
    investment_amount = st.sidebar.number_input(
        "Initial Portfolio Value ($)",
        min_value=10,
        value=1000,
        step=10
    )
    risk_per_trade_percent = st.sidebar.slider(
        "Risk per Trade (%)",
        min_value=0.5,
        max_value=5.0,
        value=2.0,
        step=0.1
    )

    # Initialize Risk Manager (Hidden logic)
    if 'risk_manager' not in st.session_state:
        st.session_state.risk_manager = RiskManager(portfolio_value=investment_amount,
                                                    global_risk_pct=risk_per_trade_percent)
    else:
        st.session_state.risk_manager.update_portfolio_value(investment_amount)
        st.session_state.risk_manager.global_risk_pct = risk_per_trade_percent
        st.session_state.risk_manager.max_risk_dollars_per_trade = investment_amount * (risk_per_trade_percent / 100.0)

    st.sidebar.markdown("---")

    # ==============================================================================
    # --- 2. SIDEBAR: STRATEGY & MODEL CONFIG (Moved UP to fix Execution Order) ---
    # ==============================================================================
    with st.sidebar.expander("🛠️ Strategy & Model Config", expanded=False):
        st.caption("Select Systems to Run")
        run_ai = st.checkbox("Run AI Advisor (Gen-3)", value=True)
        run_mico = st.checkbox("Run Micha System", value=True)

        st.caption("Technical Models")
        c1, c2 = st.columns(2)
        with c1:
            run_mean = st.checkbox("Mean Rev.", value=True)
            run_break = st.checkbox("Breakout", value=True)
            run_super = st.checkbox("SuperTrend", value=True)
        with c2:
            run_cross = st.checkbox("MA Cross", value=True)
            run_vol = st.checkbox("Vol. Mom.", value=True)

        st.divider()
        st.caption("AI Agent Profile")
        AGENT_CONFIGS = {
            'Dynamic Profit (Recommended)': "models/NASDAQ-gen3-dynamic",
            '1% Net Profit': "models/NASDAQ-gen3-1pct",
            '2% Net Profit': "models/NASDAQ-gen3-2pct",
            '3% Net Profit': "models/NASDAQ-gen3-3pct",
            '4% Net Profit': "models/NASDAQ-gen3-4pct"
        }
        selected_agent_name = st.selectbox("Select Agent", list(AGENT_CONFIGS.keys()))

        st.caption("Exit Strategy")
        use_trailing_stop = st.checkbox("Use Trailing Stop-Loss", value=True)
        atr_trailing_mult = st.slider("Trailing Stop ATR Multiple", 1.0, 5.0, 2.5, 0.1, disabled=not use_trailing_stop)

        st.caption("MICO Strategy Mode")
        strategy_mode = st.radio("A/B Testing Mode", ('Rule-Based Only (A)', 'AI-Adaptive (B)', 'Show Both (Parallel)'),
                                 index=2)

    st.sidebar.markdown("---")

    # ==============================================================================
    # --- 3. SIDEBAR: OPERATION MODE (The Big Switch) ---
    # ==============================================================================
    mode = st.sidebar.radio("Select Operation Mode",
                            ["🎯 Single Analysis", "🚀 Market Screener", "🗓️ System Backtest", "🧠 AI Training"],
                            index=0
                            )

    # Initialize variables to avoid 'not defined' errors
    # run_ai = run_mico = run_mean = run_break = run_super = run_cross = run_vol = False
    analyze_btn = scan_btn = run_full_backtest_btn = run_finetune_btn = optimize_btn = False

    # --- Initialize use_optimized ---
    use_optimized = False
    # ==============================================================================
    # --- 4. SIDEBAR: DYNAMIC CONTROLS (Based on Mode) ---
    # ==============================================================================

    if mode == "🎯 Single Analysis":
        st.sidebar.subheader("Single Stock Setup")
        stock_symbol = st.sidebar.text_input("Symbol", value="NVDA").upper().strip()

        # Date Selection
        if 'analysis_date_input' not in st.session_state:
            st.session_state.analysis_date_input = datetime.now().date()

        def set_date_to_today():
            st.session_state.analysis_date_input = datetime.now().date()

        col_d1, col_d2 = st.sidebar.columns([3, 2])
        with col_d1:
            analysis_date = st.date_input("Analysis Date", key='analysis_date_input')
        with col_d2:
            st.write("")
            st.write("")
            st.button("Today", on_click=set_date_to_today, key='analysis_today_btn')

        # Primary Action
        analyze_btn = st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True)

        # Display Options
        st.sidebar.caption("Display Settings")
        use_market_filter = st.sidebar.checkbox("Enable Market Health Filter (SPY)", value=True)
        show_mico_lines = st.sidebar.checkbox("Show Mico System Lines", value=True)

    elif mode == "🚀 Market Screener":
        st.sidebar.subheader("Screener Setup")

        # Universe Selection
        universe_options = {
            "NASDAQ 100": get_nasdaq100_tickers,
            "S&P 500": get_sp500_tickers,
            "Full NASDAQ (from file)": load_nasdaq_tickers
        }
        selected_universe_name = st.sidebar.selectbox("Universe", list(universe_options.keys()),
                                                      key="universe_selector")

        # Screener Date
        debug_mode = st.sidebar.checkbox("Enable Screener Debug Mode", value=True)
        if debug_mode:
            if 'screener_date_input' not in st.session_state:
                st.session_state.screener_date_input = datetime.now().date()

            def set_scr_date_to_today():
                st.session_state.screener_date_input = datetime.now().date()

            col1_scr, col2_scr = st.sidebar.columns([2, 1])
            with col1_scr:
                screener_analysis_date = col1_scr.date_input("🗓️ Date", key='screener_date_input')
            with col2_scr:
                st.write("")
                st.write("")
                col2_scr.button("Today", on_click=set_scr_date_to_today, key='screener_today_btn')
        else:
            screener_analysis_date = datetime.now().date()  # Fallback if not debug

        # Primary Action
        scan_btn = st.sidebar.button("Scan Universe", use_container_width=True)

        # Reuse market filter variable
        use_market_filter = st.sidebar.checkbox("Enable Market Health Filter (SPY)", value=True)

    elif mode == "🗓️ System Backtest":
        st.sidebar.subheader("Backtest Config")
        backtest_start_date = st.sidebar.date_input("Start Date", datetime.now().date() - timedelta(days=365))
        backtest_end_date = st.sidebar.date_input("End Date", datetime.now().date())

        # Universe (Needed for backtest too)
        universe_options = {"NASDAQ 100": get_nasdaq100_tickers, "S&P 500": get_sp500_tickers,
                            "Full NASDAQ (from file)": load_nasdaq_tickers}
        selected_universe_name = st.sidebar.selectbox("Universe", list(universe_options.keys()),
                                                      key="universe_selector_bt")

        run_full_backtest_btn = st.sidebar.button("Run Full System Backtest", type="primary", use_container_width=True)
        use_market_filter = True
        st.header("Full System Backtest")
        if run_full_backtest_btn:
            # --- ENFORCE OPTIMIZER USE FOR FULL BACKTEST ---
            st.error(
                "The Full System Backtest feature is now managed via the external Strategy Optimizer for robustness.")
            st.info("To run a comprehensive backtest, please use the 'Strategy Optimizer' in the Debug section.")
            st.info("Use the 'Market Screener' mode for real-time analysis.")
        else:
            st.info("Configure dates and universe, then click 'Run Full System Backtest'.")

    elif mode == "🧠 AI Training":
        st.sidebar.info("AI Adaptive Finetuning")

        # Define the variables here, before they are used in run_finetune_btn:
        predictor = st.session_state.ai_param_predictor

        finetune_start_date = st.sidebar.date_input(
            "Start Date", datetime.now().date() - timedelta(days=90)
        )
        finetune_end_date = st.sidebar.date_input(
            "End Date", datetime.now().date()  # <-- This defines finetune_end_date
        )
        # Display current state (Moved this to the sidebar block)
        st.sidebar.caption(f"Last Finetune: {predictor.get_adaptive_state()['last_finetune_date']}")
        st.sidebar.caption(
            f"Current Adaptive SL Mult: {predictor.get_adaptive_state()['last_stop_loss_multiplier']:.2f}x ATR")

        run_finetune_btn = st.sidebar.button("🧠 Run AI Finetuning", type="secondary", use_container_width=True)

        if run_finetune_btn:
            # Check for both trade history AND Drawdown before starting
            if 'screener_results' not in st.session_state or 'backtest_drawdown_pct' not in st.session_state:
                st.warning("Please run a Full System Backtest first to generate trade history for finetuning.")
            else:
                with st.spinner("Analyzing Backtest History to Fine-Tune AI Parameters..."):

                    # 1. Get the raw trade history (simulating results)
                    trade_history_df = st.session_state.screener_results

                    # --- OPTIONAL SLICING (if filtering by start date is desired) ---
                    # trade_history_df = trade_history_df[trade_history_df['Entry Date'] >= finetune_start_date.strftime('%Y-%m-%d')]

                    # Get REAL Win Rate and Drawdown
                    total_trades = len(trade_history_df)
                    if 'Actual P/L ($)' not in trade_history_df.columns:
                        st.error("Error: 'Actual P/L ($)' column missing. Rerun Backtest first.")
                        st.stop()

                    wins = trade_history_df[trade_history_df['Actual P/L ($)'] > 0]
                    win_rate = len(wins) / total_trades if total_trades > 0 else 0
                    max_drawdown_pct = st.session_state['backtest_drawdown_pct']

                    trade_summary = {
                        'total_trades': total_trades, 'win_rate': win_rate, 'max_drawdown_pct': max_drawdown_pct
                    }
                    status_msg = predictor.run_finetuning_simulation(
                        trade_summary, finetune_end_date.strftime("%Y-%m-%d")
                    )
                    st.success(f"Finetuning Complete! Status: {status_msg}")
                    st.info(
                        f"New Adaptive SL Multiplier: {predictor.get_adaptive_state()['last_stop_loss_multiplier']:.2f}x ATR")
                    st.rerun()
        else:
            st.info("Configure the learning period and click 'Run AI Finetuning'.")

    # ==============================================================================
    # --- 5. SIDEBAR: DEBUG / DEV (Collapsed) ---
    # ==============================================================================
    with st.sidebar.expander("🐞 Debug & Logs", expanded=False):
        st.checkbox("Enable Debug Logging", key="debug_logging_enabled", on_change=setup_logging_from_st)
        # --- TELEGRAM CONNECTION TEST ---
        if st.button("Verify Telegram Connection"):
            # Ensure NotificationManager is initialized in session state
            if 'notification_manager' not in st.session_state:
                st.session_state.notification_manager = NotificationManager()

            nm = st.session_state.notification_manager

            # Check if credentials exist in the environment
            if not nm.enabled:
                st.warning("Telegram is disabled: Please set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")
            else:
                with st.spinner("Testing connection to Telegram API..."):
                    success, reason = nm.check_connection()
                    if success:
                        st.success(f"✅ Telegram Connection Status: {reason}")
                        # Optional: Send a test message to the user's chat
                        try:
                            # Use the username found in the verified response for a personalized alert
                            nm.send_alert("🤖 **StockWise Test Alert:** Connection verified!",
                                          parse_mode='Markdown')
                            st.info("Test message sent to your Telegram chat!")
                        except Exception:
                            # Note: This is where the 400 error is caught, and it will still show the error
                            st.warning(
                                "Connection verified, but failed to send test message "
                                "(check chat ID or if the bot is blocked).")
                    else:
                        st.error(f"❌ Telegram Connection Failed: {reason}")

        # Optimizer (Only local)
        if not IS_CLOUD:
            st.caption("Strategy Optimizer")
            opt_symbol = st.text_input("Stock", "SPY").upper().strip()

            # --- Calibration Data Source Selector ---
            opt_data_source = st.selectbox(
                "Calibration Data Source",
                ["ALPACA", "YFINANCE", "IBKR", ],
                index=0,
                help="Select which broker data to use for this specific calibration run."
            )
            # 2. System Selector (Checkboxes logic replacement)
            calib_options = {
                "Micha System (Mico)": "MichaAdvisor",
                "SuperTrend": "SuperTrendAdvisor",
                "Mean Reversion": "MeanReversionAdvisor",
                "Breakout": "BreakoutAdvisor",
                "MA Crossover": "MovingAverageCrossoverAdvisor",
                "Volume Momentum": "VolumeMomentumAdvisor"
            }

            selected_calib_systems = st.multiselect(
                "Select Systems to Calibrate",
                options=list(calib_options.keys()),
                default=["Micha System (Mico)"]
            )

            # ---------------------------------------------

            col1_opt, col2_opt = st.columns(2)
            opt_start_date = col1_opt.date_input("Start", datetime.now().date() - timedelta(days=365))
            opt_end_date = col2_opt.date_input("End", datetime.now().date())
            optimize_btn = st.button("Run Calibration", use_container_width=True)

        # Log Download
        if os.path.exists("logs/stockwise_gen4_log.jsonl"):
            with open("logs/stockwise_gen4_log.jsonl", "r") as f:
                st.download_button("Download Logs", f, "log.jsonl")

    # ==============================================================================
    # --- MAIN CONTENT RENDERING ---
    # ==============================================================================

    # Mapping the simplified variables back to the original variables needed for logic
    run_mean_reversion = run_mean
    run_breakout = run_break
    run_supertrend = run_super
    run_ma_crossover = run_cross
    run_volume_momentum = run_vol

    # -------------------------------------------------------------------------
    # Strategy Optimizer Logic
    # -------------------------------------------------------------------------
    if optimize_btn:
        st.header("⚙️ Strategy Calibration")
        st.info(f"Running calibration on {opt_symbol}...")

        # 1. Initialize dictionary dynamically based on Sidebar Checkboxes
        all_optimizer_advisors = {}

        for friendly_name in selected_calib_systems:
            class_name = calib_options[friendly_name]  # e.g., "MichaAdvisor"

            if class_name == "MichaAdvisor":
                all_optimizer_advisors[class_name] = st.session_state.mico_advisor
            elif class_name == "SuperTrendAdvisor":
                all_optimizer_advisors[class_name] = st.session_state.supertrend_advisor
            elif class_name == "MeanReversionAdvisor":
                all_optimizer_advisors[class_name] = st.session_state.mean_reversion_advisor
            elif class_name == "BreakoutAdvisor":
                all_optimizer_advisors[class_name] = st.session_state.breakout_advisor
            elif class_name == "MovingAverageCrossoverAdvisor":
                all_optimizer_advisors[class_name] = st.session_state.ma_crossover_advisor
            elif class_name == "VolumeMomentumAdvisor":
                all_optimizer_advisors[class_name] = st.session_state.volume_momentum_advisor

        # 2. Define the Parameter Grids (Required for the optimizer)
        all_parameter_grids = {
            "MichaAdvisor": {
                'sma_short': [20, 50], 'sma_long': [100, 200], 'rsi_threshold': [70, 75],
                'stop_loss_mode': ['atr', 'ma', 'support'], 'atr_mult_stop': [2.0, 2.5],
                'min_conditions_to_buy': [3, 4], 'use_multi_timeframe': [True, False],
                'use_atr_filter': [True, False], 'atr_mult_profit': [2.5],
                'ma_stop_period': [50], 'use_volume_check': [True],
                'use_candlestick_check': [True], 'use_fundamental_filter': [False],
                'max_debt_equity': [2.0], 'min_pe_ratio': [5],
                'use_continuation_check': [False], 'max_atr_quantile': [0.75],
                'weekly_sma_short': [10], 'weekly_sma_long': [40],
                'use_earnings_filter': [True]
            },
            "SuperTrendAdvisor": {'length': [7, 10, 14], 'multiplier': [1.5, 2.0, 2.5, 3.0]},
            "MeanReversionAdvisor": {'bb_length': [20, 30], 'rsi_oversold': [25, 30, 35]},
            "BreakoutAdvisor": {'breakout_window': [20, 30, 50]},
            "MovingAverageCrossoverAdvisor": {'short_window': [20, 50], 'long_window': [100, 200]},
            "VolumeMomentumAdvisor": {'obv_window': [20, 30, 50]}
        }

        # 3. Run Optimization
        if not all_optimizer_advisors:
            st.error("Please select at least one system in 'Strategy & Model Config' sidebar to calibrate.")
        else:
            # --- NEW: MANAGE DATA SOURCE TEMPORARILY ---
            dm = st.session_state.data_manager

            # Save original state to restore later
            original_use_ibkr = dm.use_ibkr
            original_source = getattr(dm, 'data_source', 'yfinance')
            original_fallback = dm.allow_fallback

            # Apply Temporary Settings
            connection_successful = True

            # Apply Temporary Settings for this download
            if opt_data_source == "IBKR":
                dm.use_ibkr = False
                dm.allow_fallback = True
                # --- Force Connection Check ---
                if not dm.isConnected():
                    st.warning("IBKR not connected. Attempting to connect on port 7497...")
                    # Force port 7497 just to be safe
                    dm.host = '127.0.0.1'  # Fixes the "NoneType" error
                    dm.port = 7497  # Ensures Paper Trading port
                    if not dm.connect_to_ibkr():
                        st.error("Failed to connect to IBKR TWS/Gateway. Please check if it's open.")
                        connection_successful = False
            elif opt_data_source == "ALPACA":
                dm.use_ibkr = False
                dm.data_source = "alpaca"  # Requires DM to support this flag logic
            else:  # YFINANCE
                dm.use_ibkr = False
                dm.allow_fallback = True

            if connection_successful:
                try:
                    st.info(f"Downloading data for {opt_symbol} via {opt_data_source}...")

                    # Calculate start date buffer
                    fetch_start_date = opt_start_date - timedelta(days=365)

                    calibration_data = dm.get_stock_data(
                        opt_symbol,
                        start_date=fetch_start_date,
                        end_date=opt_end_date
                    )

                    if calibration_data.empty:
                        st.error(f"Could not fetch data for {opt_symbol} using {opt_data_source}.")
                    else:
                        with st.spinner(f"Calibrating {len(all_optimizer_advisors)} systems using pre-fetched data..."):
                            mico_optimizer.run_full_optimization(
                                optimizer_advisors=all_optimizer_advisors,
                                parameter_grids=all_parameter_grids,
                                symbol=opt_symbol,
                                start_date=opt_start_date,
                                end_date=opt_end_date,
                                pre_fetched_data=calibration_data
                            )
                        st.success("Calibration Complete!")

                finally:
                    # --- RESTORE ORIGINAL SETTINGS ---
                    # This ensures the rest of the app (Live Dashboard, etc.) isn't affected
                    dm.use_ibkr = original_use_ibkr
                    dm.data_source = original_source
                    dm.allow_fallback = original_fallback
                    logger.debug("DEBUG: Data Manager settings restored.")
    # -------------------------------------------------------------------------
    # [INSERT NEW CODE HERE] - Single Analysis Logic
    # -------------------------------------------------------------------------

    if mode == "🎯 Single Analysis":
        # --- SINGLE ANALYSIS LOGIC (Previously in tab_analysis) ---
        st.header(f"Analysis: {stock_symbol}")
        selected_model_dir = AGENT_CONFIGS[selected_agent_name]

        # Load Advisor if changed
        if st.session_state.advisor.model_dir != selected_model_dir:
            with st.spinner(f"Loading '{selected_agent_name}' agent..."):
                st.session_state.advisor = ProfessionalStockAdvisor(
                    model_dir=selected_model_dir,
                    data_source_manager=st.session_state.data_manager,
                    calculator=st.session_state.advisor.calculator
                )
        advisor = st.session_state.advisor

        if analyze_btn:
            if not run_ai and not run_mico:
                st.warning("Please select at least one system to run.")
            elif not stock_symbol:
                st.warning("Please enter a stock symbol.")
            else:
                ai_result = {}
                mico_result_A = {}
                mico_result_B = {}

                with st.spinner(f"Running full analysis for {stock_symbol}..."):
                    # 1. Define Charting Window
                    chart_end_date = pd.to_datetime(analysis_date) + pd.Timedelta(days=30)

                    # 2. Fetch Data
                    charting_data = st.session_state.data_manager.get_stock_data(
                        stock_symbol, days_back=365, end_date=chart_end_date
                    )

                    if charting_data is None or charting_data.empty:
                        st.error(f"No data found for {stock_symbol}.")
                    else:
                        charting_data = clean_raw_data(charting_data)
                        mico_advisor = st.session_state.mico_advisor
                        ai_predictor = st.session_state.ai_param_predictor

                        if run_ai:
                            _, ai_result = advisor.run_analysis(
                                full_stock_data=charting_data.copy(),
                                ticker_symbol=stock_symbol,
                                analysis_date=analysis_date,
                                use_market_filter=use_market_filter
                            )

                        if run_mico:
                            # Prepare data slice for MICO
                            data_up_to_date = charting_data[charting_data.index <= pd.to_datetime(analysis_date)]

                            # Load from JSON file if exists, otherwise empty dict
                            loaded_file_params = mico_advisor.get_best_params(stock_symbol)
                            # Validate (fill defaults)
                            base_params = mico_advisor._extract_and_validate_params(loaded_file_params)

                            # A: Rule-Based
                            mico_result_A = mico_advisor.analyze(
                                stock_data=charting_data.copy(),
                                symbol=stock_symbol,
                                analysis_date=analysis_date,
                                params=base_params
                            )

                            # B: AI-Adaptive
                            mico_params_B = ai_predictor.predict_optimal_params(
                                symbol=stock_symbol,
                                default_params=base_params.copy(),
                                df_slice=data_up_to_date.copy()
                            )
                            mico_result_B = mico_advisor.analyze(
                                stock_data=charting_data.copy(),
                                symbol=stock_symbol,
                                analysis_date=analysis_date,
                                params=mico_params_B
                            )
                            mico_result_B['ai_debug'] = {
                                'atr_mult_stop': mico_params_B.get('atr_mult_stop'),
                                'ai_adjustment_made': mico_params_B.get('ai_adjustment_made', False),
                                'ai_predictor_reason': mico_params_B.get('ai_predictor_reason', 'N/A'),
                                'ai_model_version': mico_params_B.get('ai_model_version', 'N/A')
                            }

                        # Determine Display Strategy
                        if strategy_mode == 'Rule-Based Only (A)':
                            final_mico_result_A_display = mico_result_A
                            final_mico_result_B_display = None
                        elif strategy_mode == 'AI-Adaptive (B)':
                            final_mico_result_A_display = mico_result_B
                            final_mico_result_B_display = None
                        else:
                            final_mico_result_A_display = mico_result_A
                            final_mico_result_B_display = mico_result_B

                        # Added the display call back with correct parameters
                        display_analysis_results(
                            ai_result,
                            final_mico_result_A_display,
                            final_mico_result_B_display,
                            stock_data=charting_data,
                            stock_symbol=stock_symbol,
                            analysis_date=analysis_date,
                            advisor=advisor,
                            show_mico_lines=show_mico_lines,
                            investment_amount=investment_amount
                        )
        else:
            st.info("Configure settings in the sidebar and click 'Run Analysis'.")

    elif mode == "🚀 Market Screener":
        st.header("Market Screener")
        if scan_btn:
            st.session_state.analysis_run = False
            if 'screener_results' in st.session_state: del st.session_state['screener_results']

            # --- 1. Initialization (Moved up to prevent NameError) ---
            load_function = universe_options[selected_universe_name]
            stock_universe = load_function()

            if not stock_universe:
                st.error(f"Could not load the '{selected_universe_name}' stock list.")
            else:
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
                    with st.spinner(f"Running parallel scan for {len(stock_universe)} symbols..."):
                        # We pass the Streamlit advisor instance to the scanner for internal fee/risk calculation access
                        recommended_trades_list = stockwise_scanner.run_full_market_scan(
                            universe_tickers=stock_universe,
                            analysis_date=screener_analysis_date,
                            main_advisor_instance=st.session_state.advisor
                        )

                    # Filter for actual trades found (BUY signal)
                    # The scanner returns a list of dictionaries.
                    recommended_trades_list = [
                        r for r in recommended_trades_list
                        if isinstance(r, dict) and r.get('signal', r.get('Signal')) == 'BUY'
                    ]

                    if recommended_trades_list:
                        recommended_trades_df = pd.DataFrame(recommended_trades_list)
                        st.session_state.screener_results = recommended_trades_df
                        # --- DEBUG: Confirm Rerun Triggered ---
                        logger.debug("[DEBUG] Scan complete. Triggering RERUN.")
                        # --- Set the flag and force rerun to display results ---
                        st.session_state.analysis_run = True
                        st.rerun()
                    else:
                        st.session_state.screener_results = pd.DataFrame()
                        st.warning("No high-confidence BUY opportunities found.")

        elif 'screener_results' in st.session_state and st.session_state.get('analysis_run', False):
            st.markdown("---")
            st.success(f"Screener found {len(st.session_state.screener_results)} trade opportunities.")

            # Reset the flag AFTER confirming the display block ran
            st.session_state.analysis_run = False  # <-- ADD THIS LINE to allow new scans

            final_df = st.session_state.screener_results
            formatter = {
                'Entry Price': '${:.2f}', 'Profit Target ($)': '${:.2f}', 'Stop-Loss': '${:.2f}',
                'Est. Net Profit ($)': '${:.2f}', 'RSI': '{:.2f}', 'PE Ratio': '{:.2f}',
                'P/S Ratio': '{:.2f}',
                'Debt/Equity': '{:.2f}'
            }
            st.dataframe(final_df.style.format(formatter, na_rep='-'), use_container_width=True)

            @st.cache_data
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')

            csv_data = convert_df_to_csv(final_df)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_data,
                file_name=f"screener_results_{screener_analysis_date}.csv",
                mime="text/csv",
                key="csv_download_btn"
            )

            # --- NEW CODE BLOCK FOR STEP 2: FUNDAMENTAL VALIDATION ---
            st.markdown("---")
            st.subheader("Step 2: Fundamental Validation")
            st.info("Run a deep fundamental and earnings analysis on the technical results found above.")

            if st.button("🔬 Run Fundamental Validation on Results", type="primary", use_container_width=True):

                # Retrieve mico_advisor from session state
                mico_advisor = st.session_state.mico_advisor

                # These are the params for our "deep" analysis
                validation_params = {
                    'use_earnings_filter': True,
                    'use_fundamental_filter': True,
                    'use_multi_timeframe': True,
                    'min_pe_ratio': 5.0,
                    'max_debt_equity': 2.0
                }

                validated_trades = []
                symbols_to_validate = final_df['Symbol'].unique()

                validation_progress = st.progress(0.0, text="Starting fundamental validation...")

                for i, symbol in enumerate(symbols_to_validate):
                    validation_progress.progress((i + 1) / len(symbols_to_validate), text=f"Validating {symbol}...")

                    # We must re-fetch the *individual* stock data here because the bulk data
                    # from the screener might not be enough for all checks.
                    # Or, even better, re-use the bulk data if it's sufficient.

                    # OPTION A (Re-use bulk data - faster but less robust):
                    # stock_data = bulk_data_df.loc[symbol]

                    # OPTION B (Re-fetch data - slower but more accurate):
                    stock_data = st.session_state.data_manager.get_stock_data(symbol,
                                                                              days_back=365,
                                                                              end_date=screener_analysis_date
                                                                              )

                    if stock_data.empty:
                        continue

                    # Re-run the analysis with fundamental checks ENABLED
                    result = mico_advisor.analyze(
                        stock_data=stock_data,
                        symbol=symbol,
                        analysis_date=screener_analysis_date,
                        params=validation_params,
                        use_market_filter=use_market_filter  # Use the same market filter setting
                    )

                    if result and (result.get('signal') or result.get('action')) == 'BUY':
                        # This stock passed both technical AND fundamental checks
                        # We can just re-use the row from the original dataframe
                        original_trade = final_df[final_df['Symbol'] == symbol].iloc[0]
                        validated_trades.append(original_trade)

                validation_progress.empty()

                if not validated_trades:
                    st.warning("No stocks passed the fundamental validation filters.")
                else:
                    st.success(f"Validation Complete: Found {len(validated_trades)} high-quality opportunities.")
                    validated_df = pd.DataFrame(validated_trades)
                    st.dataframe(validated_df.style.format(formatter, na_rep='-'), use_container_width=True)

            if st.session_state.screener_date_input == datetime.now().date():
                analyze_screener_btn = st.button("🔬 Analyze Screener Results", type="primary", use_container_width=True)
            else:
                analyze_screener_btn = True
            if analyze_screener_btn: st.session_state.analysis_run = True
            if st.session_state.get('analysis_run', False):
                with st.spinner("Running backtest simulation on results..."):
                    results_analyzer.run_backtest(
                        trades_df=st.session_state.screener_results,
                        data_manager=st.session_state.data_manager,
                        initial_portfolio_value=investment_amount,
                        risk_per_trade_percent=risk_per_trade_percent,
                        use_trailing_stop=use_trailing_stop,
                        atr_trailing_mult=atr_trailing_mult
                    )
        else:
            st.info("Select a universe and click 'Scan Universe' to begin.")

    elif mode == "🗓️ System Backtest":
        st.header("Full System Backtest")
        if run_full_backtest_btn:
            st.header("RUNNING FULL SYSTEM BACKTEST")
            st.session_state.analysis_run = False
            if 'screener_results' in st.session_state:
                del st.session_state['screener_results']

            # 1. Get active advisors (same as scan_btn)
            active_advisors = {}
            if run_ai: active_advisors["AI"] = st.session_state.advisor
            if run_mico: active_advisors["MICO"] = st.session_state.mico_advisor
            if run_mean_reversion: active_advisors["MeanReversion"] = st.session_state.mean_reversion_advisor
            if run_breakout: active_advisors["Breakout"] = st.session_state.breakout_advisor
            if run_supertrend: active_advisors["SuperTrend"] = st.session_state.supertrend_advisor
            if run_ma_crossover: active_advisors["MACrossover"] = st.session_state.ma_crossover_advisor
            if run_volume_momentum: active_advisors["VolumeMomentum"] = st.session_state.volume_momentum_advisor

            if not active_advisors:
                st.warning("Please select at least one system to run for the backtest.")
            else:
                # 2. Get stock universe (same as scan_btn)
                load_function = universe_options[selected_universe_name]
                stock_universe = load_function()

                if not stock_universe:
                    st.error(f"Could not load the '{selected_universe_name}' stock list.")
                else:
                    # 3. Create Date Range
                    trading_days = pd.bdate_range(start=backtest_start_date, end=backtest_end_date)
                    all_trades = []
                    total_trades_found = 0

                    st.info(
                        f"Running full backtest from {backtest_start_date} to {backtest_end_date} for {len(trading_days)} trading days...")

                    # --- Create the Live Dashboard Placeholders ---
                    progress_bar = st.progress(0, text="Backtest in progress...")
                    kpi_cols = st.columns(3)
                    day_counter_kpi = kpi_cols[0].empty()
                    trade_counter_kpi = kpi_cols[1].empty()
                    current_date_kpi = kpi_cols[2].empty()
                    live_results_placeholder = st.empty()
                    # --- End of Dashboard Placeholders ---

                    # 4. Loop and run screener for each day
                    for i, day in enumerate(trading_days):
                        # --- CRITICAL CHANGE: Use the consolidated scanner function logic ---
                        daily_trades_list = stockwise_scanner.run_full_market_scan(
                            universe_tickers=stock_universe,
                            analysis_date=day.date(),
                            main_advisor_instance=st.session_state.advisor
                        )
                        # Filter to only BUY signals and convert to DataFrame
                        daily_trades_df = pd.DataFrame([r for r in daily_trades_list if r['signal'] == 'BUY'])

                        if daily_trades_df is not None and not daily_trades_df.empty:
                            # Remap columns to match the expected format of results_analyzer.run_backtest
                            daily_trades_df.rename(columns={
                                'Symbol': 'Symbol',
                                'Entry Price': 'Entry Price',
                                'Stop-Loss': 'Stop-Loss',
                                'Profit Target ($)': 'Profit Target',
                                'Entry Date': 'Analysis Date',
                            }, inplace=True, errors='ignore')

                            all_trades.append(daily_trades_df)
                            total_trades_found += len(daily_trades_df)
                            # Show the *latest* trades found
                            live_results_placeholder.dataframe(daily_trades_df)

                        # --- Update Live Dashboard ---
                        progress_bar.progress((i + 1) / len(trading_days), text=f"Scanning: {day.date()}")
                        day_counter_kpi.metric("Day Processed", f"{i + 1}/{len(trading_days)}")
                        trade_counter_kpi.metric("Total Trades Found", total_trades_found)
                        current_date_kpi.metric("Scanning Date", f"{day.date()}")
                    # --- End Dashboard Update ---

                    # 5. Clean up and combine results
                    progress_bar.empty()
                    day_counter_kpi.empty()
                    trade_counter_kpi.empty()
                    current_date_kpi.empty()
                    live_results_placeholder.empty()

                    if not all_trades:
                        st.warning("No trade opportunities found in the entire date range.")
                    else:
                        full_results_df = pd.concat(all_trades, ignore_index=True)
                        st.session_state.screener_results = full_results_df
                        st.session_state.analysis_run = True  # Trigger the backtest analyzer
                        st.rerun()  # Rerun to show results
        else:
            st.info("Configure dates and universe, then click 'Run Full System Backtest'.")


def _display_mico_metrics(mico_result, advisor, investment_amount=1000, is_adaptive=False):
    """Renders MICO or AI-Adaptive MICO results and metrics in a column."""
    if not mico_result:
        st.info("System not run in this mode.")
        return

    signal = mico_result.get('signal', 'N/A')
    current_price = mico_result.get('current_price', 0)
    logger.info(f"MICO RESULT CHECK: Signal={signal}, Price={current_price}, Result keys={mico_result.keys()}")
    profit_target_price = mico_result.get('profit_target_price', 0)
    stop_loss_price = mico_result.get('stop_loss_price', 0)

    # Signal Status
    if signal == 'BUY':
        st.success("**Signal: BUY**")
    else:
        st.warning(f"**Signal: {signal}**")

    # PROFIT CALCULATION (Same as original block)
    net_profit_per_share, net_profit = 0, 0
    if signal == "BUY" and current_price > 0 and profit_target_price > 0:
        hypothetical_shares = investment_amount / current_price
        gross_profit = (profit_target_price - current_price) * hypothetical_shares
        net_profit, _ = advisor.apply_israeli_fees_and_tax(gross_profit, hypothetical_shares)
        net_profit_per_share = (net_profit / investment_amount) * 100

    # --- AI ADAPTIVE DEBUGGING INDICATORS (Requirement 2) ---
    if is_adaptive:
        ai_debug = mico_result.get('ai_debug', {})
        is_adjusted = ai_debug.get('ai_adjustment_made', False)

        m_col_ai_1, m_col_ai_2 = st.columns(2)
        m_col_ai_1.metric("AI Status", "✅ ADJUSTED" if is_adjusted else "💡 DEFAULT")
        m_col_ai_2.metric("ATR Multiplier", f"{ai_debug.get('atr_mult_stop', 'N/A')}")

        with st.expander("AI Reasoning"):
            st.caption(f"Reason: {ai_debug.get('ai_predictor_reason', 'N/A')}")
            st.caption(f"Version: {ai_debug.get('ai_model_version', 'N/A')}")

    # --- METRICS (3x3 grid) ---
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Entry Price", f"${current_price:.2f}" if current_price > 0 else "-")
    m_col2.metric("Profit Target", f"${profit_target_price:.2f}" if profit_target_price > 0 else "-")
    m_col3.metric("Hypothetical Net Profit",
                  f"${net_profit:.2f}" if net_profit > 0 else "-")  # Net Profit $

    m_col4, m_col5, m_col6 = st.columns(3)
    m_col4.metric("Risk/Reward Ratio", "N/A")
    m_col5.metric("Stop-Loss", f"${stop_loss_price:.2f}" if stop_loss_price > 0 else "-")
    m_col6.metric("Net Profit [%]",
                  f"{net_profit_per_share:.2f}%" if net_profit_per_share > 0 else "-")  # Net Profit %

    with st.expander("Show Micha Rule Analysis"):
        st.markdown(f"_{mico_result['reason']}_")


# --- Main Execution ---
if __name__ == "__main__":
    # st.set_page_config(...) # Already at top
    setup_logging_from_st()
    logger.info("StockWise AI Application starting up...")

    try:
        with open('config.yaml') as file:
            config = yaml.load(file, Loader=SafeLoader)
    except FileNotFoundError:
        logger.error("FATAL: config.yaml file not found. Please create it.", exc_info=True)
        st.error("FATAL: config.yaml file not found. Please create it.")
        st.stop()
    except Exception as e:
        logger.error(f"Error loading config.yaml: {e}", exc_info=True)
        st.error(f"Error loading config.yaml: {e}")
        st.stop()

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )
    authenticator.login('main')

    name = st.session_state.get('name')
    authentication_status = st.session_state.get('authentication_status')
    username = st.session_state.get('username')

    try:
        if authentication_status:
            authenticator.logout('Logout', 'sidebar')
            st.sidebar.write(f'Welcome *{name}*')

            DEFAULT_AGENT_MODEL_DIR = "models/NASDAQ-gen3-dynamic"

            st.session_state.setdefault('watchlist', "AAPL\nMSFT\nGOOG\nNVDA")
            st.session_state.setdefault('alerts', [])
            st.session_state.setdefault('positions', {})

            if 'log_entries' not in st.session_state:
                st.session_state.log_entries = []

            if 'IS_CLOUD' not in st.session_state:
                try:
                    _ = st.secrets["gcs_service_account"]
                    st.session_state.IS_CLOUD = True
                except:
                    st.session_state.IS_CLOUD = False

            if 'data_manager' not in st.session_state:
                logger.info("Initializing Data Manager with IBKR...")

                # 1. Enable IBKR and set the correct Paper Trading port (7497)
                dm = DataSourceManager(use_ibkr=True, host=cfg.IBKR_HOST, port=cfg.IBKR_PORT)

                # 2. Set the internal flag to 'ibkr' so the UI Status indicator sees it
                dm.data_source = 'yfinance' # Default fallback source

                # 2. Attempt IBKR connection
                if dm.use_ibkr and not dm.connect_to_ibkr():
                    logger.warning("IBKR connection failed on startup. Falling back to Alpaca/YFinance.")
                    dm.use_ibkr = False

                # 3. CRITICAL: If IBKR failed, determine the next best available fallback.
                if not dm.use_ibkr:
                    # Alpaca is the highest priority fallback if keys are available
                    if dm.stock_client:
                        dm.data_source = 'alpaca'
                    else:
                        dm.data_source = 'yfinance'  # Final fallback

                st.session_state.data_manager = dm

                # if st.session_state.IS_CLOUD:
                #     # 1. Cloud Run: Default to Alpaca
                #     dm = DataSourceManager(use_ibkr=False)
                #     dm.data_source = "alpaca"  # Manually set the attribute
                #     st.session_state.data_manager = dm
                # else:
                #     # 2. Local Run: Try IBKR first
                #     dm = DataSourceManager(use_ibkr=True)
                #     dm.data_source = "ibkr"  # Set *intent* to use IBKR
                #     dm.connect_to_ibkr()  # Attempt connection
                #
                #     # 3. THIS IS YOUR FALLBACK LOGIC
                #     if not dm.isConnected():
                #         logger.warning("IBKR connection failed on startup. Falling back to Alpaca.")
                #         dm.data_source = "alpaca"
                #         dm.use_ibkr = False
                #
                #     st.session_state.data_manager = dm

            if 'contextual_data' not in st.session_state:
                with st.spinner("Loading market context data (QQQ, VIX, TLT)..."):
                    st.session_state.contextual_data = load_contextual_data(
                        st.session_state.data_manager
                    )

            if 'advisor' not in st.session_state:
                st.session_state.advisor = ProfessionalStockAdvisor(
                    model_dir=DEFAULT_AGENT_MODEL_DIR,
                    data_source_manager=st.session_state.data_manager)

            # Pass context data to calculator
            st.session_state.advisor.calculator = FeatureCalculator(
                data_manager=st.session_state.data_manager,
                contextual_data=st.session_state.contextual_data,
                is_cloud=st.session_state.IS_CLOUD)

            if 'mico_advisor' not in st.session_state:
                st.session_state.mico_advisor = MichaAdvisor(
                    data_manager=st.session_state.data_manager)

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
                st.session_state.volume_momentum_advisor = VolumeMomentumAdvisor(
                    data_manager=st.session_state.data_manager)

            # --- Initialize AI Param Predictor (for MICO A/B Testing) ---
            if 'ai_param_predictor' not in st.session_state:
                st.session_state.ai_param_predictor = AI_ParamPredictor()

            if not st.session_state.advisor.models:
                st.warning("⚠️ AI (Gen-3) models failed to load. Only MICO and other systems will be available.")

            create_enhanced_interface(st.session_state.IS_CLOUD)
        elif authentication_status is None:
            st.warning('Please enter your username and password')
        elif not authentication_status:
            st.error('Username/password is incorrect')
    except Exception as e:
        logger.error(f"Error Authentication Status: {e}", exc_info=True)
        st.error(f"Error Authentication Status: {e}")
        st.stop()

