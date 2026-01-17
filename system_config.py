# system_config.py

"""
StockWise System Configuration
==============================
Centralized settings for the trading system.
"""

import os
from datetime import datetime, timedelta, date
import streamlit as st
import logging

# --- Import and load dotenv ---
try:
    from dotenv import load_dotenv
    load_dotenv() # Load variables from .env file immediately
except ImportError:
    pass


# logger = logging.getLogger(__name__)


# --- 1. FILE SYSTEM ---
# --- PATHS ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
CHARTS_DIR = os.path.join(PROJECT_ROOT, 'debug_charts')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# Ensure directories exist
for d in [LOGS_DIR, CHARTS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# ---2. TRADING TARGETS ---
TARGET_TICKER = "NVDA"
BENCHMARK_TICKER = "QQQ"
SECTOR_TICKER = "SMH"           # Default, can be auto-detected

# list of symbols to trade
WATCHLIST = ["NVDA", "AMD", "MSFT", "GOOGL", "AAPL", "META", 
            "TSLA", "QCOM", "AMZN", "INTC", "LLY", "JPM", "WMT", "ORCL", "PLTR", "MU"]


# # --- 3. DATA CONFIG ---
# # We want enough data for 200 EMA calculation + Testing window
# DATA_END_DATE = datetime.now().date()
# DATA_START_DATE = DATA_END_DATE - timedelta(days=1200)
# # # Total fetch range = chart window + warmup buffer
# # DATA_END_DATE = date.today()



# Minimal data required for indicators (SMA 200 needs ~1 year buffer)
##########################################
# DO NOT CHANGE THIS below 1.0
MIN_WARMUP_YEARS = 1.5
##########################################

# # score Treshold for buy signal
# SCORE_THRESHOLD_BUY = 60

# History length for indicators
# Minimum bars needed for indicators
HISTORY_LENGTH_DAYS = 200
# CHART_YEARS = 0.5          # Visualization window
# How many years of results do you want to see on the chart?
CHART_YEARS = 2

# Simulation Look Ahead
LOOK_AHEAD_DAYS = 20

# --- 4. CONNECTIVITY ---
EN_IBKR = False
IBKR_HOST = '127.0.0.1'
IBKR_PORT = 7497
IBKR_CLIENT_ID = 1

from datetime import datetime, timedelta

# --- 3. SYSTEM CONFIGURATION ---
TIMEFRAME = "1d" # User requested 1-Day Interval
# Ensure these are datetime objects
DATA_END_DATE = datetime(2025, 12, 24) # Fixed end date for simulation consistency
DATA_START_DATE = datetime(2023, 1, 1)

# Defines how aggressive the system should be.
RISK_PROFILES = {
    "Conservative": {
        "buy_threshold": 100,      # ROYAL FLUSH: Tech(80) + ML(25) = 105
        "stop_atr": 2.5,           # Tighter Stop for Precision Entries
        "reward_ratio": 1.0,       # Hit Rate Focus (1.0 ATR Target)
        "position_size_mult": 2.0  # Conviction
    },
    "Moderate": {
        "buy_threshold": 75,       # Stricter than previous
        "stop_atr": 2.5,           # Standard 2ATR stop
        "reward_ratio": 2.0,
        "position_size_mult": 1.0
    },
    "Aggressive": {
        "buy_threshold": 60,       # Loose entry (Catch everything) - changed from 55 to 60
        "stop_atr": 3.0,           # Wide stop
        "reward_ratio": 1.5,
        "position_size_mult": 1.0   # Full size - changed from 1.5 to 1.0
    }
}

# --- SELECT ACTIVE PROFILE ---
ACTIVE_PROFILE_NAME = "Conservative"            # <--- "Conservative", "Moderate", "Aggressive"
ACTIVE_PROFILE = RISK_PROFILES[ACTIVE_PROFILE_NAME]

# Apply Settings dynamically
SCORE_THRESHOLD_BUY = ACTIVE_PROFILE["buy_threshold"]
TRAILLING_STOP_ATR = ACTIVE_PROFILE["stop_atr"]
RISK_REWARD_RATIO = ACTIVE_PROFILE["reward_ratio"]

# STOP_LOSS_ATR = 2.5   # Increased from default to give trades room to breathe
TAKE_PROFIT_ATR = 5.0 # Maintain high reward target (2.0 * 2.5 = 5.0)

# --- CRITICAL: TIME INTERVAL ---
# "1h" = Sniper Mode (Swing Trading on Hourly Charts)
# Options: "1d", "1h", "4h", "15m"
TIMEFRAME = "1h"

# CORRECTED LOGIC: Calculate the earliest date needed based on the largest requirement.
REQUIRED_YEARS = max(CHART_YEARS, MIN_WARMUP_YEARS)

# Use REQUIRED_YEARS for the timedelta calculation
_total_days = int(REQUIRED_YEARS * 365)
if TIMEFRAME == "1h":
    _total_days = min(_total_days, 700) # Cap at 700 days for 1h YFinance limit
DATA_START_DATE = DATA_END_DATE - timedelta(days=_total_days)

# Print warning only if chart view is smaller than minimum required warmup
if CHART_YEARS < MIN_WARMUP_YEARS:
    logger.debug(f"⚠️ Warning: CHART_YEARS ({CHART_YEARS}) is less than MIN_WARMUP_YEARS ({MIN_WARMUP_YEARS}).")

# --- API SETTINGS ---
DATA_PROVIDER = "YFINANCE"  # Options: 'ALPACA', 'IBKR', 'YFINANCE'
# Credentials (Streamlit Secrets or Environment)
# Credentials (Streamlit Secrets or Environment or TOML)
ALPACA_KEY = None
ALPACA_SECRET = None

# 1. Try Streamlit Secrets (if running in Streamlit)
try:
    if st.secrets:
        ALPACA_KEY = st.secrets.get("APCA_API_KEY_ID")
        ALPACA_SECRET = st.secrets.get("APCA_API_SECRET_KEY")
except:
    pass

# 2. Try TOML File (for standalone scripts)
if not ALPACA_KEY:
    try:
        import tomllib
        # Check common paths
        paths = [
            os.path.join(os.getcwd(), ".streamlit", "secrets.toml"),
            os.path.join(PROJECT_ROOT, ".streamlit", "secrets.toml"),
            os.path.join(os.path.dirname(PROJECT_ROOT), ".streamlit", "secrets.toml")
        ]
        
        for p in paths:
            if os.path.exists(p):
                with open(p, "rb") as f:
                    secrets = tomllib.load(f)
                    ALPACA_KEY = secrets.get("APCA_API_KEY_ID")
                    ALPACA_SECRET = secrets.get("APCA_API_SECRET_KEY")
                    if ALPACA_KEY:
                        logger.info(f"Loaded credentials from {p}")
                        break
    except Exception as e:
        logger.warning(f"Failed to parse secrets.toml: {e}")

# 3. Fallback to Environment Variables
if not ALPACA_KEY:
    ALPACA_KEY = os.getenv("APCA_API_KEY_ID")
    ALPACA_SECRET = os.getenv("APCA_API_SECRET_KEY")

if not ALPACA_KEY:
    logger.warning("Alpaca Credentials NOT FOUND. Live Trading will fail.")

# Telegram Bot Credentials
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


# NOTE: We need a placeholder value for investment for PnL calculation
# INVESTMENT_AMOUNT = 1000 # Hardcoded investment for hypothetical PnL
PnL_THRESHOLD_PCT = 0.5  # Placeholder for PnL percentage calculation
WIN_THRESHOLD_USD = 10.0  # Minimum profit to consider a trade a "win"
WIN_THRESHOLD_PCT = 1.0  # Minimum profit percentage to consider a trade a "win"

# --- 6. INVESTMENT SETTINGS ---
INVESTMENT_AMOUNT = 1000.0 * ACTIVE_PROFILE["position_size_mult"]
FEE_PER_SHARE = 0.005 # IBKR Pro Tier approx

# # Logic:
# # 1. Standard Broker Fee is often around $2.50 min.
# # 2. But for small accounts, we simulate a "Tiered" plan (like IBKR) where min is ~$0.35-$1.00.
# # 3. SAFETY CAP: We cap the minimum fee at 0.1% (10 bps) of equity to prevent the "Death Spiral".
# # Calculate the Cap (0.1% of investment)
# _fee_cap = INVESTMENT_AMOUNT * 0.001
# # Set MINIMUM_FEE to the lower of $2.50 or the Safety Cap, but never less than $0.35 (IBKR Tiered Min)
# MINIMUM_FEE = max(0.35, min(2.50, _fee_cap))

TAX_RATE = 0.25             # 25% Capital Gains Tax
COMMISSION = 1.0            # $1 per trade
SCORE_TRIGGER_BREAKOUT = 40  # Breakout trigger score - adjusted from 55 to 40
SCORE_TRIGGER_DIP = 40    # Dip buy trigger score

# Confirmation Weights (Boosted to allow triggering)
SCORE_CONFIRM_VOLUME = 20      # Volume confirmation - increased from 15 to 20
SCORE_CONFIRM_MACD = 10       # MACD confirmation
SCORE_CONFIRM_ADX = 5        # ADX trend strength confirmation
SCORE_CONFIRM_SECTOR = 15   # Sector strength confirmation -increased from 10 to 15
SCORE_CONFIRM_SLOPE = 10    # Slope confirmation

# Penalties
SCORE_PENALTY_RSI = -25     # Overbought penalty
SCORE_PENALTY_SECTOR = -20  # Sector weakness penalty
SCORE_PENALTY_NOISE = -20   # High noise penalty

# --- GEN-7 RF SCORING ---
SCORE_CONFIRM_KALMAN = 15    # Kalman filter confirmation - increased from 10 to 15
SCORE_CONFIRM_WAVELET = 15  # Wavelet noise confirmation

# --- 5. STRATEGY PARAMETERS (Gen-6 Defaults) ---
# Used if JSON optimization file is missing
STRATEGY_PARAMS = {
    'sma_short': 20,        # Short-term SMA period
    'sma_long': 100,    # Long-term SMA period
    'rsi_threshold': 75,    # Overbought RSI threshold
    'atr_mult_stop': 2.5,   # ATR multiplier for stop loss
    'slope_threshold': 15,      # Minimum angle for breakout (ADJUSTED: WAS 20)
    'adx_threshold': 25,        # Minimum trend strength
    'vol_multiplier': 1.1,       # Volume > 110% of average
    # --- GEN-7 RF PARAMETERS ---
    'kalman_smooth_threshold': 0.5, # Max deviation (in dollars) from smooth trend for entry (local extrema)
    'wavelet_noise_max': 1.5,       # Max wavelet noise ratio for entry (1.0 = average noise)
}

# --- 7. RISK MANAGEMENT (Phase 3 Prep) ---
RISK_REWARD_RATIO = 2.0     # Aim for $2 profit for every $1 risk
# Standard stop distance: Using 2.0x ATR for volatility-adjusted sizing
TRAILLING_STOP_ATR = 2.0
TIGHT_STOP_ATR = 2.0        # Same as TRAILLING_STOP_ATR

# --- 8. TRADING COSTS (IBKR Tiered Simulation) ---
# IBKR Tiered: ~$0.0035/share, Min $0.35, Max 1% of trade value.
FEE_PER_SHARE = 0.0035      # Tiered rate (approx)
MINIMUM_FEE = 0.35          # IBKR Tiered Min per order
MAX_FEE_PCT = 0.01          # IBKR Max fee is 1% of trade value (safety cap)

TAX_RATE = 0.25             # 25% Capital Gains

# --- GEN-9 SNIPER CONFIGURATION ---
class SniperConfig:
    """
    Configuration for the Gen-9 'Fusion Sniper' System.
    Defines strict thresholds for High-Precision Swing Trading.
    """
    # 1. Targets
    TARGET_PROFIT = 0.04        # +5% Minimum Target per trade
    MAX_DRAWDOWN = -0.02        # -2% Hard Stop Loss
    SIMULATION_STARTING_CAPITAL = 1000.0 # Starting Capital for Verification Simulation
    
    # 2. AI Thresholds
    # Model is "Timid" (High Precision Loss suppresses probabilities).
    # We lower threshold to 0.40 to capture high-quality setups (previously peaked at ~45%).
    MODEL_CONFIDENCE_THRESHOLD = 0.55   # Minimum AI Probability to Pull Trigger
    
    # 3. Training Penalties
    LOSS_PENALTY_MULTIPLIER = 15.0     # Weighted Loss Penalty for False Positives
    
    # 4. Filters
    FUNDAMENTAL_MIN_SCORE = 70         # Minimum Fundamental Score (0-100)
    
    # 5. Alerting
    ENABLE_TELEGRAM_ALERTS = True

# --- 10. SCHEDULER CONFIGURATION ---
# --- 10. SCHEDULER CONFIGURATION ---
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, nearest_workday, Holiday,
    USMartinLutherKingJr, USPresidentsDay, USMemorialDay,
    USLaborDay, USThanksgivingDay, GoodFriday
)

class NYSEHolidayCalendar(AbstractHolidayCalendar):
    """
    US Stock Market Holiday Calendar (NYSE).
    Includes specific rules different from Federal holidays.
    """
    rules = [
        Holiday('New Years Day', month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday('Juneteenth', month=6, day=19, observance=nearest_workday),
        Holiday('Independence Day', month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday('Christmas', month=12, day=25, observance=nearest_workday)
    ]

def get_dynamic_holidays(years=2):
    """Generate NYSE holidays for current year + N years forward."""
    try:
        cal = NYSEHolidayCalendar()
        start_year = datetime.now().year
        end_year = start_year + years
        # Generate range
        holidays = cal.holidays(start=datetime(start_year, 1, 1), end=datetime(end_year, 12, 31))
        return [d.strftime('%Y-%m-%d') for d in holidays]
    except Exception as e:
        logger.error(f"Failed to auto-calc holidays: {e}. using Fallback.")
        return []

class SchedulerConfig:
    MARKET_TIMEZONE = 'US/Eastern'
    from datetime import time
    OPEN_TIME = time(9, 30)
    CLOSE_TIME = time(16, 0)
    PRE_BUFFER_HOURS = 2
    POST_BUFFER_HOURS = 2
    
    # Auto-Calculate Holidays (Current Year + Next 2 Years)
    MARKET_HOLIDAYS = get_dynamic_holidays(years=2)







