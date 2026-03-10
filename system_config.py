# system_config.py

"""
StockWise Gen-12 System Configuration
=====================================
Single Source of Truth.
Contains all static parameters, API keys, strategy weights, schedules, and fee/tax structures.
"""

# Import necessary standard libraries
import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, time
import streamlit as st
import sys
import re
import json
import pandas as pd


# --- LOAD ENVIRONMENT VARIABLES ---
# Attempt to load environment variables from a .env file if it exists (for local dev)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If dotenv is not installed, proceed without it (rely on system env vars)
    pass


# Feature Flags
EN_ORCHESTRAL = True  # Enable AI Orchestrator

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DATA DIRECTORY
DATA_DIR = os.path.join(BASE_DIR, "data")
# CREATE DATA DIRECTORY IF NOT EXISTS
os.makedirs(DATA_DIR, exist_ok=True)

# Sets the operational log level (Console/Basic Info)
LOG_LEVEL_CONSOLE = "INFO"
# Sets the forensic log level (File/Deep Analysis)
LOG_LEVEL_FILE = "DEBUG"
# Control switch for writing debug logs to file (True = Enabled, False = Disabled)
ENABLE_DEBUG_FILE_LOGGING = True

# --- SYSTEM FILE PATHS ---
# Default path for the debug log file
LOG_FILE_PATH = "logs/stockwise_debug.log"
# Path for the machine learning trade journal
TRADE_JOURNAL_PATH = "data/trade_journal.json"
# Path for the VIP output scan
SCANNER_OUTPUT_PATH = "data/vip_scanner_results.json"
# --- EXECUTION & RISK MANAGEMENT CONFIGURATION ---
# The target path for the time-based blacklist file to prevent wash-trade loops
COOLDOWN_FILE_PATH = "data/cooldown_list.json"
# Duration in hours a ticker stays blacklisted after stop-loss/zombie exit
# Valid range: 1-168 (1 hour to 7 days). Default: 24 hours.
COOLDOWN_PERIOD_HOURS = 24
# Path for the historical closed trades CSV for admin review and ML insights
TRADE_HISTORY_CSV_PATH = "data/trade_history.csv"

# --- ML ENGINE CONFIGURATION ---
# Maximum allowed days for a setup to mature before it gets mathematically penalized (Melting Period)
MAX_MELTING_PERIOD_DAYS = 7

# The status flag representing a trade that was actually executed by the user via Telegram
TRADE_STATUS_EXECUTED = "CONFIRMED"
TRADE_STATUS_UNFILLED = "UNFILLED"

# Base assumed friction per trade (Spread + Commissions + Margin Rate)
BASE_FRICTION = 0.003

# Strict Alpha Equation threshold: Minimum net profit after friction required to execute
MIN_NET_PROFIT = 0.005

# FULL PATHS
VIP_LIST_PATH = os.path.join(DATA_DIR, "daily_review_list.json")
LEDGER_PATH = os.path.join(DATA_DIR, "scan_ledger.json")
LOG_DIR_LOCAL = os.path.join(BASE_DIR, "logs")
# Exact requested path for Google Drive
DIR_DRIVE = r"G:\My Drive\StockWise_AI_Trading_System"
LOG_DIR_DRIVE = os.path.join(DIR_DRIVE,"logs")
CODE_DIR_DRIVE = os.path.join(DIR_DRIVE,"Code")

# Ensure Log Directories Exist
os.makedirs(LOG_DIR_LOCAL, exist_ok=True)
if os.path.exists(DIR_DRIVE): 
    os.makedirs(LOG_DIR_DRIVE, exist_ok=True)
    os.makedirs(CODE_DIR_DRIVE, exist_ok=True)

# --- LOGGING SILENCERS ---
# Silence verbose third-party libraries
logging.getLogger("yfinance").setLevel(logging.WARNING)
logging.getLogger("peewee").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# --- 1. SYSTEM PATHS ---
# Define the root directory of the project
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
STRATEGY_MAP_FILE = os.path.join(PROJECT_ROOT, "ticker_strategies.json")

# Define subdirectories for logs, models, and data
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
DB_DIR = os.path.join(PROJECT_ROOT, 'data')
TEMPLATES_DIR = os.path.join(DB_DIR, "templates")

# Ensure that the necessary directories exist; create them if they do not
for d in [LOGS_DIR, MODELS_DIR, DB_DIR, TEMPLATES_DIR]:
    os.makedirs(d, exist_ok=True)

# --- 2. API CREDENTIALS ---
## --- DATA PROVIDER SWITCHES ---
# Master toggles to Enable/Disable specific brokers.
# Set multiple to True to allow "Waterfall" fallback (e.g., Massive -> Alpaca).
# Set only one to True to force that specific provider.

EN_MASSIVE = True   # Primary: High-speed Data
EN_ALPACA = True    # Secondary: Reliable Historical
EN_IBKR = True     # Execution/Realtime (Keep False if TWS is closed)
EN_YFINANCE = True  # Fallback: Last resort

# --- DATA PACING CONFIGURATION (Seconds between requests) ---
# Defines how long the scanner waits between fetching stock data to avoid HTTP 429.
# Lower is faster, but higher risk of bans.
PROVIDER_DELAY = {
    "MASSIVE": 12.5,   # High performance
    "IBKR": 0.05,      # 50 req/sec limit (0.02s theoretical, 0.05s safe)
    "ALPACA": 2.5,     # Free tier is rate-limited (~200/min)
    "YFINANCE": 1.0,   # Aggressive scraping protection
    "DEFAULT": 0.5     # Safe fallback
}

# Validation (Optional sanity check)
if not any([EN_MASSIVE, EN_ALPACA, EN_IBKR, EN_YFINANCE]):
    logging.error("WARNING: All data providers are DISABLED. System will starve.")
    raise ValueError("At least one data provider must be enabled.")
else:
    pass

# Initialize Alpaca credentials to None
# Try loading from .streamlit/secrets.toml first (User requirement)
try:
    import toml
    # Path to the secrets file used by Streamlit
    secrets_path = os.path.join(PROJECT_ROOT, ".streamlit", "secrets.toml")
    # Check if the secrets file exists
    if os.path.exists(secrets_path):
        secrets = toml.load(secrets_path)
        # Extract Alpaca API key and secret from the TOML file
        ALPACA_KEY = secrets.get("APCA_API_KEY_ID") # Matches secrets.toml
        ALPACA_SECRET = secrets.get("APCA_API_SECRET_KEY")
        # Extract Massive API key from the TOML file
        MASSIVE_API_KEY = secrets.get("MASSIVE_API_KEY")

except Exception as e:
    # Log a warning if loading secrets fails, but continue ensuring the app doesn't crash
    logging.warning(f"Failed to load secrets.toml: {e}")

# Fallback to Environment Variables if keys were not found in secrets.toml
if not ALPACA_KEY: ALPACA_KEY = os.getenv("APCA_API_KEY_ID")
if not ALPACA_SECRET: ALPACA_SECRET = os.getenv("APCA_API_SECRET_KEY")

# Define the base URL for Alpaca Paper Trading API
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

# Telegram Bot Credentials (loaded from environment variables for security)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# --- IBKR SETTINGS ---
# Interactive Brokers Configuration
IBKR_HOST = '127.0.0.1'  # Localhost IP for TWS/Gateway
IBKR_PORT = 7497         # Default port for Paper Trading TWS
IBKR_CLIENT_ID = 1       # Client ID to identify this specific connection (ensure uniqueness if multi-process)

# investment amount
INVESTMENT_AMOUNT = 1000

# Data Routing Preferences
# Determines which provider to use for different data needs
HISTORICAL_SOURCE = "YFINANCE" # Use YFinance for bulk history (efficient for long ranges > 5 days)
REALTIME_SOURCE = "IBKR"      # Use IBKR/Alpaca for live/real-time checks

# Data Range Settings
DATA_START_DATE = "2023-01-01" # Fixed start date for consistency in backtesting/training
DATA_END_DATE = None # None implies 'Now' or 'Present' in most data fetchers

# --- 3. TRADING TARGETS ---
# Dynamic Loading Wrapper for the Watchlist
def load_dynamic_watchlist():
    """Loads the active watchlist from JSON, falls back to seed list."""
    import json
    # Use absolute path to avoid current working directory issues
    path = os.path.join(DB_DIR, "dynamic_watchlist.json")
    # Check if the dynamic watchlist file exists
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                # Load the JSON and return the 'tickers' list, defaulting to empty list on failure
                return json.load(f).get("tickers", [])
        except:
            # If read fails, fail silently and proceed to fallback
            pass
    # Fallback Seed List (Top 10 S&P 500 by weight) - Used if no dynamic list found
    return ["NVDA", "MSFT", "AAPL", "AMZN", "META", "GOOGL", "TSLA", "BRK-B", "LLY", "AVGO"]

# Global Variable WATCHLIST now calls the function to load the latest list
WATCHLIST = load_dynamic_watchlist()
BENCHMARK_TICKER = "QQQ" # Ticker used as the market benchmark (Nasdaq-100 ETF)

# --- 3.5 TRADE TYPE CONFIGURATION (NEW) ---
# Defines intervals and lookback periods for data fetching based on trade duration
TRADE_TYPE_CONFIG = {
    "SHORT": {"interval": "15m", "days_back": 30},  # For scraping/intraday
    "MID":   {"interval": "1h",  "days_back": 60},  # For swing trading
    "LONG":  {"interval": "1d",  "days_back": 730}  # For long-term investing
}

# --- 4. STRATEGY CONFIGURATION (SRS 2.A) ---
# Configuration parameters for different trading strategies (Agents)
STRATEGY_CONFIG = {
    "SNIPER": {
        "description": "Short-term momentum scalping",
        "timeframe": "1h",
        "horizon_days": (1, 5),      # Expected holding period
        "min_ai_confidence": 0.55,    # AI Model confidence threshold to enter
        "rsi_max": 70,                # RSI Ceiling for entry (avoid overbought)
        "stop_loss_atr": 1.5,         # Stop Loss based on 1.5x ATR
        "target_profit_atr": 3.0      # Take Profit based on 3.0x ATR (2:1 Reward/Risk)
    },
    "TACTICAL": {
        "description": "Mid-term trend following",
        "timeframe": "1d",
        "horizon_months": (1, 6),     # Holding period: 1-6 months
        "min_earnings_growth": 0.0,   # Logic requires positive Earnings Growth
        "stop_logic": "SMA50",        # Stop loss triggers on break of 50-day SMA
        "trailing_trigger_pct": 0.10  # Activate trailing stop after +10% gain
    },
    "STRATEGIC": {
        "description": "Long-term value investing",
        "timeframe": "1wk",
        "horizon_years": (0.5, 3.0),  # Holding period: 6 months to 3 years
        "max_peg": 1.5,               # Valuation filter: PEG Ratio under 1.5
        "exit_logic": "Fundamental Decay" # Exit if fundamentals deteriorate
    }
}

# --- 5. SCAN SCHEDULE (SRS 2.A) ---
# Schedule for running different types of market scans
SCAN_SCHEDULE = {
    "SHORT_RANGE": {
        "interval": "1h", # Runs every hour
        "type": "Sniper",
        "offset_minutes": 30    # Run at HH:30 (e.g., 9:30, 10:30)
    },
    "MID_RANGE": {
        "primary_time": "20:00", # Run primarily at 8:00 PM (Nightly)
        "intraday_check_times": ["09:45", "11:00", "13:00", "15:00"], # Specific intraday check times
        "type": "Tactical"
    },
    "LONG_RANGE": {
        "intraday_check_times": ["12:30"], # Value Dip Check at 12:30 PM
        "type": "Strategic"
    }
}

# --- NEW: MASTER PATTERN SCORING MATRIX (63 Functions) ---
# Scores assigned to identified chart patterns.
# Positive Scores (>0) imply Bullish sentiment.
# Negative Scores (<0) imply Bearish sentiment.
MASTER_SCORES = {
    # Group 1: Trend Indicators
    "GOLDEN_CROSS": 25,             # 50 SMA crosses above 200 SMA
    "DEATH_CROSS": -25,             # 50 SMA crosses below 200 SMA
    "FIB_TREND_5_8_13": 20,         # Fibonacci EMA Alignment
    
    # Group 2: Geometric Patterns (High Confidence)
    "TRIANGLE_ASCENDING": 30,       # Bullish continuation
    "TRIANGLE_DESCENDING": -30,     # Bearish continuation
    "HEAD_AND_SHOULDERS_TOP": -40,  # Strong Bearish Reversal
    "HEAD_AND_SHOULDERS_BOTTOM": 40,# Strong Bullish Reversal
    "DOUBLE_TOP": -35,              # Bearish Reversal
    "DOUBLE_BOTTOM": 35,            # Bullish Reversal
    "CUP_AND_HANDLE": 35,           # Bullish Continuation
    "RISING_WEDGE_RSI_DIV": -45,    # Strong Bearish Reversal (Wedge + Divergence)
    
    # Group 3: Candlestick Patterns (Context Aware)
    "TWEEZER_BOTTOM": 15,           # Bullish Reversal
    "TWEEZER_TOP": -15,             # Bearish Reversal
    "SMART_HAMMER": 10,             # Bullish Reversal (requires context)
    "SMART_SHOOTING_STAR": -10,     # Bearish Reversal (requires context)
    
    # Group 4: Volatility & Levels
    "BOLLINGER_SQUEEZE": 10,        # Volatility contraction (Potential Breakout Setup)
    "FIB_618_BOUNCE": 25,           # Bounce off 61.8% Retracement (Bullish)
    "CONSOLIDATION_COIL": 0,        # Neutral (Signal to wait for breakout)
    
    # Group 5: Fundamental (Strategic)
    "GRAHAM_VALUE_BUY": 50,         # Fundamental Value Signal
    "NCAV_BARGAIN": 60              # Deep Value (Net Current Asset Value)
}

# --- 6. RISK CONFIGURATION (SRS 2.A) ---
# Global risk parameters for the trading system
RISK_CONFIG = {
    "max_daily_loss_usd": 500.0,    # Max allowed dollar loss per day
    "max_daily_loss_pct": 0.015,    # Max allowed portfolio % loss per day (-1.5%)
    "target_daily_profit_usd": 1000.0, # Target dollar profit per day
    "spy_crash_trigger_pct": -0.015, # Intraday SPY drop of 1.5% triggers "Crash Mode"
    "starting_capital": 25000.0     # Baseline capital for calculations (PDT Rule threshold)
}

# class SniperConfig:
#     """Configuration for the Sniper (Momentum) Strategy."""
#     LOSS_PENALTY_MULTIPLIER = 1.5       # Penalty factor for stocks with recent losses
#     FUNDAMENTAL_MIN_SCORE = 50          # Minimum fundamental score required
#     MODEL_CONFIDENCE_THRESHOLD = 0.75   # Adjusted for High Precision / Moderate Recall

# --- GEN-13: ARCHITECTURAL THRESHOLDS & TRACKING ---
# Minimum master score for a BUY to survive evaluate_ticker gate.
# Range: 50.0-85.0. Set above TacticalSniper BUY threshold (60) but below
# unreachable levels. Friction Alpha veto is the real quality filter.
MIN_MASTER_SCORE_APPROVAL = 65.0

# Premium trades threshold (Used to track high-quality setups killed by friction)
PREMIUM_TRADE_THRESHOLD = 75.0

# Path to the Missed Opportunities Forward Testing Ledger
MISSED_OPPORTUNITIES_PATH = os.path.join(DATA_DIR, "missed_opportunities.json")

# Amnesia Loop Prevention: Cooldown periods in minutes
VETO_COOLDOWN_MINUTES = 30
DATA_STARVATION_COOLDOWN_MINUTES = 120

# --- 7. COSTS CONFIGURATION (SRS 2.A - New) ---
# Parameters for simulating trading costs in paper/backtest mode
COSTS_CONFIG = {
    "commission_per_share": 0.005,  # IBKR Pro rate structure
    "min_commission": 1.00,         # Minimum per order
    "slippage_pct": 0.001,          # 0.1% artificial slippage penalty for realism
    "tax_rate": 0.25,               # 25% Capital Gains Tax rate for net profit calc
    # Friction-Adjusted Alpha Thresholds (The Hurdle Rate)
    "min_net_profit_pct": 0.013,   # Trade must yield > 1.5% net profit
    "min_net_rr": 1.2              # Reward must be > 1.5x the Risk AFTER fees
}

# --- 7.5. INSTITUTIONAL SYSTEM CONSTRAINTS (GEN-12) ---
# We are creating the memory banks for the AI Orchestra here. 
# These parameters dictate the rules of engagement before any agent is allowed to trade.

DSP_CONFIG = {
    # The Digital Signal Processing constraints. 
    # This acts as the gatekeeper, telling the Regime Router Agent if the market is trending or chopping.
    "er_lookback_slow": 20,           # We look back 20 days to define the core, unshakeable trend.
    "er_lookback_fast": 5,            # We look back 5 days as an early warning system for violent whipsaws.
    "threshold_coherent_trend": 0.55, # If the Efficiency Ratio is above this, we activate the Trend AI.
    "threshold_stochastic_chop": 0.30 # If the Efficiency Ratio is below this, we activate the Mean-Reversion AI.
}

FRICTION_AND_ALPHA = {
    # This is our mathematical 'Hurdle Rate'. The system will refuse to trade if the broker/government takes too much.
    "min_net_profit_pct": 0.013,      # A trade MUST yield > 1.5% in pure, take-home cash.
    "min_net_rr": 1.2,                # The net reward must be at least 1.5 times the net risk.
    "max_spread_pct": 0.0005          # Microstructure Veto: We reject the stock if the Bid-Ask spread is > 0.05%.
}

VOLUMETRIC_LIMITS = {
    # This protects us from our own size. We cannot buy so much that we push the price against ourselves.
    "max_adv_participation_pct": 0.01 # We will NEVER buy more than 1% of the stock's 10-day Average Daily Volume.
}

KINETIC_STOP_CONFIG = {
    # The rules for Agent 4 (The Lifecycle Manager). The stop-loss accelerates as profit grows.
    "phase1_atr_mult": 2.0,                  # When we enter, we give the stock a wide 2.0 ATR breathing room.
    "phase2_breakeven_trigger_pct": 0.015,   # Once we hit 1.5% net profit, we instantly snap the stop to breakeven.
    "phase3_parabolic_trigger_pct": 0.03,    # At 3.0% net profit, the stock is flying. We activate the choke mechanism.
    "phase3_atr_mult": 1.0                   # The choke mechanism tightens the stop to just 1.0 ATR from the highest high.
}

MILESTONE_ALERT_CONFIG = {
    # Real Breakeven: first alert only fires after profit covers all costs
    # (commissions + slippage + tax). This buffer is added on top of actual costs.
    "safe_zone_buffer_pct": 0.002,      # 0.2% safety margin above true breakeven

    # Event-driven alert triggers -- controls when user gets notified
    "min_stop_change_pct": 0.01,         # Alert only if stop moved > 1% of current price
    "min_alert_interval_minutes": 15,    # Minimum 15 min between alerts per ticker

    # Phase 4 Runner Mode: replaces hard take_profit with ultra-tight trailing
    "runner_atr_mult": 0.5,              # Runner stop = highest_high - (ATR * 0.5)
    "runner_min_distance_pct": 0.008,    # Floor: stop never closer than 0.8% from high
                                          # Prevents noise exit when ATR is tiny
}

PORTFOLIO_DEFENSE = {
    # Structural risk management to prevent massive account blowups.
    "max_covariance_corr": 0.85,         # If the new stock moves exactly like our current portfolio (>85% correlation), we veto it.
    "zombie_trade_ttl_hours": 72,        # If a trade's regime changes, we give the "Zombie Agent" 72 hours to exit before we force-liquidate.
    "event_horizon_buffer_days": 2       # We will force-sell all active trades 2 days before any scheduled Earnings/FDA event.
}

SCAN_ROUTING_CONFIG = {
    # The Multi-Level Feedback Queue (MLFQ). This tells the Nightly Scanner how to allocate CPU power.
    "daily_scan_limit": 4000,             # We check 500 standard stocks sequentially every night to find new blood.
    "priority_scan_limit": 100,           # We check the top 50 hottest stocks first, every single night.
    "max_daily_review_stocks": 10,        # The absolute top 5 stocks are promoted to the active intraday watchlist.
    "min_vip_score_threshold": 75.0,      # A target must hit this baseline score to even be considered for VIP.
    "max_days_untraded_on_watchlist": 210, # If a stock sits on the VIP list for 7 months doing nothing, we throw it in the garbage.
    "weight_score_mult": 0.7,            # 70% of tomorrow's scan priority is based on the DSP Score.
    "weight_volatility_mult": 0.3        # 30% of tomorrow's scan priority is based on raw volatility.
}

# Mandatory Structural Templates: run on every stock before any trading analysis.
# These classify the stock's state -- they don't generate buy/sell signals.
MANDATORY_SCAN_CONFIG = {
    # Trend Direction thresholds
    "trend_bullish_min_sma_slope": 0.0,     # SMA_50 slope > 0 = rising
    "trend_weekly_confirmation": True,       # Require weekly SMA alignment too

    # Structure Detection
    "support_resistance_lookback": 60,       # Days to look back for S/R levels
    "near_level_pct": 0.02,                  # Within 2% of S/R = "near"

    # Volume Health
    "min_avg_volume": 500000,                # Minimum 500K avg daily volume
    "volume_trend_lookback": 20,             # Days for volume trend

    # Volatility State
    "squeeze_bb_width_threshold": 0.10,      # BB width < 10% = compressed
    "volatile_bb_width_threshold": 0.30,     # BB width > 30% = volatile
}

# Priority Tier Configuration -- determines intraday scan frequency based on score
SCAN_TIER_CONFIG = {
    # Tier 1 (VIP): master_score >= 85 -- scanned every 20 minutes all day
    "tier1_min_score": 85.0,
    "tier1_scan_interval_minutes": 20,

    # Tier 2 (Watch): master_score 75-84.9 -- scanned 3x/day (top 10 from this range)
    "tier2_min_score": 75.0,
    "tier2_max_count": 10,
    "tier2_scan_times": ["09:30", "12:30", "15:30"],

    # Tier 3 (Pool): below 75 -- nightly/morning scan only
    # (no special config needed -- default full scan behavior)

    # Full scan schedule (all stocks)
    "full_scan_times": {
        "morning": "08:00",     # EST -- 1.5h before market open (15:00 Israel)
        "evening": "16:15",     # EST -- 15 min after market close (23:15 Israel)
    }
}

# Default symbols for AI training when scanner results are not available
DEFAULT_TRAINING_SYMBOLS = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN',
                             'META', 'TSLA', 'AMD', 'NFLX', 'SPY']

# AI Label Generation Config: defines what counts as a "profitable" trade for training
AI_LABEL_CONFIG = {
    "lookahead_days": 5,        # How many days forward to look for profit (range: 1-20)
    "profit_target_pct": 0.02,  # Minimum gain to label as profitable (range: 0.005-0.10)
}

# --- 8. GLOBAL SETTINGS ---
MODE = "PAPER" # Operational Mode: "PAPER" (Simulated) or "LIVE" (Real Money)
LOG_LEVEL = logging.DEBUG # Default logging level - logging.INFO or logging.DEBUG
timezone = "US/Eastern"  # Market Timezone

# --- 9. INDICATOR PARAMS ---
# Default parameters for technical indicators
INDICATOR_PARAMS = {
    "supertrend_length": 10,        # Lookback period for SuperTrend
    "supertrend_multiplier": 3.0,   # ATR Multiplier for SuperTrend
    "rsi_length": 14,               # RSI Lookback
    "ichimoku_conversion": 9,       # Tenkan-sen
    "ichimoku_base": 26,            # Kijun-sen
    "ichimoku_span": 52             # Senkou Span B
}

# --- 10. LEGACY STRATEGY PARAMS (For Feature Engine Compatibility) ---
# Parameters used by older strategy logic or feature calculation
STRATEGY_PARAMS = {
    'sma_short': 20,
    'sma_long': 100,
    'rsi_threshold': 75,
    'atr_mult_stop': 2.5,
    'slope_threshold': 15,
    'adx_threshold': 25,
    'vol_multiplier': 1.1,
    'kalman_smooth_threshold': 0.5,
    'wavelet_noise_max': 1.5
}

# --- 11. AI FEATURE CONTRACT (Gen-12) ---
ML_FEATURES = [
    # 1. Base Price Action
    'close', 'volume_change', 'daily_return',
    # 2. Momentum & Trend (Must match feature_engine.py output)
    'rsi_14', 'adx_14', 'wt1', 'wt2', 'ema_spread', 'supertrend_direction',
    # 3. VSA & Patterns
    'vsa_squat_bar',
    'master_score', 'is_consolidating', 'smart_hammer', 'smart_shooting_star',
    # 4. Context (Gen-13 Additions)
    'rel_strength_qqq',
    'volatility_20d'
]

class EmojiFilter(logging.Filter):
    """
    Log Filter to remove emojis and non-ascii characters.
    Useful for consoles that don't support unicode or for clean log files.
    """
    def filter(self, record):
        # Substitute any character outside the standard ASCII range (0-127) with empty string
        record.msg = re.sub(r'[^\x00-\x7F]+', '', str(record.msg)).strip()
        return True

class LoggerSetup:
    @staticmethod
    def setup_logger(name, log_file=None, level=logging.DEBUG):
        """
        Robust Factory Function for Tiered Logging.
        Enforces Rule 2:
        - Console: INFO level (Clean high-level output).
        - File: DEBUG level (All math and decisions written to ONE Unified file).
        """
        logger = logging.getLogger(name)
        logger.setLevel(level) 
        logger.propagate = False 

        if logger.hasHandlers():
            logger.handlers.clear()

        formatter = logging.Formatter('%(asctime)s | %(levelname)s | [%(name)s.%(funcName)s] | %(message)s')
        
        # 1. Console Handler
        try:
            import sys, io
            if hasattr(sys.stdout, 'buffer'):
                console_stream = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            else:
                console_stream = sys.stdout
        except Exception:
            import sys
            console_stream = sys.stdout 
        
        console_handler = logging.StreamHandler(console_stream)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO) # Console gets INFO
        logger.addHandler(console_handler)

        emoji_filter = EmojiFilter() 
        
        # --- UNIFIED MASTER LOG ---
        # Instead of f"{name}_date.log", all agents write to the Master file.
        prefix = getattr(sys.modules[__name__], 'LOG_PREFIX', 'StockWise_Live')
        
        # Generate a unique timestamp for this specific run session
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        local_filename = f"{prefix}_{run_id}.log"

        local_file = os.path.join(LOG_DIR_LOCAL, local_filename)

        try:
            # mode='a' ensures all 4 Agents can append to the file simultaneously
            f_handler_local = logging.FileHandler(local_file, mode='a', encoding='utf-8')
            f_handler_local.setFormatter(formatter)
            f_handler_local.setLevel(logging.DEBUG) # File gets EVERYTHING
            f_handler_local.addFilter(emoji_filter)
            logger.addHandler(f_handler_local)
        except Exception as e:
            print(f"Failed to create local log: {e}")

        # --- UNIFIED DRIVE LOG ---
        if os.path.exists(LOG_DIR_DRIVE):
            drive_filename = f"StockWise_Master_{datetime.now().strftime('%Y%m%d')}.txt"
            drive_file = os.path.join(LOG_DIR_DRIVE, drive_filename)
            try:
                f_handler_drive = logging.FileHandler(drive_file, mode='a', encoding='utf-8')
                f_handler_drive.setFormatter(formatter)
                f_handler_drive.setLevel(logging.DEBUG)
                f_handler_drive.addFilter(emoji_filter)
                logger.addHandler(f_handler_drive)
            except Exception as e:
                pass

        return logger

    @staticmethod
    def read_logs(log_file='system_thoughts.log'):
        """Reads the content of a specific log file."""
        try:
            with open(os.path.join("logs", log_file), 'r', encoding='utf-8') as f:
                return f.readlines()
        except:
            return []
            
            
class SystemActionLogger:
    """
    Singleton logger for tracking high-level system actions.
    Saves to logs/system_actions.log
    """
    _logger = None

    @classmethod
    def get_logger(cls):
        """Returns the singleton logger instance, creating it if needed."""
        if cls._logger is None:
            cls._logger = LoggerSetup.setup_logger("SystemActions", "system_actions.log")
        return cls._logger
    
    @classmethod
    def _setup(cls):
        """Internal setup method for the logger."""
        if not os.path.exists("logs"):
            os.makedirs("logs")

        cls._logger = logging.getLogger("SystemActions")
        cls._logger.setLevel(logging.INFO)
        cls._logger.handlers.clear()

        # File Handler
        log_file = os.path.join("logs", "system_actions.log")
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        
        # Format: TIMESTAMP | COMPONENT | ACTION | DETAILS
        formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
        file_handler.setFormatter(formatter)
        
        cls._logger.addHandler(file_handler)


    @staticmethod
    def log_action(component, action, details=""):
        """
        Log a formatted system action.
        Usage: SystemActionLogger.log_action("LiveTrader", "EXECUTION", "Bought AAPL @ 150")
        """
        logger = SystemActionLogger.get_logger()
        msg = f"[{component}] {action}: {details}"
        logger.info(msg)
        
def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and validates a DataFrame from any source.
    - Handles empty or None DataFrames.
    - Flattens multi-level column indexes (often from yfinance).
    - Converts all column names to lowercase.
    - Standardizes 'adj close' to 'close'.
    - FIX: Removes all timezone information to prevent comparison errors.
    - Validates that essential columns are present.
    """
    # Return empty DataFrame if input is invalid
    if df is None or df.empty:
        return pd.DataFrame()

    # Deduplicate index: keep the first occurrence of timestamp
    # This prevents the "ambiguous value" error during backtesting or merging.
    if not df.index.is_unique:
        df = df[~df.index.duplicated(keep='first')]

    # Flatten MultiIndex columns (common in yfinance bulk downloads)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    # Normalize column names to lowercase for consistency
    df.columns = [col.lower() for col in df.columns]

    # Remove duplicate columns if any exist
    df = df.loc[:, ~df.columns.duplicated(keep='first')]

    # --- Handle Adjusted Close ---
    # Logic: If 'adj close' exists, prefer it as the source of truth for 'close'
    # This accounts for splits/dividends in historical data.
    if 'adj close' in df.columns:
        df['close'] = df['adj close']
        # Remove the 'adj close' column(s) after merging to avoid confusion
        df = df.drop(columns=[col for col in df.columns if col == 'adj close'])

    # 4. Remove all timezone info from the index
    # We standardize on naive datetimes to avoid pytz inconsistencies
    if pd.api.types.is_datetime64_any_dtype(df.index) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Enforce numeric types for standard OHLCV columns
    standard_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in standard_cols:
        if col in df.columns:
            # Coerce errors to NaN (non-numeric becomes NaN)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Identify which standard columns exist in the dataframe
    existing_cols = [col for col in standard_cols if col in df.columns]

    # Critical check: Ensure we have ALL required columns *after* cleaning
    if not all(col in df.columns for col in standard_cols):
        # If any required column is missing (e.g. Volume), likely bad data from source
        # This can happen if yfinance returns a weird format or delisted ticker
        return pd.DataFrame()  # Return empty, not a subset

    # Remove rows where any of the standard required columns are NaN
    if existing_cols:
        df.dropna(subset=existing_cols, inplace=True)

    # Return only the standard columns (strips extra columns like 'dividends')
    return df[existing_cols]


def snapshot_configuration():
    """
    Saves the current system configuration to a JSON file for debugging.
    Proposed Usage: Run this at system startup to log the exact config used.
    """
    try:
        # Collect all config dictionaries into a single snapshot object
        snapshot = {
            "TIMESTAMP": datetime.now().isoformat(),
            "MODE": MODE,
            "STRATEGY_CONFIG": STRATEGY_CONFIG,
            "RISK_CONFIG": RISK_CONFIG,
            "SCAN_SCHEDULE": SCAN_SCHEDULE,
            "COSTS_CONFIG": COSTS_CONFIG,
            # Handle MASTER_SCORES dynamically in case it's missing (though it shouldn't be)
            "MASTER_SCORES": globals().get('MASTER_SCORES', {}) 
        }
        
        # Create a timestamped filename
        filename = f"config_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = os.path.join(LOGS_DIR, filename)
        
        # Write to JSON file
        with open(path, 'w') as f:
            json.dump(snapshot, f, indent=4)
            
        print(f"Configuration snapshot saved to: {path}")
    except Exception as e:
        print(f"Failed to save config snapshot: {e}")
