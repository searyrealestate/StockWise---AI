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

# Signal generation pipeline mode
# "legacy": use orchestra.evaluate_ticker() (original 6 hardcoded setups)
# "templates": use template_matcher.scan_ticker() (block-based templates)
# "dual": run both and log comparison (testing mode)
SIGNAL_PIPELINE_MODE = "templates"

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
# Google Drive removed — all logs saved locally in logs/ directory

# Ensure Log Directories Exist
os.makedirs(LOG_DIR_LOCAL, exist_ok=True)

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

# Maximum number of active trading templates (SPEC v13.4 §4 — ceiling, not floor)
MAX_TEMPLATES = 5

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
# --- DATA PACING CONFIGURATION (Seconds between requests per provider) ---
# [2026-03-14] UPDATED: Reduced delays after adding escalating circuit breaker.
# Old values: MASSIVE=12.5, ALPACA=2.5, YFINANCE=1.0
# Reason: Circuit breaker now handles Massive 429s properly (1h/4h lockout),
# so per-request delays no longer need to compensate for rate-limit risk.
PROVIDER_DELAY = {
    "MASSIVE": 1.0,    # Pre-request courtesy delay; circuit breaker handles 429 lockout
    "IBKR": 0.05,      # 50 req/sec limit (0.02s theoretical, 0.05s safe)
    "ALPACA": 0.5,     # Free tier allows ~200/min; 0.5s = 2/sec, safely within limit
    "YFINANCE": 1.5,   # Slightly conservative against scraping protection
    "DEFAULT": 0.5,    # Safe fallback
    # ═══ MASSIVE_TIMEOUT (2026-03-18) ═══════════════════════════════════
    # DO NOT DELETE: Hard timeout in seconds for MASSIVE (Polygon) API calls.
    # The Polygon SDK has no built-in timeout — without this, a 429 rate
    # limit response causes 30-60 seconds of hidden internal retries.
    # This value is read by _download_from_massive() in data_source_manager.py.
    # Reduce to 5 if MASSIVE is consistently failing; increase to 15 for slow networks.
    # ═══════════════════════════════════════════════════════════════════════
    "MASSIVE_TIMEOUT": 10
}

# ═══ WATERFALL ROUTING (2026-03-24 — DDR #2) ════════════════════════════
# ARCHITECTURAL CHANGE: DATA_PROVIDER = "ALPACA" was removed here.
# Previous comment said "DO NOT DELETE" — reason was to prevent DSM from
# silently disabling Alpaca. That concern is now resolved differently:
# DSM uses EN_ALPACA / EN_MASSIVE / EN_IBKR / EN_YFINANCE flags directly.
# The system no longer depends on a single provider. Waterfall routing
# (Massive → Alpaca → IBKR → YFinance) is the standard per SPEC v13.4 §2.
# See: data_source_manager.py get_stock_data() priority_list.
# ═════════════════════════════════════════════════════════════════════════

# Validation (Optional sanity check)
if not any([EN_MASSIVE, EN_ALPACA, EN_IBKR, EN_YFINANCE]):
    logging.error("WARNING: All data providers are DISABLED. System will starve.")
    raise ValueError("At least one data provider must be enabled.")
else:
    pass

# Initialize Alpaca credentials to None
# Try loading from .streamlit/secrets.toml first (User requirement)
ALPACA_KEY = None
ALPACA_SECRET = None
MASSIVE_API_KEY = None

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
# ═══ CORE SYMBOL LIST (2026-03-20) ═══════════════════════════════════
# DO NOT DELETE: Single source of truth for initial symbols.
# Used as: (1) VIP fallback on first run, (2) WATCHLIST seed for scanner
# priority queue, (3) AI training baseline.
# SPY must always be first — it's the benchmark for Relative Strength.
# After first scan, VIP is managed by the scanner. Symbols exit VIP
# after 210 days without recommendation (TTL in SCAN_ROUTING_CONFIG).
# ═════════════════════════════════════════════════════════════════════
DEFAULT_TRAINING_SYMBOLS = [
    'SPY', 'NVDA', 'MSFT', 'AAPL', 'AMZN', 'META', 'GOOGL',
    'TSLA', 'AMD', 'NFLX', 'BRK-B', 'LLY', 'AVGO'
]

# Dynamic Loading Wrapper for the Watchlist
def load_dynamic_watchlist():
    """Loads the active watchlist from JSON, falls back to seed list (atomic read via safe_json_io)."""
    from safe_json_io import safe_json_read
    path = os.path.join(DB_DIR, "dynamic_watchlist.json")
    data = safe_json_read(path, default={})
    tickers = data.get("tickers", [])
    if tickers:
        return tickers
    # Fallback: use DEFAULT_TRAINING_SYMBOLS as the seed list
    return list(DEFAULT_TRAINING_SYMBOLS)

# Global Variable WATCHLIST now calls the function to load the latest list
WATCHLIST = load_dynamic_watchlist()
# ═══ BENCHMARK (2026-03-19) ═══════════════════════════════════════════
# DO NOT DELETE: SPY is the S&P500 benchmark used for Relative Strength
# calculation. All stocks are measured against SPY to determine if they
# outperform or underperform the market. Do not change without updating
# stock_hunter.py RS calculation.
# ═══════════════════════════════════════════════════════════════════════
BENCHMARK_TICKER = "SPY"

# --- 3.5 TRADE TYPE CONFIGURATION (NEW) ---
# Defines intervals and lookback periods for data fetching based on trade duration
TRADE_TYPE_CONFIG = {
    "SHORT": {"interval": "15m", "days_back": 30},  # For scraping/intraday
    "MID":   {"interval": "1h",  "days_back": 60},  # For swing trading
    "LONG":  {"interval": "1d",  "days_back": 730}  # For long-term investing
}

# Data Guard: minimum candles required for statistical validity (SPEC v13.4 §2)
# Prevents processing stocks with insufficient history for VSA and moving averages.
MIN_CANDLES_FOR_PROCESSING = 200

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
    "starting_capital": 5000.0      # Baseline capital for calculations
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
    "min_net_profit_pct": 0.005,   # Trade must yield > 0.5% net profit (SPEC v13.4 DDR #3)
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
    "min_net_profit_pct": 0.005,      # A trade MUST yield > 0.5% net profit (SPEC v13.4 DDR #3 — unified threshold)
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
    "phase3_atr_mult": 1.0,                  # The choke mechanism tightens the stop to just 1.0 ATR from the highest high.
    "runner_atr_mult": 0.5,               # Phase 4 Runner: ultra-tight trailing (DDR #4)
    "runner_min_distance_pct": 0.008,      # Phase 4 Runner: floor distance from high (DDR #4)
}

MILESTONE_ALERT_CONFIG = {
    # Real Breakeven: first alert only fires after profit covers all costs
    # (commissions + slippage + tax). This buffer is added on top of actual costs.
    "safe_zone_buffer_pct": 0.002,      # 0.2% safety margin above true breakeven

    # Event-driven alert triggers -- controls when user gets notified
    "min_stop_change_pct": 0.01,         # Alert only if stop moved > 1% of current price
    "min_alert_interval_minutes": 15,    # Minimum 15 min between alerts per ticker

    # Phase 4 Runner Mode: replaces hard take_profit with ultra-tight trailing
    "runner_atr_mult": 0.5,              # DEPRECATED — canonical source is now KINETIC_STOP_CONFIG
    "runner_min_distance_pct": 0.008,    # DEPRECATED — canonical source is now KINETIC_STOP_CONFIG
                                          # Kept for backward compatibility with live_trading_engine.py
}

# Position Management Configuration
POSITION_MANAGEMENT_CONFIG = {
    # PHASE_PAUSE: Healthy pullback detection
    "max_healthy_pullback_pct": 0.03,    # Up to 3% pullback = still healthy
    "min_er_for_pause": 0.45,            # ER must be above this for "trend intact"
    "min_rsi_for_pause": 40,             # RSI must be above this for "not oversold"

    # Re-entry: after stop-loss exit, when to recommend re-entry
    "re_entry_enabled": True,
    "re_entry_min_wait_candles": 3,      # Wait at least 3 candles after exit
    "re_entry_requires_new_signal": True, # Must get a fresh template signal
}

# Pre-Market Gap Validator (SPEC v13.4 §5)
PRE_MARKET_CONFIG = {
    "enabled": True,
    "check_time": "09:25",           # ET — window: check_time-5m to check_time+10m
    "max_gap_pct": 0.05,             # Veto if overnight gap > 5%
    "min_gap_pct": 0.001,            # Ignore gaps < 0.1% (noise floor)
    "use_ibkr_for_premarket": True,  # Prefer IBKR real pre-market price
    "fallback_to_last_close": True,  # Use last daily close if IBKR unavailable
    "veto_cooldown_minutes": 60,     # Suppress repeat vetoes for 60 min
}

# Shadow Ledger: Candle-by-Candle Learning Engine (SPEC v13.4 §4)
# Runs OFFLINE (weekends) — evaluates all templates across historical data bar-by-bar.
# Output: per-symbol, per-template win rates used by template_matcher (W4-4).
# Phase 2 planned: MTFA (Multi-Timeframe Analysis) — daily only for now.
SHADOW_LEDGER_CONFIG = {
    "enabled": True,
    "ledger_path": "data/shadow_ledger.json",
    "eval_days_back": 1095,              # 3 years — enough for ~26 signals per template with cooldown
    "max_templates": 5,                  # Matches MAX_TEMPLATES ceiling
    "lookahead_candles": 20,             # How many candles forward to check target/stop
    "min_candles_for_eval": 200,         # Matches MIN_CANDLES_FOR_PROCESSING — indicator warmup
    "min_bars_between_signals": 20,      # Cooldown: prevent correlated signals from same template
    "run_mode": "offline",               # "offline" = weekend batch only
}

# Asset-Specific Optimization (DDR #1)
# Uses per-symbol template win rates from Shadow Ledger instead of global averages.
# Cold start: symbols with < cold_start_min_signals fall back to global average.
# Blended: per_stock_weight% per-stock + global_weight% global for established symbols.
ASSET_SPECIFIC_CONFIG = {
    "enabled": True,
    "cold_start_min_signals": 5,                    # Below this → use global average only
    "per_stock_weight": 0.7,                        # 70% weight to per-stock stats
    "global_weight": 0.3,                           # 30% weight to global stats
    "shadow_ledger_path": "data/shadow_ledger.json",  # Must match SHADOW_LEDGER_CONFIG.ledger_path
}

# Vectorized Decay: per-template-category aging rates (SPEC v13.4 §4)
# Momentum signals lose relevance quickly. VSA/institutional patterns persist.
# Applied by Shadow Ledger after each full evaluation run.
# Phase 2: MTFA will introduce per-timeframe decay rates.
VECTORIZED_DECAY_CONFIG = {
    "enabled": True,
    "decay_rates": {
        "momentum": 0.90,           # Fast decay — momentum signals lose relevance quickly
        "breakout": 0.92,           # Medium-fast decay
        "mean_reversion": 0.93,     # Medium decay
        "vsa_institutional": 0.99,  # Slow decay — institutional accumulation patterns persist
        "default": 0.95,            # Default for uncategorized templates
    },
    "decay_period_days": 7,         # Decay applied per this many days of age
    "min_weight": 0.05,             # Floor — signals never fully forgotten
}

# Portfolio Risk Management (Phase 5)
PORTFOLIO_RISK_CONFIG = {
    # --- Correlation Check ---
    "max_sector_positions": 2,           # Max positions in same sector
    "correlation_lookback_days": 60,     # Days to calculate correlation
    "max_correlation": 0.80,             # Don't hold 2 stocks with corr > 0.80

    # --- Max Drawdown Protection ---
    "max_portfolio_drawdown_pct": 0.10,  # 10% max drawdown -> stop all new entries
    "max_single_position_pct": 0.20,     # No single position > 20% of portfolio
    "max_total_exposure_pct": 0.60,      # Total invested <= 60% of portfolio
    "drawdown_cooldown_hours": 24,       # After circuit breaker, wait 24h before resuming

    # --- Multi-Timeframe ---
    "weekly_trend_enabled": True,
    "weekly_sma_period": 40,             # 40 weeks ~= 200 days (same as SMA_200 daily)
    "weekly_trend_must_be_bullish": True, # Only enter if weekly trend is up

    # --- Zombie & Event Horizon (merged from PORTFOLIO_DEFENSE) ---
    "zombie_trade_ttl_hours": 72,        # If regime changes, 72h grace before force-exit
    "event_horizon_buffer_days": 2,      # Force-sell 2 days before earnings/FDA events
}

SCAN_ROUTING_CONFIG = {
    "daily_scan_limit": 4000,             # Maximum symbols to scan per nightly run
    "priority_scan_limit": 100,           # Top N symbols from ledger scanned first (by score)
    "max_daily_review_stocks": 10,        # Top 10 stocks promoted to VIP list
    "min_vip_score_threshold": 75.0,      # Minimum master_score to qualify for VIP
    "max_days_untraded_on_watchlist": 210, # TTL: symbols exit VIP after 7 months without recommendation
    "max_vip_list_size": 50,              # Maximum symbols in cumulative VIP list
    "weight_score_mult": 0.7,             # 70% of scan priority = DSP Score
    "weight_volatility_mult": 0.3         # 30% of scan priority = raw volatility
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

# Template Discovery Engine: finds new profitable templates from historical data
DISCOVERY_CONFIG = {
    # Data collection
    "history_days": 500,             # ~2 years of trading history
    "api_throttle_seconds": 1.0,     # Delay between API calls (respect rate limits)
    "min_history_rows": 200,         # Skip stock if less than 200 rows

    # Combination generation
    "min_blocks_per_combo": 3,       # Minimum blocks in a template
    "max_blocks_per_combo": 5,       # Maximum blocks in a template
    "max_combos_to_test": 5000,      # Cap total combos (safety limit)

    # Quality thresholds for saving a discovered template
    "min_activations": 10,           # Must trigger at least 10 times in history
    "min_win_rate": 55.0,            # Must win > 55% of the time
    "min_avg_profit_pct": 1.0,       # Average profit must be > 1%
    "min_profit_factor": 1.5,        # Total profits / total losses > 1.5
    "min_stocks_profitable": 3,      # Must work on at least 3 different stocks

    # Lookahead for labeling (did the stock go up after signal?)
    "lookahead_days": 5,             # Check if price rose within 5 days
    "profit_target_pct": 0.02,       # 2% minimum gain to count as "win"
    "stop_target_pct": 0.03,         # 3% drop = "loss" (wider than profit to account for noise)
}

# Parameter ranges for block optimization.
# Discovery Engine tests each variation to find optimal params per ticker.
# Format: block_name -> list of param variations to try
PARAM_RANGES = {
    # Trend blocks
    "close_above_sma": [[20], [50], [100], [150], [200]],
    "sma_above_sma": [[20, 50], [20, 100], [50, 100], [50, 200], [100, 200]],
    "close_above_ema": [[8], [12], [21], [26], [50]],
    "er_slow_above": [[0.40], [0.45], [0.50], [0.55], [0.60]],
    "trend_alignment": [[]],  # No params to vary

    # Momentum blocks
    "rsi_between": [[30, 60], [35, 65], [40, 70], [45, 75], [50, 80]],
    "rsi_below": [[25], [30], [35]],
    "rsi_above": [[45], [50], [55], [60]],
    "macd_above_signal": [[]],
    "macd_histogram_positive": [[]],

    # Volume blocks
    "volume_surge": [[1.2], [1.3], [1.5], [2.0], [2.5]],
    "rvol_above": [[1.1], [1.2], [1.3], [1.5]],

    # Volatility blocks
    "squeeze_active": [[]],
    "squeeze_momentum_positive": [[]],
    "bb_width_below": [[0.08], [0.10], [0.12], [0.15], [0.20]],
    "atr_percent_above": [[0.005], [0.008], [0.01], [0.015], [0.02]],

    # Price action blocks
    "bullish_candle": [[]],
    "close_above_ref": [["bb_upper"], ["sma_50"], ["ema_12"]],
    "close_below_ref": [["bb_lower"], ["sma_50"]],
}

# Relative Strength Configuration
RELATIVE_STRENGTH_CONFIG = {
    "lookback_days": [20, 60, 120],  # Calculate RS over these periods
    "outperform_threshold": 1.05,     # RS > 1.05 = outperforming
    "underperform_threshold": 0.95,   # RS < 0.95 = underperforming
}

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

OBSERVABILITY_CONFIG = {
    # ── Storage ──────────────────────────────────────────────────────────────
    "log_dir": "data/decision_logs",          # Directory for JSONL decision logs
    "log_filename": "decisions.jsonl",        # Append-only JSONL file
    "max_log_size_mb": 50,                    # Rotate when file exceeds this size
    "max_rotated_files": 5,                   # How many rotated files to keep

    # ── What to capture ──────────────────────────────────────────────────────
    "log_signal_events": True,                # Log every template signal generated
    "log_veto_events": True,                  # Log every veto-gate decision (pass/block)
    "log_risk_events": True,                  # Log every risk-gate decision
    "log_execution_events": True,             # Log every execute_ticket call
    "log_exit_events": True,                  # Log every position exit

    # ── Async / performance ───────────────────────────────────────────────────
    "async_write": False,                     # False = synchronous (safe default)
    "flush_every_n_events": 1,               # Flush to disk after every N events (1 = immediate)

    # ── Schema version ────────────────────────────────────────────────────────
    "schema_version": "1.0",
}

# --- 11. AI FEATURE CONTRACT (Gen-12) ---
ML_FEATURES = [
    # 1. Base Price Action
    'close', 'volume_change', 'daily_return',
    # 2. Momentum & Trend (Must match feature_engine.py output)
    'rsi_14', 'adx_14', 'ema_spread', 'supertrend_direction',
    # 3. VSA & Patterns
    'vsa_squat_bar',
    'is_consolidating', 'smart_hammer', 'smart_shooting_star',
    # 4. Context (Gen-13 Additions)
    'volatility_20d'
]

# ════════════════════════════════════════════════════════════════
# TEMPLATE ENGINE CONFIG
# SPEC v13.4 §4: Max 5 condition blocks per template to prevent
# overfitting. No limit on total number of templates.
# ════════════════════════════════════════════════════════════════
TEMPLATE_CONFIG = {
    "max_conditions_per_template": 5,   # SPEC v13.4 §4 ceiling
}

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
        
        local_filename = f"{prefix}_{run_id}.txt"

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

        # Google Drive log removed — all logs go to local logs/ directory only

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
        
def snapshot_configuration(logger_instance=None):
    """
    Saves complete system configuration snapshot.
    Output A: Line in Master Log (parseable by Simulator)
    Output B: Separate JSON file in logs/ directory

    Called automatically at session startup by live_trading_engine.py

    Args:
        logger_instance: Logger to write the CONFIG_SNAPSHOT line to.
                         If None, only writes JSON file.
    """
    try:
        snapshot = {
            "config_version": "13.1",
            "timestamp": datetime.now().isoformat(),

            # Strategy & Signals
            "signal_pipeline_mode": SIGNAL_PIPELINE_MODE,
            "min_master_score_approval": MIN_MASTER_SCORE_APPROVAL,
            "premium_trade_threshold": PREMIUM_TRADE_THRESHOLD,
            "strategy_config": STRATEGY_CONFIG,
            "master_scores": MASTER_SCORES,

            # Risk Management
            "risk_config": RISK_CONFIG,
            "costs_config": COSTS_CONFIG,
            "friction_and_alpha": FRICTION_AND_ALPHA,
            "volumetric_limits": VOLUMETRIC_LIMITS,

            # Position Lifecycle
            "kinetic_stop_config": KINETIC_STOP_CONFIG,
            "milestone_alert_config": MILESTONE_ALERT_CONFIG,
            "position_management_config": POSITION_MANAGEMENT_CONFIG,

            # Portfolio Level
            "portfolio_risk_config": PORTFOLIO_RISK_CONFIG,

            # DSP & Regime
            "dsp_config": DSP_CONFIG,

            # Scanner
            "scan_routing_config": SCAN_ROUTING_CONFIG,

            # Data Providers
            "provider_delay": PROVIDER_DELAY,
            "historical_source": HISTORICAL_SOURCE,
            "realtime_source": REALTIME_SOURCE,
            "en_ibkr": EN_IBKR,
            "en_alpaca": EN_ALPACA,
            "en_massive": EN_MASSIVE,
            "en_yfinance": EN_YFINANCE,

            # Indicator Parameters (for Simulator reproducibility)
            "indicators": {
                "rsi_period": 14,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "atr_period": 14,
                "stochastic_k_period": 14,
                "stochastic_d_period": 3,
                "bb_period": 20,
                "bb_std_dev": 2,
                "er_lookback_slow": DSP_CONFIG.get("er_lookback_slow", 20),
                "er_lookback_fast": DSP_CONFIG.get("er_lookback_fast", 5)
            },

            # Cooldowns
            "cooldown_period_hours": COOLDOWN_PERIOD_HOURS,
            "veto_cooldown_minutes": VETO_COOLDOWN_MINUTES,
            "data_starvation_cooldown_minutes": DATA_STARVATION_COOLDOWN_MINUTES,

            # Investment
            "investment_amount": INVESTMENT_AMOUNT,
            "base_friction": BASE_FRICTION,
            "min_net_profit": MIN_NET_PROFIT,
        }

        # --- Output A: Write to Master Log ---
        if logger_instance:
            # Compact JSON on one line for easy parsing
            snapshot_json = json.dumps(snapshot, separators=(',', ':'))
            logger_instance.info(f"CONFIG_SNAPSHOT|{snapshot_json}")

        # --- Output B: Save as separate JSON file ---
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"config_snapshot_{run_id}.json"
        filepath = os.path.join(LOGS_DIR, filename)

        # TODO: migrate to safe_json_io (needs ensure_ascii=False support not in safe_json_write)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)

        print(f"Config snapshot saved: {filepath}")
        return snapshot

    except Exception as e:
        print(f"Failed to save config snapshot: {e}")
        if logger_instance:
            logger_instance.error(f"Config snapshot failed: {e}")
        return None
