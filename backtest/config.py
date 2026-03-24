"""Backtest pipeline configuration — all tunable constants in one place."""

# ── Date splits ────────────────────────────────────────────────────────────
TRAIN_START = "2024-01-01"
TRAIN_END   = "2025-02-28"    # 14 months
VAL_START   = "2025-03-01"
VAL_END     = "2025-07-31"    # 5 months
TEST_START  = "2025-08-01"
TEST_END    = "2026-03-21"    # ~5 months

INTERVAL   = "1d"
API_DELAY  = 0.5              # seconds between Alpaca calls

# ── Performance thresholds ─────────────────────────────────────────────────
MIN_WIN_RATE          = 70.0  # %
TARGET_DAILY_RETURN   = 2.0   # %
MIN_TRADES            = 50

# ── Template discovery ─────────────────────────────────────────────────────
MIN_BLOCKS             = 3
MAX_BLOCKS             = 5
MAX_COMBOS             = 10_000
MIN_ACTIVATIONS        = 15
MIN_PROFIT_FACTOR      = 1.5
MIN_STOCKS_PROFITABLE  = 10

# ── Portfolio simulation ────────────────────────────────────────────────────
STARTING_CAPITAL   = 5_000.0
MAX_POSITION_PCT   = 0.20
COMMISSION         = 0.005
SLIPPAGE           = 0.001
TAX_RATE           = 0.25
