# setup_templates.py

"""
StockWise Gen-13 Setup Template Engine
======================================
Defines the data model for trading templates and provides
load/save/validate operations.

A Template is a reusable pattern that describes:
- WHEN to enter (conditions on indicators)
- WHERE to place stop-loss and take-profit
- WHAT market state it works best in

Templates are stored as JSON in data/templates/ directory.
They can be:
  1. Seed templates (manually defined, ship with the system)
  2. Discovered templates (found by backtesting historical data)
"""

import os
import json
from safe_json_io import safe_json_read, safe_json_write
import logging
from datetime import datetime
import system_config as cfg

logger = logging.getLogger("TemplateEngine")


# ============================================================
# CONDITION BLOCK REGISTRY
# ============================================================
# Each block is a reusable function: (row, params) -> bool
# Templates reference blocks by name + params.
# To add a new block: just add an entry here. No template changes needed.
#
# Block categories:
#   trend_*      -- Trend direction filters
#   momentum_*   -- RSI, MACD, momentum indicators
#   volume_*     -- Volume analysis
#   volatility_* -- BB, ATR, squeeze
#   price_*      -- Price action and candle patterns
# ============================================================

def _safe_get(row, key, default=0):
    """Safely get a value from a row, handling NaN."""
    import math
    val = row.get(key, default)
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return default
    return val


# --- TREND BLOCKS ---

def block_close_above_sma(row, params):
    """Close > SMA_N. params: [sma_period]  e.g. [50]"""
    period = params[0]
    return _safe_get(row, 'close') > _safe_get(row, f'sma_{period}')

def block_sma_above_sma(row, params):
    """SMA_A > SMA_B (trend alignment). params: [fast_period, slow_period]  e.g. [50, 200]"""
    fast = _safe_get(row, f'sma_{params[0]}')
    slow = _safe_get(row, f'sma_{params[1]}')
    return fast > slow and slow > 0

def block_close_above_ema(row, params):
    """Close > EMA_N. params: [ema_period]  e.g. [12]"""
    period = params[0]
    return _safe_get(row, 'close') > _safe_get(row, f'ema_{period}')

def block_er_slow_above(row, params):
    """Efficiency Ratio (slow) above threshold. params: [threshold]  e.g. [0.55]"""
    return _safe_get(row, 'er_slow') >= params[0]

def block_trend_alignment(row, params):
    """Full trend alignment flag is active. params: [] (no params)"""
    return _safe_get(row, 'trend_alignment') == 1


# --- MOMENTUM BLOCKS ---

def block_rsi_between(row, params):
    """RSI is in range. params: [min, max]  e.g. [40, 65]"""
    rsi = _safe_get(row, 'rsi', 50)
    return params[0] <= rsi <= params[1]

def block_rsi_below(row, params):
    """RSI below threshold (oversold). params: [threshold]  e.g. [30]"""
    return _safe_get(row, 'rsi', 50) < params[0]

def block_rsi_above(row, params):
    """RSI above threshold. params: [threshold]  e.g. [50]"""
    return _safe_get(row, 'rsi', 50) > params[0]

def block_macd_above_signal(row, params):
    """MACD line above signal line (bullish). params: []"""
    return _safe_get(row, 'macd') > _safe_get(row, 'macd_signal')

def block_macd_histogram_positive(row, params):
    """MACD histogram > 0 (momentum building). params: []"""
    return _safe_get(row, 'macd_hist') > 0


# --- VOLUME BLOCKS ---

def block_volume_surge(row, params):
    """Current volume > avg * multiplier. params: [multiplier]  e.g. [1.5]"""
    vol = _safe_get(row, 'volume')
    avg = _safe_get(row, 'vol_avg_20', 1)
    return vol > avg * params[0] if avg > 0 else False

def block_rvol_above(row, params):
    """Relative Volume above threshold. params: [threshold]  e.g. [1.3]"""
    return _safe_get(row, 'rvol', 1.0) > params[0]


# --- VOLATILITY BLOCKS ---

def block_squeeze_active(row, params):
    """Bollinger Squeeze is on (BB inside KC). params: []"""
    return _safe_get(row, 'squeeze_on') == 1

def block_squeeze_momentum_positive(row, params):
    """Squeeze momentum (MACD hist proxy) is positive. params: []"""
    return _safe_get(row, 'mom_sqz') > 0

def block_bb_width_below(row, params):
    """BB width below threshold (narrow bands). params: [threshold]  e.g. [0.15]
    Uses bb_width_pct (bb_width / bb_mid) so threshold is a fraction of price,
    not raw dollars. Falls back to bb_width if bb_width_pct is absent."""
    width = _safe_get(row, 'bb_width_pct', None)
    if width is None or (isinstance(width, float) and width != width):  # None or NaN
        width = _safe_get(row, 'bb_width', 1.0)
    return width < params[0]

def block_atr_percent_above(row, params):
    """ATR as % of price above threshold (enough volatility for profit). params: [min_pct]  e.g. [0.01]"""
    close = _safe_get(row, 'close', 1)
    atr = _safe_get(row, 'atr', 0)
    return (atr / close) >= params[0] if close > 0 else False


# --- PRICE ACTION BLOCKS ---

def block_bullish_candle(row, params):
    """Close > Open (green candle). params: []"""
    return _safe_get(row, 'close') > _safe_get(row, 'open')

def block_close_above_ref(row, params):
    """Close above a named reference column. params: [column_name]  e.g. ['bb_upper']"""
    return _safe_get(row, 'close') > _safe_get(row, params[0])

def block_close_below_ref(row, params):
    """Close below a named reference column. params: [column_name]  e.g. ['bb_lower']"""
    return _safe_get(row, 'close') < _safe_get(row, params[0])


# --- NEW BLOCKS: TREND (expanded) ---

def block_adx_above(row, params):
    """ADX above threshold (strong trend). params: [threshold]  e.g. [25]"""
    return _safe_get(row, 'adx') > params[0]

def block_supertrend_bullish(row, params):
    """SuperTrend direction is bullish (+1). params: []"""
    return _safe_get(row, 'supertrend_direction', 0) > 0

def block_golden_cross_active(row, params):
    """Golden Cross detected (SMA50 crossed above SMA200). params: []"""
    return bool(_safe_get(row, 'golden_cross', False))


# --- NEW BLOCKS: MOMENTUM (expanded) ---

def block_stoch_oversold(row, params):
    """Stochastic %K below threshold (oversold). params: [threshold]  e.g. [20]"""
    return _safe_get(row, 'stoch_k', 50) < params[0]

def block_cci_between(row, params):
    """CCI is in range. params: [min, max]  e.g. [-100, 100]"""
    cci = _safe_get(row, 'cci', 0)
    return params[0] <= cci <= params[1]

def block_roc_positive(row, params):
    """Rate of Change is positive (upward momentum). params: []"""
    return _safe_get(row, 'roc', 0) > 0


# --- NEW BLOCKS: VOLUME (expanded) ---

def block_obv_rising(row, params):
    """OBV is rising (net accumulation proxy). params: [sma_period]  e.g. [20]
    Uses OBV > 0 combined with volume >= 80% of average as accumulation signal."""
    obv = _safe_get(row, 'obv', 0)
    vol = _safe_get(row, 'volume', 0)
    avg = _safe_get(row, 'vol_avg_20', 1)
    return obv > 0 and vol > avg * 0.8

def block_cmf_positive(row, params):
    """Chaikin Money Flow is positive (buying pressure). params: []"""
    return _safe_get(row, 'cmf', 0) > 0

def block_vwap_above(row, params):
    """Close is above VWAP (trading above fair value). params: []"""
    close = _safe_get(row, 'close', 0)
    vwap  = _safe_get(row, 'vwap', 0)
    return close > vwap > 0


# --- NEW BLOCKS: PRICE ACTION (expanded) ---

def block_gap_up_today(row, params):
    """Gap up detected. params: []"""
    return bool(_safe_get(row, 'gap_up', False))

def block_fib_near_support(row, params):
    """Price within tolerance_pct of Fibonacci 61.8% level. params: [tolerance_pct]  e.g. [0.02]"""
    close  = _safe_get(row, 'close', 0)
    fib618 = _safe_get(row, 'fib_618', 0)
    if close <= 0 or fib618 <= 0:
        return False
    return abs(close - fib618) / close <= params[0]

def block_double_bottom_active(row, params):
    """Double bottom pattern detected. params: []"""
    return bool(_safe_get(row, 'double_bottom', False))


# --- STOP-LOSS METHOD BLOCKS ---

def stop_atr(row, params):
    """Stop = close - ATR * multiplier. params: [atr_multiplier]  e.g. [1.5]"""
    close = _safe_get(row, 'close')
    atr = _safe_get(row, 'atr', close * 0.02)
    return round(close - atr * params[0], 2)

def stop_swing_low(row, params):
    """Stop = recent low - ATR * buffer. params: [atr_buffer]  e.g. [0.5]
    Note: requires 'recent_low' or falls back to close - 2*ATR"""
    close = _safe_get(row, 'close')
    atr = _safe_get(row, 'atr', close * 0.02)
    recent_low = _safe_get(row, 'low', close)
    return round(recent_low - atr * params[0], 2)

def stop_fixed_pct(row, params):
    """Stop = close * (1 - pct). params: [pct]  e.g. [0.02] for 2%"""
    close = _safe_get(row, 'close')
    return round(close * (1 - params[0]), 2)

def stop_sma(row, params):
    """Stop = SMA_N - ATR * buffer. params: [sma_period, atr_buffer]  e.g. [50, 0.5]"""
    sma = _safe_get(row, f'sma_{params[0]}')
    atr = _safe_get(row, 'atr', _safe_get(row, 'close', 100) * 0.02)
    buffer = params[1] if len(params) > 1 else 0.5
    return round(sma - atr * buffer, 2) if sma > 0 else stop_atr(row, [2.0])


# --- TARGET METHOD BLOCKS ---

def target_atr(row, params):
    """Target = close + ATR * multiplier. params: [atr_multiplier]  e.g. [3.0]"""
    close = _safe_get(row, 'close')
    atr = _safe_get(row, 'atr', close * 0.02)
    return round(close + atr * params[0], 2)

def target_fixed_pct(row, params):
    """Target = close * (1 + pct). params: [pct]  e.g. [0.05] for 5%"""
    close = _safe_get(row, 'close')
    return round(close * (1 + params[0]), 2)


# ============================================================
# BLOCK REGISTRY -- Maps block names to functions
# ============================================================
# To add a new block: 1) Write the function above  2) Add it here
# Templates reference blocks by the KEY in this dict.

CONDITION_BLOCKS = {
    # Trend
    "close_above_sma":        block_close_above_sma,
    "sma_above_sma":          block_sma_above_sma,
    "close_above_ema":        block_close_above_ema,
    "er_slow_above":          block_er_slow_above,
    "trend_alignment":        block_trend_alignment,
    "adx_above":              block_adx_above,              # NEW
    "supertrend_bullish":     block_supertrend_bullish,     # NEW
    "golden_cross_active":    block_golden_cross_active,    # NEW

    # Momentum
    "rsi_between":            block_rsi_between,
    "rsi_below":              block_rsi_below,
    "rsi_above":              block_rsi_above,
    "macd_above_signal":      block_macd_above_signal,
    "macd_histogram_positive": block_macd_histogram_positive,
    "stoch_oversold":         block_stoch_oversold,         # NEW
    "cci_between":            block_cci_between,            # NEW
    "roc_positive":           block_roc_positive,           # NEW

    # Volume
    "volume_surge":           block_volume_surge,
    "rvol_above":             block_rvol_above,
    "obv_rising":             block_obv_rising,             # NEW
    "cmf_positive":           block_cmf_positive,           # NEW
    "vwap_above":             block_vwap_above,             # NEW

    # Volatility
    "squeeze_active":         block_squeeze_active,
    "squeeze_momentum_positive": block_squeeze_momentum_positive,
    "bb_width_below":         block_bb_width_below,
    "atr_percent_above":      block_atr_percent_above,

    # Price Action
    "bullish_candle":         block_bullish_candle,
    "close_above_ref":        block_close_above_ref,
    "close_below_ref":        block_close_below_ref,
    "gap_up_today":           block_gap_up_today,           # NEW
    "fib_near_support":       block_fib_near_support,       # NEW
    "double_bottom_active":   block_double_bottom_active,   # NEW
}

STOP_BLOCKS = {
    "atr": stop_atr,
    "swing_low": stop_swing_low,
    "fixed_pct": stop_fixed_pct,
    "sma": stop_sma,
}

TARGET_BLOCKS = {
    "atr": target_atr,
    "fixed_pct": target_fixed_pct,
}


class SetupTemplate:
    """
    A single trading setup template.

    Structure:
    {
        "id": "TREND_PULLBACK_EMA",
        "name": "Trend Pullback to EMA",
        "description": "Buy on pullback to EMA_12 in confirmed uptrend",
        "version": 1,
        "source": "seed",          # "seed" or "discovered"
        "enabled": true,

        "required_state": {
            "trend": ["BULLISH"],          # Which trend directions this works in
            "structure": ["OPEN_FIELD", "NEAR_SUPPORT"],  # Acceptable structures
            "volume": ["HEALTHY", "SURGING"],              # Required volume states
            "volatility": ["NORMAL", "COMPRESSED"]         # Acceptable volatility
        },

        "conditions": [
            {"indicator": "rsi", "operator": "between", "value": [40, 65]},
            {"indicator": "close", "operator": ">", "reference": "ema_12"},
            {"indicator": "macd", "operator": ">", "reference": "macd_signal"},
            {"indicator": "volume", "operator": ">", "reference": "vol_avg_20", "multiplier": 1.2}
        ],

        "entry": {
            "type": "close",               # Enter at close of signal candle
            "confirmation_candles": 1       # Wait 1 candle to confirm
        },

        "stop_loss": {
            "method": "atr",               # "atr", "swing_low", "sma", "fixed_pct"
            "atr_multiplier": 1.5,
            "fallback_pct": 0.02           # 2% fallback if method fails
        },

        "take_profit": {
            "method": "atr",               # "atr", "resistance", "fixed_pct"
            "atr_multiplier": 3.0,
            "use_runner_mode": true         # Let Phase 4 Runner handle exit
        },

        "statistics": {
            "total_activations": 0,
            "wins": 0,
            "losses": 0,
            "avg_profit_pct": 0.0,
            "avg_loss_pct": 0.0,
            "win_rate": 0.0,
            "last_activated": null,
            "best_tickers": {},            # {"AAPL": {"wins": 5, "losses": 1}}
            "worst_tickers": {},
            "best_conditions": [],         # ["earnings_season", "low_vix"]
            "created_at": null,
            "updated_at": null
        }
    }
    """

    # Required fields that every template must have
    REQUIRED_FIELDS = ['id', 'name', 'conditions', 'stop_loss', 'take_profit']

    # Valid operators for conditions
    VALID_OPERATORS = ['>', '<', '>=', '<=', '==', '!=', 'between', 'crosses_above', 'crosses_below']

    def __init__(self, data):
        """Initialize from a dictionary (loaded from JSON)."""
        self.data = data
        self.id = data.get('id', 'UNKNOWN')
        self.name = data.get('name', 'Unnamed Template')
        self.enabled = data.get('enabled', True)
        self.source = data.get('source', 'seed')
        self.category = data.get('category', 'default')
        self.required_state = data.get('required_state', {})
        self.conditions = data.get('conditions', [])
        self.entry = data.get('entry', {"type": "close", "confirmation_candles": 0})
        self.stop_loss = data.get('stop_loss', {"method": "atr", "atr_multiplier": 2.0, "fallback_pct": 0.02})
        self.take_profit = data.get('take_profit', {"method": "atr", "atr_multiplier": 3.0, "use_runner_mode": True})
        self.statistics = data.get('statistics', self._empty_stats())

    def _empty_stats(self):
        """Returns a fresh extended statistics block."""
        return {
            # --- Basic Performance ---
            "total_activations": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "avg_profit_pct": 0.0,
            "avg_loss_pct": 0.0,
            "max_profit_pct": 0.0,
            "max_loss_pct": 0.0,
            "avg_hold_duration_hours": 0.0,

            # --- Per-Ticker Performance ---
            "ticker_stats": {},
            # e.g. {"AAPL": {"wins": 5, "losses": 1, "avg_profit": 2.1, "total": 6}}

            # --- Per-Volume-Range Performance ---
            "volume_range_stats": {},
            # e.g. {"high": {"wins": 8, "losses": 2}, "mid": {...}, "low": {...}}

            # --- Per-Trend-Direction Performance ---
            "trend_stats": {},
            # e.g. {"BULLISH": {"wins": 12, "losses": 3}, "SIDEWAYS": {...}}

            # --- Per-Volatility-State Performance ---
            "volatility_stats": {},
            # e.g. {"COMPRESSED": {"wins": 8, "losses": 1}, "NORMAL": {...}}

            # --- Per-Regime Performance ---
            "regime_stats": {},
            # e.g. {"TREND": {"wins": 10, "losses": 2}, "CHOP": {...}}

            # --- Time-Based Performance ---
            "month_stats": {},
            # e.g. {"01": {"wins": 2, "losses": 1}, "02": {...}}

            "day_of_week_stats": {},
            # e.g. {"Mon": {"wins": 5, "losses": 2}, "Tue": {...}}

            # --- Streak Tracking ---
            "consecutive_wins": 0,
            "consecutive_losses": 0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,

            # --- Meta ---
            "last_activated": None,
            "last_win_ticker": None,
            "last_loss_ticker": None,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),

            # --- Per-Block Performance (P1 #7A) ---
            "block_stats": {},
            # Structure per block:
            # "rsi_between": {
            #     "evaluated": 0, "passed": 0, "failed": 0, "pass_rate": 0.0,
            #     "was_the_blocker": 0, "blocker_rate": 0.0,
            #     "when_passed": {"total_trades": 0, "wins": 0, "losses": 0,
            #                     "wr": 0.0, "avg_pnl": 0.0, "total_pnl": 0.0},
            #     "per_symbol": {}
            # }
        }

    def get_category(self):
        """
        Return template category for vectorized decay (SPEC v13.4 §4).

        If template data contains a 'category' field, use it.
        Otherwise infer from template ID naming conventions so existing seed
        templates get correct decay rates without modifying JSON files.
        """
        # Explicit category field takes priority
        explicit = self.data.get('category', None)
        if explicit:
            return explicit
        # Infer from ID — covers all current seed templates
        tid = self.id.upper()
        if 'VSA' in tid or 'INSTITUTIONAL' in tid:
            return 'vsa_institutional'
        if 'BREAKOUT' in tid:
            return 'breakout'
        if 'BOUNCE' in tid or 'REVERSION' in tid or 'OVERSOLD' in tid:
            return 'mean_reversion'
        if 'MOMENTUM' in tid or 'TREND' in tid or 'PULLBACK' in tid:
            return 'momentum'
        return 'default'

    def validate(self):
        """
        Validates the template structure. Returns (is_valid, errors_list).
        """
        errors = []

        for field in self.REQUIRED_FIELDS:
            if field not in self.data:
                errors.append(f"Missing required field: {field}")

        # Validate conditions (block-based)
        for i, cond in enumerate(self.conditions):
            block_name = cond.get('block')
            if not block_name:
                # Support legacy operator-based format too
                if 'indicator' not in cond:
                    errors.append(f"Condition {i}: missing 'block' or 'indicator'")
                continue
            if block_name not in CONDITION_BLOCKS:
                errors.append(f"Condition {i}: unknown block '{block_name}'")
            if 'params' not in cond:
                errors.append(f"Condition {i}: missing 'params' for block '{block_name}'")

        # Validate stop_loss
        if self.stop_loss.get('method') not in ['atr', 'swing_low', 'sma', 'fixed_pct']:
            errors.append(f"Invalid stop_loss method: {self.stop_loss.get('method')}")

        # Validate take_profit
        if self.take_profit.get('method') not in ['atr', 'resistance', 'fixed_pct']:
            errors.append(f"Invalid take_profit method: {self.take_profit.get('method')}")

        # ── Anti-Overfitting Validation ──────────────────────────────
        tmpl_cfg = getattr(cfg, 'TEMPLATE_CONFIG', {})

        # Rule 1: Hard ceiling (safety net)
        hard_limit = tmpl_cfg.get('max_conditions_hard_limit',
                                  tmpl_cfg.get('max_conditions_per_template', 7))
        if len(self.conditions) > hard_limit:
            errors.append(
                f"Too many conditions: {len(self.conditions)} > hard limit {hard_limit}"
            )
            logger.warning(
                f"[{self.id}] Template rejected: {len(self.conditions)} conditions "
                f"exceeds hard limit of {hard_limit}"
            )

        # Rule 2: Category diversity — max N blocks from same category
        max_per_cat = tmpl_cfg.get('max_conditions_per_category', 2)
        categories  = tmpl_cfg.get('block_categories', {})
        if categories and self.conditions:
            block_to_cat = {}
            for cat, blocks in categories.items():
                for b in blocks:
                    block_to_cat[b] = cat

            cat_counts = {}
            for cond in self.conditions:
                bn  = cond.get('block', '')
                cat = block_to_cat.get(bn, 'unknown')
                cat_counts[cat] = cat_counts.get(cat, 0) + 1

            for cat, count in cat_counts.items():
                if cat != 'unknown' and count > max_per_cat:
                    errors.append(
                        f"Category '{cat}' has {count} blocks (max {max_per_cat}) — "
                        f"reduces diversity, risk of redundancy"
                    )
                    logger.warning(
                        f"[{self.id}] Category diversity violation: "
                        f"'{cat}' has {count} blocks (max {max_per_cat})"
                    )

        return len(errors) == 0, errors

    def get_category(self):
        """Return the template's strategy category (e.g. 'mean_reversion', 'breakout')."""
        return self.category

    def get_win_rate(self):
        """Returns win rate as a percentage."""
        total = self.statistics.get('total_activations', 0)
        if total == 0:
            return 0.0
        return (self.statistics.get('wins', 0) / total) * 100.0

    def get_best_context(self):
        """
        Analyzes statistics to determine the best context for this template.
        Returns a dict describing when/where this template performs best.

        Example return:
        {
            "best_trend": "BULLISH",
            "best_trend_win_rate": 82.0,
            "best_volatility": "COMPRESSED",
            "best_volume": "high",
            "best_ticker": "AAPL",
            "best_ticker_win_rate": 85.0,
            "best_month": "03",
            "best_day": "Tue",
            "avoid_trend": "BEARISH",
            "avoid_ticker": "TSLA",
        }
        """
        result = {}
        stats = self.statistics

        def _best_from_dict(d):
            """Find key with highest win rate from a {key: {wins, losses}} dict."""
            best_key, best_wr = None, 0
            worst_key, worst_wr = None, 100
            for key, data in d.items():
                total = data.get('wins', 0) + data.get('losses', 0)
                if total < 3:  # Need minimum sample
                    continue
                wr = (data['wins'] / total) * 100
                if wr > best_wr:
                    best_wr = wr
                    best_key = key
                if wr < worst_wr:
                    worst_wr = wr
                    worst_key = key
            return best_key, best_wr, worst_key, worst_wr

        # Best/worst trend
        trend_stats = stats.get('trend_stats', {})
        if trend_stats:
            b, bwr, w, wwr = _best_from_dict(trend_stats)
            if b:
                result['best_trend'] = b
                result['best_trend_win_rate'] = round(bwr, 1)
            if w:
                result['avoid_trend'] = w

        # Best/worst volatility
        vol_stats = stats.get('volatility_stats', {})
        if vol_stats:
            b, bwr, w, wwr = _best_from_dict(vol_stats)
            if b:
                result['best_volatility'] = b
            if w:
                result['avoid_volatility'] = w

        # Best volume range
        volume_stats = stats.get('volume_range_stats', {})
        if volume_stats:
            b, bwr, w, wwr = _best_from_dict(volume_stats)
            if b:
                result['best_volume'] = b

        # Best/worst ticker
        ticker_stats = stats.get('ticker_stats', {})
        if ticker_stats:
            b, bwr, w, wwr = _best_from_dict(ticker_stats)
            if b:
                result['best_ticker'] = b
                result['best_ticker_win_rate'] = round(bwr, 1)
            if w:
                result['avoid_ticker'] = w

        # Best month
        month_stats = stats.get('month_stats', {})
        if month_stats:
            b, bwr, w, wwr = _best_from_dict(month_stats)
            if b:
                result['best_month'] = b

        # Best day
        day_stats = stats.get('day_of_week_stats', {})
        if day_stats:
            b, bwr, w, wwr = _best_from_dict(day_stats)
            if b:
                result['best_day'] = b

        return result

    def evaluate_conditions(self, row):
        """
        Evaluate all conditions against a DataFrame row using the Block Registry.
        Returns: (all_passed: bool, details: list of dicts)
        """
        details = []
        all_passed = True

        for cond in self.conditions:
            block_name = cond.get('block')
            params = cond.get('params', [])

            if block_name and block_name in CONDITION_BLOCKS:
                try:
                    result = CONDITION_BLOCKS[block_name](row, params)
                    details.append({"block": block_name, "params": params, "passed": result})
                    if not result:
                        all_passed = False
                except Exception as e:
                    logger.debug(f"Block {block_name} error: {e}")
                    details.append({"block": block_name, "params": params, "passed": False, "error": str(e)})
                    all_passed = False
            else:
                logger.warning(f"Unknown block: {block_name}")
                details.append({"block": block_name, "passed": False, "error": "unknown block"})
                all_passed = False

        return all_passed, details

    def calculate_stop_loss(self, row):
        """Calculate stop-loss price using the configured stop block."""
        method = self.stop_loss.get('method', 'atr')
        params = []

        if method == 'atr':
            params = [self.stop_loss.get('atr_multiplier', 2.0)]
        elif method == 'swing_low':
            params = [self.stop_loss.get('atr_multiplier', 0.5)]
        elif method == 'fixed_pct':
            params = [self.stop_loss.get('fallback_pct', 0.02)]
        elif method == 'sma':
            params = [self.stop_loss.get('sma_period', 50), self.stop_loss.get('atr_multiplier', 0.5)]

        if method in STOP_BLOCKS:
            try:
                return STOP_BLOCKS[method](row, params)
            except Exception as e:
                logger.debug(f"Stop block {method} error: {e}")

        # Fallback
        fallback_pct = self.stop_loss.get('fallback_pct', 0.02)
        return stop_fixed_pct(row, [fallback_pct])

    def calculate_take_profit(self, row):
        """Calculate take-profit price using the configured target block."""
        method = self.take_profit.get('method', 'atr')
        params = []

        if method == 'atr':
            params = [self.take_profit.get('atr_multiplier', 3.0)]
        elif method == 'fixed_pct':
            params = [self.take_profit.get('target_pct', 0.05)]

        if method in TARGET_BLOCKS:
            try:
                return TARGET_BLOCKS[method](row, params)
            except Exception as e:
                logger.debug(f"Target block {method} error: {e}")

        # Fallback
        return target_atr(row, [3.0])

    def record_result(self, ticker, profit_pct, won, context=None):
        """
        Records the outcome of a template activation with full context.

        Args:
            ticker: Stock symbol (e.g., "AAPL")
            profit_pct: Realized profit/loss percentage (e.g., 2.5 or -1.3)
            won: True if profitable, False if loss
            context: Optional dict with additional info:
                {
                    "stock_state": {"trend": "BULLISH", "volume": "SURGING", ...},
                    "regime": "TREND",
                    "hold_duration_hours": 48.5,
                    "avg_volume": 5000000,
                }
        """
        if context is None:
            context = {}

        stats = self.statistics
        stats['total_activations'] = stats.get('total_activations', 0) + 1
        stats['last_activated'] = datetime.now().isoformat()
        stats['updated_at'] = datetime.now().isoformat()

        # --- Basic Win/Loss ---
        if won:
            stats['wins'] = stats.get('wins', 0) + 1
            old_avg = stats.get('avg_profit_pct', 0.0)
            n_wins = stats['wins']
            stats['avg_profit_pct'] = old_avg + (profit_pct - old_avg) / n_wins
            stats['max_profit_pct'] = max(stats.get('max_profit_pct', 0.0), profit_pct)
            stats['last_win_ticker'] = ticker
            stats['consecutive_wins'] = stats.get('consecutive_wins', 0) + 1
            stats['consecutive_losses'] = 0
            stats['max_consecutive_wins'] = max(stats.get('max_consecutive_wins', 0), stats['consecutive_wins'])
        else:
            stats['losses'] = stats.get('losses', 0) + 1
            old_avg = stats.get('avg_loss_pct', 0.0)
            n_losses = stats['losses']
            stats['avg_loss_pct'] = old_avg + (profit_pct - old_avg) / n_losses
            stats['max_loss_pct'] = min(stats.get('max_loss_pct', 0.0), profit_pct)
            stats['last_loss_ticker'] = ticker
            stats['consecutive_losses'] = stats.get('consecutive_losses', 0) + 1
            stats['consecutive_wins'] = 0
            stats['max_consecutive_losses'] = max(stats.get('max_consecutive_losses', 0), stats['consecutive_losses'])

        # --- Win Rate ---
        total = stats['total_activations']
        stats['win_rate'] = (stats['wins'] / total) * 100.0 if total > 0 else 0.0

        # --- Hold Duration ---
        hold_hours = context.get('hold_duration_hours', 0)
        if hold_hours > 0:
            old_avg_hold = stats.get('avg_hold_duration_hours', 0.0)
            stats['avg_hold_duration_hours'] = old_avg_hold + (hold_hours - old_avg_hold) / total

        # --- Per-Ticker Stats ---
        ticker_stats = stats.get('ticker_stats', {})
        if ticker not in ticker_stats:
            ticker_stats[ticker] = {"wins": 0, "losses": 0, "total": 0, "avg_profit": 0.0}
        ts = ticker_stats[ticker]
        ts['total'] += 1
        if won:
            ts['wins'] += 1
            old = ts.get('avg_profit', 0.0)
            ts['avg_profit'] = old + (profit_pct - old) / ts['wins']
        else:
            ts['losses'] += 1
        stats['ticker_stats'] = ticker_stats

        # --- Per-Volume-Range Stats ---
        avg_volume = context.get('avg_volume', 0)
        if avg_volume > 0:
            if avg_volume >= 5_000_000:
                vol_key = "high"
            elif avg_volume >= 1_000_000:
                vol_key = "mid"
            else:
                vol_key = "low"
            vol_stats = stats.get('volume_range_stats', {})
            if vol_key not in vol_stats:
                vol_stats[vol_key] = {"wins": 0, "losses": 0}
            vol_stats[vol_key]["wins" if won else "losses"] += 1
            stats['volume_range_stats'] = vol_stats

        # --- Per-Trend Stats ---
        stock_state = context.get('stock_state', {})
        trend = stock_state.get('trend', '')
        if trend:
            trend_stats = stats.get('trend_stats', {})
            if trend not in trend_stats:
                trend_stats[trend] = {"wins": 0, "losses": 0}
            trend_stats[trend]["wins" if won else "losses"] += 1
            stats['trend_stats'] = trend_stats

        # --- Per-Volatility Stats ---
        volatility = stock_state.get('volatility', '')
        if volatility:
            vol_st = stats.get('volatility_stats', {})
            if volatility not in vol_st:
                vol_st[volatility] = {"wins": 0, "losses": 0}
            vol_st[volatility]["wins" if won else "losses"] += 1
            stats['volatility_stats'] = vol_st

        # --- Per-Regime Stats ---
        regime = context.get('regime', '')
        if regime:
            reg_stats = stats.get('regime_stats', {})
            if regime not in reg_stats:
                reg_stats[regime] = {"wins": 0, "losses": 0}
            reg_stats[regime]["wins" if won else "losses"] += 1
            stats['regime_stats'] = reg_stats

        # --- Time-Based Stats ---
        now = datetime.now()
        month_key = now.strftime("%m")
        month_stats = stats.get('month_stats', {})
        if month_key not in month_stats:
            month_stats[month_key] = {"wins": 0, "losses": 0}
        month_stats[month_key]["wins" if won else "losses"] += 1
        stats['month_stats'] = month_stats

        day_key = now.strftime("%a")  # Mon, Tue, Wed...
        day_stats = stats.get('day_of_week_stats', {})
        if day_key not in day_stats:
            day_stats[day_key] = {"wins": 0, "losses": 0}
        day_stats[day_key]["wins" if won else "losses"] += 1
        stats['day_of_week_stats'] = day_stats

    def record_block_results(self, details, symbol="", all_passed=False,
                             outcome=None):
        """
        Record per-block pass/fail statistics from evaluate_conditions.

        Called by shadow_ledger during candle-by-candle evaluation.

        Args:
            details: list of dicts from evaluate_conditions(), each:
                     {"block": "rsi_between", "params": [40,65], "passed": True}
            symbol: ticker symbol for per-symbol tracking
            all_passed: whether ALL conditions passed (signal generated)
            outcome: dict with trade outcome if all_passed, e.g.:
                     {"hit": "target", "pnl_pct": 2.5} or
                     {"hit": "stop", "pnl_pct": -1.2} or None
        """
        if not details:
            return

        stats = self.statistics
        if "block_stats" not in stats:
            stats["block_stats"] = {}

        block_stats = stats["block_stats"]

        # Determine if there's exactly one blocker
        passed_blocks = [d for d in details if d.get("passed", False)]
        failed_blocks = [d for d in details if not d.get("passed", False)]
        single_blocker = None
        if len(failed_blocks) == 1 and len(passed_blocks) == len(details) - 1:
            single_blocker = failed_blocks[0].get("block", "")

        for detail in details:
            block_name = detail.get("block", "")
            if not block_name:
                continue
            passed = detail.get("passed", False)

            # Initialize block entry if new
            if block_name not in block_stats:
                block_stats[block_name] = {
                    "evaluated": 0,
                    "passed": 0,
                    "failed": 0,
                    "pass_rate": 0.0,
                    "was_the_blocker": 0,
                    "blocker_rate": 0.0,
                    "when_passed": {
                        "total_trades": 0,
                        "wins": 0,
                        "losses": 0,
                        "wr": 0.0,
                        "avg_pnl": 0.0,
                        "total_pnl": 0.0,
                    },
                    "per_symbol": {},
                }

            bs = block_stats[block_name]

            # Level 1: basic counts
            bs["evaluated"] += 1
            if passed:
                bs["passed"] += 1
            else:
                bs["failed"] += 1

            # Blocker detection
            if block_name == single_blocker:
                bs["was_the_blocker"] += 1

            # Recalculate rates
            if bs["evaluated"] > 0:
                bs["pass_rate"] = round(bs["passed"] / bs["evaluated"] * 100, 1)
            if bs["failed"] > 0:
                bs["blocker_rate"] = round(
                    bs["was_the_blocker"] / bs["failed"] * 100, 1
                )

            # Level 2: outcome correlation (only when all conditions passed)
            if all_passed and passed and outcome is not None:
                wp = bs["when_passed"]
                hit = outcome.get("hit", "neither")
                pnl = outcome.get("pnl_pct", 0.0)

                if hit in ("target", "stop"):
                    wp["total_trades"] += 1
                    wp["total_pnl"] += pnl
                    if hit == "target":
                        wp["wins"] += 1
                    else:
                        wp["losses"] += 1

                    if wp["total_trades"] > 0:
                        wp["wr"] = round(
                            wp["wins"] / wp["total_trades"] * 100, 1
                        )
                        wp["avg_pnl"] = round(
                            wp["total_pnl"] / wp["total_trades"], 2
                        )

            # Level 3: per-symbol
            if symbol:
                if symbol not in bs["per_symbol"]:
                    bs["per_symbol"][symbol] = {
                        "evaluated": 0,
                        "passed": 0,
                        "pass_rate": 0.0,
                        "trades_when_passed": 0,
                        "wins_when_passed": 0,
                        "wr_when_passed": 0.0,
                    }

                ps = bs["per_symbol"][symbol]
                ps["evaluated"] += 1
                if passed:
                    ps["passed"] += 1

                if ps["evaluated"] > 0:
                    ps["pass_rate"] = round(
                        ps["passed"] / ps["evaluated"] * 100, 1
                    )

                # Per-symbol outcome
                if all_passed and passed and outcome is not None:
                    hit = outcome.get("hit", "neither")
                    if hit in ("target", "stop"):
                        ps["trades_when_passed"] += 1
                        if hit == "target":
                            ps["wins_when_passed"] += 1
                        if ps["trades_when_passed"] > 0:
                            ps["wr_when_passed"] = round(
                                ps["wins_when_passed"] / ps["trades_when_passed"] * 100, 1
                            )

        logger.debug(
            f"[{self.id}] Block stats updated: "
            f"{len(passed_blocks)} passed, {len(failed_blocks)} failed"
            f"{f', blocker={single_blocker}' if single_blocker else ''}"
        )

    def to_dict(self):
        """Serialize back to dictionary for JSON storage."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.data.get('description', ''),
            "version": self.data.get('version', 1),
            "source": self.source,
            "category": self.category,
            "enabled": self.enabled,
            "required_state": self.required_state,
            "conditions": self.conditions,
            "entry": self.entry,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "statistics": self.statistics,
        }


class TemplateManager:
    """
    Loads, saves, and manages the library of trading templates.
    Templates are stored as individual JSON files in data/templates/.
    """

    def __init__(self):
        self.templates_dir = os.path.join(cfg.DB_DIR, "templates")
        os.makedirs(self.templates_dir, exist_ok=True)
        self.templates = {}  # id -> SetupTemplate
        self.load_all()

    def load_all(self):
        """Load all template JSON files from the templates directory."""
        self.templates = {}

        if not os.path.exists(self.templates_dir):
            logger.warning(f"Templates directory not found: {self.templates_dir}")
            return

        for filename in os.listdir(self.templates_dir):
            if not filename.endswith('.json'):
                continue
            filepath = os.path.join(self.templates_dir, filename)
            try:
                data = safe_json_read(filepath, default={})
                template = SetupTemplate(data)
                is_valid, errors = template.validate()
                if is_valid:
                    if not template.enabled:
                        logger.info(f"Skipping disabled template: {template.id}")
                        continue
                    self.templates[template.id] = template
                    logger.debug(
                        f"Loaded template: {template.id} ({template.name}), "
                        f"{len(template.conditions)} conditions"
                    )
                else:
                    logger.warning(f"Invalid template {filename}: {errors}")
            except Exception as e:
                logger.error(f"Failed to load template {filename}: {e}")

        logger.info(f"Template library loaded: {len(self.templates)} templates")

    def save_template(self, template):
        """Save a single template to disk."""
        filepath = os.path.join(self.templates_dir, f"{template.id}.json")
        try:
            safe_json_write(filepath, template.to_dict())
            logger.debug(f"Saved template: {template.id}")
        except Exception as e:
            logger.error(f"Failed to save template {template.id}: {e}")

    def save_all(self):
        """Save all templates to disk (useful after updating statistics)."""
        for template in self.templates.values():
            self.save_template(template)
        logger.info(f"Saved {len(self.templates)} templates to disk")

    def disable_template(self, template_id):
        """Set enabled=false on a template JSON file and remove it from the active cache.

        Used by the quality gate to disable newly generated templates that fail WF validation.
        Does NOT delete the file — preserves it for future analysis and re-enabling.

        Returns:
            True if the file was found and updated, False on error.
        """
        filepath = os.path.join(self.templates_dir, f"{template_id}.json")
        try:
            data = safe_json_read(filepath, default={})
            if not data:
                logger.warning(f"Quality Gate: template file not found for {template_id}")
                return False
            data["enabled"] = False
            safe_json_write(filepath, data)
            self.templates.pop(template_id, None)
            logger.info(f"Quality Gate: disabled template {template_id}")
            return True
        except Exception as e:
            logger.error(f"Quality Gate: failed to disable template {template_id}: {e}")
            return False

    def get_enabled(self):
        """Return list of all enabled templates."""
        return [t for t in self.templates.values() if t.enabled]

    def get_for_state(self, stock_state, symbol=""):
        """
        Return templates that match the stock's current state.
        Filters enabled templates by required_state compatibility.
        Logs match/reject reasons per template for analysis.
        """
        matching = []
        enabled = self.get_enabled()
        for template in enabled:
            if self._state_matches(template.required_state, stock_state):
                matching.append(template)
                logger.debug(f"[{symbol}] [REGIME] ✓ {template.name} — state match")
            else:
                mismatch = self._get_mismatch_reason(template.required_state, stock_state)
                logger.debug(f"[{symbol}] [REGIME] ✗ {template.name} — {mismatch}")
        logger.info(f"[{symbol}] [REGIME] Template filtering: {len(enabled)} enabled → {len(matching)} matched state")
        return matching

    def _state_matches(self, required_state, stock_state):
        """
        Check if a stock's state matches a template's requirements.
        Each required_state field is a list of acceptable values.
        If a field is missing from required_state, any value is accepted.
        """
        for key, acceptable_values in required_state.items():
            actual_value = stock_state.get(key, '')
            if actual_value not in acceptable_values:
                return False
        return True

    def _get_mismatch_reason(self, required_state, stock_state):
        """Return human-readable string describing which state fields don't match."""
        reasons = []
        for key, acceptable in required_state.items():
            actual = stock_state.get(key, '')
            if actual not in acceptable:
                reasons.append(f"{key} mismatch (required: {','.join(acceptable)} | actual: {actual})")
        return "; ".join(reasons) if reasons else "unknown mismatch"

    def get_template_by_id(self, template_id):
        """Get a specific template by ID."""
        return self.templates.get(template_id)

    def add_template(self, template_data):
        """Add a new template from a dictionary. Validates before adding."""
        template = SetupTemplate(template_data)
        is_valid, errors = template.validate()
        if not is_valid:
            logger.error(f"Cannot add invalid template: {errors}")
            return False
        self.templates[template.id] = template
        self.save_template(template)
        logger.info(
            f"Added new template: {template.id} ({template.name}) "
            f"with {len(template.conditions)} conditions"
        )
        return True

    def get_statistics_summary(self):
        """Return a summary of all template statistics for logging/reporting."""
        summary = []
        for t in self.templates.values():
            summary.append({
                "id": t.id,
                "name": t.name,
                "enabled": t.enabled,
                "source": t.source,
                "total": t.statistics.get('total_activations', 0),
                "win_rate": t.get_win_rate(),
                "avg_profit": t.statistics.get('avg_profit_pct', 0.0),
            })
        return sorted(summary, key=lambda x: x['win_rate'], reverse=True)


# ── TEMPLATE GENERATOR (CP-4) ────────────────────────────────────────────────

class TemplateGenerator:
    """
    Generate new templates from coverage gaps using recipe-based approach.

    Each recipe is a predefined trading strategy (mean reversion, squeeze breakout, etc.)
    mapped to specific market states. The generator:
    1. Reads coverage_gaps from shadow_ledger.json
    2. Matches recipes to gap states
    3. Builds template dicts from recipes
    4. Validates via SetupTemplate.validate()
    5. Saves via TemplateManager.add_template()

    All templates created with source="generated" and start in BURN_IN lifecycle.
    """

    def __init__(self, template_manager=None):
        self.tm = template_manager or TemplateManager()
        self._gen_counter = {}  # recipe_id -> count for unique ID generation

    def generate_from_gaps(self, coverage_gaps=None):
        """Main entry: read gaps → match recipes → create templates → validate → save.

        Args:
            coverage_gaps: list of gap dicts from shadow_ledger. If None, loads from file.

        Returns:
            dict: {
                "created": [list of template IDs created],
                "skipped_duplicate": [list of recipe IDs skipped],
                "skipped_low_score": [list],
                "skipped_low_bars": [list],
                "validation_failed": [list],
                "total_gaps_evaluated": int,
            }
        """
        gen_cfg = getattr(cfg, 'TEMPLATE_EVOLUTION_CONFIG', {}).get("generation", {})
        if not gen_cfg.get("enabled", True):
            logger.info("[GENERATE] Template generation is disabled in config")
            return {"created": [], "skipped_duplicate": [], "skipped_low_score": [],
                    "skipped_low_bars": [], "validation_failed": [], "total_gaps_evaluated": 0}

        if coverage_gaps is None:
            coverage_gaps = self._load_coverage_gaps()

        if not coverage_gaps:
            logger.info("[GENERATE] No coverage gaps found — nothing to generate")
            return {"created": [], "skipped_duplicate": [], "skipped_low_score": [],
                    "skipped_low_bars": [], "validation_failed": [], "total_gaps_evaluated": 0}

        min_score = gen_cfg.get("min_opportunity_score", 0.30)
        min_bars = gen_cfg.get("min_bars_for_generation", 30)
        max_per_gap = gen_cfg.get("max_templates_per_gap", 2)
        max_total = gen_cfg.get("max_total_generated", 10)
        recipes = getattr(cfg, 'TEMPLATE_GENERATION_RECIPES', {})

        report = {
            "created": [],
            "skipped_duplicate": [],
            "skipped_low_score": [],
            "skipped_low_bars": [],
            "validation_failed": [],
            "total_gaps_evaluated": len(coverage_gaps),
        }

        # Sort gaps by opportunity_score descending (prioritize biggest opportunities)
        sorted_gaps = sorted(coverage_gaps, key=lambda g: g.get("opportunity_score", 0), reverse=True)

        for gap in sorted_gaps:
            if len(report["created"]) >= max_total:
                logger.info(f"[GENERATE] Reached max_total_generated={max_total} — stopping")
                break

            score = gap.get("opportunity_score", 0)
            bars = gap.get("bar_count", gap.get("bars", 0))
            state_str = gap.get("state", "")

            logger.info(f"[GENERATE] Evaluating gap: {state_str} | bars={bars} | score={score:.3f}")

            if score < min_score:
                report["skipped_low_score"].append(state_str)
                logger.debug(f"[GENERATE] Skipped {state_str} — score {score:.3f} < min {min_score}")
                continue

            if bars < min_bars:
                report["skipped_low_bars"].append(state_str)
                logger.debug(f"[GENERATE] Skipped {state_str} — bars {bars} < min {min_bars}")
                continue

            # Parse gap state into dict
            gap_state = self._parse_gap_state(state_str)

            # Find matching recipes
            matched_recipes = self._match_recipes_to_gap(gap_state, recipes)

            created_for_gap = 0
            for recipe_id, recipe in matched_recipes:
                if created_for_gap >= max_per_gap:
                    break
                if len(report["created"]) >= max_total:
                    break

                # Build template from recipe
                template_data = self._build_template_from_recipe(recipe_id, recipe, gap_state, gen_cfg)

                # Check for functional duplicate
                if self._is_duplicate(template_data):
                    report["skipped_duplicate"].append(recipe_id)
                    logger.info(f"[GENERATE] Skipped recipe {recipe_id} — duplicate of existing template")
                    continue

                # Validate
                test_template = SetupTemplate(template_data)
                is_valid, errors = test_template.validate()
                if not is_valid:
                    report["validation_failed"].append({"recipe": recipe_id, "errors": errors})
                    logger.warning(f"[GENERATE] REJECTED {template_data['id']} — validation errors: {errors}")
                    continue

                # Save via TemplateManager
                success = self.tm.add_template(template_data)
                if success:
                    report["created"].append(template_data["id"])
                    created_for_gap += 1
                    logger.info(
                        f"[GENERATE] Created {template_data['id']} | "
                        f"recipe={recipe_id} | state={state_str} | "
                        f"blocks={len(template_data['conditions'])} | "
                        f"source=generated | category={template_data.get('category', 'unknown')}"
                    )

        # Summary log
        logger.info(
            f"[GENERATE-SUMMARY] gaps_evaluated={report['total_gaps_evaluated']} | "
            f"templates_created={len(report['created'])} | "
            f"duplicates_skipped={len(report['skipped_duplicate'])} | "
            f"low_score_skipped={len(report['skipped_low_score'])} | "
            f"low_bars_skipped={len(report['skipped_low_bars'])} | "
            f"validation_failed={len(report['validation_failed'])}"
        )

        return report

    def _load_coverage_gaps(self):
        """Load gaps_by_state from shadow_ledger.json."""
        evo_cfg = getattr(cfg, 'TEMPLATE_EVOLUTION_CONFIG', {})
        path = evo_cfg.get("auto_disable", {}).get("disable_list_path", "data/shadow_ledger.json")
        try:
            data = safe_json_read(path, default={})
            return data.get("coverage_gaps", {}).get("gaps_by_state", [])
        except Exception as e:
            logger.error(f"[GENERATE] Failed to load coverage gaps: {e}")
            return []

    def _parse_gap_state(self, state_str):
        """Parse 'BEARISH:OPEN_FIELD:HEALTHY:COMPRESSED' into state dict."""
        parts = state_str.split(":") if state_str else []
        keys = ["trend", "structure", "volume", "volatility"]
        return {keys[i]: parts[i] for i in range(min(len(parts), len(keys)))}

    def _match_recipes_to_gap(self, gap_state, recipes):
        """Return list of (recipe_id, recipe_dict) that apply to this gap state.

        A recipe matches if:
        - gap trend is in recipe's applicable_trends
        - gap volatility is in recipe's applicable_volatility
        - gap structure is NOT in recipe's excluded_structure
        - IF recipe has required_structure: gap structure must be in it
        """
        matched = []
        gap_trend = gap_state.get("trend", "")
        gap_vol = gap_state.get("volatility", "")
        gap_struct = gap_state.get("structure", "")

        for recipe_id, recipe in recipes.items():
            # Trend match
            if gap_trend not in recipe.get("applicable_trends", []):
                continue
            # Volatility match
            if gap_vol not in recipe.get("applicable_volatility", []):
                continue
            # Required structure (if specified)
            req_struct = recipe.get("required_structure", [])
            if req_struct and gap_struct not in req_struct:
                continue
            # Excluded structure
            excl_struct = recipe.get("excluded_structure", [])
            if excl_struct and gap_struct in excl_struct:
                continue

            matched.append((recipe_id, recipe))

        if not matched:
            logger.debug(f"[GENERATE] No recipe matched gap state: {gap_state}")

        return matched

    def _build_template_from_recipe(self, recipe_id, recipe, gap_state, gen_cfg):
        """Construct full template dict from recipe + gap state."""
        template_id = self._generate_template_id(recipe_id)

        # Build required_state from recipe applicable values + gap specifics
        required_state = {}
        if recipe.get("applicable_trends"):
            required_state["trend"] = list(recipe["applicable_trends"])
        if recipe.get("required_structure"):
            required_state["structure"] = list(recipe["required_structure"])
        elif gap_state.get("structure"):
            required_state["structure"] = self._infer_structures(gap_state)
        if recipe.get("applicable_volatility"):
            required_state["volatility"] = list(recipe["applicable_volatility"])
        # Volume: always require HEALTHY or SURGING for generated templates
        required_state["volume"] = ["HEALTHY", "SURGING"]

        stop_mult = recipe.get("stop_atr_mult", gen_cfg.get("default_stop_atr_mult", 1.5))
        target_mult = recipe.get("target_atr_mult", gen_cfg.get("default_target_atr_mult", 2.5))
        confirmation = recipe.get("confirmation_candles", gen_cfg.get("default_confirmation_candles", 1))
        use_runner = recipe.get("use_runner", gen_cfg.get("use_runner_for_reversal", False))

        return {
            "id": template_id,
            "name": f"{recipe.get('description', recipe_id)} (Generated)",
            "description": recipe.get("description", f"Auto-generated from recipe {recipe_id}"),
            "version": 1,
            "source": gen_cfg.get("source_label", "generated"),
            "category": recipe.get("category", "default"),
            "enabled": True,
            "required_state": required_state,
            "conditions": [dict(c) for c in recipe.get("conditions", [])],
            "entry": {
                "type": "close",
                "confirmation_candles": confirmation,
            },
            "stop_loss": {
                "method": gen_cfg.get("default_stop_method", "atr"),
                "atr_multiplier": stop_mult,
                "fallback_pct": 0.02,
            },
            "take_profit": {
                "method": "atr",
                "atr_multiplier": target_mult,
                "use_runner_mode": use_runner,
            },
            "statistics": SetupTemplate({})._empty_stats(),
        }

    def _infer_structures(self, gap_state):
        """Infer reasonable structure values based on trend.

        For BEARISH: OPEN_FIELD, NEAR_RESISTANCE (common bearish structures).
        For SIDEWAYS: OPEN_FIELD, NEAR_SUPPORT, NEAR_RESISTANCE.
        """
        trend = gap_state.get("trend", "")
        if trend == "BEARISH":
            return ["OPEN_FIELD", "NEAR_RESISTANCE"]
        elif trend == "SIDEWAYS":
            return ["OPEN_FIELD", "NEAR_SUPPORT", "NEAR_RESISTANCE"]
        return ["OPEN_FIELD"]

    def _generate_template_id(self, recipe_id):
        """Generate unique ID: GEN_{RECIPE} or GEN_{RECIPE}_{NNN}."""
        count = self._gen_counter.get(recipe_id, 0) + 1
        self._gen_counter[recipe_id] = count

        base_id = f"GEN_{recipe_id}"
        candidate = base_id if count == 1 else f"{base_id}_{count:03d}"

        existing_ids = set(self.tm.templates.keys())
        while candidate in existing_ids:
            count += 1
            self._gen_counter[recipe_id] = count
            candidate = f"{base_id}_{count:03d}"

        return candidate

    def _is_duplicate(self, new_template):
        """Check if a functionally equivalent template already exists.

        Two templates are duplicates if they have:
        - Same set of block names (order doesn't matter)
        - Any overlapping trend values (covers same market regime)
        """
        new_blocks = set(c.get("block", "") for c in new_template.get("conditions", []))
        new_trends = set(new_template.get("required_state", {}).get("trend", []))

        for existing in self.tm.templates.values():
            existing_blocks = set(c.get("block", "") for c in existing.conditions)
            existing_trends = set(existing.required_state.get("trend", []))

            if new_blocks == existing_blocks and bool(new_trends & existing_trends):
                return True

        return False

    def generate_all_recipes(self):
        """Bootstrap: create one template per recipe, regardless of gap data.

        Use when coverage_gaps is empty or stale. Creates templates for ALL
        defined recipes so they can start collecting data in backtests.

        Returns: same report format as generate_from_gaps()
        """
        gen_cfg = getattr(cfg, 'TEMPLATE_EVOLUTION_CONFIG', {}).get("generation", {})
        if not gen_cfg.get("enabled", True):
            logger.info("[GENERATE] Template generation is disabled in config")
            return {"created": [], "skipped_duplicate": [], "validation_failed": [],
                    "total_gaps_evaluated": 0, "mode": "bootstrap"}

        recipes = getattr(cfg, 'TEMPLATE_GENERATION_RECIPES', {})
        report = {"created": [], "skipped_duplicate": [], "validation_failed": [],
                  "total_gaps_evaluated": len(recipes), "mode": "bootstrap"}

        for recipe_id, recipe in recipes.items():
            # Build a synthetic gap state from recipe's target conditions
            trends  = recipe.get("applicable_trends", ["BEARISH"])
            vols    = recipe.get("applicable_volatility", ["NORMAL"])
            structs = recipe.get("required_structure", ["OPEN_FIELD"])
            if not structs:
                structs = ["OPEN_FIELD"]

            gap_state = {
                "trend":      trends[0],
                "structure":  structs[0],
                "volume":     "HEALTHY",
                "volatility": vols[0],
            }

            template_data = self._build_template_from_recipe(recipe_id, recipe, gap_state, gen_cfg)

            if self._is_duplicate(template_data):
                report["skipped_duplicate"].append(recipe_id)
                logger.info(f"[GENERATE-BOOTSTRAP] Skipped {recipe_id} — duplicate")
                continue

            test_template = SetupTemplate(template_data)
            is_valid, errors = test_template.validate()
            if not is_valid:
                report["validation_failed"].append({"recipe": recipe_id, "errors": errors})
                logger.warning(f"[GENERATE-BOOTSTRAP] REJECTED {template_data['id']} — {errors}")
                continue

            if self.tm.add_template(template_data):
                report["created"].append(template_data["id"])
                logger.info(f"[GENERATE-BOOTSTRAP] Created {template_data['id']} | recipe={recipe_id}")

        logger.info(
            f"[GENERATE-BOOTSTRAP-SUMMARY] created={len(report['created'])} | "
            f"skipped={len(report['skipped_duplicate'])} | "
            f"failed={len(report['validation_failed'])}"
        )
        return report

    def get_generation_report(self):
        """Return summary of all generated templates currently loaded."""
        generated = [
            {"id": t.id, "name": t.name, "category": t.get_category(),
             "conditions": len(t.conditions), "enabled": t.enabled}
            for t in self.tm.templates.values()
            if t.source == "generated"
        ]
        return {
            "total_generated": len(generated),
            "templates": generated,
        }
