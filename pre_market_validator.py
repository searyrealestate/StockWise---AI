# pre_market_validator.py

"""
StockWise Gen-13 Pre-Market Gap Validator
==========================================
Vetoes new entries when the overnight gap exceeds threshold (SPEC v13.4 §5 GAP-07).

Checked at 09:25 ET, BEFORE execute_ticket() is called.
Only active in the 09:20–09:35 ET window to avoid stale checks.

Gap direction (up or down) both count — large gaps in either direction
represent elevated risk and unpredictable opening price action.
"""

import logging
from datetime import datetime, timedelta
import pytz
import system_config as cfg

logger = logging.getLogger("PreMarketValidator")

try:
    from decision_logger import DecisionLogger as _DecisionLogger
    _dl = _DecisionLogger()
except Exception:
    _dl = None

ET = pytz.timezone("America/New_York")


class PreMarketValidator:
    """
    Pre-entry gate: detects overnight gap at the 09:25 ET window.

    Called by live_trading_engine BEFORE execute_ticket().

    Args (constructor):
        data_source_manager: DataSourceManager instance (for IBKR pre-market fetch).
                             May be None — will skip IBKR path and use close fallback.
    """

    def __init__(self, data_source_manager=None):
        self.dsm = data_source_manager
        self.config = getattr(cfg, 'PRE_MARKET_CONFIG', {})
        # symbol -> veto_until (datetime)
        self._veto_cache = {}

    # ========================================
    # PUBLIC: check_gap
    # ========================================
    def check_gap(self, symbol, df):
        """
        Returns (approved, reason).

        approved=True  → gap acceptable, proceed with entry.
        approved=False → gap too large, veto this entry.

        Args:
            symbol: Ticker symbol string
            df:     DataFrame with at least 'close' column and >= 2 rows of daily data
        """
        if not self.config.get('enabled', True):
            return True, ""

        # Check cooldown cache (avoid hammering veto logs every candle)
        if symbol in self._veto_cache:
            if datetime.now() < self._veto_cache[symbol]:
                remaining = (self._veto_cache[symbol] - datetime.now()).seconds // 60
                return False, f"Pre-market gap veto cooldown active for {symbol} ({remaining}m remaining)"
            else:
                del self._veto_cache[symbol]

        # Only run inside the 09:20–09:35 ET window
        now_et = datetime.now(ET)
        check_time_str = self.config.get('check_time', '09:25')
        h, m = map(int, check_time_str.split(':'))
        window_start = now_et.replace(hour=h, minute=max(0, m - 5), second=0, microsecond=0)
        window_end = now_et.replace(hour=h, minute=min(59, m + 10), second=0, microsecond=0)

        if not (window_start <= now_et <= window_end):
            return True, ""  # Outside window — don't block

        max_gap = self.config.get('max_gap_pct', 0.05)
        min_gap = self.config.get('min_gap_pct', 0.001)
        cooldown_minutes = self.config.get('veto_cooldown_minutes', 60)

        try:
            gap_pct = self._calculate_gap(symbol, df)

            if gap_pct is None:
                return True, ""  # Cannot calculate — fail open

            abs_gap = abs(gap_pct)

            if abs_gap < min_gap:
                return True, ""  # Negligible gap — noise floor

            if abs_gap > max_gap:
                direction = "UP" if gap_pct > 0 else "DOWN"
                reason = (
                    f"Pre-market gap {direction} {abs_gap:.1%} exceeds max "
                    f"{max_gap:.0%} for {symbol}"
                )
                logger.warning(f"[{symbol}] PRE-MARKET VETO: {reason}")
                self._veto_cache[symbol] = datetime.now() + timedelta(minutes=cooldown_minutes)
                if _dl:
                    try: _dl.log_veto(symbol=symbol, gate="premarket_gap", passed=False, reason=reason, gap_pct=round(gap_pct, 4), max_gap=max_gap)
                    except Exception: pass
                return False, reason

            logger.debug(f"[{symbol}] Pre-market gap {gap_pct:+.2%} — within limit")
            return True, ""

        except Exception as e:
            logger.debug(f"[{symbol}] Pre-market check error (fail open): {e}")
            return True, ""  # Fail open — never block on unexpected errors

    # ========================================
    # PRIVATE: _calculate_gap
    # ========================================
    def _calculate_gap(self, symbol, df):
        """
        Calculate overnight gap as (current_price - prev_close) / prev_close.

        Priority order:
          1. IBKR real pre-market price (if use_ibkr_for_premarket=True and dsm available)
          2. Last daily close vs. second-to-last close (fallback_to_last_close=True)

        Returns:
            float gap_pct, or None if unable to calculate.
        """
        use_ibkr = self.config.get('use_ibkr_for_premarket', True)
        fallback = self.config.get('fallback_to_last_close', True)

        if df is None or len(df) < 2:
            return None

        if 'close' not in df.columns:
            return None

        prev_close = float(df['close'].iloc[-2])
        if prev_close <= 0:
            return None

        # 1. Try IBKR for live pre-market price
        if use_ibkr and self.dsm is not None:
            try:
                pre_price = self._get_ibkr_premarket_price(symbol)
                if pre_price is not None and pre_price > 0:
                    return (pre_price - prev_close) / prev_close
            except Exception as e:
                logger.debug(f"[{symbol}] IBKR pre-market price unavailable: {e}")

        # 2. Fallback: last daily close vs. second-to-last
        if fallback:
            last_close = float(df['close'].iloc[-1])
            if last_close > 0:
                return (last_close - prev_close) / prev_close

        return None

    def _get_ibkr_premarket_price(self, symbol):
        """
        Attempt to fetch the most recent pre-market price via DSM (IBKR path).
        Returns float or None.
        """
        if self.dsm is None:
            return None
        try:
            df_rt = self.dsm.get_stock_data(symbol, days_back=1)
            if df_rt is not None and not df_rt.empty and 'close' in df_rt.columns:
                return float(df_rt['close'].iloc[-1])
        except Exception:
            pass
        return None
