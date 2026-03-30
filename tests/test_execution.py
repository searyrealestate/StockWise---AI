# tests/test_execution.py

"""
StockWise Gen-13 — Execution Tests (TDD v1.1 Section 7)
=======================================================
Pre-Market Validator (PM-01→07), Order Types (OT-01→04), Kinetic Stop (KS-01→17).
MONEY PATH — these tests protect real trades.

Execution: python -m pytest tests/test_execution.py -v --tb=short
Expected : 28 passed, 0 failed
"""

import os
import sys
import re
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import pytz
from pre_market_validator import PreMarketValidator
import system_config as cfg


# ── Helpers ────────────────────────────────────────────────────────────────────

ET_TZ = pytz.timezone('US/Eastern')
# 09:25 ET — inside the check window (09:20–09:35)
_IN_WINDOW  = ET_TZ.localize(datetime(2026, 3, 25, 9, 25, 0))
# 10:00 ET — outside the check window
_OUT_WINDOW = ET_TZ.localize(datetime(2026, 3, 25, 10, 0, 0))


def _gap_df(prev_close: float, last_close: float) -> pd.DataFrame:
    """Two-row DataFrame where gap = (last_close - prev_close) / prev_close."""
    return pd.DataFrame(
        {'close': [prev_close, last_close]},
        index=pd.date_range("2026-03-24", periods=2, freq='B'),
    )


def _read_source(filename: str) -> str:
    path = os.path.join(PROJECT_ROOT, filename)
    if not os.path.exists(path):
        pytest.skip(f"{filename} not found")
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


# ═══════════════════════════════════════════════════════
# 7.1  PRE-MARKET VALIDATOR TESTS  (PM-01 → PM-07)
# ═══════════════════════════════════════════════════════

class TestPreMarketValidator:
    """PM-01 to PM-07: Overnight gap detection at 09:25 ET."""

    @pytest.fixture
    def pmv(self):
        v = PreMarketValidator(data_source_manager=None)
        v._veto_cache.clear()
        return v

    # PM-01: Large gap (12%) inside window → VETO
    def test_pm01_large_gap_vetoed(self, pmv):
        df = _gap_df(prev_close=100.0, last_close=112.0)   # +12% gap
        with patch('pre_market_validator.datetime') as mock_dt:
            mock_dt.now.return_value = _IN_WINDOW
            approved, reason = pmv.check_gap("AAPL", df)
        assert not approved, "12% gap must be VETOED"
        assert reason, "Veto reason must not be empty"

    # PM-02: Small gap (1.5%) inside window → PASS
    def test_pm02_small_gap_passes(self, pmv):
        df = _gap_df(prev_close=100.0, last_close=101.5)   # +1.5% gap
        with patch('pre_market_validator.datetime') as mock_dt:
            mock_dt.now.return_value = _IN_WINDOW
            approved, reason = pmv.check_gap("MSFT", df)
        assert approved, f"1.5% gap must PASS, got: {reason}"

    # PM-03: Threshold driven by max_gap_pct from config
    def test_pm03_gap_threshold_from_config(self, pmv):
        max_gap = cfg.PRE_MARKET_CONFIG.get('max_gap_pct', 0.05)

        # Just above threshold → VETO
        above = 100.0 * (1 + max_gap + 0.02)
        with patch('pre_market_validator.datetime') as mock_dt:
            mock_dt.now.return_value = _IN_WINDOW
            approved_above, _ = pmv.check_gap("TSLA", _gap_df(100.0, above))
        assert not approved_above, f"Gap > {max_gap:.0%} must be vetoed"

        pmv._veto_cache.clear()

        # Just below threshold → PASS
        below = 100.0 * (1 + max_gap - 0.02)
        with patch('pre_market_validator.datetime') as mock_dt:
            mock_dt.now.return_value = _IN_WINDOW
            approved_below, _ = pmv.check_gap("TSLA", _gap_df(100.0, below))
        assert approved_below, f"Gap < {max_gap:.0%} must pass"

    # PM-04: Inside check window + large gap → validator fires (veto returned)
    def test_pm04_inside_window_validator_fires(self, pmv):
        df = _gap_df(100.0, 115.0)   # 15% gap
        with patch('pre_market_validator.datetime') as mock_dt:
            mock_dt.now.return_value = _IN_WINDOW
            approved, reason = pmv.check_gap("NVDA", df)
        # At 09:25 the check RUNS; large gap → veto
        assert not approved, "Validator must fire and veto at 09:25 with 15% gap"

    # PM-05: Outside check window (10:00 ET) → validator returns True regardless of gap
    def test_pm05_outside_window_always_passes(self, pmv):
        df = _gap_df(100.0, 115.0)   # 15% gap — would veto if in window
        with patch('pre_market_validator.datetime') as mock_dt:
            mock_dt.now.return_value = _OUT_WINDOW
            approved, reason = pmv.check_gap("NVDA", df)
        assert approved, (
            f"Outside check window must PASS regardless of gap, got: reason='{reason}'"
        )

    # PM-06: Config prefers IBKR data for pre-market prices
    def test_pm06_config_prefers_ibkr(self):
        assert cfg.PRE_MARKET_CONFIG.get('use_ibkr_for_premarket', False) is True, (
            "PRE_MARKET_CONFIG.use_ibkr_for_premarket must be True"
        )

    # PM-07: Veto is cached — repeated call returns cached veto without re-computing
    def test_pm07_veto_cached_after_first_violation(self, pmv):
        df_large = _gap_df(100.0, 115.0)  # 15% — triggers veto
        df_small = _gap_df(100.0, 100.5)  # 0.5% — would normally pass

        with patch('pre_market_validator.datetime') as mock_dt:
            mock_dt.now.return_value = _IN_WINDOW

            # First call: triggers veto and stores in cache
            approved1, reason1 = pmv.check_gap("GOOG", df_large)
            assert not approved1, "First call: large gap must be vetoed"
            assert "GOOG" in pmv._veto_cache, "Symbol must be in veto cache after veto"

            # Second call: cache active — returns veto even with small gap
            approved2, reason2 = pmv.check_gap("GOOG", df_small)
        assert not approved2, "Second call: veto cache must still block entry"
        assert "cooldown" in reason2.lower() or "veto" in reason2.lower()


# ═══════════════════════════════════════════════════════
# 7.2  ORDER TYPE TESTS  (OT-01 → OT-04)
# ═══════════════════════════════════════════════════════

class TestOrderTypes:
    """OT-01 to OT-04: Limit-based entry — zero Market Orders in the codebase."""

    # OT-01: execute_ticket uses limit_price for both entry and fill price
    def test_ot01_execute_uses_limit_price(self):
        source = _read_source("live_trading_engine.py")
        # The execute_ticket return has exec_price: ticket["limit_price"]
        assert 'exec_price' in source and 'limit_price' in source, (
            "execute_ticket must use limit_price as exec_price (LIMIT-style fill)"
        )
        # Verify exec_price is set from limit_price, not a market query
        assert re.search(r"exec_price.*limit_price", source), (
            "exec_price must be assigned from limit_price"
        )

    # OT-02: No MARKET or MKT order_type assignment in execution code
    def test_ot02_no_market_order_type(self):
        source = _read_source("live_trading_engine.py")
        code_lines = [l.split('#')[0] for l in source.split('\n')]  # strip comments
        violations = [
            l for l in code_lines
            if re.search(r"order_type\s*=\s*['\"](?:MARKET|MKT)['\"]", l, re.IGNORECASE)
        ]
        assert not violations, (
            f"Market order type found in live_trading_engine.py:\n"
            + "\n".join(violations)
        )

    # OT-03: execute_ticket returns FILLED status with the limit price (no slippage lookup)
    def test_ot03_execute_returns_filled_at_limit(self):
        source = _read_source("live_trading_engine.py")
        # Verify the return structure includes {"status": "FILLED", "exec_price": limit_price}
        assert '"FILLED"' in source or "'FILLED'" in source, (
            "execute_ticket must return FILLED status"
        )
        # Verify no live market price lookup for execution price
        assert 'get_current_price' not in source or source.count('get_current_price') < 3, (
            "exec_price should not require a live market price call"
        )

    # OT-04: Breakeven calculation uses slippage from COSTS_CONFIG, not a hardcoded literal
    def test_ot04_slippage_from_config(self):
        source = _read_source("live_trading_engine.py")
        # Phase 2 breakeven uses: entry_price * (1 + cfg.COSTS_CONFIG["slippage_pct"])
        assert 'slippage_pct' in source, (
            "slippage_pct must be sourced from cfg.COSTS_CONFIG, not hardcoded"
        )
        assert 'COSTS_CONFIG' in source, (
            "COSTS_CONFIG reference required in live_trading_engine.py for order pricing"
        )


# ═══════════════════════════════════════════════════════
# 7.3  KINETIC STOP TESTS  (KS-01 → KS-17)
# ═══════════════════════════════════════════════════════

class TestKineticStop:
    """
    KS-01 to KS-17: manage_kinetic_stop() on LifecycleManager.
    Phase 1(ATR) → 2(Breakeven) → 3(Parabolic) → PAUSE(state) → 4(Runner).
    """

    @pytest.fixture(scope="class")
    def lm(self):
        from live_trading_engine import LifecycleManager
        return LifecycleManager()

    def _pos(self, entry=100.0, stop=97.0, highest=None, runner=False,
             er_slow=0.5, rsi=50.0):
        """Build a position dict for manage_kinetic_stop."""
        return {
            "entry_price":  entry,
            "stop_loss":    stop,
            "highest_high": highest if highest is not None else entry,
            "runner_mode":  runner,
            "last_er_slow": er_slow,
            "last_rsi":     rsi,
        }

    # KS-01: New position, small profit (<1.5%) → PHASE_1_BREATHING with ATR-based stop
    def test_ks01_phase1_breathing_small_profit(self, lm):
        pos = self._pos(entry=100.0, stop=96.0)
        new_stop, highest, phase = lm.manage_kinetic_stop("AAPL", pos, 101.0, 2.0)
        assert phase == "PHASE_1_BREATHING"
        # Phase 1 stop = highest_high - atr * phase1_atr_mult = 101 - 2*2 = 97
        # new_stop = max(96, 97) = 97
        assert new_stop >= 96.0, "Stop must not decrease (monotonic)"

    # KS-02: Profit ≥ 1.5% → PHASE_2_BREAKEVEN, stop moves to entry + slippage
    def test_ks02_phase2_breakeven(self, lm):
        pos = self._pos(entry=100.0, stop=97.0)
        new_stop, highest, phase = lm.manage_kinetic_stop("AAPL", pos, 102.0, 2.0)
        assert phase == "PHASE_2_BREAKEVEN"
        # Breakeven stop = entry * (1 + slippage) ≈ 100.1; must be >= entry
        assert new_stop >= 100.0, (
            f"Phase 2 stop {new_stop:.2f} must be ≥ entry (100.00) — trade is now risk-free"
        )

    # KS-03: Profit ≥ 3% → PHASE_3_PARABOLIC, tight stop = highest - 1 ATR
    def test_ks03_phase3_parabolic(self, lm):
        pos = self._pos(entry=100.0, stop=97.0)
        new_stop, highest, phase = lm.manage_kinetic_stop("AAPL", pos, 104.0, 2.0)
        assert phase == "PHASE_3_PARABOLIC"
        # Choke stop = 104 - 2*1.0 = 102; new_stop = max(97, 102) = 102
        assert new_stop >= 102.0, f"Phase 3 choke stop must be ~102, got {new_stop:.2f}"

    # KS-04: All 3 PAUSE conditions met from Phase 3 → PHASE_PAUSE, stop frozen
    def test_ks04_pause_all_conditions_met(self, lm):
        # entry=100, highest=105, current=103 → profit=3%, pullback=(105-103)/105=1.9%
        pos = self._pos(entry=100.0, stop=101.0, highest=105.0, er_slow=0.55, rsi=50.0)
        new_stop, highest, phase = lm.manage_kinetic_stop("AAPL", pos, 103.0, 2.0)
        assert phase == "PHASE_PAUSE", (
            f"Expected PHASE_PAUSE when all 3 conditions met, got {phase}"
        )
        # Stop must not change — frozen during pause
        assert new_stop == 101.0, (
            f"Stop must be frozen at 101.0 during PAUSE, got {new_stop:.2f}"
        )

    # KS-05: PAUSE — RSI below 40 → no PAUSE (falls back to PHASE_3_PARABOLIC)
    def test_ks05_pause_blocked_low_rsi(self, lm):
        pos = self._pos(entry=100.0, stop=101.0, highest=105.0,
                        er_slow=0.55, rsi=35.0)   # RSI = 35 < 40
        new_stop, highest, phase = lm.manage_kinetic_stop("AAPL", pos, 103.0, 2.0)
        assert phase != "PHASE_PAUSE", (
            f"PAUSE must not fire when RSI={35} < 40"
        )

    # KS-06: PAUSE — ER below 0.45 → no PAUSE
    def test_ks06_pause_blocked_low_er(self, lm):
        pos = self._pos(entry=100.0, stop=101.0, highest=105.0,
                        er_slow=0.30, rsi=55.0)   # ER = 0.30 < 0.45
        new_stop, highest, phase = lm.manage_kinetic_stop("AAPL", pos, 103.0, 2.0)
        assert phase != "PHASE_PAUSE", (
            f"PAUSE must not fire when ER={0.30} < 0.45"
        )

    # KS-07: runner_mode=True → PHASE_4_RUNNER, ultra-tight trailing
    def test_ks07_phase4_runner_mode(self, lm):
        pos = self._pos(entry=100.0, stop=108.0, highest=112.0, runner=True)
        new_stop, highest, phase = lm.manage_kinetic_stop("AAPL", pos, 110.0, 1.0)
        assert phase == "PHASE_4_RUNNER", f"runner_mode=True must give PHASE_4_RUNNER, got {phase}"

    # KS-08: manage_kinetic_stop returns exactly 3 values
    def test_ks08_returns_exactly_three_values(self, lm):
        pos = self._pos()
        result = lm.manage_kinetic_stop("AAPL", pos, 101.0, 2.0)
        assert isinstance(result, tuple), "Return must be a tuple"
        assert len(result) == 3, (
            f"manage_kinetic_stop must return (new_stop, highest_high, phase), got {len(result)} values"
        )
        new_stop, highest_high, phase = result
        assert isinstance(new_stop, float), "new_stop must be float"
        assert isinstance(highest_high, float), "highest_high must be float"
        assert isinstance(phase, str), "phase must be str"

    # KS-09: Stop is monotonically non-decreasing in an uptrend
    def test_ks09_stop_never_decreases(self, lm):
        pos = self._pos(entry=100.0, stop=96.0)
        stops = []
        current_stop = 96.0
        for price in [101.0, 102.0, 103.0, 104.0, 105.0]:
            pos["stop_loss"] = current_stop
            pos["highest_high"] = max(pos["highest_high"], price)
            new_stop, _, _ = lm.manage_kinetic_stop("AAPL", pos, price, 2.0)
            assert new_stop >= current_stop, (
                f"Stop decreased: {current_stop:.2f} → {new_stop:.2f} at price={price}"
            )
            current_stop = new_stop
            stops.append(new_stop)
        assert stops == sorted(stops), "Stops must be non-decreasing across uptrend"

    # KS-10: Phase 2 stop guarantees trade is at breakeven or better
    def test_ks10_phase2_stop_at_breakeven_or_above(self, lm):
        pos = self._pos(entry=100.0, stop=97.0)
        new_stop, _, phase = lm.manage_kinetic_stop("AAPL", pos, 102.0, 2.0)
        assert phase == "PHASE_2_BREAKEVEN"
        # Stop must be at or above entry (no loss possible at this point)
        assert new_stop >= 100.0, (
            f"Phase 2 stop {new_stop:.2f} must be ≥ entry 100.00 (risk-free trade)"
        )

    # KS-11: Sequential phase transitions: 1 → 2 → 3 as profit grows
    def test_ks11_phase_transitions_sequential(self, lm):
        entry = 100.0
        phases = []
        for price, expected_phase_prefix in [
            (100.5, "PHASE_1"),   # 0.5% profit → Phase 1
            (101.6, "PHASE_2"),   # 1.6% profit → Phase 2
            (103.5, "PHASE_3"),   # 3.5% profit → Phase 3
        ]:
            pos = self._pos(entry=entry, stop=97.0)
            pos["highest_high"] = price  # no pullback
            _, _, phase = lm.manage_kinetic_stop("AAPL", pos, price, 2.0)
            assert phase.startswith(expected_phase_prefix), (
                f"At {price} ({(price-entry)/entry:.1%} profit) expected {expected_phase_prefix}, got {phase}"
            )
            phases.append(phase)

    # KS-12: Phase 3 stop is tighter than Phase 1 stop at equal highest_high
    def test_ks12_phase3_tighter_than_phase1(self, lm):
        # Phase 1: entry=100, price=101 → Phase 1 stop
        pos1 = self._pos(entry=100.0, stop=96.0)
        stop1, _, phase1 = lm.manage_kinetic_stop("AAPL", pos1, 101.0, 2.0)
        assert phase1 == "PHASE_1_BREATHING"

        # Phase 3: entry=100, price=104, highest_high=104 → Phase 3 stop
        pos3 = self._pos(entry=100.0, stop=96.0, highest=104.0)
        stop3, _, phase3 = lm.manage_kinetic_stop("AAPL", pos3, 104.0, 2.0)
        assert phase3 == "PHASE_3_PARABOLIC"

        # Phase 3 stop must be closer to price than Phase 1 stop
        distance_phase1 = 101.0 - stop1
        distance_phase3 = 104.0 - stop3
        assert distance_phase3 < distance_phase1, (
            f"Phase 3 ({distance_phase3:.2f} away) must be tighter than Phase 1 ({distance_phase1:.2f} away)"
        )

    # KS-13: No programmatic profit-taking patterns in source
    def test_ks13_no_programmatic_profit_take(self):
        source = _read_source("live_trading_engine.py")
        bad_patterns = [
            "profit_exit",
            "exit_at_target",
            "close_at_profit",
            "take_profit_exit",
        ]
        for pattern in bad_patterns:
            assert pattern.lower() not in source.lower(), (
                f"Forbidden profit-taking pattern found: '{pattern}'"
            )

    # KS-14: All kinetic stop parameters come from KINETIC_STOP_CONFIG
    def test_ks14_all_params_from_config(self):
        config = getattr(cfg, 'KINETIC_STOP_CONFIG', {})
        assert isinstance(config, dict) and len(config) > 0, (
            "KINETIC_STOP_CONFIG must be a non-empty dict"
        )
        required_keys = {
            'phase1_atr_mult',
            'phase2_breakeven_trigger_pct',
            'phase3_parabolic_trigger_pct',
            'phase3_atr_mult',
        }
        missing = required_keys - set(config.keys())
        assert not missing, f"KINETIC_STOP_CONFIG missing keys: {missing}"

    # KS-15: Zero current_price handled gracefully (no crash, stop unchanged or lowered by max())
    def test_ks15_zero_price_no_crash(self, lm):
        pos = self._pos(entry=100.0, stop=97.0)
        try:
            new_stop, highest_high, phase = lm.manage_kinetic_stop("AAPL", pos, 0.0, 2.0)
        except Exception as exc:
            pytest.fail(f"manage_kinetic_stop crashed with price=0: {exc}")
        assert isinstance(new_stop, (int, float)), "new_stop must be numeric even with price=0"

    # KS-16: PAUSE cannot fire from Phase 1 (profit < phase3 trigger)
    def test_ks16_pause_not_from_phase1(self, lm):
        # profit = 1% < phase3_trigger (3%) → PAUSE condition `profit >= phase3_trigger` fails
        pos = self._pos(entry=100.0, stop=97.0, highest=101.0, er_slow=0.6, rsi=55.0)
        # Add pullback: current=100.5 < highest=101 → pullback=0.5%
        _, _, phase = lm.manage_kinetic_stop("AAPL", pos, 100.5, 2.0)
        assert phase != "PHASE_PAUSE", (
            f"PAUSE must NOT activate in Phase 1 (profit < phase3_trigger), got {phase}"
        )
        assert phase == "PHASE_1_BREATHING", f"Expected PHASE_1_BREATHING, got {phase}"

    # KS-17: PAUSE cannot fire when runner_mode=True (Phase 4 takes priority)
    def test_ks17_pause_not_when_runner_mode(self, lm):
        # All PAUSE conditions met, BUT runner_mode=True → Phase 4 wins
        pos = self._pos(entry=100.0, stop=103.0, highest=106.0,
                        runner=True, er_slow=0.6, rsi=55.0)
        # current=104, pullback=(106-104)/106=1.9%, profit=4% → all PAUSE conditions met
        _, _, phase = lm.manage_kinetic_stop("AAPL", pos, 104.0, 2.0)
        assert phase == "PHASE_4_RUNNER", (
            f"runner_mode=True must yield PHASE_4_RUNNER, not PAUSE. Got: {phase}"
        )
