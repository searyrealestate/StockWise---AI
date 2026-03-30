# tests/test_portfolio_risk.py

"""
StockWise Gen-13 — Portfolio Risk Tests (TDD v1.1 Section 8)
============================================================
Gate 1: Correlation & Sector (G1-01→07)
Gate 2: Drawdown & Exposure (G2-01→07)
Gate 3: Weekly Trend (G3-01→05)
Combined Gates (GC-01→06)
Called AFTER signal detection, BEFORE execution. MONEY PATH.

Execution: python -m pytest tests/test_portfolio_risk.py -v --tb=short
Expected : 25 passed, 0 failed
"""

import os
import sys
import re
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from portfolio_risk import PortfolioRiskManager
import system_config as cfg


# ── Helpers ────────────────────────────────────────────────────────────────────

def _prm() -> PortfolioRiskManager:
    """Fresh PortfolioRiskManager per test (stateful — must not share)."""
    return PortfolioRiskManager()


def _get_config() -> dict:
    return getattr(cfg, 'PORTFOLIO_RISK_CONFIG', {})


def _read_source(filename: str) -> str:
    path = os.path.join(PROJECT_ROOT, filename)
    if not os.path.exists(path):
        pytest.skip(f"{filename} not found")
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def _daily_df(rows: int = 300, trend: str = 'bullish') -> pd.DataFrame:
    """Build a DatetimeIndex daily OHLCV DataFrame for weekly trend tests."""
    idx = pd.date_range('2020-01-01', periods=rows, freq='B')
    if trend == 'bullish':
        close = np.linspace(80.0, 200.0, rows)
    elif trend == 'bearish':
        close = np.linspace(200.0, 80.0, rows)
    else:  # flat / constant
        close = np.full(rows, 100.0)
    return pd.DataFrame({
        'open':   close * 0.999,
        'high':   close * 1.005,
        'low':    close * 0.995,
        'close':  close,
        'volume': np.full(rows, 1_000_000.0),
    }, index=idx)


def _mock_md_with_corr(symbol_a: str, symbol_b: str, corr_target: float) -> MagicMock:
    """
    Build a mock DataSourceManager whose get_stock_data() returns price series
    with a known correlation between symbol_a and symbol_b.

    corr_target ≈ 1.0  → use near-identical monotone series (high correlation)
    corr_target ≈ 0.0  → use orthogonal random series (low correlation)
    """
    n = 65
    idx = pd.date_range('2025-01-01', periods=n, freq='B')

    if corr_target > 0.8:
        # Both monotonically increasing → corr ≈ 1.0
        close_a = pd.Series(np.linspace(100.0, 160.0, n), index=idx)
        close_b = pd.Series(np.linspace(200.0, 320.0, n), index=idx)
    else:
        # Uncorrelated: one up, one random/flat
        rng = np.random.default_rng(42)
        close_a = pd.Series(np.linspace(100.0, 160.0, n), index=idx)
        close_b = pd.Series(100.0 + np.cumsum(rng.normal(0, 0.5, n)), index=idx)

    def _get(sym, **kwargs):
        if sym == symbol_a:
            return pd.DataFrame({'close': close_a}, index=idx)
        return pd.DataFrame({'close': close_b}, index=idx)

    md = MagicMock()
    md.get_stock_data.side_effect = _get
    return md


def _positions_with_exposure(exposure_pct: float,
                              portfolio_value: float = 100_000.0) -> dict:
    """Return open_positions dict where total invested == exposure_pct * portfolio_value."""
    invested = exposure_pct * portfolio_value
    return {'AAPL': {'entry_price': 100.0, 'qty': int(invested / 100.0)}}


# ═══════════════════════════════════════════════════════
# 8.1  GATE 1: CORRELATION & SECTOR  (G1-01 → G1-07)
# ═══════════════════════════════════════════════════════

class TestGate1CorrelationSector:
    """check_correlation_gate: sector concentration + price correlation."""

    # G1-01: 2 tech positions + new tech signal → BLOCKED (sector count ≥ max_sector)
    def test_g101_same_sector_limit_blocks(self):
        prm = _prm()
        # AAPL and MSFT are both Technology
        positions = {'AAPL': {}, 'MSFT': {}}
        # GOOGL is also Technology → sector_count = 2 ≥ max_sector(2) → BLOCKED
        ok, reason = prm.check_correlation_gate('GOOGL', positions, market_data=None)
        assert not ok, "2 existing tech + new tech must be BLOCKED"
        assert 'Technology' in reason or 'sector' in reason.lower()

    # G1-02: 1 tech + new healthcare → ALLOWED (different sector)
    def test_g102_different_sector_allowed(self):
        prm = _prm()
        positions = {'AAPL': {}}   # 1 tech
        # JNJ is Healthcare → only 0 existing healthcare < max_sector(2) → ALLOWED
        ok, _ = prm.check_correlation_gate('JNJ', positions, market_data=None)
        assert ok, "Different sector (Healthcare) must be ALLOWED"

    # G1-03: High correlation (≈1.0) → BLOCKED (> 0.80)
    def test_g103_high_correlation_blocks(self):
        prm = _prm()
        positions = {'AAPL': {}}
        md = _mock_md_with_corr('NVDA', 'AAPL', corr_target=1.0)
        ok, reason = prm.check_correlation_gate('NVDA', positions, market_data=md)
        assert not ok, "High correlation must be BLOCKED"
        assert 'correlation' in reason.lower() or 'corr' in reason.lower()

    # G1-04: Low correlation (≈0.0) → ALLOWED
    def test_g104_low_correlation_allowed(self):
        prm = _prm()
        positions = {'AAPL': {}}
        md = _mock_md_with_corr('XOM', 'AAPL', corr_target=0.0)
        ok, _ = prm.check_correlation_gate('XOM', positions, market_data=md)
        assert ok, "Low correlation must be ALLOWED"

    # G1-05: Boundary — max_correlation=0.80; condition is corr > 0.80, so ≤0.80 passes
    def test_g105_boundary_max_correlation_is_strict(self):
        cfg_corr = _get_config().get('max_correlation', 0.80)
        assert cfg_corr == 0.80, f"max_correlation should be 0.80, got {cfg_corr}"
        # Source inspection: correlation check uses > not >=
        source = _read_source("portfolio_risk.py")
        assert 'corr > max_corr' in source, (
            "Correlation gate must use strict '>' so exactly 0.80 is allowed"
        )

    # G1-06: Required config keys exist
    def test_g106_config_keys_present(self):
        config = _get_config()
        assert config.get('max_sector_positions') is not None, (
            "PORTFOLIO_RISK_CONFIG must have max_sector_positions"
        )
        assert config.get('max_correlation') is not None, (
            "PORTFOLIO_RISK_CONFIG must have max_correlation"
        )
        assert config.get('correlation_lookback_days') is not None, (
            "PORTFOLIO_RISK_CONFIG must have correlation_lookback_days"
        )

    # G1-07: Unknown symbol (not in SECTOR_MAP) → no sector block, gate passes
    def test_g107_unknown_sector_no_crash(self):
        prm = _prm()
        positions = {'AAPL': {}, 'MSFT': {}}  # 2 tech
        # 'UNKN' not in SECTOR_MAP → sector='Unknown' → no sector count → gate passes
        ok, reason = prm.check_correlation_gate('UNKN', positions, market_data=None)
        assert ok, (
            f"Unknown-sector symbol must NOT be sector-blocked (no crash): reason='{reason}'"
        )


# ═══════════════════════════════════════════════════════
# 8.2  GATE 2: DRAWDOWN & EXPOSURE  (G2-01 → G2-07)
# ═══════════════════════════════════════════════════════

class TestGate2DrawdownExposure:
    """check_drawdown_gate: circuit breaker + exposure ceiling."""

    # G2-01: 12% drawdown (≥ 10% threshold) → BLOCKED, circuit breaker fires
    def test_g201_drawdown_above_10pct_blocks(self):
        prm = _prm()
        prm.portfolio_high_water_mark = 100_000.0   # simulate previous high
        ok, reason = prm.check_drawdown_gate({}, portfolio_value=88_000.0)  # -12%
        assert not ok, "12% drawdown must BLOCK (circuit breaker)"
        assert 'CIRCUIT BREAKER' in reason.upper() or 'drawdown' in reason.lower()

    # G2-02: 8% drawdown (< 10%) → ALLOWED
    def test_g202_drawdown_below_10pct_allowed(self):
        prm = _prm()
        prm.portfolio_high_water_mark = 100_000.0
        ok, _ = prm.check_drawdown_gate({}, portfolio_value=92_000.0)  # -8%
        assert ok, "8% drawdown must be ALLOWED"

    # G2-03: Total exposure 62% (≥ 60% ceiling) → BLOCKED
    def test_g203_exposure_above_60pct_blocks(self):
        prm = _prm()
        positions = _positions_with_exposure(0.62, portfolio_value=100_000.0)
        ok, reason = prm.check_drawdown_gate(positions, portfolio_value=100_000.0)
        assert not ok, "62% exposure must BLOCK"
        assert '62' in reason or 'exposure' in reason.lower() or 'Exposure' in reason

    # G2-04: Total exposure 55% (< 60%) → ALLOWED
    def test_g204_exposure_below_60pct_allowed(self):
        prm = _prm()
        positions = _positions_with_exposure(0.55, portfolio_value=100_000.0)
        ok, _ = prm.check_drawdown_gate(positions, portfolio_value=100_000.0)
        assert ok, "55% exposure must be ALLOWED"

    # G2-05: Circuit breaker fires → subsequent calls still blocked (24h cooldown)
    def test_g205_circuit_breaker_persists(self):
        prm = _prm()
        prm.portfolio_high_water_mark = 100_000.0

        # First call — triggers circuit breaker (12% down)
        ok1, _ = prm.check_drawdown_gate({}, portfolio_value=88_000.0)
        assert not ok1
        assert prm.circuit_breaker_active is True

        # Immediate second call — circuit breaker still hot (elapsed < 24h)
        ok2, reason2 = prm.check_drawdown_gate({}, portfolio_value=95_000.0)
        assert not ok2, "Circuit breaker must still block within 24h window"
        assert 'circuit breaker' in reason2.lower() or 'cooldown' in reason2.lower()

    # G2-06: Portfolio value = 0 → BLOCKED, no ZeroDivisionError
    def test_g206_zero_portfolio_value_blocks_gracefully(self):
        prm = _prm()
        try:
            ok, reason = prm.check_drawdown_gate({}, portfolio_value=0)
        except ZeroDivisionError:
            pytest.fail("check_drawdown_gate raised ZeroDivisionError with portfolio_value=0")
        assert not ok, "Zero portfolio value must BLOCK new entries"

    # G2-07: Config has all required drawdown/exposure keys
    def test_g207_config_keys_present(self):
        config = _get_config()
        required = {
            'max_portfolio_drawdown_pct',
            'max_total_exposure_pct',
            'drawdown_cooldown_hours',
        }
        missing = required - set(config.keys())
        assert not missing, f"PORTFOLIO_RISK_CONFIG missing keys: {missing}"
        assert config['max_portfolio_drawdown_pct'] == 0.10, "drawdown threshold must be 10%"
        assert config['max_total_exposure_pct'] == 0.60, "exposure ceiling must be 60%"


# ═══════════════════════════════════════════════════════
# 8.3  GATE 3: WEEKLY TREND  (G3-01 → G3-05)
# ═══════════════════════════════════════════════════════

class TestGate3WeeklyTrend:
    """check_weekly_trend_gate: weekly resample vs 40-week SMA."""

    # G3-01: Declining price series → weekly close < SMA_40 → BLOCKED
    def test_g301_bearish_weekly_trend_blocks(self):
        prm = _prm()
        df = _daily_df(rows=320, trend='bearish')   # 200→80 decline over 320 days
        ok, reason = prm.check_weekly_trend_gate('AAPL', df)
        assert not ok, "Bearish weekly trend (close < SMA_40) must BLOCK"
        assert 'BEARISH' in reason.upper() or 'weekly' in reason.lower()

    # G3-02: Rising price series → weekly close > SMA_40 → ALLOWED
    def test_g302_bullish_weekly_trend_allowed(self):
        prm = _prm()
        df = _daily_df(rows=320, trend='bullish')   # 80→200 rise over 320 days
        ok, _ = prm.check_weekly_trend_gate('AAPL', df)
        assert ok, "Bullish weekly trend (close > SMA_40) must be ALLOWED"

    # G3-03: Constant price series → weekly close == SMA_40 → ALLOWED (strict < not <=)
    def test_g303_equal_sma_is_allowed(self):
        prm = _prm()
        df = _daily_df(rows=320, trend='flat')   # constant 100.0 → SMA_40 = 100.0 = close
        ok, reason = prm.check_weekly_trend_gate('AAPL', df)
        # weekly_close (100) < weekly_sma_val (100) → False → ALLOWED
        assert ok, (
            f"Weekly close == SMA_40 must PASS (strict < check), got reason='{reason}'"
        )

    # G3-04: Config has weekly_sma_period key set to 40
    def test_g304_config_has_weekly_sma_period(self):
        config = _get_config()
        assert 'weekly_sma_period' in config, (
            "PORTFOLIO_RISK_CONFIG must have weekly_sma_period"
        )
        assert config['weekly_sma_period'] == 40, (
            f"weekly_sma_period must be 40 (≈ SMA_200 daily), got {config['weekly_sma_period']}"
        )
        assert config.get('weekly_trend_enabled', True) is True

    # G3-05: Fewer than 50 daily rows → ALLOWED (benefit of the doubt)
    def test_g305_insufficient_data_allows(self):
        prm = _prm()
        df = _daily_df(rows=30, trend='bearish')   # < 50 rows → gate short-circuits
        ok, reason = prm.check_weekly_trend_gate('AAPL', df)
        assert ok, (
            f"Insufficient data (< 50 daily rows) must ALLOW, not block. Got: '{reason}'"
        )


# ═══════════════════════════════════════════════════════
# 8.4  COMBINED GATES TESTS  (GC-01 → GC-06)
# ═══════════════════════════════════════════════════════

class TestCombinedGates:
    """check_all_gates: all 3 gates together via the combined entry point."""

    def _bullish_df(self) -> pd.DataFrame:
        return _daily_df(rows=320, trend='bullish')

    def _bearish_df(self) -> pd.DataFrame:
        return _daily_df(rows=320, trend='bearish')

    # GC-01: All 3 gates pass → approved=True, reasons=[]
    def test_gc01_all_gates_pass(self):
        prm = _prm()
        # Gate 1: empty positions → no sector/corr check
        # Gate 2: good portfolio, no drawdown, no exposure
        # Gate 3: bullish weekly
        approved, reasons = prm.check_all_gates(
            symbol='AAPL',
            df=self._bullish_df(),
            open_positions={},
            market_data=None,
            portfolio_value=100_000.0,
        )
        assert approved is True, f"All gates should pass, got reasons: {reasons}"
        assert reasons == [], f"No veto reasons expected, got: {reasons}"

    # GC-02: Gate 2 blocks (12% drawdown) → approved=False, one reason
    def test_gc02_gate2_blocks_execution(self):
        prm = _prm()
        prm.portfolio_high_water_mark = 100_000.0
        approved, reasons = prm.check_all_gates(
            symbol='AAPL',
            df=self._bullish_df(),
            open_positions={},
            market_data=None,
            portfolio_value=88_000.0,   # -12% → circuit breaker
        )
        assert not approved, "Gate 2 drawdown must block execution"
        assert len(reasons) >= 1, "Must have at least one reason"
        assert any('CIRCUIT' in r.upper() or 'drawdown' in r.lower() for r in reasons)

    # GC-03: Gate 1 AND Gate 3 both fail → reasons list has 2 entries
    def test_gc03_multiple_gate_failures_all_reported(self):
        prm = _prm()
        # Gate 1 fail: 2 tech positions + new tech signal
        positions = {'AAPL': {}, 'MSFT': {}}
        # Gate 3 fail: bearish weekly trend
        approved, reasons = prm.check_all_gates(
            symbol='GOOGL',          # Technology → sector blocked
            df=self._bearish_df(),   # bearish weekly → Gate 3 blocked
            open_positions=positions,
            market_data=None,
            portfolio_value=100_000.0,
        )
        assert not approved, "Multiple gate failures must block"
        assert len(reasons) >= 2, (
            f"Both Gate 1 and Gate 3 failures must be reported; got {len(reasons)}: {reasons}"
        )

    # GC-04: Veto triggers logger.warning with RISK VETO message
    def test_gc04_veto_logs_warning(self, caplog):
        import logging
        prm = _prm()
        prm.portfolio_high_water_mark = 100_000.0
        with caplog.at_level(logging.WARNING):
            prm.check_all_gates(
                symbol='AAPL',
                df=self._bullish_df(),
                open_positions={},
                market_data=None,
                portfolio_value=88_000.0,  # triggers circuit breaker
            )
        assert any(
            'veto' in rec.message.lower() or 'RISK' in rec.message
            for rec in caplog.records
        ), f"Expected RISK VETO warning, log records: {[r.message for r in caplog.records]}"

    # GC-05: Circuit breaker fires → next fresh call for same portfolio still blocked
    def test_gc05_circuit_breaker_blocks_subsequent_entries(self):
        prm = _prm()
        prm.portfolio_high_water_mark = 100_000.0

        # Trigger circuit breaker
        prm.check_all_gates('AAPL', self._bullish_df(), {}, None, 88_000.0)
        assert prm.circuit_breaker_active

        # New signal, different symbol — still blocked
        approved, reasons = prm.check_all_gates(
            'MSFT', self._bullish_df(), {}, None, 90_000.0
        )
        assert not approved, "Circuit breaker must block all new entries until 24h elapses"
        assert any('circuit breaker' in r.lower() for r in reasons)

    # GC-06: Source inspection — check_all_gates called in live_trading_engine.py
    def test_gc06_risk_check_wired_into_execution_pipeline(self):
        source = _read_source("live_trading_engine.py")
        assert 'check_all_gates' in source, (
            "check_all_gates must be called in live_trading_engine.py execution pipeline"
        )
        assert 'PortfolioRiskManager' in source, (
            "PortfolioRiskManager must be imported/used in live_trading_engine.py"
        )
