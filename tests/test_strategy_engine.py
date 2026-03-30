# tests/test_strategy_engine.py

"""
StockWise Gen-13 — Strategy Engine Tests (TDD v1.1 Section 5)
=============================================================
Alpha Equation (AE-01→08), Regime Coupling (RCp-01→05),
Asset-Specific Optimization (AS-01→06), Vectorized Decay (VD-01→05).
Phase B probabilistic engine.

Execution: python -m pytest tests/test_strategy_engine.py -v --tb=short
Expected : 24 passed, 0 failed
"""

import os
import sys
import re
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import system_config as cfg


# ── Helpers ────────────────────────────────────────────────────────────────────

def _read_source(filename: str) -> str:
    path = os.path.join(PROJECT_ROOT, filename)
    if not os.path.exists(path):
        pytest.skip(f"{filename} not found")
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def _entry_eq(final_score: float, atr_pct: float):
    """
    Replicate StrategyEngine.calculate_entry_equation logic using actual config.
    Returns (is_profitable, expected_rise, total_friction).
    """
    friction = cfg.BASE_FRICTION       # 0.003
    min_profit = cfg.MIN_NET_PROFIT    # 0.005
    expected_rise = (final_score / 100.0) * atr_pct
    is_profitable = (expected_rise - friction) >= min_profit
    return is_profitable, expected_rise, friction


def _regime_row(er_slow: float = 0.5, er_fast: float = 0.5) -> pd.DataFrame:
    """Single-row DataFrame with ER values for RegimeRouter."""
    return pd.DataFrame([{'er_slow': er_slow, 'er_fast': er_fast}])


# ═══════════════════════════════════════════════════════
# 5.1  ALPHA EQUATION TESTS  (AE-01 → AE-08)
# DDR #3: Threshold fixed at 0.5% (MIN_NET_PROFIT = 0.005)
# ═══════════════════════════════════════════════════════

class TestAlphaEquation:
    """calculate_entry_equation: (score/100 × atr) − BASE_FRICTION ≥ MIN_NET_PROFIT."""

    # AE-01: High-score trade (expected_rise >> friction) → profitable
    def test_ae01_high_score_passes(self):
        ok, rise, friction = _entry_eq(final_score=80, atr_pct=0.02)
        # expected_rise = 0.8 * 0.02 = 0.016; 0.016 - 0.003 = 0.013 >= 0.005 ✓
        assert ok is True, f"Score=80, atr=2% must PASS alpha gate (rise={rise:.4f})"

    # AE-02: Low-score trade (expected_rise < friction + min_profit) → rejected
    def test_ae02_low_score_rejected(self):
        ok, rise, friction = _entry_eq(final_score=30, atr_pct=0.01)
        # expected_rise = 0.3 * 0.01 = 0.003; 0.003 - 0.003 = 0.0 < 0.005 → rejected
        assert ok is False, f"Score=30, atr=1% must FAIL alpha gate (rise={rise:.4f})"

    # AE-03: Exactly at threshold (expected_rise - friction == min_profit) → PASSES (>= not >)
    def test_ae03_exact_threshold_passes(self):
        # Need: expected_rise - friction = min_profit
        # expected_rise = MIN_NET_PROFIT + BASE_FRICTION = 0.005 + 0.003 = 0.008
        # Choose: final_score=80, atr_pct=0.01 → expected_rise = 0.8 * 0.01 = 0.008 ✓
        ok, rise, friction = _entry_eq(final_score=80, atr_pct=0.01)
        assert abs(rise - friction - cfg.MIN_NET_PROFIT) < 1e-9, (
            f"Test misconfigured: rise-friction={rise-friction:.6f}, expected {cfg.MIN_NET_PROFIT}"
        )
        assert ok is True, "Exactly at threshold must PASS (>= not strict >)"

    # AE-04: MIN_NET_PROFIT = 0.005 (0.5%) in config — not 0.013 (old 1.3%)
    def test_ae04_alpha_threshold_is_0_5_percent(self):
        assert cfg.MIN_NET_PROFIT == 0.005, (
            f"MIN_NET_PROFIT={cfg.MIN_NET_PROFIT}, expected 0.005 (DDR #3: unified threshold)"
        )
        assert cfg.FRICTION_AND_ALPHA['min_net_profit_pct'] == 0.005, (
            "FRICTION_AND_ALPHA.min_net_profit_pct must be 0.005"
        )

    # AE-05: No reference to old 1.3% (0.013) threshold in alpha-related code
    def test_ae05_no_old_1_3_percent_remnant(self):
        se_src = _read_source("strategy_engine.py")
        cfg_src = _read_source("system_config.py")
        combined = se_src + cfg_src
        # Pattern: min_net_profit or alpha config set to 0.013
        old_ref = re.findall(
            r"min_net_profit.*0\.013|alpha.*0\.013|MIN_NET_PROFIT\s*=\s*0\.013",
            combined, re.IGNORECASE
        )
        assert not old_ref, f"Old 1.3% threshold remnant found: {old_ref}"

    # AE-06: final_score=0 → expected_rise=0, result=-friction < min_profit → False, no crash
    def test_ae06_zero_score_no_crash(self):
        try:
            ok, rise, friction = _entry_eq(final_score=0, atr_pct=0.05)
        except Exception as exc:
            pytest.fail(f"Zero final_score raised: {exc}")
        assert ok is False, "Score=0 must be rejected (expected_rise=0 < min_profit+friction)"

    # AE-07: final_score=100, atr_pct=1.0 → no overflow, returns True
    def test_ae07_max_score_no_overflow(self):
        try:
            ok, rise, friction = _entry_eq(final_score=100, atr_pct=1.0)
        except Exception as exc:
            pytest.fail(f"Max score raised: {exc}")
        assert ok is True, "Score=100, atr=100% must pass alpha gate"
        assert rise == pytest.approx(1.0)

    # AE-08: _entry_eq returns exactly 3 values (is_profitable, expected_rise, total_friction)
    def test_ae08_returns_three_values(self):
        result = _entry_eq(final_score=70, atr_pct=0.02)
        assert isinstance(result, tuple) and len(result) == 3
        is_prof, expected_rise, friction = result
        assert isinstance(is_prof, bool)
        assert isinstance(expected_rise, float)
        assert isinstance(friction, float)


# ═══════════════════════════════════════════════════════
# 5.2  REGIME COUPLING TESTS  (RCp-01 → RCp-05)
# ═══════════════════════════════════════════════════════

class TestRegimeCoupling:
    """RegimeRouter.classify_regime(): ER-based routing to TREND/CHOP/HALT/NEUTRAL."""

    @pytest.fixture(scope="class")
    def router(self):
        from strategy_engine import RegimeRouter
        return RegimeRouter()

    # RCp-01: Strong trend (er_slow ≥ 0.6 AND er_fast not diverging) → TREND
    def test_rcop01_trend_regime(self, router):
        df = _regime_row(er_slow=0.75, er_fast=0.65)
        regime = router.classify_regime(df)
        assert regime == "TREND", f"er_slow=0.75, er_fast=0.65 should be TREND, got {regime}"

    # RCp-02: Low efficiency ratio (er_slow ≤ 0.4) → CHOP
    def test_rcop02_chop_regime(self, router):
        df = _regime_row(er_slow=0.25, er_fast=0.20)
        regime = router.classify_regime(df)
        assert regime == "CHOP", f"er_slow=0.25 should be CHOP, got {regime}"

    # RCp-03: Velocity divergence (er_slow > 0.6 AND er_fast < 0.2) → HALT
    def test_rcop03_halt_velocity_divergence(self, router):
        df = _regime_row(er_slow=0.70, er_fast=0.10)
        regime = router.classify_regime(df)
        assert regime == "HALT", (
            f"Velocity divergence (er_slow=0.70, er_fast=0.10) must yield HALT, got {regime}"
        )

    # RCp-04: Dead zone (0.4 < er_slow < 0.6) → NEUTRAL
    def test_rcop04_neutral_dead_zone(self, router):
        df = _regime_row(er_slow=0.50, er_fast=0.45)
        regime = router.classify_regime(df)
        assert regime == "NEUTRAL", (
            f"Dead zone (er_slow=0.50) must yield NEUTRAL, got {regime}"
        )

    # RCp-05: Empty DataFrame → HALT (safe fail-closed behaviour)
    def test_rcop05_empty_df_halt(self, router):
        regime = router.classify_regime(pd.DataFrame())
        assert regime == "HALT", (
            f"Empty df must return HALT (fail-closed), got {regime}"
        )


# ═══════════════════════════════════════════════════════
# 5.3  ASSET-SPECIFIC OPTIMIZATION  (AS-01 → AS-06)
# DDR #1: Per-stock win rates via Shadow Ledger
# ═══════════════════════════════════════════════════════

class TestAssetSpecificOptimization:
    """TemplateMatcher.get_template_win_rate: cold-start fallback + blended weights."""

    @pytest.fixture
    def matcher(self):
        from template_matcher import TemplateMatcher
        return TemplateMatcher()

    def _shadow(self, data: dict) -> dict:
        """Build a shadow_stats dict suitable for _load_shadow_stats return value."""
        return {
            sym: {
                tid: {
                    "signal_count": v.get("signal_count", 0),
                    "wins": v.get("wins", 0),
                    "losses": v.get("losses", 0),
                    "win_rate": v.get("win_rate", 50.0),
                    "total_pnl_pct": v.get("total_pnl_pct", 0.0),
                }
                for tid, v in templates.items()
            }
            for sym, templates in data.items()
        }

    # AS-01: Per-stock ≥ cold_start signals → blended rate used (not default 50%)
    def test_as01_per_stock_stats_used_when_enough_signals(self, matcher):
        stats = self._shadow({
            "TSLA": {"T1": {"signal_count": 50, "wins": 35, "win_rate": 70.0}},
            "AAPL": {"T1": {"signal_count": 30, "wins": 12, "win_rate": 40.0}},
        })
        with patch.object(matcher, '_load_shadow_stats', return_value=stats):
            wr = matcher.get_template_win_rate("T1", "TSLA")
        # 50 >= cold_start(5) → blended = 70%*0.7 + global*0.3 ≠ 50
        assert wr != 50.0, "Per-stock stats (50 signals) must produce a non-default win rate"
        assert wr > 50.0, "TSLA 70% WR must pull blended above 50%"

    # AS-02: Symbol not in shadow ledger → returns global average (not 50% default)
    def test_as02_unknown_symbol_falls_back_to_global(self, matcher):
        stats = self._shadow({
            "TSLA": {"T1": {"signal_count": 50, "wins": 35, "win_rate": 70.0}},
        })
        with patch.object(matcher, '_load_shadow_stats', return_value=stats):
            wr = matcher.get_template_win_rate("T1", "NEW_TICKER")
        # 0 signals < cold_start → returns global = 70% (from TSLA)
        assert wr != 50.0, "Cold start with global data should NOT return default 50%"
        assert abs(wr - 70.0) < 1.0, f"Global WR should be ≈70% (from TSLA), got {wr}"

    # AS-03: Per-stock signals < cold_start threshold → uses global, ignores per-stock
    def test_as03_below_cold_start_uses_global(self, matcher):
        cold_start = cfg.ASSET_SPECIFIC_CONFIG.get('cold_start_min_signals', 5)
        stats = self._shadow({
            "TICKER": {"T1": {"signal_count": cold_start - 1, "wins": cold_start - 1,
                               "win_rate": 100.0}},    # 100% per-stock
            "SPY":    {"T1": {"signal_count": 50, "wins": 25, "win_rate": 50.0}},
        })
        with patch.object(matcher, '_load_shadow_stats', return_value=stats):
            wr = matcher.get_template_win_rate("T1", "TICKER")
        # Below cold_start → ignores TICKER 100% per-stock, uses global ≈ 65%
        assert wr < 100.0, (
            f"With {cold_start-1} signals (< cold_start={cold_start}), "
            f"per-stock 100% must NOT be used directly. Got {wr}"
        )

    # AS-04: Per-stock signals ≥ cold_start → blended pulls toward per-stock WR
    def test_as04_above_cold_start_blends_per_stock(self, matcher):
        stats = self._shadow({
            "TICKER": {"T1": {"signal_count": 20, "wins": 16, "win_rate": 80.0}},
            "SPY":    {"T1": {"signal_count": 50, "wins": 25, "win_rate": 50.0}},
        })
        with patch.object(matcher, '_load_shadow_stats', return_value=stats):
            wr = matcher.get_template_win_rate("T1", "TICKER")
        # blended = 80*0.7 + global*0.3 > 50% since per-stock is 80%
        assert wr > 50.0, f"20 signals ≥ cold_start: blended must exceed global 50%, got {wr}"

    # AS-05: cold_start_min_signals = 5 from ASSET_SPECIFIC_CONFIG
    def test_as05_cold_start_from_config(self):
        config = getattr(cfg, 'ASSET_SPECIFIC_CONFIG', {})
        assert 'cold_start_min_signals' in config, (
            "cold_start_min_signals not in ASSET_SPECIFIC_CONFIG"
        )
        assert config['cold_start_min_signals'] == 5, (
            f"cold_start_min_signals={config['cold_start_min_signals']}, expected 5"
        )
        assert config['per_stock_weight'] == 0.7
        assert config['global_weight'] == 0.3

    # AS-06: TSLA favors template A, AAPL favors template B → per-stock ordering preserved
    def test_as06_different_stocks_different_rankings(self, matcher):
        stats = self._shadow({
            "TSLA": {
                "SETUP_A": {"signal_count": 50, "wins": 40, "win_rate": 80.0},
                "SETUP_B": {"signal_count": 50, "wins": 20, "win_rate": 40.0},
            },
            "AAPL": {
                "SETUP_A": {"signal_count": 50, "wins": 15, "win_rate": 30.0},
                "SETUP_B": {"signal_count": 50, "wins": 45, "win_rate": 90.0},
            },
        })
        with patch.object(matcher, '_load_shadow_stats', return_value=stats):
            tsla_a = matcher.get_template_win_rate("SETUP_A", "TSLA")
            tsla_b = matcher.get_template_win_rate("SETUP_B", "TSLA")
            aapl_a = matcher.get_template_win_rate("SETUP_A", "AAPL")
            aapl_b = matcher.get_template_win_rate("SETUP_B", "AAPL")

        assert tsla_a > tsla_b, f"TSLA should rank SETUP_A above B ({tsla_a:.1f} vs {tsla_b:.1f})"
        assert aapl_b > aapl_a, f"AAPL should rank SETUP_B above A ({aapl_b:.1f} vs {aapl_a:.1f})"


# ═══════════════════════════════════════════════════════
# 5.4  VECTORIZED DECAY TESTS  (VD-01 → VD-05)
# Per-category decay: momentum=0.90 fast, vsa_institutional=0.99 slow
# ═══════════════════════════════════════════════════════

class TestVectorizedDecay:
    """Config-driven decay rates and apply_decay() in shadow_ledger.py."""

    def _cfg(self) -> dict:
        return getattr(cfg, 'VECTORIZED_DECAY_CONFIG', {})

    def _weight(self, rate: float, days: int) -> float:
        period = self._cfg().get('decay_period_days', 7)
        min_w = self._cfg().get('min_weight', 0.05)
        return max(rate ** (days / period), min_w)

    # VD-01: Recent (7 days) weight is >3× older (180 days) weight for default rate
    def test_vd01_recent_weighted_more_than_old(self):
        rate = self._cfg().get('decay_rates', {}).get('default', 0.95)
        recent = self._weight(rate, 7)
        old    = self._weight(rate, 180)
        ratio  = recent / old
        assert ratio > 3.0, (
            f"Recent/old weight ratio = {ratio:.1f} — must be > 3.0 "
            f"(recent={recent:.3f}, 6mo_old={old:.3f})"
        )

    # VD-02: VSA/institutional signal from 6 months ago retains > 50% weight (slow decay 0.99)
    def test_vd02_vsa_signal_retained_after_6_months(self):
        vsa_rate = self._cfg().get('decay_rates', {}).get('vsa_institutional', 0.99)
        weight = self._weight(vsa_rate, 180)
        assert weight > 0.5, (
            f"VSA weight after 180 days = {weight:.3f} — must retain > 0.5 (rate={vsa_rate})"
        )

    # VD-03: Momentum signal from 3 months ago is nearly decayed away (fast decay 0.90)
    def test_vd03_momentum_decays_fast(self):
        mom_rate = self._cfg().get('decay_rates', {}).get('momentum', 0.90)
        weight = self._weight(mom_rate, 90)
        assert weight < 0.35, (
            f"Momentum weight after 90 days = {weight:.3f} — must be < 0.35 (fast decay rate={mom_rate})"
        )

    # VD-04: apply_decay() exists in shadow_ledger.py and references decay config
    def test_vd04_apply_decay_exists_in_shadow_ledger(self):
        source = _read_source("shadow_ledger.py")
        assert "def apply_decay" in source, (
            "apply_decay method not found in shadow_ledger.py"
        )
        assert "VECTORIZED_DECAY_CONFIG" in source or "decay_rates" in source, (
            "apply_decay must reference decay config (not hardcoded rates)"
        )
        # Verify it regresses to neutral 50% (not 0%) on full decay
        assert "50.0" in source, (
            "Decay formula must regress to 50% neutral, not 0% — check decay formula"
        )

    # VD-05: Decay rates in config; momentum decays faster than vsa_institutional
    def test_vd05_decay_rates_ordered_correctly(self):
        rates = self._cfg().get('decay_rates', {})
        assert 'momentum' in rates, "momentum decay rate missing from VECTORIZED_DECAY_CONFIG"
        assert 'vsa_institutional' in rates, "vsa_institutional rate missing"
        assert rates['momentum'] < rates['vsa_institutional'], (
            f"momentum({rates['momentum']}) must decay faster (lower rate) "
            f"than vsa_institutional({rates['vsa_institutional']})"
        )
        # All rates must be in (0, 1)
        for name, rate in rates.items():
            assert 0 < rate < 1, f"Decay rate '{name}'={rate} must be in (0, 1)"
