"""
StockWise AI - Unit Tests for Phase 1 Bug Fixes
=================================================
Each test class validates one specific bug fix in isolation.
Run: python tests/unit_tests.py  OR  pytest tests/unit_tests.py -v

IMPORT STRATEGY:
Since feature_engine.py imports pandas_ta (not pip-installable in this environment;
only pandas-ta-classic is available under a different import name), we stub it at
sys.modules level BEFORE importing strategy_engine.
TacticalSniper.__init__ loads .pkl models from disk — they won't exist in the test
environment, but _load_model() already returns None gracefully, so analyze() still
works (AI score defaults to 50.0 when model is None).
"""
import sys
import os
import types
from unittest.mock import MagicMock

# === DEPENDENCY STUBBING (must happen before any project imports) ===
# Stub pandas_ta regardless of whether it's installed — tests should not
# depend on a 200MB TA library for unit-testing column name logic.
_pandas_ta_stub = types.ModuleType('pandas_ta')
for _fn in ['rsi', 'sma', 'ema', 'macd', 'bbands', 'kc', 'donchian', 'atr',
            'adx', 'stoch', 'squeeze', 'squeeze_pro']:
    setattr(_pandas_ta_stub, _fn, MagicMock(return_value=None))
sys.modules['pandas_ta'] = _pandas_ta_stub

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np

# Now safe to import project modules
from strategy_engine import TacticalSniper, StrategyEngine


def _make_row(**overrides):
    """
    Factory: creates a single-row DataFrame with all columns TacticalSniper needs.
    Defaults represent a neutral stock that triggers NO setups.
    Override specific fields to test individual setups.
    """
    defaults = {
        # OHLCV
        'open': 100.0, 'high': 102.0, 'low': 98.0, 'close': 101.0,
        'volume': 500_000, 'vol_avg_20': 400_000,
        # Trend
        'sma_50': 95.0, 'sma_200': 90.0,
        'er_slow': 0.30, 'er_fast': 0.25, 'trend_alignment': 0,
        # Volatility
        'bb_width': 0.25, 'bb_upper': 110.0, 'bb_lower': 90.0,
        'kc_upper': 108.0, 'kc_lower': 92.0,
        'squeeze_on': 0, 'mom_sqz': 0.0, 'atr': 2.0,
        # Momentum
        'rvol': 1.0, 'rsi': 50.0,
        'macd': 0.0, 'macdsignal': 0.0, 'macd_hist': 0.0,
        # Regime helpers
        'er_slow': 0.30,
        'is_consolidating': False,
    }
    defaults.update(overrides)
    return pd.DataFrame([defaults])


# ============================================================
# BUG 1.2: Column Name Case Mismatch in apply_checklist_bonus
# ============================================================
class TestBug1_2_ColumnCaseMismatch:
    """Verify apply_checklist_bonus reads lowercase column names from feature_engine."""

    def test_sma_alignment_bonus_fires_with_lowercase_columns(self):
        """Bonus +10 for close > sma_50 > sma_200 using lowercase key names."""
        engine = StrategyEngine()
        row = _make_row(close=150.0, sma_50=140.0, sma_200=130.0).iloc[0]
        result = engine.apply_checklist_bonus(row, 50.0)
        assert result > 50.0, \
            f"SMA alignment bonus not applied (expected > 50, got {result}). " \
            f"Likely still using uppercase 'SMA_50'/'SMA_200'."

    def test_sma_alignment_bonus_absent_when_not_aligned(self):
        """No SMA bonus when close < sma_50 (downtrend)."""
        engine = StrategyEngine()
        row_down = _make_row(close=80.0, sma_50=140.0, sma_200=130.0).iloc[0]
        row_up = _make_row(close=150.0, sma_50=140.0, sma_200=130.0).iloc[0]
        result_down = engine.apply_checklist_bonus(row_down, 50.0)
        result_up = engine.apply_checklist_bonus(row_up, 50.0)
        assert result_up > result_down, \
            "Uptrend should score higher than downtrend via SMA bonus."

    def test_rsi_bonus_fires_with_lowercase_rsi_column(self):
        """Bonus +7 for rsi in 40-65 range using lowercase key 'rsi', not 'rsi_14'."""
        engine = StrategyEngine()
        row_sweet = _make_row(rsi=55.0).iloc[0]
        row_high = _make_row(rsi=80.0).iloc[0]
        result_sweet = engine.apply_checklist_bonus(row_sweet, 50.0)
        result_high = engine.apply_checklist_bonus(row_high, 50.0)
        assert result_sweet > result_high, \
            f"RSI sweet-spot bonus not applied. Sweet: {result_sweet}, Overbought: {result_high}. " \
            f"Likely still using 'rsi_14'."

    def test_bb_upper_bonus_fires_with_lowercase_column(self):
        """Bonus +5 for (bb_upper - close) / close > 3% using key 'bb_upper'."""
        engine = StrategyEngine()
        row_room = _make_row(close=100.0, bb_upper=115.0).iloc[0]   # 15% room → bonus
        row_tight = _make_row(close=100.0, bb_upper=101.0).iloc[0]  # 1% room → no bonus
        result_room = engine.apply_checklist_bonus(row_room, 50.0)
        result_tight = engine.apply_checklist_bonus(row_tight, 50.0)
        assert result_room > result_tight, \
            f"BB distance bonus not applied. Room: {result_room}, Tight: {result_tight}. " \
            f"Likely still using 'BBU_20_2.0'."

    def test_old_uppercase_column_names_absent_from_source(self):
        """
        Guard: SMA_50, SMA_200, BBU_20 must not appear as dict key lookups.
        rsi_14 is checked specifically as a .get() key — the local variable
        name rsi_14 = ... is acceptable; only the lookup key must be 'rsi'.
        """
        path = os.path.join(PROJECT_ROOT, 'strategy_engine.py')
        with open(path, 'r') as f:
            code = f.read()
        # These should never appear as column key lookups
        for old_key in ["'SMA_50'", "'SMA_200'", "'BBU_20", "'rsi_14'"]:
            assert old_key not in code, \
                f"Dead column key {old_key} still used in strategy_engine.py!"


# ============================================================
# BUG 1.3: Non-Existent Column 'er_trend'
# ============================================================
class TestBug1_3_ErTrend:
    """Verify Setup 1 (DSP_SUPER_TREND) uses er_slow threshold instead of er_trend."""

    def test_setup1_fires_when_er_slow_above_threshold(self):
        """Setup 1 should activate when er_slow >= 0.55 AND trend_alignment == 1."""
        sniper = TacticalSniper()
        df = _make_row(er_slow=0.65, trend_alignment=1)
        result = sniper.analyze("TEST", df, "TREND")
        assert "DSP_SUPER_TREND" in result.get('setups_found', []), \
            f"Setup 1 should fire when er_slow=0.65 >= 0.55. Got: {result.get('setups_found')}"

    def test_setup1_blocked_when_er_slow_below_threshold(self):
        """Setup 1 should NOT activate when er_slow < 0.55."""
        sniper = TacticalSniper()
        df = _make_row(er_slow=0.30, trend_alignment=1)
        result = sniper.analyze("TEST", df, "TREND")
        assert "DSP_SUPER_TREND" not in result.get('setups_found', []), \
            f"Setup 1 should NOT fire when er_slow=0.30. Got: {result.get('setups_found')}"

    def test_setup1_blocked_when_trend_alignment_zero(self):
        """Setup 1 requires BOTH conditions: high er_slow AND trend_alignment == 1."""
        sniper = TacticalSniper()
        df = _make_row(er_slow=0.80, trend_alignment=0)
        result = sniper.analyze("TEST", df, "TREND")
        assert "DSP_SUPER_TREND" not in result.get('setups_found', []), \
            f"Setup 1 should NOT fire without trend_alignment. Got: {result.get('setups_found')}"

    def test_setup1_at_exact_threshold_boundary(self):
        """Setup 1 should fire at exactly the threshold value (>= is inclusive)."""
        sniper = TacticalSniper()
        df = _make_row(er_slow=0.55, trend_alignment=1)
        result = sniper.analyze("TEST", df, "TREND")
        assert "DSP_SUPER_TREND" in result.get('setups_found', []), \
            f"Setup 1 should fire at er_slow=0.55 (boundary, inclusive). Got: {result.get('setups_found')}"

    def test_er_trend_string_fully_removed(self):
        """Guard: the string 'er_trend' must not appear anywhere in strategy_engine.py."""
        path = os.path.join(PROJECT_ROOT, 'strategy_engine.py')
        with open(path, 'r') as f:
            code = f.read()
        assert 'er_trend' not in code, \
            "'er_trend' still found in strategy_engine.py — bug not fully removed!"


# ============================================================
# RUNNER (also compatible with pytest)
# ============================================================
if __name__ == '__main__':
    passed = 0
    failed = 0
    errors = []

    test_classes = [TestBug1_2_ColumnCaseMismatch, TestBug1_3_ErTrend]

    for cls in test_classes:
        print(f"\n--- {cls.__name__} ---")
        instance = cls()
        for method_name in sorted(dir(instance)):
            if not method_name.startswith('test_'):
                continue
            try:
                getattr(instance, method_name)()
                passed += 1
                print(f"  PASS: {method_name}")
            except AssertionError as e:
                failed += 1
                errors.append((cls.__name__, method_name, str(e)))
                print(f"  FAIL: {method_name}\n       {e}")
            except Exception as e:
                failed += 1
                errors.append((cls.__name__, method_name, f"ERROR: {e}"))
                print(f"  ERROR: {method_name}\n        {e}")

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed} tests")

    if errors:
        print("\nFailed tests detail:")
        for cls_name, method, msg in errors:
            print(f"  {cls_name}.{method}:\n    {msg}")
        sys.exit(1)
    else:
        print("All tests PASSED!")
        sys.exit(0)
