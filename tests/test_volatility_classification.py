"""
StockWise — Volatility Classification Tests
=============================================
Validates _classify_volatility_state uses bb_width_pct (not bb_width).
Ref: P0 #1 fix, SPEC v13.4 §3 Regime Classification.
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_df(bb_width_pct=None, bb_width=None):
    """Helper: create minimal DataFrame for volatility classification."""
    data = {
        "close":  [150.0],
        "open":   [149.0],
        "high":   [151.0],
        "low":    [148.0],
        "volume": [1_000_000],
    }
    if bb_width_pct is not None:
        data["bb_width_pct"] = [bb_width_pct]
    if bb_width is not None:
        data["bb_width"] = [bb_width]
    return pd.DataFrame(data)


@pytest.fixture
def hunter():
    from stock_hunter import StockHunter
    return StockHunter.__new__(StockHunter)


# ═══════════════════════════════════════════════════════════
# Unit tests
# ═══════════════════════════════════════════════════════════

class TestClassifyVolatilityState:
    """Tests for _classify_volatility_state using bb_width_pct."""

    def test_compressed_below_threshold(self, hunter):
        """T1: bb_width_pct < 0.10 → COMPRESSED"""
        df = _make_df(bb_width_pct=0.05)
        assert hunter._classify_volatility_state(df) == "COMPRESSED"

    def test_normal_in_range(self, hunter):
        """T2: bb_width_pct between 0.10 and 0.30 → NORMAL"""
        df = _make_df(bb_width_pct=0.20)
        assert hunter._classify_volatility_state(df) == "NORMAL"

    def test_volatile_above_threshold(self, hunter):
        """T3: bb_width_pct > 0.30 → VOLATILE"""
        df = _make_df(bb_width_pct=0.45)
        assert hunter._classify_volatility_state(df) == "VOLATILE"

    def test_nan_bb_width_pct_falls_back(self, hunter):
        """T4: bb_width_pct = NaN → fallback to bb_width → safe result"""
        df = _make_df(bb_width_pct=float("nan"), bb_width=0.20)
        result = hunter._classify_volatility_state(df)
        assert result in ("COMPRESSED", "NORMAL", "VOLATILE")

    def test_missing_bb_width_pct_falls_back(self, hunter):
        """T5: bb_width_pct column missing → fallback to bb_width"""
        df = _make_df(bb_width=0.20)
        result = hunter._classify_volatility_state(df)
        assert result in ("COMPRESSED", "NORMAL", "VOLATILE")

    def test_boundary_exact_squeeze_threshold(self, hunter):
        """T6: bb_width_pct = 0.10 exactly → NORMAL (not < 0.10)"""
        df = _make_df(bb_width_pct=0.10)
        assert hunter._classify_volatility_state(df) == "NORMAL"

    def test_boundary_exact_volatile_threshold(self, hunter):
        """T7: bb_width_pct = 0.30 exactly → NORMAL (not > 0.30)"""
        df = _make_df(bb_width_pct=0.30)
        assert hunter._classify_volatility_state(df) == "NORMAL"

    def test_config_overrides_defaults(self, hunter, monkeypatch):
        """T8: Custom thresholds from config are respected."""
        import system_config as cfg

        custom_config = {
            "squeeze_bb_width_threshold": 0.05,
            "volatile_bb_width_threshold": 0.20,
        }
        monkeypatch.setattr(cfg, "MANDATORY_SCAN_CONFIG", custom_config, raising=False)

        # 0.08 is between 0.05 and 0.20 → NORMAL with custom thresholds
        df = _make_df(bb_width_pct=0.08)
        assert hunter._classify_volatility_state(df) == "NORMAL"

        # 0.03 < 0.05 → COMPRESSED with custom thresholds
        df2 = _make_df(bb_width_pct=0.03)
        assert hunter._classify_volatility_state(df2) == "COMPRESSED"


# ═══════════════════════════════════════════════════════════
# Regression guards
# ═══════════════════════════════════════════════════════════

class TestVolatilityClassificationRegression:
    """Regression guards to prevent this bug from returning."""

    def test_uses_bb_width_pct_not_bb_width(self):
        """R1: Source code inspection — must use bb_width_pct as primary field."""
        import inspect
        from stock_hunter import StockHunter

        source = inspect.getsource(StockHunter._classify_volatility_state)
        assert "bb_width_pct" in source, (
            "_classify_volatility_state must use bb_width_pct, not raw bb_width"
        )

    def test_dollar_bb_width_not_compared_to_pct_threshold(self):
        """R2: A stock with bb_width=$21 (dollars) must NOT be classified
        solely by that value — bb_width_pct should be used instead."""
        from stock_hunter import StockHunter

        hunter = StockHunter.__new__(StockHunter)
        # bb_width_pct=0.12 (NORMAL), but raw bb_width=$21 (VOLATILE with old bug)
        df = _make_df(bb_width_pct=0.12, bb_width=21.0)
        assert hunter._classify_volatility_state(df) == "NORMAL", (
            "Must use bb_width_pct (0.12=NORMAL), not bb_width ($21=VOLATILE)"
        )

    def test_compressed_enables_squeeze_template(self):
        """R3: End-to-end — COMPRESSED state must allow SQUEEZE_BREAKOUT match."""
        from stock_hunter import StockHunter
        from setup_templates import TemplateManager

        hunter = StockHunter.__new__(StockHunter)
        df = _make_df(bb_width_pct=0.05)
        state = {
            "trend":      "BULLISH",
            "structure":  "OPEN_FIELD",
            "volume":     "SURGING",
            "volatility": hunter._classify_volatility_state(df),
        }

        tm = TemplateManager()
        matching = tm.get_for_state(state)
        template_ids = [t.id for t in matching]

        assert "SQUEEZE_BREAKOUT" in template_ids, (
            f"COMPRESSED state must match SQUEEZE_BREAKOUT, got: {template_ids}"
        )
