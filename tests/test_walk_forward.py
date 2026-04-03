# tests/test_walk_forward.py

"""
StockWise Gen-13 — Walk-Forward Validator Tests
================================================
Tests for WalkForwardValidator: 70/30 split, CP-2 checkpoint, overfit detection.

Execution: python -m pytest tests/test_walk_forward.py -v --tb=short
Expected : 12 passed, 0 failed
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from backtest_engine import WalkForwardValidator
import system_config as cfg


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_dummy_data(n_days=500, seed=42):
    """Create minimal OHLCV DataFrame for backtest split testing."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=n_days, freq="B")
    close = np.maximum(10.0, 100.0 + np.cumsum(rng.standard_normal(n_days) * 0.5))
    return pd.DataFrame({
        "open":   close * 0.999,
        "high":   close * 1.010,
        "low":    close * 0.990,
        "close":  close,
        "volume": rng.integers(1_000_000, 5_000_000, n_days).astype(float),
    }, index=dates)


def _per_tmpl(tid, *, test_t, test_wr, test_pf, test_pnl,
              train_t=10, train_wr=0.50, train_pf=2.0, train_pnl=5.0,
              wr_delta=None, is_generated=None, overfit_flag=False):
    """Build a per_template entry dict."""
    is_gen = is_generated if is_generated is not None else tid.startswith("GEN_")
    delta = wr_delta if wr_delta is not None else train_wr - test_wr
    return {
        "train_trades": train_t, "train_wr": train_wr, "train_pf": train_pf,
        "train_pnl": train_pnl, "test_trades": test_t, "test_wr": test_wr,
        "test_pf": test_pf, "test_pnl": test_pnl, "wr_delta": delta,
        "is_generated": is_gen, "overfit_flag": overfit_flag,
    }


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestWalkForwardValidator:
    """Tests for Walk-Forward Validation."""

    def setup_method(self):
        self.wf = WalkForwardValidator(symbols=["TEST"])

    # WF-01: Data splits at ~70% of timeline
    def test_wf01_split_70_30(self):
        data = {"TEST": _make_dummy_data(500)}
        train, test, info = self.wf._split_data(data)
        assert train is not None, "train data must not be None"
        assert test is not None, "test data must not be None"
        total = info["train_days"] + info["test_days"]
        ratio = info["train_days"] / total
        assert 0.65 <= ratio <= 0.75, f"Expected ~70% train split, got {ratio:.2%}"

    # WF-02: Train data ends at or before split_date (no look-ahead)
    def test_wf02_no_lookahead(self):
        data = {"TEST": _make_dummy_data(500)}
        train, test, info = self.wf._split_data(data)
        split_str = info["split_date"]
        train_max = str(train["TEST"].index.max().date())
        assert train_max <= split_str, \
            f"Train data leaks into test: train_max={train_max} > split={split_str}"

    # WF-03: split_date present in split_info (needed for audit)
    def test_wf03_split_date_in_info(self):
        data = {"TEST": _make_dummy_data(500)}
        _, _, info = self.wf._split_data(data)
        assert info.get("split_date") is not None
        assert len(info["split_date"]) == 10  # "YYYY-MM-DD"

    # WF-04: empty data → returns (None, None, {})
    def test_wf04_empty_data_graceful(self):
        train, test, info = self.wf._split_data({})
        assert train is None
        assert test is None

    # WF-05: per-template comparison calculates WR and trade counts correctly
    def test_wf05_compare_per_template(self):
        train_trades = [
            {"template_id": "T1", "pnl_pct": 2.0},
            {"template_id": "T1", "pnl_pct": -1.0},
        ]
        test_trades = [
            {"template_id": "T1", "pnl_pct": 1.5},
        ]
        result = self.wf._compare_per_template(train_trades, test_trades)
        assert "T1" in result
        assert result["T1"]["train_trades"] == 2
        assert result["T1"]["test_trades"] == 1
        assert result["T1"]["train_wr"] == 0.5
        assert result["T1"]["test_wr"] == 1.0

    # WF-06: CP-2 passes when generated templates meet thresholds
    def test_wf06_cp2_pass(self):
        per_template = {
            "GEN_TEST": _per_tmpl("GEN_TEST", test_t=10, test_wr=0.40,
                                  test_pf=2.0, test_pnl=5.0),
        }
        cp2 = self.wf._evaluate_cp2(per_template)
        assert cp2["verdict"] == "PASS"
        assert cp2["pass_wr"] is True
        assert cp2["pass_pf"] is True

    # WF-07: CP-2 fails when generated WR < 25%
    def test_wf07_cp2_fail_low_wr(self):
        per_template = {
            "GEN_TEST": _per_tmpl("GEN_TEST", test_t=10, test_wr=0.10,
                                  test_pf=0.5, test_pnl=-3.0,
                                  train_wr=0.50, wr_delta=0.40, overfit_flag=True),
        }
        cp2 = self.wf._evaluate_cp2(per_template)
        assert cp2["verdict"] == "FAIL"
        assert cp2["pass_wr"] is False

    # WF-08: overfit detection flags train WR >> test WR
    def test_wf08_overfit_detection(self):
        per_template = {
            "GEN_OVERFIT": _per_tmpl("GEN_OVERFIT", test_t=10, test_wr=0.20,
                                     test_pf=0.5, test_pnl=-2.0,
                                     train_t=20, train_wr=0.60,
                                     wr_delta=0.40, overfit_flag=True),
        }
        warnings = self.wf._detect_overfitting(per_template)
        assert len(warnings) == 1
        assert "GEN_OVERFIT" in warnings[0]
        assert "OVERFIT" in warnings[0]

    # WF-09: train_pct comes from WALK_FORWARD_CONFIG, not hardcoded
    def test_wf09_config_respected(self):
        wf_cfg = getattr(cfg, 'WALK_FORWARD_CONFIG', {})
        assert self.wf.train_pct == wf_cfg.get("train_pct", 0.70)

    # WF-10: seed and generated templates both counted separately in CP-2
    def test_wf10_seed_and_generated_reported(self):
        per_template = {
            "MOMENTUM_BREAKOUT": _per_tmpl(
                "MOMENTUM_BREAKOUT", test_t=5, test_wr=0.40, test_pf=1.5,
                test_pnl=3.0, is_generated=False),
            "GEN_BEARISH_REVERSAL": _per_tmpl(
                "GEN_BEARISH_REVERSAL", test_t=5, test_wr=0.30, test_pf=1.2,
                test_pnl=1.0),
        }
        cp2 = self.wf._evaluate_cp2(per_template)
        assert cp2["seed_test_trades"] == 5
        assert cp2["generated_test_trades"] == 5

    # WF-11: no generated templates → CP-2 passes vacuously
    def test_wf11_no_generated_templates_pass(self):
        per_template = {
            "MOMENTUM_BREAKOUT": _per_tmpl(
                "MOMENTUM_BREAKOUT", test_t=10, test_wr=0.40, test_pf=1.8,
                test_pnl=5.0, is_generated=False),
        }
        cp2 = self.wf._evaluate_cp2(per_template)
        assert cp2["verdict"] == "PASS"
        assert cp2["generated_test_trades"] == 0

    # WF-12: insufficient test trades → reason in cp2["reasons"]
    def test_wf12_insufficient_test_trades_reason(self):
        per_template = {
            "GEN_TEST": _per_tmpl("GEN_TEST", test_t=2, test_wr=0.50,
                                  test_pf=1.5, test_pnl=1.0),
        }
        cp2 = self.wf._evaluate_cp2(per_template)
        assert any("only 2 test trades" in r for r in cp2["reasons"])
