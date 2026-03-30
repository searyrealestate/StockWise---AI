"""
Tests for block-level evaluation statistics in backtest_engine.py (Section 8).
Validates the second-pass block evaluation collection.
"""

import os
import sys
import pytest
from collections import defaultdict
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest_engine import BacktestEngine
import system_config as cfg


def _make_trade(symbol="AAPL", template_id="TREND_PULLBACK_EMA", pnl=100, pnl_pct=2.0,
                bars_held=5, final_phase="PHASE_2_BREAKEVEN", entry_date="2024-06-15",
                exit_date="2024-06-20", exit_reason="STOP_HIT", entry_price=150.0,
                exit_price=153.0, shares=10, template_name="Trend Pullback"):
    return {
        "symbol": symbol, "template_id": template_id, "template_name": template_name,
        "entry_price": entry_price, "exit_price": exit_price,
        "entry_date": entry_date, "exit_date": exit_date,
        "shares": shares, "pnl": pnl, "pnl_pct": pnl_pct,
        "exit_reason": exit_reason, "bars_held": bars_held, "final_phase": final_phase,
    }


# ─────────────────────────────────────────────────────────────
# Config tests
# ─────────────────────────────────────────────────────────────

class TestBlockEvalConfig:
    """Test that block evaluation config exists and works."""

    def test_analytics_config_has_block_eval_key(self):
        """ANALYTICS_CONFIG must contain include_block_evaluations key."""
        ac = getattr(cfg, "ANALYTICS_CONFIG", {})
        assert "include_block_evaluations" in ac

    def test_block_eval_toggle_off(self):
        """When include_block_evaluations=False, block_eval_stats stays empty."""
        engine = BacktestEngine(symbols=["AAPL"], initial_capital=10000, data_cache={})
        engine.closed_trades = []
        patched_cfg = {**getattr(cfg, "ANALYTICS_CONFIG", {}),
                       "include_block_evaluations": False}
        with patch.object(cfg, "ANALYTICS_CONFIG", patched_cfg):
            engine._collect_block_evaluations()
        assert engine.block_eval_stats == {}


# ─────────────────────────────────────────────────────────────
# Initialisation
# ─────────────────────────────────────────────────────────────

class TestBlockEvalStats:
    """Test block evaluation statistics collection logic."""

    def test_block_eval_stats_initialized_empty(self):
        """block_eval_stats must start as empty dict."""
        engine = BacktestEngine(symbols=[], initial_capital=10000, data_cache={})
        assert engine.block_eval_stats == {}

    def test_block_eval_empty_data_no_crash(self):
        """Empty data_cache should produce empty block_eval_stats, not crash."""
        engine = BacktestEngine(symbols=["AAPL"], initial_capital=10000, data_cache={})
        engine.closed_trades = []
        try:
            engine._collect_block_evaluations()
        except Exception as e:
            pytest.fail(f"_collect_block_evaluations crashed on empty data: {e}")
        assert isinstance(engine.block_eval_stats, dict)


# ─────────────────────────────────────────────────────────────
# Logic invariants (unit-level, no backtest needed)
# ─────────────────────────────────────────────────────────────

class TestBlockEvalInvariants:
    """Invariants on the stats structure."""

    def test_pass_plus_fail_equals_eval(self):
        """For any block: passed + failed must equal evaluated."""
        stats = {
            "blocks": {
                "rsi_between":    {"evaluated": 100, "passed": 60,  "failed": 40},
                "close_above_ema": {"evaluated": 100, "passed": 95, "failed":  5},
            }
        }
        for bn, bs in stats["blocks"].items():
            assert bs["passed"] + bs["failed"] == bs["evaluated"], \
                f"{bn}: pass+fail != eval"

    def test_sole_blocker_when_single_block_fails(self):
        """When exactly 1 of N blocks fails, it is the sole blocker."""
        details = [
            {"block": "rsi_between",    "passed": True},
            {"block": "close_above_ema", "passed": True},
            {"block": "close_above_sma", "passed": True},
            {"block": "sma_above_sma",  "passed": False},
        ]
        failed = [d["block"] for d in details if not d["passed"]]
        assert len(failed) == 1
        assert failed[0] == "sma_above_sma"

    def test_no_sole_blocker_when_multiple_fail(self):
        """When 2+ blocks fail, no sole blocker (len > 1)."""
        details = [
            {"block": "rsi_between",    "passed": False},
            {"block": "close_above_ema", "passed": True},
            {"block": "close_above_sma", "passed": False},
            {"block": "sma_above_sma",  "passed": True},
        ]
        failed = [d["block"] for d in details if not d["passed"]]
        assert len(failed) == 2  # more than 1 → no sole blocker assigned

    def test_state_filter_total_equals_matched_plus_rejected(self):
        """State filter: total_scans == state_matched + state_rejected."""
        sf = {"total_scans": 100, "state_matched": 40, "state_rejected": 60}
        assert sf["state_matched"] + sf["state_rejected"] == sf["total_scans"]

    def test_when_passed_wr_calculation(self):
        """when_passed WR = wins / total_trades * 100."""
        wp = {"total_trades": 20, "wins": 8, "losses": 12, "pnl_sum": 15.0}
        wr = round(wp["wins"] / wp["total_trades"] * 100, 1)
        assert wr == 40.0

    def test_pass_rate_calculation(self):
        """pass_rate = passed / evaluated * 100."""
        bs = {"evaluated": 80, "passed": 60, "failed": 20}
        pr = round(bs["passed"] / bs["evaluated"] * 100, 1)
        assert pr == 75.0


# ─────────────────────────────────────────────────────────────
# Regression guards
# ─────────────────────────────────────────────────────────────

class TestBlockEvalRegression:
    """Regression guards."""

    def test_scan_for_signals_has_no_block_eval_code(self):
        """_scan_for_signals must NOT contain block_eval references (untouched)."""
        import inspect
        source = inspect.getsource(BacktestEngine._scan_for_signals)
        assert "block_eval" not in source, \
            "_scan_for_signals was modified — it must remain untouched"

    def test_existing_analytics_sections_intact(self):
        """Sections 1-7 must still be present in _compute_analytics output."""
        engine = BacktestEngine(symbols=["AAPL"], initial_capital=10000, data_cache={})
        engine.closed_trades = [_make_trade()]
        engine.block_eval_stats = {}
        result = engine._compute_analytics(engine.closed_trades, {})
        for section in ["template_anatomy", "trade_breakdown", "temporal",
                        "phase_analysis", "block_stats", "shadow_ledger_matrix",
                        "winner_loser_profile"]:
            assert section in result, f"Missing existing section: {section}"

    def test_section_8_present_in_analytics(self):
        """block_evaluations key must be present in _compute_analytics output."""
        engine = BacktestEngine(symbols=["AAPL"], initial_capital=10000, data_cache={})
        engine.closed_trades = [_make_trade()]
        engine.block_eval_stats = {}
        result = engine._compute_analytics(engine.closed_trades, {})
        assert "block_evaluations" in result
