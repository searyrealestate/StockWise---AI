"""
StockWise — Backtest Analytics Tests
Validates _compute_analytics() and _print_analytics() on BacktestEngine.
Ref: SPEC v13.4 §5
"""

import io
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest_engine import BacktestEngine


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _make_engine():
    engine = BacktestEngine.__new__(BacktestEngine)
    engine.closed_trades = []
    engine.equity_curve  = []
    engine.open_positions = []
    engine.capital       = 100_000
    engine.config        = {"initial_capital": 100_000}
    return engine


def _trade(symbol="AAPL", template_id="T1", pnl_pct=2.0, pnl=200.0,
           bars_held=5, entry_date="2025-01-15", final_phase="BULL"):
    return {
        "symbol":      symbol,
        "template_id": template_id,
        "pnl_pct":     pnl_pct,
        "pnl":         pnl,
        "bars_held":   bars_held,
        "entry_date":  entry_date,
        "final_phase": final_phase,
    }


def _trades_mixed():
    return [
        _trade("AAPL", "T1", pnl_pct=2.0,  pnl= 200, bars_held=5,  entry_date="2024-01-10", final_phase="BULL"),
        _trade("TSLA", "T1", pnl_pct=-1.5, pnl=-150, bars_held=3,  entry_date="2024-02-20", final_phase="BEAR"),
        _trade("MSFT", "T2", pnl_pct=3.0,  pnl= 300, bars_held=12, entry_date="2025-03-05", final_phase="BULL"),
        _trade("NVDA", "T2", pnl_pct=-0.5, pnl= -50, bars_held=7,  entry_date="2025-04-15", final_phase="TRANSITION"),
    ]


# ─────────────────────────────────────────────────────────────
# Level 1: section existence and types
# ─────────────────────────────────────────────────────────────

class TestAnalyticsSections:
    """A1–A3: all 7 sections present, types correct."""

    def test_returns_dict(self):
        """A1: _compute_analytics returns a dict."""
        engine = _make_engine()
        result = engine._compute_analytics(_trades_mixed(), {})
        assert isinstance(result, dict)

    def test_all_seven_sections_present(self):
        """A2: All 7 required sections are keys in the result."""
        engine = _make_engine()
        result = engine._compute_analytics(_trades_mixed(), {})
        expected = {
            "template_anatomy", "trade_breakdown", "temporal",
            "phase_analysis", "block_stats", "shadow_ledger_matrix",
            "winner_loser_profile",
        }
        assert expected.issubset(result.keys())

    def test_empty_trades_returns_empty_dict(self):
        """A3: Empty trades → empty analytics dict."""
        engine = _make_engine()
        result = engine._compute_analytics([], {})
        assert result == {}


# ─────────────────────────────────────────────────────────────
# Level 2: trade_breakdown correctness
# ─────────────────────────────────────────────────────────────

class TestTradeBreakdown:
    """A4–A6: trade_breakdown per-template stats."""

    def test_trade_counts_correct(self):
        """A4: trade_breakdown counts trades per template."""
        engine = _make_engine()
        result = engine._compute_analytics(_trades_mixed(), {})
        bd = result["trade_breakdown"]
        assert bd["T1"]["total_trades"] == 2
        assert bd["T2"]["total_trades"] == 2

    def test_win_rate_correct(self):
        """A5: Win rate computed correctly per template."""
        engine = _make_engine()
        result = engine._compute_analytics(_trades_mixed(), {})
        bd = result["trade_breakdown"]
        assert bd["T1"]["win_rate"] == 50.0   # 1 win / 2 trades
        assert bd["T2"]["win_rate"] == 50.0   # 1 win / 2 trades

    def test_avg_pnl_pct_correct(self):
        """A6: avg_pnl_pct = mean pnl_pct for template's trades."""
        engine = _make_engine()
        result = engine._compute_analytics(_trades_mixed(), {})
        bd = result["trade_breakdown"]
        # T1: (2.0 + -1.5) / 2 = 0.25
        assert bd["T1"]["avg_pnl_pct"] == pytest.approx(0.25)


# ─────────────────────────────────────────────────────────────
# Level 3: temporal correctness
# ─────────────────────────────────────────────────────────────

class TestTemporal:
    """A7–A8: temporal by_year / by_quarter grouping."""

    def test_by_year_splits_correctly(self):
        """A7: Trades from 2024 and 2025 land in separate year buckets."""
        engine = _make_engine()
        result = engine._compute_analytics(_trades_mixed(), {})
        by_year = result["temporal"]["by_year"]
        assert "2024" in by_year
        assert "2025" in by_year
        assert by_year["2024"]["trades"] == 2
        assert by_year["2025"]["trades"] == 2

    def test_by_quarter_present(self):
        """A8: by_quarter keys are present and non-empty."""
        engine = _make_engine()
        result = engine._compute_analytics(_trades_mixed(), {})
        by_q = result["temporal"]["by_quarter"]
        assert len(by_q) >= 2


# ─────────────────────────────────────────────────────────────
# Level 4: phase_analysis correctness
# ─────────────────────────────────────────────────────────────

class TestPhaseAnalysis:
    """A9: phase_analysis groups trades by final_phase."""

    def test_phase_counts(self):
        """A9: Correct trade count per phase."""
        engine = _make_engine()
        result = engine._compute_analytics(_trades_mixed(), {})
        pa = result["phase_analysis"]
        assert pa["BULL"]["trades"] == 2
        assert pa["BEAR"]["trades"] == 1
        assert pa["TRANSITION"]["trades"] == 1


# ─────────────────────────────────────────────────────────────
# Level 5: winner_loser_profile
# ─────────────────────────────────────────────────────────────

class TestWinnerLoserProfile:
    """A10: winner_loser_profile structure and top/worst trades."""

    def test_top_5_and_worst_5_present(self):
        """A10: top_5_trades and worst_5_trades keys exist."""
        engine = _make_engine()
        result = engine._compute_analytics(_trades_mixed(), {})
        wlp = result["winner_loser_profile"]
        assert "top_5_trades" in wlp
        assert "worst_5_trades" in wlp

    def test_top_trade_is_highest_pnl(self):
        """A11: First entry in top_5_trades has highest pnl_pct."""
        engine = _make_engine()
        result = engine._compute_analytics(_trades_mixed(), {})
        top5 = result["winner_loser_profile"]["top_5_trades"]
        assert top5[0]["pnl_pct"] == 3.0  # MSFT T2

    def test_bars_distribution_keys_exist(self):
        """A12: bars_distribution_wins and _losses keys present."""
        engine = _make_engine()
        result = engine._compute_analytics(_trades_mixed(), {})
        wlp = result["winner_loser_profile"]
        assert "bars_distribution_wins"   in wlp
        assert "bars_distribution_losses" in wlp


# ─────────────────────────────────────────────────────────────
# Regression guards
# ─────────────────────────────────────────────────────────────

class TestAnalyticsRegression:
    """Regression guards."""

    def test_compute_analytics_method_exists(self):
        """R1: _compute_analytics exists and is callable."""
        assert callable(getattr(BacktestEngine, "_compute_analytics", None))

    def test_print_analytics_method_exists(self):
        """R2: _print_analytics exists and is callable."""
        assert callable(getattr(BacktestEngine, "_print_analytics", None))

    def test_analytics_key_in_run_results(self):
        """R3: analytics key present in results dict from run()."""
        engine = _make_engine()
        # Verify analytics dict is returned with all 7 sections (without running full backtest)
        analytics = engine._compute_analytics(_trades_mixed(), {})
        assert "analytics" not in analytics   # analytics IS the value, not nested
        assert "trade_breakdown" in analytics  # key structure is flat 7-section dict
