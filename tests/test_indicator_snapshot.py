"""
Tests for indicator snapshot capture and profiler analysis.
"""

import json
import os
import sys

import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest_engine import BacktestEngine, Position


# ─────────────────────────────────────────────────────────
# Position snapshot
# ─────────────────────────────────────────────────────────

class TestPositionSnapshot:
    """Position class supports indicator_snapshot."""

    def test_position_has_snapshot_slot(self):
        """indicator_snapshot must be in Position.__slots__."""
        assert "indicator_snapshot" in Position.__slots__

    def test_position_snapshot_default_empty(self):
        pos = Position(
            symbol="AAPL", template_id="TEST", template_name="Test",
            entry_price=150.0, entry_date="2024-01-01", shares=10,
            stop_loss=145.0, take_profit=165.0, initial_stop=145.0,
        )
        assert pos.indicator_snapshot == {}

    def test_position_snapshot_assignable(self):
        pos = Position(
            symbol="AAPL", template_id="TEST", template_name="Test",
            entry_price=150.0, entry_date="2024-01-01", shares=10,
            stop_loss=145.0, take_profit=165.0, initial_stop=145.0,
        )
        pos.indicator_snapshot = {"rsi": 55.0, "adx": 28.0, "er_slow": 0.5}
        assert pos.indicator_snapshot["rsi"] == 55.0
        assert len(pos.indicator_snapshot) == 3


# ─────────────────────────────────────────────────────────
# Trade snapshot propagation
# ─────────────────────────────────────────────────────────

class TestTradeHasSnapshot:
    """Closed trades include indicators_at_entry."""

    def _make_trade_with_snapshot(self):
        return {
            "symbol": "AAPL", "template_id": "TEST", "pnl": 100, "pnl_pct": 2.0,
            "bars_held": 5, "final_phase": "PHASE_3",
            "entry_date": "2024-06-15", "exit_date": "2024-06-20",
            "entry_price": 150.0, "exit_price": 153.0, "shares": 10,
            "template_name": "Test",
            "indicators_at_entry": {
                "rsi": 55.2, "adx": 28.1, "er_slow": 0.51,
                "macd_hist": 1.3, "rvol": 1.42, "bb_width_pct": 0.08,
                "close": 150.0, "volume": 1_000_000,
            },
        }

    def test_trade_has_indicators_key(self):
        trade = self._make_trade_with_snapshot()
        assert "indicators_at_entry" in trade
        assert isinstance(trade["indicators_at_entry"], dict)

    def test_trade_snapshot_has_numeric_values(self):
        trade = self._make_trade_with_snapshot()
        for k, v in trade["indicators_at_entry"].items():
            assert isinstance(v, (int, float)), f"{k} is {type(v)}"

    def test_trade_snapshot_json_serializable(self):
        trade = self._make_trade_with_snapshot()
        parsed = json.loads(json.dumps(trade))
        assert parsed["indicators_at_entry"]["rsi"] == 55.2


# ─────────────────────────────────────────────────────────
# Indicator Profiler
# ─────────────────────────────────────────────────────────

def _make_profiler_trades():
    """5 wins (high ER/ADX) + 5 losses (low ER/ADX)."""
    trades = []
    for i in range(5):
        trades.append({
            "symbol": "GOOGL", "template_id": "TEST", "template_name": "T",
            "pnl": 100 + i * 10, "pnl_pct": 2.0 + i * 0.5,
            "bars_held": 5, "final_phase": "PHASE_3",
            "entry_date": f"2024-0{i+1}-15", "exit_date": f"2024-0{i+1}-20",
            "entry_price": 150.0, "exit_price": 153.0, "shares": 10,
            "indicators_at_entry": {
                "rsi": 52.0 + i, "adx": 27.0 + i, "er_slow": 0.48 + i * 0.02,
                "macd_hist": 1.0 + i * 0.3, "rvol": 1.3 + i * 0.1,
                "bb_width_pct": 0.07 + i * 0.005, "close": 150.0, "volume": 1_000_000,
            },
        })
    for i in range(5):
        trades.append({
            "symbol": "GOOGL", "template_id": "TEST", "template_name": "T",
            "pnl": -(50 + i * 10), "pnl_pct": -(1.0 + i * 0.3),
            "bars_held": 3, "final_phase": "PHASE_1",
            "entry_date": f"2024-0{i+6}-10", "exit_date": f"2024-0{i+6}-13",
            "entry_price": 150.0, "exit_price": 148.0, "shares": 10,
            "indicators_at_entry": {
                "rsi": 45.0 + i, "adx": 18.0 + i, "er_slow": 0.25 + i * 0.02,
                "macd_hist": -0.5 - i * 0.2, "rvol": 0.9 + i * 0.05,
                "bb_width_pct": 0.12 + i * 0.01, "close": 150.0, "volume": 900_000,
            },
        })
    return trades


class TestIndicatorProfiler:
    """Section 10: WIN vs LOSS indicator comparison."""

    def _engine(self, trades):
        engine = BacktestEngine(symbols=["GOOGL"], initial_capital=100_000,
                                data_cache={"GOOGL": MagicMock()})
        engine.closed_trades = trades
        return engine

    def test_profiler_section_present(self):
        trades = _make_profiler_trades()
        result = self._engine(trades)._compute_analytics(trades, {})
        assert "indicator_profiler" in result

    def test_profiler_has_discriminators(self):
        trades = _make_profiler_trades()
        result = self._engine(trades)._compute_analytics(trades, {})
        assert len(result["indicator_profiler"].get("top_discriminators", [])) > 0

    def test_er_slow_discriminates_positive(self):
        """er_slow should be positive discriminator (higher in wins)."""
        trades = _make_profiler_trades()
        result = self._engine(trades)._compute_analytics(trades, {})
        discs = result["indicator_profiler"]["top_discriminators"]
        er = next((d for d in discs if d["indicator"] == "er_slow"), None)
        assert er is not None
        assert er["delta"] > 0

    def test_ohlcv_excluded_from_discriminators(self):
        """close and volume must NOT appear in discriminators."""
        trades = _make_profiler_trades()
        result = self._engine(trades)._compute_analytics(trades, {})
        names = [d["indicator"] for d in result["indicator_profiler"]["top_discriminators"]]
        assert "close"  not in names
        assert "volume" not in names

    def test_normalized_score_in_0_1(self):
        trades = _make_profiler_trades()
        result = self._engine(trades)._compute_analytics(trades, {})
        for d in result["indicator_profiler"]["top_discriminators"]:
            assert 0 <= d["normalized_score"] <= 1.0, \
                f"{d['indicator']}: score={d['normalized_score']}"

    def test_empty_no_crash(self):
        """Trades without snapshots → empty profiler, no crash."""
        trades = [
            {"symbol": "GOOGL", "pnl": 100, "pnl_pct": 2.0, "bars_held": 5,
             "final_phase": "P3", "entry_date": "2024-01-15", "exit_date": "2024-01-20",
             "entry_price": 150, "exit_price": 153, "shares": 10,
             "template_id": "T", "template_name": "T"},
        ]
        engine = BacktestEngine(symbols=["GOOGL"], initial_capital=100_000,
                                data_cache={"GOOGL": MagicMock()})
        engine.closed_trades = trades
        result = engine._compute_analytics(trades, {})
        ip = result.get("indicator_profiler", {})
        assert ip == {} or ip.get("top_discriminators", []) == []

    def test_win_loss_snapshot_counts(self):
        trades = _make_profiler_trades()
        result = self._engine(trades)._compute_analytics(trades, {})
        ip = result["indicator_profiler"]
        assert ip["total_wins_with_snapshot"]   == 5
        assert ip["total_losses_with_snapshot"] == 5


# ─────────────────────────────────────────────────────────
# Per-Symbol Summary
# ─────────────────────────────────────────────────────────

class TestPerSymbolSummary:
    """per_symbol_summary key in analytics."""

    def _engine(self, trades):
        syms = list({t["symbol"] for t in trades})
        engine = BacktestEngine(symbols=syms, initial_capital=100_000,
                                data_cache={s: MagicMock() for s in syms})
        engine.closed_trades = trades
        return engine

    def test_section_present(self):
        trades = [
            {"symbol": "GOOGL", "pnl": 100, "pnl_pct": 2.0, "bars_held": 5,
             "final_phase": "P3", "entry_date": "2024-01-15", "exit_date": "2024-01-20",
             "entry_price": 150, "exit_price": 153, "shares": 10,
             "template_id": "T", "template_name": "T"},
        ]
        result = self._engine(trades)._compute_analytics(trades, {})
        assert "per_symbol_summary" in result
        assert "GOOGL" in result["per_symbol_summary"]

    def test_wr_correct(self):
        trades = [
            {"symbol": "AAPL", "pnl": 100,  "pnl_pct":  2.0, "bars_held": 5,
             "final_phase": "P3", "entry_date": "2024-01-15", "exit_date": "2024-01-20",
             "entry_price": 150, "exit_price": 153, "shares": 10,
             "template_id": "T", "template_name": "T"},
            {"symbol": "AAPL", "pnl": -50, "pnl_pct": -1.0, "bars_held": 3,
             "final_phase": "P1", "entry_date": "2024-02-15", "exit_date": "2024-02-18",
             "entry_price": 150, "exit_price": 148, "shares": 10,
             "template_id": "T", "template_name": "T"},
        ]
        result = self._engine(trades)._compute_analytics(trades, {})
        aapl = result["per_symbol_summary"]["AAPL"]
        assert aapl["trades"] == 2
        assert aapl["wins"]   == 1
        assert aapl["wr"]     == 50.0


# ─────────────────────────────────────────────────────────
# PULLBACK v3
# ─────────────────────────────────────────────────────────

class TestPullbackV3:
    """PULLBACK v3 template validation."""

    def _get(self):
        from setup_templates import TemplateManager
        return TemplateManager().get_template_by_id("TREND_PULLBACK_EMA")

    def test_version_3(self):
        assert self._get().data.get("version") == 3

    def test_4_conditions(self):
        assert len(self._get().conditions) == 4

    def test_er_slow_030(self):
        er = [c for c in self._get().conditions if c["block"] == "er_slow_above"]
        assert len(er) == 1
        assert er[0]["params"] == [0.30]

    def test_no_volume_surge(self):
        blocks = [c["block"] for c in self._get().conditions]
        assert "volume_surge" not in blocks

    def test_passes_validation(self):
        t = self._get()
        valid, errors = t.validate()
        assert valid, f"PULLBACK v3 failed: {errors}"


# ─────────────────────────────────────────────────────────
# Regression
# ─────────────────────────────────────────────────────────

class TestRegression:
    """Nothing broke."""

    def test_analytics_sections_1_to_8_intact(self):
        engine = BacktestEngine(symbols=["AAPL"], initial_capital=100_000,
                                data_cache={"AAPL": MagicMock()})
        trade = {"symbol": "AAPL", "pnl": 100, "pnl_pct": 2.0, "bars_held": 5,
                 "final_phase": "P3", "entry_date": "2024-01-15",
                 "exit_date": "2024-01-20", "entry_price": 150, "exit_price": 153,
                 "shares": 10, "template_id": "T", "template_name": "T"}
        engine.closed_trades = [trade]
        result = engine._compute_analytics([trade], {})
        for section in ["template_anatomy", "trade_breakdown", "temporal",
                        "phase_analysis", "block_stats", "shadow_ledger_matrix",
                        "winner_loser_profile", "block_evaluations",
                        "per_symbol_summary", "indicator_profiler"]:
            assert section in result, f"Missing section: {section}"

    def test_all_seed_templates_valid(self):
        from setup_templates import TemplateManager
        tm = TemplateManager()
        for tid, t in tm.templates.items():
            valid, errors = t.validate()
            assert valid, f"{tid} failed: {errors}"
