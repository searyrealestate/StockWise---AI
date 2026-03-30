"""
StockWise — Backtest → Shadow Ledger Feed Tests
Validates backtest results are additively merged into shadow_ledger.json
for DDR #1 Asset-Specific win rates.
"""

import inspect
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safe_json_io import safe_json_read, safe_json_write


@pytest.fixture
def engine_with_tmp_ledger(monkeypatch, tmp_path):
    """Create a BacktestEngine that writes to a temp shadow_ledger.json."""
    import system_config as cfg

    tmp_ledger = str(tmp_path / "shadow_ledger.json")

    sl_config = dict(getattr(cfg, 'SHADOW_LEDGER_CONFIG', {}))
    sl_config['ledger_path'] = tmp_ledger
    monkeypatch.setattr(cfg, 'SHADOW_LEDGER_CONFIG', sl_config, raising=False)

    from backtest_engine import BacktestEngine
    engine = BacktestEngine.__new__(BacktestEngine)
    engine.closed_trades = []
    engine.feed_shadow_ledger = True

    return engine, tmp_ledger


# ═══════════════════════════════════════════════════════════
# Unit tests
# ═══════════════════════════════════════════════════════════

class TestBacktestShadowFeed:
    """Tests for _feed_shadow_ledger additive merge."""

    def test_creates_entries_for_traded_symbols(self, engine_with_tmp_ledger):
        """T1: Feed creates entries for symbols that had trades."""
        engine, ledger_path = engine_with_tmp_ledger

        trades = [
            {"symbol": "AAPL", "template_id": "MOMENTUM_BREAKOUT", "pnl_pct": 2.5, "won": True},
            {"symbol": "AAPL", "template_id": "MOMENTUM_BREAKOUT", "pnl_pct": -1.0, "won": False},
            {"symbol": "MSFT", "template_id": "TREND_PULLBACK_EMA", "pnl_pct": 1.5, "won": True},
        ]
        engine._feed_shadow_ledger(trades)

        ledger = safe_json_read(ledger_path, default={})
        stats = ledger.get("template_stats", {})

        assert "AAPL" in stats, "AAPL should be in template_stats"
        assert "MSFT" in stats, "MSFT should be in template_stats"
        assert stats["AAPL"]["MOMENTUM_BREAKOUT"]["signal_count"] == 2
        assert stats["AAPL"]["MOMENTUM_BREAKOUT"]["wins"] == 1
        assert stats["AAPL"]["MOMENTUM_BREAKOUT"]["losses"] == 1
        assert stats["MSFT"]["TREND_PULLBACK_EMA"]["signal_count"] == 1

    def test_additive_merge_doubles_on_second_run(self, engine_with_tmp_ledger):
        """T2: Running feed twice with same data → signal_count doubles."""
        engine, ledger_path = engine_with_tmp_ledger

        trades = [
            {"symbol": "NVDA", "template_id": "MOMENTUM_BREAKOUT", "pnl_pct": 3.0, "won": True},
        ]

        engine._feed_shadow_ledger(trades)
        engine._feed_shadow_ledger(trades)

        ledger = safe_json_read(ledger_path, default={})
        stats = ledger["template_stats"]["NVDA"]["MOMENTUM_BREAKOUT"]

        assert stats["signal_count"] == 2, "Should accumulate, not overwrite"
        assert stats["wins"] == 2

    def test_win_rate_recalculated_correctly(self, engine_with_tmp_ledger):
        """T3: win_rate is recalculated after merge, not accumulated."""
        engine, ledger_path = engine_with_tmp_ledger

        # First run: 1 win
        trades1 = [
            {"symbol": "TSLA", "template_id": "MOMENTUM_BREAKOUT", "pnl_pct": 2.0, "won": True},
        ]
        engine._feed_shadow_ledger(trades1)

        # Second run: 1 loss
        trades2 = [
            {"symbol": "TSLA", "template_id": "MOMENTUM_BREAKOUT", "pnl_pct": -1.0, "won": False},
        ]
        engine._feed_shadow_ledger(trades2)

        ledger = safe_json_read(ledger_path, default={})
        stats = ledger["template_stats"]["TSLA"]["MOMENTUM_BREAKOUT"]

        assert stats["signal_count"] == 2
        assert stats["wins"] == 1
        assert stats["losses"] == 1
        assert stats["win_rate"] == 50.0, "WR should be recalculated as 1/2 = 50%"

    def test_no_trades_no_modification(self, engine_with_tmp_ledger):
        """T4: Empty trades list → no crash, no file created."""
        engine, ledger_path = engine_with_tmp_ledger

        engine._feed_shadow_ledger([])

        # File should not exist or contain empty template_stats
        if os.path.exists(ledger_path):
            ledger = safe_json_read(ledger_path, default={})
            assert ledger.get("template_stats", {}) == {}

    def test_preserves_existing_symbols(self, engine_with_tmp_ledger):
        """T5: Existing shadow_ledger data for other symbols is preserved."""
        engine, ledger_path = engine_with_tmp_ledger

        existing = {
            "metadata": {"last_run": "2026-03-28T00:00:00", "version": "13.4"},
            "template_stats": {
                "GOOGL": {
                    "MOMENTUM_BREAKOUT": {
                        "signal_count": 10, "wins": 6, "losses": 4,
                        "total_pnl_pct": 5.0, "win_rate": 60.0, "avg_pnl_pct": 0.5,
                    }
                }
            }
        }
        safe_json_write(ledger_path, existing)

        trades = [
            {"symbol": "AAPL", "template_id": "TREND_PULLBACK_EMA", "pnl_pct": 1.0, "won": True},
        ]
        engine._feed_shadow_ledger(trades)

        ledger = safe_json_read(ledger_path, default={})
        stats = ledger["template_stats"]

        # GOOGL should be untouched
        assert stats["GOOGL"]["MOMENTUM_BREAKOUT"]["signal_count"] == 10
        assert stats["GOOGL"]["MOMENTUM_BREAKOUT"]["wins"] == 6
        # AAPL should be added
        assert "AAPL" in stats

    def test_merges_into_existing_symbol(self, engine_with_tmp_ledger):
        """T6: Feed adds to existing symbol stats, not overwrites."""
        engine, ledger_path = engine_with_tmp_ledger

        existing = {
            "metadata": {"last_run": None, "version": "13.4"},
            "template_stats": {
                "AAPL": {
                    "MOMENTUM_BREAKOUT": {
                        "signal_count": 5, "wins": 3, "losses": 2,
                        "total_pnl_pct": 4.0, "win_rate": 60.0, "avg_pnl_pct": 0.8,
                    }
                }
            }
        }
        safe_json_write(ledger_path, existing)

        trades = [
            {"symbol": "AAPL", "template_id": "MOMENTUM_BREAKOUT", "pnl_pct": 2.0, "won": True},
        ]
        engine._feed_shadow_ledger(trades)

        ledger = safe_json_read(ledger_path, default={})
        stats = ledger["template_stats"]["AAPL"]["MOMENTUM_BREAKOUT"]

        assert stats["signal_count"] == 6, "Should be 5 + 1"
        assert stats["wins"] == 4, "Should be 3 + 1"
        assert stats["losses"] == 2, "Losses unchanged"
        assert abs(stats["win_rate"] - 66.7) < 0.1, "WR = 4/6 = 66.7%"

    def test_avg_pnl_recalculated(self, engine_with_tmp_ledger):
        """T7: avg_pnl_pct is recalculated from total_pnl_pct / signal_count."""
        engine, ledger_path = engine_with_tmp_ledger

        trades = [
            {"symbol": "AMD", "template_id": "MOMENTUM_BREAKOUT", "pnl_pct": 3.0, "won": True},
            {"symbol": "AMD", "template_id": "MOMENTUM_BREAKOUT", "pnl_pct": -1.0, "won": False},
            {"symbol": "AMD", "template_id": "MOMENTUM_BREAKOUT", "pnl_pct": 2.0, "won": True},
        ]
        engine._feed_shadow_ledger(trades)

        ledger = safe_json_read(ledger_path, default={})
        stats = ledger["template_stats"]["AMD"]["MOMENTUM_BREAKOUT"]

        assert stats["total_pnl_pct"] == 4.0, "3.0 + (-1.0) + 2.0 = 4.0"
        assert abs(stats["avg_pnl_pct"] - 1.33) < 0.02, "4.0 / 3 = 1.33"


# ═══════════════════════════════════════════════════════════
# Regression guards
# ═══════════════════════════════════════════════════════════

class TestBacktestShadowFeedRegression:
    """Regression guards."""

    def test_uses_safe_json_io(self):
        """R1: _feed_shadow_ledger uses safe_json_io, not raw json."""
        from backtest_engine import BacktestEngine
        source = inspect.getsource(BacktestEngine._feed_shadow_ledger)
        assert "safe_json_read" in source, "Must use safe_json_read"
        assert "safe_json_write" in source, "Must use safe_json_write"

    def test_run_calls_feed(self):
        """R2: run() method calls _feed_shadow_ledger."""
        from backtest_engine import BacktestEngine
        source = inspect.getsource(BacktestEngine.run)
        assert "_feed_shadow_ledger" in source, \
            "run() must call _feed_shadow_ledger"

    def test_feed_flag_exists(self):
        """R3: feed_shadow_ledger flag exists on BacktestEngine."""
        from backtest_engine import BacktestEngine
        source = inspect.getsource(BacktestEngine.__init__)
        assert "feed_shadow_ledger" in source, \
            "BacktestEngine must have feed_shadow_ledger flag"
