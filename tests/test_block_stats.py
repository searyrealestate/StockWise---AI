"""
StockWise — Block-Level Statistics Tests
Validates per-block pass/fail/blocker tracking in templates.
Ref: P1 #7A, SPEC v13.4 §4
"""

import inspect
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from setup_templates import SetupTemplate


def _make_details(blocks_passed):
    """Helper: create evaluate_conditions details.
    blocks_passed: dict of {block_name: bool}
    """
    return [
        {"block": name, "params": [], "passed": passed}
        for name, passed in blocks_passed.items()
    ]


def _make_template():
    """Create a minimal template for testing."""
    data = {
        "id": "TEST_BLOCK_STATS",
        "name": "Test Block Stats",
        "description": "Test",
        "version": 1,
        "source": "test",
        "enabled": True,
        "required_state": {},
        "conditions": [
            {"block": "rsi_between", "params": [40, 65]},
            {"block": "close_above_sma", "params": [50]},
            {"block": "volume_surge", "params": [1.2]},
        ],
        "entry": {"type": "close", "confirmation_candles": 0},
        "stop_loss": {"method": "atr", "atr_multiplier": 2.0, "fallback_pct": 0.02},
        "take_profit": {"method": "atr", "atr_multiplier": 3.0, "use_runner_mode": False},
    }
    return SetupTemplate(data)


# ═══════════════════════════════════════════════════════════
# Level 1: evaluated/passed/failed/pass_rate/blocker
# ═══════════════════════════════════════════════════════════

class TestBlockStatsBasic:
    """Level 1: evaluated/passed/failed/pass_rate/blocker."""

    def test_increments_evaluated_and_passed(self):
        """T1: Block that passes → evaluated+1, passed+1."""
        t = _make_template()
        details = _make_details({"rsi_between": True, "close_above_sma": True, "volume_surge": True})
        t.record_block_results(details, all_passed=True)

        bs = t.statistics["block_stats"]["rsi_between"]
        assert bs["evaluated"] == 1
        assert bs["passed"] == 1
        assert bs["failed"] == 0
        assert bs["pass_rate"] == 100.0

    def test_increments_failed(self):
        """T2: Block that fails → failed+1."""
        t = _make_template()
        details = _make_details({"rsi_between": False, "close_above_sma": True, "volume_surge": True})
        t.record_block_results(details, all_passed=False)

        bs = t.statistics["block_stats"]["rsi_between"]
        assert bs["evaluated"] == 1
        assert bs["passed"] == 0
        assert bs["failed"] == 1
        assert bs["pass_rate"] == 0.0

    def test_single_blocker_detected(self):
        """T3: When exactly one block fails → it's marked as blocker."""
        t = _make_template()
        details = _make_details({"rsi_between": True, "close_above_sma": True, "volume_surge": False})
        t.record_block_results(details, all_passed=False)

        assert t.statistics["block_stats"]["volume_surge"]["was_the_blocker"] == 1
        assert t.statistics["block_stats"]["rsi_between"]["was_the_blocker"] == 0

    def test_multiple_failures_no_single_blocker(self):
        """T4: When 2+ blocks fail → no single blocker assigned."""
        t = _make_template()
        details = _make_details({"rsi_between": False, "close_above_sma": False, "volume_surge": True})
        t.record_block_results(details, all_passed=False)

        assert t.statistics["block_stats"]["rsi_between"]["was_the_blocker"] == 0
        assert t.statistics["block_stats"]["close_above_sma"]["was_the_blocker"] == 0

    def test_additive_across_calls(self):
        """T5: Multiple calls accumulate counts."""
        t = _make_template()
        for _ in range(5):
            details = _make_details({"rsi_between": True, "close_above_sma": True, "volume_surge": True})
            t.record_block_results(details, all_passed=True)
        for _ in range(3):
            details = _make_details({"rsi_between": False, "close_above_sma": True, "volume_surge": True})
            t.record_block_results(details, all_passed=False)

        bs = t.statistics["block_stats"]["rsi_between"]
        assert bs["evaluated"] == 8
        assert bs["passed"] == 5
        assert bs["failed"] == 3
        assert bs["pass_rate"] == 62.5

    def test_empty_details_no_crash(self):
        """T6: Empty details → no crash, no stats."""
        t = _make_template()
        t.record_block_results([], all_passed=False)
        assert len(t.statistics.get("block_stats", {})) == 0


# ═══════════════════════════════════════════════════════════
# Level 2: outcome correlation
# ═══════════════════════════════════════════════════════════

class TestBlockStatsOutcome:
    """Level 2: outcome correlation when template passes."""

    def test_win_recorded(self):
        """T7: When all pass + target hit → wins+1, WR updated."""
        t = _make_template()
        details = _make_details({"rsi_between": True, "close_above_sma": True, "volume_surge": True})
        outcome = {"hit": "target", "pnl_pct": 2.5}
        t.record_block_results(details, all_passed=True, outcome=outcome)

        wp = t.statistics["block_stats"]["rsi_between"]["when_passed"]
        assert wp["total_trades"] == 1
        assert wp["wins"] == 1
        assert wp["wr"] == 100.0
        assert wp["avg_pnl"] == 2.5

    def test_loss_recorded(self):
        """T8: When all pass + stop hit → losses+1."""
        t = _make_template()
        details = _make_details({"rsi_between": True, "close_above_sma": True, "volume_surge": True})
        outcome = {"hit": "stop", "pnl_pct": -1.5}
        t.record_block_results(details, all_passed=True, outcome=outcome)

        wp = t.statistics["block_stats"]["rsi_between"]["when_passed"]
        assert wp["total_trades"] == 1
        assert wp["losses"] == 1
        assert wp["wr"] == 0.0

    def test_neither_not_counted(self):
        """T9: Outcome 'neither' → not counted as trade."""
        t = _make_template()
        details = _make_details({"rsi_between": True, "close_above_sma": True, "volume_surge": True})
        outcome = {"hit": "neither", "pnl_pct": 0.0}
        t.record_block_results(details, all_passed=True, outcome=outcome)

        wp = t.statistics["block_stats"]["rsi_between"]["when_passed"]
        assert wp["total_trades"] == 0

    def test_no_outcome_when_failed(self):
        """T10: Failed template → outcome not recorded even if provided."""
        t = _make_template()
        details = _make_details({"rsi_between": False, "close_above_sma": True, "volume_surge": True})
        outcome = {"hit": "target", "pnl_pct": 5.0}
        t.record_block_results(details, all_passed=False, outcome=outcome)

        wp = t.statistics["block_stats"]["rsi_between"]["when_passed"]
        assert wp["total_trades"] == 0


# ═══════════════════════════════════════════════════════════
# Level 3: per-symbol
# ═══════════════════════════════════════════════════════════

class TestBlockStatsPerSymbol:
    """Level 3: per-symbol breakdown."""

    def test_per_symbol_tracked(self):
        """T11: Per-symbol evaluated/passed tracked correctly."""
        t = _make_template()
        details = _make_details({"rsi_between": True, "close_above_sma": True, "volume_surge": True})
        t.record_block_results(details, symbol="AAPL", all_passed=True)
        t.record_block_results(details, symbol="AAPL", all_passed=True)

        details2 = _make_details({"rsi_between": False, "close_above_sma": True, "volume_surge": True})
        t.record_block_results(details2, symbol="TSLA", all_passed=False)

        ps_aapl = t.statistics["block_stats"]["rsi_between"]["per_symbol"]["AAPL"]
        assert ps_aapl["evaluated"] == 2
        assert ps_aapl["passed"] == 2
        assert ps_aapl["pass_rate"] == 100.0

        ps_tsla = t.statistics["block_stats"]["rsi_between"]["per_symbol"]["TSLA"]
        assert ps_tsla["evaluated"] == 1
        assert ps_tsla["passed"] == 0
        assert ps_tsla["pass_rate"] == 0.0

    def test_per_symbol_wr_when_passed(self):
        """T12: Per-symbol WR tracked when block passes + trade completes."""
        t = _make_template()
        details = _make_details({"rsi_between": True, "close_above_sma": True, "volume_surge": True})

        t.record_block_results(details, symbol="AAPL", all_passed=True,
                               outcome={"hit": "target", "pnl_pct": 2.0})
        t.record_block_results(details, symbol="AAPL", all_passed=True,
                               outcome={"hit": "stop", "pnl_pct": -1.0})

        ps = t.statistics["block_stats"]["rsi_between"]["per_symbol"]["AAPL"]
        assert ps["trades_when_passed"] == 2
        assert ps["wins_when_passed"] == 1
        assert ps["wr_when_passed"] == 50.0

    def test_no_symbol_no_per_symbol(self):
        """T13: No symbol provided → per_symbol not populated."""
        t = _make_template()
        details = _make_details({"rsi_between": True, "close_above_sma": True, "volume_surge": True})
        t.record_block_results(details, symbol="", all_passed=True)

        assert t.statistics["block_stats"]["rsi_between"]["per_symbol"] == {}


# ═══════════════════════════════════════════════════════════
# Regression guards
# ═══════════════════════════════════════════════════════════

class TestBlockStatsRegression:
    """Regression guards."""

    def test_block_stats_in_empty_stats(self):
        """R1: block_stats field exists in _empty_stats."""
        t = _make_template()
        assert "block_stats" in t.statistics

    def test_record_block_results_method_exists(self):
        """R2: record_block_results method exists on SetupTemplate."""
        assert hasattr(SetupTemplate, "record_block_results")
        assert callable(getattr(SetupTemplate, "record_block_results"))

    def test_to_dict_preserves_block_stats(self):
        """R3: to_dict includes block_stats in statistics."""
        t = _make_template()
        details = _make_details({"rsi_between": True, "close_above_sma": True, "volume_surge": True})
        t.record_block_results(details, all_passed=True,
                               outcome={"hit": "target", "pnl_pct": 1.0})

        d = t.to_dict()
        assert "block_stats" in d["statistics"]
        assert "rsi_between" in d["statistics"]["block_stats"]
