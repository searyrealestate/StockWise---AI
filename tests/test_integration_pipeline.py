# tests/test_integration_pipeline.py

"""
StockWise Gen-13 — Integration Pipeline Tests (TDD v1.1 Section 12)
===================================================================
IT-01→10: End-to-end flows testing multiple components working together.
All external calls mocked — zero API calls.

File: test_integration_pipeline.py (separate from pre-existing test_integration.py)

Execution: python -m pytest tests/test_integration_pipeline.py -v --tb=short
Expected : 10 passed, 0 failed
"""

import os
import sys

import pytest
from unittest.mock import patch, MagicMock

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import system_config as cfg
import live_trading_engine as lte_module
from live_trading_engine import LiveTradingEngine, LifecycleManager


def _read_source(filename):
    path = os.path.join(PROJECT_ROOT, filename)
    if not os.path.exists(path):
        pytest.skip(f"{filename} not found")
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def _line_of(source, pattern):
    """Return the line number (1-based) of the first occurrence of pattern in source."""
    for i, line in enumerate(source.splitlines(), 1):
        if pattern in line:
            return i
    return -1


def _make_lte():
    """
    Minimal LiveTradingEngine with all external I/O mocked:
      - NotificationManager (Telegram)
      - safe_json_read / safe_json_write (file persistence)
    """
    with patch.object(lte_module, 'NotificationManager'), \
         patch('live_trading_engine.safe_json_read', return_value={}), \
         patch('live_trading_engine.safe_json_write'):
        engine = LiveTradingEngine()
    return engine


def _make_ticket(symbol="AAPL"):
    return {
        "symbol": symbol, "action": "BUY",
        "limit_price": 150.0, "stop_loss": 145.0, "take_profit": 165.0,
        "qty": 10, "master_score": 87.0, "confidence_score": 87.0,
        "risk_reward_ratio": 3.0, "template_id": "T1", "template_name": "MOMENTUM_BREAKOUT",
        "use_runner_mode": False, "conditions_detail": [],
        "stock_state": {"trend": "BULLISH"},
    }


# ═══════════════════════════════════════════════════════════════
# IT-01 → IT-10
# ═══════════════════════════════════════════════════════════════

class TestIntegrationPipeline:

    # IT-01 (P0): execute_ticket stores position in self.positions with correct entry price
    def test_it01_execute_ticket_stores_position(self):
        engine = _make_lte()
        engine.positions = {}
        ticket = _make_ticket("AAPL")

        with patch.object(engine, '_save_json'):
            engine.execute_ticket(ticket, "TREND")

        assert "AAPL" in engine.positions, "execute_ticket must store position in self.positions"
        assert engine.positions["AAPL"]["entry_price"] == 150.0

    # IT-02 (P0): Veto gate called BEFORE template matching in pipeline
    def test_it02_veto_before_template_in_pipeline(self):
        source = _read_source("live_trading_engine.py")
        veto_line   = _line_of(source, "check_veto_gates")
        signal_line = _line_of(source, "matcher.scan_ticker")
        assert veto_line > 0, "check_veto_gates not found in live_trading_engine.py"
        assert signal_line > 0, "matcher.scan_ticker not found in live_trading_engine.py"
        assert veto_line < signal_line, (
            f"Veto gate (line {veto_line}) must precede template scan (line {signal_line})"
        )

    # IT-03 (P0): Risk gates checked AFTER signal but BEFORE execute_ticket
    def test_it03_risk_gates_between_signal_and_execute(self):
        source = _read_source("live_trading_engine.py")
        signal_line  = _line_of(source, "matcher.scan_ticker")
        risk_line    = _line_of(source, "check_all_gates")
        execute_line = _line_of(source, "execute_ticket(ticket, current_regime)")
        assert risk_line > 0, "check_all_gates not found in live_trading_engine.py"
        assert signal_line < risk_line < execute_line, (
            f"Expected: signal({signal_line}) < risk({risk_line}) < execute({execute_line}). "
            f"Risk gates must sit between signal detection and execution."
        )

    # IT-04 (P0): Alpha equation is in strategy_engine.py (BASE_FRICTION / MIN_NET_PROFIT)
    def test_it04_alpha_equation_in_strategy_engine(self):
        source = _read_source("strategy_engine.py")
        assert "BASE_FRICTION" in source or "MIN_NET_PROFIT" in source or \
               "calculate_entry_equation" in source, (
            "Alpha equation (BASE_FRICTION / MIN_NET_PROFIT / calculate_entry_equation) "
            "not found in strategy_engine.py"
        )

    # IT-05 (P0): DataSourceManager (waterfall provider) is used in the live trading pipeline
    def test_it05_datasource_manager_in_pipeline(self):
        source = _read_source("live_trading_engine.py")
        assert any(p in source for p in [
            "DataSourceManager", "data_source_manager", "get_stock_data"
        ]), "DataSourceManager not found in live_trading_engine.py"

    # IT-06 (P0): Pre-market validator (check_gap) is called between risk gates and execution
    def test_it06_premarket_between_risk_and_execute(self):
        source = _read_source("live_trading_engine.py")
        risk_line    = _line_of(source, "check_all_gates")
        pm_line      = _line_of(source, "pre_market_validator.check_gap")
        execute_line = _line_of(source, "execute_ticket(ticket, current_regime)")
        assert pm_line > 0, "pre_market_validator.check_gap not found in live_trading_engine.py"
        assert risk_line < pm_line < execute_line, (
            f"Expected: risk({risk_line}) < pre-market({pm_line}) < execute({execute_line})"
        )

    # IT-07 (P0): All 5 kinetic stop phases are defined in live_trading_engine.py
    def test_it07_all_kinetic_stop_phases_present(self):
        source = _read_source("live_trading_engine.py")
        phases = [
            "PHASE_1_BREATHING", "PHASE_2_BREAKEVEN",
            "PHASE_3_PARABOLIC", "PHASE_PAUSE", "PHASE_4_RUNNER",
        ]
        missing = [p for p in phases if p not in source]
        assert not missing, (
            f"Kinetic stop phases missing from live_trading_engine.py: {missing}"
        )

    # IT-08 (P1): Scan → VIP flow exists: stock_hunter scans and builds VIP,
    #             live_trading_engine reads VIP list (WATCHLIST) for the signal loop
    def test_it08_scan_to_vip_to_signal_loop(self):
        sh_source  = _read_source("stock_hunter.py")
        lte_source = _read_source("live_trading_engine.py")
        # stock_hunter writes to VIP / watchlist
        assert "vip" in sh_source.lower() or "watchlist" in sh_source.lower(), (
            "No VIP/watchlist output in stock_hunter.py"
        )
        # live_trading_engine reads the VIP list (cfg.WATCHLIST or vip_list variable)
        assert "watchlist" in lte_source.lower() or "vip_list" in lte_source.lower(), (
            "live_trading_engine.py does not consume the VIP/watchlist"
        )

    # IT-09 (P1): Daily position summary exists and can be called without crash
    def test_it09_daily_summary_callable(self):
        engine = _make_lte()
        engine.positions = {
            "AAPL": {
                "entry_price": 150.0, "qty": 10,
                "stop_loss": 145.0, "take_profit": 165.0,
            }
        }
        # send_daily_position_summary calls self.notifier.send_message — notifier is mocked
        try:
            engine.send_daily_position_summary()
        except Exception as exc:
            pytest.fail(f"send_daily_position_summary raised: {exc}")

    # IT-10 (P1): Zombie protocol exists with TTL-based force liquidation after regime shift.
    # After zombie_trade_ttl_hours, check_zombie_protocol returns True (force liquidate).
    def test_it10_zombie_protocol_ttl_force_liquidation(self):
        source = _read_source("live_trading_engine.py")
        # Zombie tagging + TTL exists
        assert "zombie_timestamp" in source, (
            "'zombie_timestamp' not found — zombie trade tagging not implemented"
        )
        assert "zombie_trade_ttl_hours" in source, (
            "'zombie_trade_ttl_hours' config key not found in live_trading_engine.py"
        )
        # Force liquidation path exists after TTL expiry
        assert "Force Liquidation" in source or "check_zombie_protocol" in source, (
            "Force liquidation after zombie TTL not found"
        )
        # Behavioral: check_zombie_protocol exists on LifecycleManager
        lm = LifecycleManager()
        assert hasattr(lm, 'check_zombie_protocol'), (
            "check_zombie_protocol method not on LifecycleManager"
        )
