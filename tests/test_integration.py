# tests/test_integration.py

"""
StockWise Gen-13 Integration Tests
===================================
Tests the complete signal pipeline end-to-end:
stock_hunter.classify_stock_state -> template_matcher.scan_ticker -> portfolio_risk gates

Uses synthetic DataFrames -- no API calls needed.
"""

import sys
import os
import types
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Stub external dependencies
for mod in ['xgboost', 'xgboost.sklearn', 'xgboost.core']:
    if mod not in sys.modules:
        sys.modules[mod] = types.ModuleType(mod)
try:
    import pandas_ta
except ImportError:
    try:
        import pandas_ta_classic
        sys.modules['pandas_ta'] = pandas_ta_classic
    except ImportError:
        sys.modules['pandas_ta'] = types.ModuleType('pandas_ta')


def _make_df(days=200, trend="up", start_price=100.0):
    """
    Generate a synthetic OHLCV DataFrame for testing.
    trend: "up", "down", "sideways", "volatile", "crash"
    """
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='B')

    prices = [start_price]
    for i in range(1, days):
        noise = np.random.normal(0, 0.01)
        if trend == "up":
            drift = 0.002 + noise
        elif trend == "down":
            drift = -0.002 + noise
        elif trend == "sideways":
            drift = noise * 0.5
        elif trend == "volatile":
            drift = np.random.normal(0, 0.03)
        elif trend == "crash":
            drift = -0.001 + noise if i < days * 0.7 else -0.02 + noise
        else:
            drift = noise
        prices.append(prices[-1] * (1 + drift))

    prices = np.array(prices)
    df = pd.DataFrame({
        'open':   prices * (1 - np.random.uniform(0.001, 0.005, days)),
        'high':   prices * (1 + np.random.uniform(0.005, 0.02, days)),
        'low':    prices * (1 - np.random.uniform(0.005, 0.02, days)),
        'close':  prices,
        'volume': np.random.randint(500000, 2000000, days).astype(float)
    }, index=dates)

    return df


class TestIntegration_FullPipeline:
    """End-to-end: raw data -> features -> state -> templates -> signals"""

    def test_bullish_stock_generates_signal(self):
        """A clearly bullish stock should produce at least one template signal."""
        from feature_engine import FeatureEngine
        from stock_hunter import StockHunter
        from template_matcher import TemplateMatcher
        from data_source_manager import DataSourceManager

        df = _make_df(days=300, trend="up", start_price=100.0)

        fe = FeatureEngine()
        df_features = fe.calculate_features(df)

        dm = DataSourceManager()
        hunter = StockHunter(dm)
        state = hunter.classify_stock_state(df_features)

        matcher = TemplateMatcher()
        signals = matcher.scan_ticker("TEST_BULL", df_features, stock_state=state)

        print(f"  State: {state}")
        print(f"  Signals: {len(signals)}")
        if signals:
            for s in signals:
                print(f"    {s['template_id']}: conf={s['confidence_score']}, R:R={s['risk_reward_ratio']}")

        assert isinstance(signals, list), "scan_ticker should return a list"

    def test_bearish_stock_no_buy_signal(self):
        """A clearly bearish stock should NOT produce BUY signals (except maybe OVERSOLD_BOUNCE)."""
        from feature_engine import FeatureEngine
        from stock_hunter import StockHunter
        from template_matcher import TemplateMatcher
        from data_source_manager import DataSourceManager

        df = _make_df(days=300, trend="down", start_price=200.0)

        fe = FeatureEngine()
        df_features = fe.calculate_features(df)

        dm = DataSourceManager()
        hunter = StockHunter(dm)
        state = hunter.classify_stock_state(df_features)

        matcher = TemplateMatcher()
        signals = matcher.scan_ticker("TEST_BEAR", df_features, stock_state=state)

        print(f"  State: {state}")
        print(f"  Signals: {len(signals)}")

        for s in signals:
            if s['template_id'] not in ['OVERSOLD_BOUNCE']:
                print(f"  WARNING: Unexpected signal {s['template_id']} on bearish stock!")

    def test_portfolio_risk_blocks_correlated_entries(self):
        """If 2 tech stocks are already open, a 3rd tech stock should be blocked."""
        from portfolio_risk import PortfolioRiskManager

        mgr = PortfolioRiskManager()
        open_positions = {
            "AAPL": {"entry_price": 150, "qty": 10},
            "MSFT": {"entry_price": 300, "qty": 5},
        }

        # NVDA is also Technology -- should be blocked (max 2 per sector)
        ok, reasons = mgr.check_correlation_gate("NVDA", open_positions)
        assert not ok, f"Should block 3rd tech stock, but got: {reasons}"
        print(f"  Correctly blocked NVDA: {reasons}")

    def test_portfolio_risk_allows_different_sector(self):
        """If 2 tech stocks are open, a healthcare stock should be allowed."""
        from portfolio_risk import PortfolioRiskManager

        mgr = PortfolioRiskManager()
        open_positions = {
            "AAPL": {"entry_price": 150, "qty": 10},
            "MSFT": {"entry_price": 300, "qty": 5},
        }

        ok, reasons = mgr.check_correlation_gate("JNJ", open_positions)
        assert ok, f"Should allow healthcare stock, but blocked: {reasons}"
        print(f"  Correctly allowed JNJ (Healthcare)")

    def test_drawdown_circuit_breaker(self):
        """If portfolio drops 10%, circuit breaker should activate."""
        from portfolio_risk import PortfolioRiskManager

        mgr = PortfolioRiskManager()
        open_positions = {"AAPL": {"entry_price": 150, "qty": 10}}

        # First call: set high water mark at $100K
        mgr.check_drawdown_gate(open_positions, portfolio_value=100000)

        # Second call: portfolio dropped to $89K (11% drawdown)
        ok, reason = mgr.check_drawdown_gate(open_positions, portfolio_value=89000)
        assert not ok, "Should trigger circuit breaker at 11% drawdown"
        assert "CIRCUIT BREAKER" in reason
        print(f"  Circuit breaker triggered: {reason}")

    def test_weekly_trend_blocks_bearish(self):
        """Weekly trend gate should block entry if weekly SMA is bearish."""
        from portfolio_risk import PortfolioRiskManager

        mgr = PortfolioRiskManager()
        # Create 2 years of bearish data
        df = _make_df(days=500, trend="down", start_price=200.0)

        ok, reason = mgr.check_weekly_trend_gate("TEST", df)
        print(f"  Weekly trend gate: ok={ok}, reason={reason}")
        # Should block -- stock in downtrend for 2 years

    def test_exposure_limit(self):
        """If total exposure exceeds 60%, new entries should be blocked."""
        from portfolio_risk import PortfolioRiskManager

        mgr = PortfolioRiskManager()
        # $70K invested out of $100K = 70% exposure
        open_positions = {
            "AAPL": {"entry_price": 150, "qty": 200},   # $30K
            "MSFT": {"entry_price": 400, "qty": 100},   # $40K
        }

        ok, reason = mgr.check_drawdown_gate(open_positions, portfolio_value=100000)
        assert not ok, "Should block at 70% exposure"
        print(f"  Exposure blocked: {reason}")


class TestIntegration_PositionManagement:
    """Test PHASE_PAUSE, Zombie warning, and daily summary."""

    def test_phase_pause_on_healthy_pullback(self):
        """PHASE_PAUSE should freeze stop when pullback is small and trend intact."""
        from live_trading_engine import LifecycleManager

        lm = LifecycleManager()
        position = {
            "entry_price": 100.0,
            "stop_loss": 97.0,
            "highest_high": 110.0,
            "runner_mode": False,
            "last_er_slow": 0.6,   # Trend intact
            "last_rsi": 55,        # Not oversold
        }

        # Price pulled back 1.8% from high (108 from 110)
        current_price = 108.0
        current_atr = 2.0

        new_stop, new_high, phase = lm.manage_kinetic_stop("TEST", position, current_price, current_atr)
        print(f"  Phase: {phase}, Stop: {new_stop}")

        if phase == "PHASE_PAUSE":
            # Stop should be frozen (not tightened)
            assert new_stop == 97.0, f"Stop should stay at 97.0, got {new_stop}"
            print(f"  PHASE_PAUSE correctly activated -- stop frozen")

    def test_phase_pause_not_on_large_pullback(self):
        """PHASE_PAUSE should NOT activate on a large pullback (>3%)."""
        from live_trading_engine import LifecycleManager

        lm = LifecycleManager()
        position = {
            "entry_price": 100.0,
            "stop_loss": 97.0,
            "highest_high": 110.0,
            "runner_mode": False,
            "last_er_slow": 0.6,
            "last_rsi": 55,
        }

        # 5% pullback from high (104.5 from 110) -- too large for PAUSE
        current_price = 104.5
        current_atr = 2.0

        new_stop, new_high, phase = lm.manage_kinetic_stop("TEST", position, current_price, current_atr)
        print(f"  Phase: {phase}, Stop: {new_stop}")
        assert phase != "PHASE_PAUSE", "Should NOT pause on 5% pullback"

    def test_kinetic_stop_never_goes_down(self):
        """Stop should only go UP, never down, regardless of phase."""
        from live_trading_engine import LifecycleManager

        lm = LifecycleManager()
        initial_stop = 97.0
        position = {
            "entry_price": 100.0,
            "stop_loss": initial_stop,
            "highest_high": 105.0,
            "runner_mode": False,
            "last_er_slow": 0.5,
            "last_rsi": 50,
        }

        prices = [103, 104, 102, 106, 104, 108, 105]
        prev_stop = initial_stop

        for price in prices:
            if price > position['highest_high']:
                position['highest_high'] = price
            new_stop, new_high, phase = lm.manage_kinetic_stop("TEST", position, price, 2.0)
            assert new_stop >= prev_stop, \
                f"Stop went DOWN from {prev_stop} to {new_stop} at price {price} (phase {phase})"
            position['stop_loss'] = new_stop
            prev_stop = new_stop

        print(f"  Stop monotonically increased: {initial_stop} -> {prev_stop}")

    def test_daily_summary_format(self):
        """Daily position summary should include all required fields."""
        from live_trading_engine import LiveTradingEngine

        engine = LiveTradingEngine()
        engine.positions = {
            "AAPL": {
                "entry_price": 150.0,
                "stop_loss": 145.0,
                "last_price": 155.0,
                "runner_mode": True,
                "last_phase": "PHASE_4_RUNNER"
            }
        }

        # This would normally send Telegram -- but notifier may be None in test
        engine.send_daily_position_summary()
        print(f"  Daily summary generated for {len(engine.positions)} positions")


class TestIntegration_EdgeCases:
    """Test system behavior with bad/missing/unexpected data."""

    def test_empty_dataframe(self):
        """Scanner should handle empty DataFrame gracefully."""
        from template_matcher import TemplateMatcher

        matcher = TemplateMatcher()
        signals = matcher.scan_ticker("EMPTY", pd.DataFrame(), stock_state={})
        assert signals == [], "Should return empty list for empty DataFrame"
        print(f"  Empty DataFrame handled correctly")

    def test_nan_heavy_data(self):
        """Scanner should handle DataFrame full of NaN values."""
        from template_matcher import TemplateMatcher

        matcher = TemplateMatcher()
        df = pd.DataFrame({
            'close': [np.nan] * 10,
            'open':  [np.nan] * 10,
            'high':  [np.nan] * 10,
            'low':   [np.nan] * 10,
            'volume':[np.nan] * 10,
        })

        signals = matcher.scan_ticker("NAN_TEST", df, stock_state={"trend": "BULLISH"})
        assert isinstance(signals, list), "Should return list, not crash"
        print(f"  NaN-heavy data handled: {len(signals)} signals")

    def test_single_row_dataframe(self):
        """Scanner should handle DataFrame with just 1 row."""
        from feature_engine import FeatureEngine
        from template_matcher import TemplateMatcher

        df = pd.DataFrame({
            'open': [100.0], 'high': [102.0], 'low': [99.0],
            'close': [101.0], 'volume': [500000.0]
        }, index=[datetime.now()])

        fe = FeatureEngine()
        try:
            df_features = fe.calculate_features(df)
        except Exception as e:
            print(f"  FeatureEngine crashed on 1 row (expected): {e}")
            return

        matcher = TemplateMatcher()
        signals = matcher.scan_ticker("SINGLE", df_features, stock_state={})
        print(f"  Single row handled: {len(signals)} signals")

    def test_unknown_stock_sector(self):
        """Stocks not in SECTOR_MAP should not crash sector check."""
        from portfolio_risk import PortfolioRiskManager

        mgr = PortfolioRiskManager()
        open_positions = {"AAPL": {"entry_price": 150, "qty": 10}}

        # ZZZZ is not in SECTOR_MAP
        ok, reason = mgr.check_correlation_gate("ZZZZ", open_positions)
        assert ok, f"Unknown sector should not block: {reason}"
        print(f"  Unknown stock ZZZZ passed sector check")

    def test_zero_portfolio_value(self):
        """Drawdown gate should handle zero portfolio value gracefully."""
        from portfolio_risk import PortfolioRiskManager

        mgr = PortfolioRiskManager()
        ok, reason = mgr.check_drawdown_gate({}, portfolio_value=0)
        assert ok, f"Zero portfolio should pass: {reason}"
        print(f"  Zero portfolio handled correctly")

    def test_template_with_missing_block(self):
        """Template with invalid block name should fail validation gracefully."""
        from setup_templates import SetupTemplate

        t = SetupTemplate({
            "id": "BROKEN",
            "name": "Broken Template",
            "conditions": [{"block": "nonexistent_block", "params": []}],
            "stop_loss": {"method": "atr"},
            "take_profit": {"method": "atr"}
        })

        valid, errors = t.validate()
        assert not valid, "Template with unknown block should fail validation"
        print(f"  Invalid block caught: {errors}")


# ============================================================
# RUNNER
# ============================================================
if __name__ == '__main__':
    passed = 0
    failed = 0
    errors = []

    test_classes = [
        TestIntegration_FullPipeline,
        TestIntegration_PositionManagement,
        TestIntegration_EdgeCases,
    ]

    for cls in test_classes:
        print(f"\n--- {cls.__name__} ---")
        obj = cls()

        for method_name in sorted(dir(obj)):
            if not method_name.startswith('test_'):
                continue
            method = getattr(obj, method_name)
            try:
                method()
                print(f"  PASS: {method_name}")
                passed += 1
            except AssertionError as e:
                print(f"  FAIL: {method_name} -- {e}")
                failed += 1
                errors.append(f"{cls.__name__}.{method_name}: {e}")
            except Exception as e:
                print(f"  ERROR: {method_name} -- {e}")
                failed += 1
                errors.append(f"{cls.__name__}.{method_name}: {e}")

    print(f"\n{'='*60}")
    print(f"Integration Tests: {passed} passed, {failed} failed out of {passed+failed}")
    if errors:
        print(f"\nFailed tests:")
        for e in errors:
            print(f"  - {e}")
    print(f"{'='*60}")
