"""
StockWise — Shadow Ledger CLI & Wiring Tests
Validates shadow_ledger.py can be executed standalone and
produces valid template_stats for DDR #1 Asset-Specific.
"""

import inspect
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════
# Unit tests
# ═══════════════════════════════════════════════════════════

class TestShadowLedgerCLI:
    """Tests for Shadow Ledger CLI wiring."""

    def test_main_block_exists(self):
        """T1: shadow_ledger.py has __main__ block."""
        import shadow_ledger
        source = inspect.getsource(shadow_ledger)
        assert 'if __name__ == "__main__"' in source, \
            "shadow_ledger.py must have __main__ block for CLI execution"

    def test_print_summary_exists(self):
        """T2: _print_summary function exists."""
        from shadow_ledger import _print_summary
        assert callable(_print_summary)

    def test_evaluate_history_populates_stats(self):
        """T3: evaluate_history produces per-symbol template_stats."""
        import numpy as np
        import pandas as pd
        from shadow_ledger import ShadowLedger

        sl = ShadowLedger()

        np.random.seed(42)
        n = 300
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        volume = np.random.randint(500000, 5000000, n)

        df = pd.DataFrame({
            'open': close - np.random.randn(n) * 0.1,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
            'rsi': 30 + np.random.rand(n) * 40,
            'macd': np.random.randn(n) * 0.5,
            'macd_signal': np.random.randn(n) * 0.3,
            'macd_hist': np.random.randn(n) * 0.2,
            'sma_50': close - np.random.randn(n) * 2,
            'sma_200': close - 5,
            'ema_12': close - np.random.randn(n) * 0.5,
            'atr': np.full(n, 1.5),
            'bb_width': np.full(n, 5.0),
            'bb_width_pct': np.full(n, 0.15),
            'vol_avg_20': volume * 0.8,
            'squeeze_on': np.zeros(n),
            'mom_sqz': np.random.randn(n) * 0.1,
        })

        result = sl.evaluate_history("TEST_STOCK", df)
        assert isinstance(result, dict), "evaluate_history should return a dict"

        all_stats = sl.ledger.get("template_stats", {})
        assert "TEST_STOCK" in all_stats, \
            "template_stats should contain TEST_STOCK entry"

    def test_stats_have_required_fields(self):
        """T4: per-symbol template stats have required fields."""
        import numpy as np
        import pandas as pd
        from shadow_ledger import ShadowLedger

        sl = ShadowLedger()

        np.random.seed(123)
        n = 300
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)

        df = pd.DataFrame({
            'open': close - 0.1,
            'high': close + 1,
            'low': close - 1,
            'close': close,
            'volume': np.random.randint(1000000, 5000000, n),
            'rsi': 50 + np.random.randn(n) * 10,
            'macd': np.random.randn(n) * 0.5,
            'macd_signal': np.random.randn(n) * 0.3,
            'macd_hist': np.random.randn(n) * 0.2,
            'sma_50': close - 2,
            'sma_200': close - 5,
            'ema_12': close - 0.5,
            'atr': np.full(n, 1.5),
            'bb_width': np.full(n, 5.0),
            'bb_width_pct': np.full(n, 0.15),
            'vol_avg_20': np.full(n, 2000000),
            'squeeze_on': np.zeros(n),
            'mom_sqz': np.random.randn(n) * 0.1,
        })

        sl.evaluate_history("FIELD_TEST", df)
        sym_stats = sl.ledger.get("template_stats", {}).get("FIELD_TEST", {})

        required_fields = ["signal_count", "wins", "losses", "win_rate", "avg_pnl_pct"]
        for tid, stats in sym_stats.items():
            for field in required_fields:
                assert field in stats, \
                    f"Template {tid} missing field '{field}'"

    def test_empty_df_no_crash(self):
        """T5: Empty DataFrame → graceful skip, no crash."""
        import pandas as pd
        from shadow_ledger import ShadowLedger

        sl = ShadowLedger()
        result = sl.evaluate_history("EMPTY_STOCK", pd.DataFrame())
        assert result == {}, "Empty DF should return empty dict"

    def test_none_df_no_crash(self):
        """T6: None DataFrame → graceful skip, no crash."""
        from shadow_ledger import ShadowLedger

        sl = ShadowLedger()
        result = sl.evaluate_history("NONE_STOCK", None)
        assert result == {}, "None DF should return empty dict"

    def test_print_summary_no_crash_empty(self):
        """T7: _print_summary with empty ledger → no crash."""
        from shadow_ledger import ShadowLedger, _print_summary

        sl = ShadowLedger()
        # Should not raise
        _print_summary(sl)


# ═══════════════════════════════════════════════════════════
# Regression guards
# ═══════════════════════════════════════════════════════════

class TestShadowLedgerWiringRegression:
    """Regression guards for Shadow Ledger wiring."""

    def test_ledger_path_matches_config(self):
        """R1: ShadowLedger ledger_path matches ASSET_SPECIFIC_CONFIG."""
        import system_config as cfg
        from shadow_ledger import ShadowLedger

        sl = ShadowLedger()
        asset_path = getattr(cfg, 'ASSET_SPECIFIC_CONFIG', {}).get(
            'shadow_ledger_path', 'data/shadow_ledger.json'
        )
        ledger_path = getattr(cfg, 'SHADOW_LEDGER_CONFIG', {}).get(
            'ledger_path', 'data/shadow_ledger.json'
        )
        assert asset_path == ledger_path, \
            f"Path mismatch: ASSET_SPECIFIC={asset_path} vs SHADOW_LEDGER={ledger_path}"

    def test_run_full_evaluation_has_logging(self):
        """R2: run_full_evaluation contains per-symbol logging."""
        import shadow_ledger as sl_module
        source = inspect.getsource(sl_module.ShadowLedger.run_full_evaluation)
        assert "total signals" in source.lower() or "evaluation complete" in source.lower(), \
            "run_full_evaluation must log per-symbol signal counts"
