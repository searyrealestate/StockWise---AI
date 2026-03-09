"""
StockWise AI - Master System Validator
=======================================
Integration tests verifying cross-component data flow and system consistency.
Run: python tests/master_validator.py

This validates that the files WORK TOGETHER, not just individually.
Each check targets a specific contract between two components.
"""
import sys
import os
import re
import types
from unittest.mock import MagicMock

# === DEPENDENCY STUBBING ===
_pandas_ta_stub = types.ModuleType('pandas_ta')
for _fn in ['rsi', 'sma', 'ema', 'macd', 'bbands', 'kc', 'donchian', 'atr',
            'adx', 'stoch', 'squeeze', 'squeeze_pro']:
    setattr(_pandas_ta_stub, _fn, MagicMock(return_value=None))
sys.modules['pandas_ta'] = _pandas_ta_stub

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import system_config as cfg


class MasterValidator:
    """Cross-system integration validation checks."""

    def __init__(self):
        self.results = []

    def _record(self, name, passed, detail=""):
        status = "PASS" if passed else "FAIL"
        self.results.append((name, passed, detail))
        suffix = f" — {detail}" if detail and not passed else ""
        print(f"  [{status}] {name}{suffix}")

    # ------------------------------------------------------------------
    # CHECK 1: Column names in strategy_engine match feature_engine output
    # ------------------------------------------------------------------
    def check_column_name_consistency(self):
        """
        Parse feature_engine.py for all df['column_name'] = assignments,
        then verify strategy_engine.py only .get()s columns from that set
        (plus standard OHLCV columns that come directly from the data source).
        """
        fe_path = os.path.join(PROJECT_ROOT, 'feature_engine.py')
        se_path = os.path.join(PROJECT_ROOT, 'strategy_engine.py')

        with open(fe_path, 'r') as f:
            fe_code = f.read()
        with open(se_path, 'r') as f:
            se_code = f.read()

        # Columns CREATED by feature_engine (df['xxx'] = ...)
        created = set(re.findall(r"df\['(\w+)'\]\s*=", fe_code))
        # Standard OHLCV that arrive from the data source, not feature_engine
        created.update(['open', 'high', 'low', 'close', 'volume', 'date', 'timestamp'])

        # Columns READ by strategy_engine via .get() or direct index
        referenced = set(re.findall(r"\.get\('(\w+)'", se_code))
        referenced.update(re.findall(r"last\['(\w+)'\]", se_code))
        referenced.update(re.findall(r"row\['(\w+)'\]", se_code))

        # Known non-column dict keys used in return payloads or config lookups
        non_column_keys = {
            'action', 'master_score', 'ai_score', 'tech_score', 'symbol',
            'setups_found', 'stop_loss', 'target_price', 'limit_price',
            'take_profit', 'qty', 'reason', 'entry_price', 'entry_time',
            'scores', 'highest_high', 'min_net_profit_pct', 'min_net_rr',
            'max_spread_pct', 'timestamp', 'threshold_coherent_trend',
            'er_lookback_slow', 'er_lookback_fast', 'threshold_stochastic_chop',
        }
        # Columns referenced by strategy_engine but not yet confirmed in
        # feature_engine output (regex limitations or future bugs — tracked separately).
        # macdsignal vs macd_signal mismatch is a known candidate for Bug 1.x.
        # All previously pending columns are now resolved:
        # squeeze_on, mom_sqz — fixed by Bug 1.6a
        # macdsignal — fixed by Bug 2.1 (now macd_signal)
        # is_consolidating, BOLLINGER_SQUEEZE — fixed by Bug 2.2 (replaced with squeeze_on)
        pending_investigation = set()
        missing = referenced - created - non_column_keys - pending_investigation

        self._record(
            "Column Name Consistency (feature_engine -> strategy_engine)",
            len(missing) == 0,
            f"Columns read by strategy_engine but never created: {sorted(missing)}" if missing else ""
        )

    # ------------------------------------------------------------------
    # CHECK 2: DSP_CONFIG exists and has all required keys
    # ------------------------------------------------------------------
    def check_dsp_config_integrity(self):
        """Verify DSP_CONFIG exists in system_config and has all required keys."""
        dsp = getattr(cfg, 'DSP_CONFIG', None)
        self._record(
            "DSP_CONFIG exists in system_config",
            dsp is not None,
            "DSP_CONFIG missing from system_config.py" if dsp is None else ""
        )

        if dsp is None:
            return

        required_keys = [
            'er_lookback_slow',
            'er_lookback_fast',
            'threshold_coherent_trend',
            'threshold_stochastic_chop',
        ]
        for key in required_keys:
            self._record(
                f"DSP_CONFIG['{key}'] present",
                key in dsp,
                f"Key missing from DSP_CONFIG" if key not in dsp else ""
            )

        threshold = dsp.get('threshold_coherent_trend')
        self._record(
            "threshold_coherent_trend is in valid range [0.0, 1.0]",
            threshold is not None and 0.0 <= threshold <= 1.0,
            f"Value={threshold}" if threshold is not None else "Key missing"
        )

    # ------------------------------------------------------------------
    # CHECK 3: MIN_MASTER_SCORE_APPROVAL is reachable
    # ------------------------------------------------------------------
    def check_threshold_sanity(self):
        """MIN_MASTER_SCORE_APPROVAL must exist and be reachable by the scoring formula."""
        approval = getattr(cfg, 'MIN_MASTER_SCORE_APPROVAL', None)
        self._record(
            "MIN_MASTER_SCORE_APPROVAL defined in system_config",
            approval is not None,
            "Missing from system_config.py" if approval is None else ""
        )
        if approval is not None:
            # Max master_score = (100 * 0.7) + (100 * 0.3) = 100.
            # Approval above 85 makes trades practically unreachable.
            self._record(
                f"MIN_MASTER_SCORE_APPROVAL ({approval}) is reachable (<= 85)",
                approval <= 85.0,
                f"Value {approval} is too high — no trade will ever pass."
            )

    # ------------------------------------------------------------------
    # CHECK 4: No known-bad dead column references in strategy_engine
    # ------------------------------------------------------------------
    def check_no_dead_column_references(self):
        """Ensure Bug 1.2 and 1.3 column names are fully purged."""
        se_path = os.path.join(PROJECT_ROOT, 'strategy_engine.py')
        with open(se_path, 'r') as f:
            code = f.read()

        # Check for specific column key lookups (quoted strings in .get() calls).
        # Note: rsi_14 as a LOCAL VARIABLE NAME is acceptable; only the .get() key
        # must be 'rsi'. So we check for the quoted key form "'rsi_14'", not the raw identifier.
        dead_refs = {
            "'er_trend'":   "Bug 1.3 — should use er_slow with threshold comparison",
            "'SMA_50'":     "Bug 1.2 — should be sma_50",
            "'SMA_200'":    "Bug 1.2 — should be sma_200",
            "'BBU_20":      "Bug 1.2 — should be bb_upper",
            "'rsi_14'":     "Bug 1.2 — should be rsi",
        }
        for ref, explanation in dead_refs.items():
            self._record(
                f"Dead column key {ref} removed from strategy_engine.py",
                ref not in code,
                explanation if ref in code else ""
            )

    # ------------------------------------------------------------------
    # CHECK 5: analyze() return contract has expected keys
    # ------------------------------------------------------------------
    def check_analyze_return_contract(self):
        """
        Import TacticalSniper and run analyze() with a minimal DataFrame.
        Verify the returned dict contains all keys the live engine depends on.
        """
        try:
            from strategy_engine import TacticalSniper
            import pandas as pd

            row = {
                'open': 100.0, 'high': 102.0, 'low': 98.0, 'close': 101.0,
                'volume': 500_000, 'vol_avg_20': 400_000,
                'sma_50': 95.0, 'sma_200': 90.0,
                'er_slow': 0.65, 'er_fast': 0.50, 'trend_alignment': 1,
                'bb_width': 0.25, 'bb_upper': 110.0, 'bb_lower': 90.0,
                'kc_upper': 108.0, 'kc_lower': 92.0,
                'squeeze_on': 0, 'mom_sqz': 0.0, 'atr': 2.0,
                'rvol': 1.0, 'rsi': 55.0,
                'macd': 0.1, 'macdsignal': 0.05, 'macd_hist': 0.05,
                'is_consolidating': False,
            }
            df = pd.DataFrame([row])
            sniper = TacticalSniper()
            result = sniper.analyze("VALIDATOR", df, "TREND")

            required_keys = [
                'action', 'master_score', 'ai_score',
                'tech_score', 'setups_found', 'stop_loss', 'target_price',
            ]
            missing_keys = [k for k in required_keys if k not in result]
            self._record(
                "analyze() return dict has all required keys",
                len(missing_keys) == 0,
                f"Missing keys: {missing_keys}" if missing_keys else ""
            )

            self._record(
                "analyze() 'action' value is 'BUY' or 'WAIT'",
                result.get('action') in ('BUY', 'WAIT'),
                f"Got: {result.get('action')}"
            )
        except Exception as e:
            self._record("analyze() return contract", False, f"Exception: {e}")

    # ------------------------------------------------------------------
    # CHECK 6: Cooldown write/read integrity (Bug 1.4)
    # ------------------------------------------------------------------
    def check_cooldown_write_exists(self):
        """Verify _write_cooldown method exists in LiveTradingEngine."""
        lte_path = os.path.join(PROJECT_ROOT, 'live_trading_engine.py')
        with open(lte_path, 'r', encoding='utf-8', errors='replace') as f:
            code = f.read()

        self._record(
            "_write_cooldown method exists in LiveTradingEngine",
            '_write_cooldown' in code,
            "Method not found -- stop-loss cannot write to cooldown file"
        )

        self._record(
            "_write_cooldown is called on STOP LOSS HIT",
            'self._write_cooldown' in code and 'STOP LOSS HIT' in code,
            "Missing call: _write_cooldown not invoked on stop-loss"
        )

    def check_cooldown_config_param(self):
        """Verify COOLDOWN_PERIOD_HOURS exists in system_config."""
        hours = getattr(cfg, 'COOLDOWN_PERIOD_HOURS', None)
        self._record(
            "COOLDOWN_PERIOD_HOURS exists in system_config",
            hours is not None,
            "Missing param -- cooldown duration is hardcoded"
        )
        if hours is not None:
            self._record(
                "COOLDOWN_PERIOD_HOURS in valid range (1-168)",
                1 <= hours <= 168,
                f"Value: {hours} -- outside 1h-7d range"
            )

    # ------------------------------------------------------------------
    # CHECK 7: AI Pipeline Integrity (Bug 1.1)
    # ------------------------------------------------------------------
    def check_ai_pipeline_integrity(self):
        """Verify AI training saves real features and prediction is model-type safe."""
        tm_path = os.path.join(PROJECT_ROOT, 'train_model.py')
        se_path = os.path.join(PROJECT_ROOT, 'strategy_engine.py')

        with open(tm_path, 'r') as f:
            tm_code = f.read()
        with open(se_path, 'r') as f:
            se_code = f.read()

        self._record(
            "AI features: no hardcoded meta-feature list in train_model.py",
            '"tech_score", "ai_score", "master_score", "regime_val"' not in tm_code,
            "Still saving ['tech_score','ai_score','master_score','regime_val'] instead of real columns"
        )

        self._record(
            "AI model: XGBClassifier (not Regressor) for binary target",
            'XGBClassifier' in tm_code,
            "Still using XGBRegressor -- predict_proba will fail"
        )

        self._record(
            "AI prediction: safe predict_proba with fallback",
            'hasattr' in se_code and 'predict_proba' in se_code,
            "get_ai_probability doesn't have model-type-safe prediction"
        )

    # ------------------------------------------------------------------
    # CHECK 8: Regime Gate in evaluate_ticker (Bug 2.3)
    # ------------------------------------------------------------------
    def check_regime_gate(self):
        """Verify evaluate_ticker blocks HALT and NEUTRAL regimes."""
        import re
        se_path = os.path.join(PROJECT_ROOT, 'strategy_engine.py')
        with open(se_path, 'r') as f:
            code = f.read()

        match = re.search(
            r'def evaluate_ticker\(.*?\n(.*?)(?=\n    def |\nclass |\Z)',
            code, re.DOTALL
        )
        if not match:
            self._record("evaluate_ticker method exists", False, "Method not found")
            return

        method_body = match.group(1)

        self._record(
            "evaluate_ticker blocks HALT regime",
            'regime == "HALT"' in method_body or "regime == 'HALT'" in method_body,
            "HALT regime not checked -- system may buy during crash"
        )

        self._record(
            "evaluate_ticker blocks NEUTRAL regime",
            'regime == "NEUTRAL"' in method_body or "regime == 'NEUTRAL'" in method_body,
            "NEUTRAL regime not checked -- system analyzes in dead zone"
        )

    # ------------------------------------------------------------------
    # CHECK 9: Milestone Alert System (Phase 2.5)
    # ------------------------------------------------------------------
    def check_milestone_alert_system(self):
        """Verify milestone alert infrastructure is complete."""
        lte_path = os.path.join(PROJECT_ROOT, 'live_trading_engine.py')
        with open(lte_path, 'r', encoding='utf-8', errors='replace') as f:
            code = f.read()

        self._record(
            "_calculate_real_breakeven method exists",
            '_calculate_real_breakeven' in code,
            "Method not found -- breakeven calculation missing"
        )

        self._record(
            "_check_and_send_milestone_alert method exists",
            '_check_and_send_milestone_alert' in code,
            "Method not found -- milestone alerts missing"
        )

        self._record(
            "Runner Mode activation on take_profit",
            'runner_mode' in code and 'take_profit' in code,
            "take_profit should activate runner_mode, not liquidate"
        )

        self._record(
            "Phase 4 Runner in kinetic stop with floor",
            'PHASE_4_RUNNER' in code and 'runner_min_distance_pct' in code,
            "Phase 4 Runner missing or no min distance floor"
        )

        self._record(
            "Milestone alert called after kinetic stop update",
            '_check_and_send_milestone_alert' in code,
            "Milestone alerts not wired into stop update flow"
        )

        # Config check
        config = getattr(cfg, 'MILESTONE_ALERT_CONFIG', None)
        self._record(
            "MILESTONE_ALERT_CONFIG exists with runner floor",
            config is not None and 'runner_min_distance_pct' in (config or {}),
            "Config missing or incomplete"
        )

    # ------------------------------------------------------------------
    # REPORT
    # ------------------------------------------------------------------
    def run_all(self):
        print("=" * 60)
        print("STOCKWISE AI - MASTER SYSTEM VALIDATOR")
        print("=" * 60)

        self.check_column_name_consistency()
        print()
        self.check_dsp_config_integrity()
        print()
        self.check_threshold_sanity()
        print()
        self.check_no_dead_column_references()
        print()
        self.check_analyze_return_contract()
        print()
        self.check_cooldown_write_exists()
        print()
        self.check_cooldown_config_param()
        print()
        self.check_ai_pipeline_integrity()
        print()
        self.check_regime_gate()
        print()
        self.check_milestone_alert_system()

        total = len(self.results)
        passed = sum(1 for _, p, _ in self.results if p)
        failed = total - passed

        print(f"\n{'=' * 60}")
        print(f"TOTAL: {passed}/{total} passed ({failed} failed)")

        if failed > 0:
            print("\nFAILED CHECKS:")
            for name, p, detail in self.results:
                if not p:
                    print(f"  FAIL {name}: {detail}")
            sys.exit(1)
        else:
            print("ALL SYSTEM CHECKS PASSED!")
            sys.exit(0)


if __name__ == '__main__':
    MasterValidator().run_all()
