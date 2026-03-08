"""
StockWise AI - Unit Tests for Phase 1 Bug Fixes
=================================================
Each test class validates one specific bug fix in isolation.
Run: python tests/unit_tests.py  OR  pytest tests/unit_tests.py -v

IMPORT STRATEGY:
Since feature_engine.py imports pandas_ta (not pip-installable in this environment;
only pandas-ta-classic is available under a different import name), we stub it at
sys.modules level BEFORE importing strategy_engine.
TacticalSniper.__init__ loads .pkl models from disk — they won't exist in the test
environment, but _load_model() already returns None gracefully, so analyze() still
works (AI score defaults to 50.0 when model is None).
"""
import sys
import os
import types
from unittest.mock import MagicMock

# === DEPENDENCY STUBBING (must happen before any project imports) ===
# Stub pandas_ta regardless of whether it's installed — tests should not
# depend on a 200MB TA library for unit-testing column name logic.
_pandas_ta_stub = types.ModuleType('pandas_ta')
for _fn in ['rsi', 'sma', 'ema', 'macd', 'bbands', 'kc', 'donchian', 'atr',
            'adx', 'stoch', 'squeeze', 'squeeze_pro']:
    setattr(_pandas_ta_stub, _fn, MagicMock(return_value=None))
sys.modules['pandas_ta'] = _pandas_ta_stub

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np

# Now safe to import project modules
from strategy_engine import TacticalSniper, StrategyEngine


def _make_row(**overrides):
    """
    Factory: creates a single-row DataFrame with all columns TacticalSniper needs.
    Defaults represent a neutral stock that triggers NO setups.
    Override specific fields to test individual setups.
    """
    defaults = {
        # OHLCV
        'open': 100.0, 'high': 102.0, 'low': 98.0, 'close': 101.0,
        'volume': 500_000, 'vol_avg_20': 400_000,
        # Trend
        'sma_50': 95.0, 'sma_200': 90.0,
        'er_slow': 0.30, 'er_fast': 0.25, 'trend_alignment': 0,
        # Volatility
        'bb_width': 0.25, 'bb_upper': 110.0, 'bb_lower': 90.0,
        'kc_upper': 108.0, 'kc_lower': 92.0,
        'squeeze_on': 0, 'mom_sqz': 0.0, 'atr': 2.0,
        # Momentum
        'rvol': 1.0, 'rsi': 50.0,
        'macd': 0.0, 'macdsignal': 0.0, 'macd_hist': 0.0,
        # Regime helpers
        'er_slow': 0.30,
        'is_consolidating': False,
    }
    defaults.update(overrides)
    return pd.DataFrame([defaults])


# ============================================================
# BUG 1.2: Column Name Case Mismatch in apply_checklist_bonus
# ============================================================
class TestBug1_2_ColumnCaseMismatch:
    """Verify apply_checklist_bonus reads lowercase column names from feature_engine."""

    def test_sma_alignment_bonus_fires_with_lowercase_columns(self):
        """Bonus +10 for close > sma_50 > sma_200 using lowercase key names."""
        engine = StrategyEngine()
        row = _make_row(close=150.0, sma_50=140.0, sma_200=130.0).iloc[0]
        result = engine.apply_checklist_bonus(row, 50.0)
        assert result > 50.0, \
            f"SMA alignment bonus not applied (expected > 50, got {result}). " \
            f"Likely still using uppercase 'SMA_50'/'SMA_200'."

    def test_sma_alignment_bonus_absent_when_not_aligned(self):
        """No SMA bonus when close < sma_50 (downtrend)."""
        engine = StrategyEngine()
        row_down = _make_row(close=80.0, sma_50=140.0, sma_200=130.0).iloc[0]
        row_up = _make_row(close=150.0, sma_50=140.0, sma_200=130.0).iloc[0]
        result_down = engine.apply_checklist_bonus(row_down, 50.0)
        result_up = engine.apply_checklist_bonus(row_up, 50.0)
        assert result_up > result_down, \
            "Uptrend should score higher than downtrend via SMA bonus."

    def test_rsi_bonus_fires_with_lowercase_rsi_column(self):
        """Bonus +7 for rsi in 40-65 range using lowercase key 'rsi', not 'rsi_14'."""
        engine = StrategyEngine()
        row_sweet = _make_row(rsi=55.0).iloc[0]
        row_high = _make_row(rsi=80.0).iloc[0]
        result_sweet = engine.apply_checklist_bonus(row_sweet, 50.0)
        result_high = engine.apply_checklist_bonus(row_high, 50.0)
        assert result_sweet > result_high, \
            f"RSI sweet-spot bonus not applied. Sweet: {result_sweet}, Overbought: {result_high}. " \
            f"Likely still using 'rsi_14'."

    def test_bb_upper_bonus_fires_with_lowercase_column(self):
        """Bonus +5 for (bb_upper - close) / close > 3% using key 'bb_upper'."""
        engine = StrategyEngine()
        row_room = _make_row(close=100.0, bb_upper=115.0).iloc[0]   # 15% room → bonus
        row_tight = _make_row(close=100.0, bb_upper=101.0).iloc[0]  # 1% room → no bonus
        result_room = engine.apply_checklist_bonus(row_room, 50.0)
        result_tight = engine.apply_checklist_bonus(row_tight, 50.0)
        assert result_room > result_tight, \
            f"BB distance bonus not applied. Room: {result_room}, Tight: {result_tight}. " \
            f"Likely still using 'BBU_20_2.0'."

    def test_old_uppercase_column_names_absent_from_source(self):
        """
        Guard: SMA_50, SMA_200, BBU_20 must not appear as dict key lookups.
        rsi_14 is checked specifically as a .get() key — the local variable
        name rsi_14 = ... is acceptable; only the lookup key must be 'rsi'.
        """
        path = os.path.join(PROJECT_ROOT, 'strategy_engine.py')
        with open(path, 'r') as f:
            code = f.read()
        # These should never appear as column key lookups
        for old_key in ["'SMA_50'", "'SMA_200'", "'BBU_20", "'rsi_14'"]:
            assert old_key not in code, \
                f"Dead column key {old_key} still used in strategy_engine.py!"


# ============================================================
# BUG 1.3: Non-Existent Column 'er_trend'
# ============================================================
class TestBug1_3_ErTrend:
    """Verify Setup 1 (DSP_SUPER_TREND) uses er_slow threshold instead of er_trend."""

    def test_setup1_fires_when_er_slow_above_threshold(self):
        """Setup 1 should activate when er_slow >= 0.55 AND trend_alignment == 1."""
        sniper = TacticalSniper()
        df = _make_row(er_slow=0.65, trend_alignment=1)
        result = sniper.analyze("TEST", df, "TREND")
        assert "DSP_SUPER_TREND" in result.get('setups_found', []), \
            f"Setup 1 should fire when er_slow=0.65 >= 0.55. Got: {result.get('setups_found')}"

    def test_setup1_blocked_when_er_slow_below_threshold(self):
        """Setup 1 should NOT activate when er_slow < 0.55."""
        sniper = TacticalSniper()
        df = _make_row(er_slow=0.30, trend_alignment=1)
        result = sniper.analyze("TEST", df, "TREND")
        assert "DSP_SUPER_TREND" not in result.get('setups_found', []), \
            f"Setup 1 should NOT fire when er_slow=0.30. Got: {result.get('setups_found')}"

    def test_setup1_blocked_when_trend_alignment_zero(self):
        """Setup 1 requires BOTH conditions: high er_slow AND trend_alignment == 1."""
        sniper = TacticalSniper()
        df = _make_row(er_slow=0.80, trend_alignment=0)
        result = sniper.analyze("TEST", df, "TREND")
        assert "DSP_SUPER_TREND" not in result.get('setups_found', []), \
            f"Setup 1 should NOT fire without trend_alignment. Got: {result.get('setups_found')}"

    def test_setup1_at_exact_threshold_boundary(self):
        """Setup 1 should fire at exactly the threshold value (>= is inclusive)."""
        sniper = TacticalSniper()
        df = _make_row(er_slow=0.55, trend_alignment=1)
        result = sniper.analyze("TEST", df, "TREND")
        assert "DSP_SUPER_TREND" in result.get('setups_found', []), \
            f"Setup 1 should fire at er_slow=0.55 (boundary, inclusive). Got: {result.get('setups_found')}"

    def test_er_trend_string_fully_removed(self):
        """Guard: the string 'er_trend' must not appear anywhere in strategy_engine.py."""
        path = os.path.join(PROJECT_ROOT, 'strategy_engine.py')
        with open(path, 'r') as f:
            code = f.read()
        assert 'er_trend' not in code, \
            "'er_trend' still found in strategy_engine.py — bug not fully removed!"


# ============================================================
# BUG 1.4: Cooldown File Never Written
# ============================================================
import json as _json  # alias to avoid shadowing in test bodies
import tempfile
import shutil

class TestBug1_4_CooldownWrite:
    """Verify stop-loss triggers cooldown file write and strategy engine reads it."""

    def setup_method(self, _method=None):
        self.temp_dir = tempfile.mkdtemp()
        self.cooldown_path = os.path.join(self.temp_dir, 'cooldown_list.json')

    def teardown_method(self, _method=None):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_write_cooldown_creates_file(self):
        """_write_cooldown should create the cooldown JSON file."""
        import system_config as cfg
        from live_trading_engine import LiveTradingEngine
        engine = LiveTradingEngine.__new__(LiveTradingEngine)
        original = cfg.COOLDOWN_FILE_PATH
        cfg.COOLDOWN_FILE_PATH = self.cooldown_path
        try:
            engine._write_cooldown("AAPL", reason="STOP LOSS HIT")
            assert os.path.exists(self.cooldown_path), "Cooldown file was not created!"
            with open(self.cooldown_path, 'r') as f:
                data = _json.load(f)
            assert "AAPL" in data, f"AAPL not found in cooldown file! Got: {list(data.keys())}"
            assert data["AAPL"]["reason"] == "STOP LOSS HIT"
            assert "timestamp" in data["AAPL"]
        finally:
            cfg.COOLDOWN_FILE_PATH = original

    def test_write_cooldown_appends_not_overwrites(self):
        """Writing a second ticker should keep the first."""
        import system_config as cfg
        from live_trading_engine import LiveTradingEngine
        engine = LiveTradingEngine.__new__(LiveTradingEngine)
        original = cfg.COOLDOWN_FILE_PATH
        cfg.COOLDOWN_FILE_PATH = self.cooldown_path
        try:
            engine._write_cooldown("AAPL", reason="STOP LOSS HIT")
            engine._write_cooldown("TSLA", reason="ZOMBIE PROTOCOL (Time Expired)")
            with open(self.cooldown_path, 'r') as f:
                data = _json.load(f)
            assert "AAPL" in data and "TSLA" in data, \
                f"Both tickers should be in cooldown. Got: {list(data.keys())}"
        finally:
            cfg.COOLDOWN_FILE_PATH = original

    def test_is_in_cooldown_reads_written_file(self):
        """strategy_engine._is_in_cooldown should detect a ticker written by live_trading_engine."""
        import system_config as cfg
        from live_trading_engine import LiveTradingEngine
        from strategy_engine import StrategyEngine
        original = cfg.COOLDOWN_FILE_PATH
        cfg.COOLDOWN_FILE_PATH = self.cooldown_path
        try:
            live = LiveTradingEngine.__new__(LiveTradingEngine)
            live._write_cooldown("NVDA", reason="STOP LOSS HIT")

            strategy = StrategyEngine.__new__(StrategyEngine)
            strategy.cooldown_file = self.cooldown_path
            assert strategy._is_in_cooldown("NVDA") is True, \
                "Strategy engine should detect NVDA in cooldown!"
            assert strategy._is_in_cooldown("GOOG") is False, \
                "GOOG should NOT be in cooldown!"
        finally:
            cfg.COOLDOWN_FILE_PATH = original

    def test_cooldown_period_from_config(self):
        """Verify _is_in_cooldown uses COOLDOWN_PERIOD_HOURS from config, not hardcoded 86400."""
        import re
        path = os.path.join(PROJECT_ROOT, 'strategy_engine.py')
        with open(path, 'r') as f:
            code = f.read()
        match = re.search(
            r'def _is_in_cooldown\(self.*?\n(.*?)(?=\n    def |\nclass |\Z)',
            code, re.DOTALL
        )
        assert match, "_is_in_cooldown method not found in strategy_engine.py"
        method_body = match.group(1)
        assert '86400' not in method_body, \
            "Hardcoded 86400 still in _is_in_cooldown! Should use COOLDOWN_PERIOD_HOURS * 3600"
        assert 'COOLDOWN_PERIOD_HOURS' in method_body, \
            "_is_in_cooldown should reference COOLDOWN_PERIOD_HOURS from config"


# ============================================================
# BUG 1.5: Dual Threshold Conflict
# ============================================================
class TestBug1_5_DualThreshold:
    """Verify MIN_MASTER_SCORE_APPROVAL is reachable and not blocking valid BUYs."""

    def test_approval_threshold_lowered(self):
        """MIN_MASTER_SCORE_APPROVAL should be 65.0 (not 80.0)."""
        import system_config as cfg
        val = getattr(cfg, 'MIN_MASTER_SCORE_APPROVAL', None)
        assert val is not None, "MIN_MASTER_SCORE_APPROVAL missing from config!"
        assert val == 65.0, \
            f"MIN_MASTER_SCORE_APPROVAL should be 65.0, got {val}"

    def test_approval_above_sniper_buy_threshold(self):
        """Approval must be above TacticalSniper's BUY threshold (60) to add value."""
        import system_config as cfg
        val = cfg.MIN_MASTER_SCORE_APPROVAL
        assert val > 60.0, \
            f"Approval ({val}) should be above Sniper BUY threshold (60)"

    def test_approval_below_unreachable(self):
        """Approval must be <= 85 to be realistically achievable."""
        import system_config as cfg
        val = cfg.MIN_MASTER_SCORE_APPROVAL
        assert val <= 85.0, \
            f"Approval ({val}) is too high -- most signals won't reach it"

    def test_buy_survives_with_score_above_approval(self):
        """A BUY verdict with master_score > MIN_MASTER_SCORE_APPROVAL should survive."""
        from strategy_engine import StrategyEngine
        engine = StrategyEngine()

        df = _make_row(
            er_slow=0.70, trend_alignment=1,
            volume=800000, vol_avg_20=400000,
            rsi=60.0, macd=1.0, macdsignal=0.5,
            close=100.0, sma_50=95.0, sma_200=90.0,
            bb_upper=115.0, rvol=1.5, atr=2.0
        )

        result = engine.evaluate_ticker("TEST_BUY", df)
        if result.get('action') == 'WAIT':
            reason = result.get('reason', '')
            assert 'Calibrated Threshold' not in reason, \
                f"Good signal killed by threshold gate! Score: {result.get('master_score')}, Reason: {reason}"


# ============================================================
# BUG 1.6a: Missing squeeze_on and mom_sqz Columns
# ============================================================
class TestBug1_6a_SqueezeColumns:
    """Verify squeeze columns are created by feature_engine and Setup 2 can fire."""

    def test_squeeze_on_column_created(self):
        """feature_engine.py should now contain df['squeeze_on'] assignment."""
        path = os.path.join(PROJECT_ROOT, 'feature_engine.py')
        with open(path, 'r') as f:
            code = f.read()
        assert "df['squeeze_on']" in code, \
            "squeeze_on column not created in feature_engine.py!"

    def test_mom_sqz_column_created(self):
        """feature_engine.py should now contain df['mom_sqz'] assignment."""
        path = os.path.join(PROJECT_ROOT, 'feature_engine.py')
        with open(path, 'r') as f:
            code = f.read()
        assert "df['mom_sqz']" in code, \
            "mom_sqz column not created in feature_engine.py!"

    def test_setup2_squeeze_prep_fires(self):
        """Setup 2 VOLATILITY_SQUEEZE_PREP should fire when bb_width < 0.15 and squeeze_on=1."""
        sniper = TacticalSniper()
        df = _make_row(bb_width=0.10, squeeze_on=1, mom_sqz=0.0)
        result = sniper.analyze("TEST", df, "TREND")
        assert "VOLATILITY_SQUEEZE_PREP" in result.get('setups_found', []), \
            f"Squeeze prep should fire. Got: {result.get('setups_found')}"

    def test_setup2_squeeze_firing_fires(self):
        """Setup 2 SQUEEZE_FIRING_LONG should fire when bb_width < 0.15 and mom_sqz > 0."""
        sniper = TacticalSniper()
        df = _make_row(bb_width=0.10, squeeze_on=0, mom_sqz=1.5)
        result = sniper.analyze("TEST", df, "TREND")
        assert "SQUEEZE_FIRING_LONG" in result.get('setups_found', []), \
            f"Squeeze firing should fire. Got: {result.get('setups_found')}"

    def test_setup2_does_not_fire_wide_bands(self):
        """Setup 2 should NOT fire when bb_width >= 0.15."""
        sniper = TacticalSniper()
        df = _make_row(bb_width=0.25, squeeze_on=1, mom_sqz=2.0)
        result = sniper.analyze("TEST", df, "TREND")
        setups = result.get('setups_found', [])
        assert "VOLATILITY_SQUEEZE_PREP" not in setups and "SQUEEZE_FIRING_LONG" not in setups, \
            f"Setup 2 should NOT fire with wide bands. Got: {setups}"


# ============================================================
# BUG 1.6b: Dangerous df.fillna(0) on Price Columns
# ============================================================
class TestBug1_6b_SafeFillna:
    """Verify feature_engine no longer zero-fills price columns."""

    def test_no_blanket_fillna_zero(self):
        """The pattern 'df = df.fillna(0)' should not appear in feature_engine.py."""
        path = os.path.join(PROJECT_ROOT, 'feature_engine.py')
        with open(path, 'r') as f:
            code = f.read()
        assert 'df = df.fillna(0)' not in code, \
            "Dangerous 'df = df.fillna(0)' still in feature_engine.py!"

    def test_price_columns_use_ffill(self):
        """feature_engine.py should use ffill() for price columns."""
        path = os.path.join(PROJECT_ROOT, 'feature_engine.py')
        with open(path, 'r') as f:
            code = f.read()
        assert 'ffill()' in code, \
            "ffill() not found in feature_engine.py — price columns may still be zero-filled"

    def test_ffill_preserves_prices(self):
        """Forward-fill should carry last known price, not zero."""
        df = pd.DataFrame({
            'open': [100.0, np.nan, np.nan],
            'high': [105.0, np.nan, np.nan],
            'low': [95.0, np.nan, np.nan],
            'close': [102.0, np.nan, np.nan],
            'volume': [50000, np.nan, np.nan],
            'rsi': [np.nan, np.nan, np.nan],
            'sma_50': [np.nan, np.nan, np.nan],
        })

        price_cols = ['open', 'high', 'low', 'close', 'volume']
        indicator_cols = [c for c in df.columns if c not in price_cols]
        df[price_cols] = df[price_cols].ffill()
        df[indicator_cols] = df[indicator_cols].fillna(0)

        assert df['close'].iloc[2] == 102.0, \
            f"Close should be ffilled to 102.0, got {df['close'].iloc[2]}"
        assert df['volume'].iloc[1] == 50000, \
            f"Volume should be ffilled to 50000, got {df['volume'].iloc[1]}"
        assert df['rsi'].iloc[0] == 0, \
            f"RSI NaN should become 0, got {df['rsi'].iloc[0]}"


# ============================================================
# BUG 1.6c: buy_date_raw.split("T") Returns List Instead of String
# ============================================================
class TestBug1_6c_DateSplit:
    """Verify ISO date parsing returns a string, not a list."""

    def test_iso_date_extracts_string(self):
        """split('T')[0] should return '2025-02-05', not a list."""
        raw = "2025-02-05T14:30:00"
        result = raw.split("T")[0] if "T" in raw else raw
        assert isinstance(result, str), f"Expected str, got {type(result)}"
        assert result == "2025-02-05", f"Expected '2025-02-05', got '{result}'"

    def test_date_without_T_passthrough(self):
        """A date without 'T' should pass through unchanged."""
        raw = "2025-02-05"
        result = raw.split("T")[0] if "T" in raw else raw
        assert result == "2025-02-05"

    def test_unknown_passthrough(self):
        """'UNKNOWN' fallback should pass through."""
        raw = "UNKNOWN"
        result = raw.split("T")[0] if "T" in raw else raw
        assert result == "UNKNOWN"

    def test_source_code_has_index_zero(self):
        """The split('T') call in live_trading_engine.py must include [0]."""
        import re
        path = os.path.join(PROJECT_ROOT, 'live_trading_engine.py')
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            code = f.read()
        assert '.split("T")[0]' in code or ".split('T')[0]" in code, \
            "split('T') without [0] still in live_trading_engine.py!"
        # Must NOT find bare split("T") without [0] outside comments
        lines = code.split('\n')
        real_bare = []
        for line in lines:
            if line.strip().startswith('#'):
                continue
            if re.search(r'\.split\(["\']T["\']\)(?!\[0\])', line):
                real_bare.append(line.strip())
        assert len(real_bare) == 0, \
            f"Bare split('T') without [0] found: {real_bare}"


# ============================================================
# BUG 1.1: AI Feature Mismatch (Core B Dead)
# ============================================================
class TestBug1_1_AIFeatureMismatch:
    """Verify AI training saves real features and prediction handles both model types."""

    def test_train_and_save_does_not_hardcode_meta_features(self):
        """train_and_save should NOT write ['tech_score', 'ai_score', 'master_score', 'regime_val']."""
        path = os.path.join(PROJECT_ROOT, 'train_model.py')
        with open(path, 'r') as f:
            code = f.read()
        assert '"tech_score", "ai_score", "master_score", "regime_val"' not in code, \
            "Hardcoded meta-feature list still in train_model.py!"

    def test_train_saves_actual_columns(self):
        """train_and_save should save X.columns (the actual DataFrame column names)."""
        path = os.path.join(PROJECT_ROOT, 'train_model.py')
        with open(path, 'r') as f:
            code = f.read()
        assert 'X.columns' in code or 'feature_list = list(' in code, \
            "train_and_save doesn't extract actual column names from training data"

    def test_model_is_classifier_not_regressor(self):
        """train_and_save should use XGBClassifier (target is binary 0/1)."""
        path = os.path.join(PROJECT_ROOT, 'train_model.py')
        with open(path, 'r') as f:
            code = f.read()
        assert 'XGBClassifier' in code, \
            "Model should be XGBClassifier, not XGBRegressor"

    def test_get_ai_probability_handles_both_model_types(self):
        """get_ai_probability should have predict_proba AND predict fallback."""
        path = os.path.join(PROJECT_ROOT, 'strategy_engine.py')
        with open(path, 'r') as f:
            code = f.read()
        assert 'predict_proba' in code and 'hasattr' in code, \
            "get_ai_probability should check hasattr(model, 'predict_proba') for safe fallback"

    def test_execute_pipeline_uses_universal_dataset(self):
        """execute_training_pipeline should call build_universal_dataset, not prepare_training_data."""
        import re
        path = os.path.join(PROJECT_ROOT, 'train_model.py')
        with open(path, 'r') as f:
            code = f.read()
        match = re.search(
            r'def execute_training_pipeline\(.*?\n(.*?)(?=\n    def |\nclass |\Z)',
            code, re.DOTALL
        )
        assert match, "execute_training_pipeline method not found"
        method_body = match.group(1)
        assert 'build_universal_dataset' in method_body, \
            "execute_training_pipeline should use build_universal_dataset (real features)"
        assert 'prepare_training_data' not in method_body, \
            "execute_training_pipeline should NOT call prepare_training_data (meta-features path)"

    def test_default_training_symbols_in_config(self):
        """DEFAULT_TRAINING_SYMBOLS should exist in system_config."""
        import system_config as cfg
        symbols = getattr(cfg, 'DEFAULT_TRAINING_SYMBOLS', None)
        assert symbols is not None, "DEFAULT_TRAINING_SYMBOLS missing from system_config"
        assert len(symbols) >= 5, f"Need at least 5 default symbols, got {len(symbols)}"


# ============================================================
# BUG 2.1: macdsignal vs macd_signal Column Name Mismatch
# ============================================================
class TestBug2_1_MacdSignalName:
    """Verify Setup 5 uses correct macd_signal column name."""

    def test_macd_signal_column_name_correct(self):
        """strategy_engine.py should reference 'macd_signal', not 'macdsignal'."""
        path = os.path.join(PROJECT_ROOT, 'strategy_engine.py')
        with open(path, 'r') as f:
            code = f.read()
        assert "'macdsignal'" not in code, \
            "Old column name 'macdsignal' still in strategy_engine.py!"
        assert "'macd_signal'" in code, \
            "'macd_signal' not found -- Setup 5 can't read MACD signal line"

    def test_momentum_breakout_fires_correctly(self):
        """Setup 5 should fire when RSI 50-75 AND macd > macd_signal (real crossover)."""
        sniper = TacticalSniper()
        df = _make_row(rsi=60.0, macd=1.5, macd_signal=0.5)
        result = sniper.analyze("TEST", df, "TREND")
        assert "MOMENTUM_BREAKOUT" in result.get('setups_found', []), \
            f"Setup 5 should fire on MACD crossover. Got: {result.get('setups_found')}"

    def test_momentum_breakout_blocked_when_macd_below_signal(self):
        """Setup 5 should NOT fire when macd < macd_signal (bearish)."""
        sniper = TacticalSniper()
        df = _make_row(rsi=60.0, macd=0.5, macd_signal=1.5)
        result = sniper.analyze("TEST", df, "TREND")
        assert "MOMENTUM_BREAKOUT" not in result.get('setups_found', []), \
            f"Setup 5 should NOT fire when MACD below signal. Got: {result.get('setups_found')}"

    def test_no_false_positive_from_zero_default(self):
        """With correct column name, a negative MACD should not trigger breakout."""
        sniper = TacticalSniper()
        df = _make_row(rsi=60.0, macd=-0.5, macd_signal=0.3)
        result = sniper.analyze("TEST", df, "TREND")
        assert "MOMENTUM_BREAKOUT" not in result.get('setups_found', []), \
            f"Negative MACD should not trigger breakout. Got: {result.get('setups_found')}"


# ============================================================
# BUG 2.2: is_consolidating / BOLLINGER_SQUEEZE Don't Exist
# ============================================================
class TestBug2_2_SqueezeBonus:
    """Verify checklist squeeze bonus uses squeeze_on column."""

    def test_squeeze_bonus_fires_with_squeeze_on(self):
        """Squeeze bonus +10 should fire when squeeze_on == 1."""
        from strategy_engine import StrategyEngine
        engine = StrategyEngine()
        row_yes = _make_row(squeeze_on=1).iloc[0]
        row_no = _make_row(squeeze_on=0).iloc[0]
        result_yes = engine.apply_checklist_bonus(row_yes, 50.0)
        result_no = engine.apply_checklist_bonus(row_no, 50.0)
        assert result_yes > result_no, \
            f"Squeeze bonus not applied. With squeeze_on=1: {result_yes}, Without: {result_no}"

    def test_old_column_names_removed(self):
        """'is_consolidating' and 'BOLLINGER_SQUEEZE' should not be used as column lookups."""
        path = os.path.join(PROJECT_ROOT, 'strategy_engine.py')
        with open(path, 'r') as f:
            code = f.read()
        assert "'is_consolidating'" not in code, \
            "'is_consolidating' still referenced in strategy_engine.py!"
        assert "'BOLLINGER_SQUEEZE'" not in code, \
            "'BOLLINGER_SQUEEZE' still referenced in strategy_engine.py!"

    def test_squeeze_on_used_in_checklist(self):
        """apply_checklist_bonus should reference 'squeeze_on'."""
        path = os.path.join(PROJECT_ROOT, 'strategy_engine.py')
        with open(path, 'r') as f:
            code = f.read()
        assert "'squeeze_on'" in code, \
            "'squeeze_on' not found in strategy_engine.py checklist bonus"


# ============================================================
# RUNNER (also compatible with pytest)
# ============================================================
if __name__ == '__main__':
    passed = 0
    failed = 0
    errors = []

    test_classes = [TestBug1_1_AIFeatureMismatch, TestBug1_2_ColumnCaseMismatch, TestBug1_3_ErTrend, TestBug1_4_CooldownWrite, TestBug1_5_DualThreshold, TestBug1_6a_SqueezeColumns, TestBug1_6b_SafeFillna, TestBug1_6c_DateSplit, TestBug2_1_MacdSignalName, TestBug2_2_SqueezeBonus]

    for cls in test_classes:
        print(f"\n--- {cls.__name__} ---")
        for method_name in sorted(dir(cls)):
            if not method_name.startswith('test_'):
                continue
            instance = cls()
            # Honour pytest-style setup/teardown if defined
            if hasattr(instance, 'setup_method'):
                instance.setup_method()
            try:
                getattr(instance, method_name)()
                passed += 1
                print(f"  PASS: {method_name}")
            except AssertionError as e:
                failed += 1
                errors.append((cls.__name__, method_name, str(e)))
                print(f"  FAIL: {method_name}\n       {e}")
            except Exception as e:
                failed += 1
                errors.append((cls.__name__, method_name, f"ERROR: {e}"))
                print(f"  ERROR: {method_name}\n        {e}")
            finally:
                if hasattr(instance, 'teardown_method'):
                    instance.teardown_method()

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed} tests")

    if errors:
        print("\nFailed tests detail:")
        for cls_name, method, msg in errors:
            print(f"  {cls_name}.{method}:\n    {msg}")
        sys.exit(1)
    else:
        print("All tests PASSED!")
        sys.exit(0)
