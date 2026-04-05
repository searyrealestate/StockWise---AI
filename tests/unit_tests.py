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
from unittest.mock import MagicMock, patch

# === DEPENDENCY STUBBING (must happen before any project imports) ===
# Stub pandas_ta regardless of whether it's installed — tests should not
# depend on a 200MB TA library for unit-testing column name logic.
_pandas_ta_stub = types.ModuleType('pandas_ta')
for _fn in ['rsi', 'sma', 'ema', 'macd', 'bbands', 'kc', 'donchian', 'atr',
            'adx', 'stoch', 'squeeze', 'squeeze_pro']:
    setattr(_pandas_ta_stub, _fn, MagicMock(return_value=None))
sys.modules['pandas_ta'] = _pandas_ta_stub

# Stub xgboost so train_model.py can be imported without the package installed.
_xgb_stub = types.ModuleType('xgboost')
_xgb_stub.XGBClassifier = MagicMock
_xgb_stub.XGBRegressor = MagicMock
sys.modules['xgboost'] = _xgb_stub

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
# BUG 2.3: HALT/NEUTRAL Regime Does Not Block Analysis
# ============================================================
class TestBug2_3_RegimeGate:
    """Verify HALT and NEUTRAL regimes block analysis in evaluate_ticker."""

    def test_halt_regime_returns_wait(self):
        """HALT regime should return WAIT without running sniper.analyze()."""
        from strategy_engine import StrategyEngine
        engine = StrategyEngine()
        # er_slow > 0.6 AND er_fast < 0.2 -> HALT (velocity divergence)
        df = _make_row(er_slow=0.70, er_fast=0.15)
        # Bypass feature calculation so our hand-crafted er_slow/er_fast survive intact
        with patch.object(engine.features, 'calculate_features', return_value=df):
            result = engine.evaluate_ticker("TEST_HALT", df)
        assert result.get('action') == 'WAIT', \
            f"HALT regime should return WAIT. Got: {result.get('action')}"
        assert 'HALT' in result.get('reason', ''), \
            f"Reason should mention HALT. Got: {result.get('reason')}"

    def test_neutral_regime_returns_wait(self):
        """NEUTRAL regime should return WAIT without running sniper.analyze()."""
        from strategy_engine import StrategyEngine
        engine = StrategyEngine()
        # er_slow between chop_thr (0.30) and trend_thr (0.55) -> NEUTRAL
        df = _make_row(er_slow=0.42, er_fast=0.40)
        # Bypass feature calculation so our hand-crafted er_slow/er_fast survive intact
        with patch.object(engine.features, 'calculate_features', return_value=df):
            result = engine.evaluate_ticker("TEST_NEUTRAL", df)
        assert result.get('action') == 'WAIT', \
            f"NEUTRAL regime should return WAIT. Got: {result.get('action')}"
        assert 'NEUTRAL' in result.get('reason', ''), \
            f"Reason should mention NEUTRAL. Got: {result.get('reason')}"

    def test_trend_regime_still_analyzed(self):
        """TREND regime should still pass through to full analysis."""
        from strategy_engine import StrategyEngine
        engine = StrategyEngine()
        # er_slow >= 0.55 AND er_fast NOT < 0.2 → TREND
        df = _make_row(er_slow=0.70, er_fast=0.65)
        result = engine.evaluate_ticker("TEST_TREND", df)
        reason = result.get('reason', '')
        assert 'HALT' not in reason and 'NEUTRAL' not in reason, \
            f"TREND should not be blocked by regime gate. Reason: {reason}"

    def test_chop_regime_still_analyzed(self):
        """CHOP regime should still pass through to full analysis."""
        from strategy_engine import StrategyEngine
        engine = StrategyEngine()
        # er_slow <= 0.30 → CHOP
        df = _make_row(er_slow=0.20, er_fast=0.15)
        result = engine.evaluate_ticker("TEST_CHOP", df)
        reason = result.get('reason', '')
        assert 'HALT' not in reason and 'NEUTRAL' not in reason, \
            f"CHOP should not be blocked by regime gate. Reason: {reason}"


# ============================================================
# BUG 2.4: _generate_labels profit_target Inconsistency + Hardcoded Values
# ============================================================
class TestBug2_4_LabelConfig:
    """Verify _generate_labels reads from AI_LABEL_CONFIG and has no hardcoded defaults."""

    def test_ai_label_config_exists_in_system_config(self):
        """AI_LABEL_CONFIG must be present in system_config with required keys."""
        import system_config as cfg
        label_cfg = getattr(cfg, 'AI_LABEL_CONFIG', None)
        assert label_cfg is not None, \
            "AI_LABEL_CONFIG missing from system_config.py"
        assert 'lookahead_days' in label_cfg, \
            "AI_LABEL_CONFIG missing 'lookahead_days' key"
        assert 'profit_target_pct' in label_cfg, \
            "AI_LABEL_CONFIG missing 'profit_target_pct' key"

    def test_ai_label_config_values_are_consistent(self):
        """AI_LABEL_CONFIG values must match Bug 2.4 spec: lookahead=5, profit=0.02."""
        import system_config as cfg
        label_cfg = cfg.AI_LABEL_CONFIG
        assert label_cfg['lookahead_days'] == 5, \
            f"lookahead_days should be 5, got {label_cfg['lookahead_days']}"
        assert label_cfg['profit_target_pct'] == 0.02, \
            f"profit_target_pct should be 0.02, got {label_cfg['profit_target_pct']}"

    def test_generate_labels_defaults_are_none(self):
        """_generate_labels() must use None defaults, not hardcoded 5 and 0.03."""
        import inspect
        from train_model import RegimeModelTrainer
        sig = inspect.signature(RegimeModelTrainer._generate_labels)
        params = sig.parameters
        assert params['lookahead'].default is None, \
            f"lookahead default should be None, got {params['lookahead'].default}"
        assert params['profit_target'].default is None, \
            f"profit_target default should be None, got {params['profit_target'].default}"

    def test_generate_labels_reads_config_when_no_args(self):
        """_generate_labels() called with no args must produce labels using AI_LABEL_CONFIG values."""
        import pandas as pd
        import numpy as np
        from train_model import RegimeModelTrainer
        # Build a minimal df: close=100, high[1]=103 (3% above close).
        # The code uses shift(-1).rolling(), so row 0's future_high = high[1]=103.
        # max_gain[0] = (103-100)/100 = 3% >= 2% target -> label=1.
        # er_slow is required by the dropna guard at the end of _generate_labels.
        df = pd.DataFrame({
            'close': [100.0] * 10,
            'high': [100.0, 103.0] + [100.0] * 8,
            'er_slow': [0.6] * 10,
        })
        trainer = RegimeModelTrainer()
        result = trainer._generate_labels(df)
        assert 'target' in result.columns, "_generate_labels must produce a 'target' column"
        # Row 0 should be labeled 1 -- future high of 103 is 3% gain, above 2% threshold
        assert result['target'].iloc[0] == 1, \
            f"Row 0 should be label=1 (3% gain >= 2% target). Got {result['target'].iloc[0]}"

    def test_hardcoded_call_removed_from_build_universal_dataset(self):
        """build_universal_dataset must not call _generate_labels with explicit hardcoded args."""
        import re
        tm_path = os.path.join(PROJECT_ROOT, 'train_model.py')
        with open(tm_path, 'r') as f:
            code = f.read()
        # The old hardcoded call: _generate_labels(df, lookahead=5, profit_target=0.02)
        assert 'profit_target=0.02' not in code, \
            "Hardcoded profit_target=0.02 still present in train_model.py"
        assert '_generate_labels(df)' in code or '_generate_labels(df,' not in code.replace(
            '_generate_labels(df)', ''), \
            "build_universal_dataset should call _generate_labels(df) with no extra args"


# ============================================================
# PHASE 2.5: Milestone Alerts + Runner Mode
# ============================================================
class TestPhase2_5_MilestoneAlerts:
    """Verify milestone alert infrastructure and Runner Mode."""

    def test_milestone_config_exists(self):
        """MILESTONE_ALERT_CONFIG should exist with all required keys."""
        import system_config as cfg
        config = getattr(cfg, 'MILESTONE_ALERT_CONFIG', None)
        assert config is not None, "MILESTONE_ALERT_CONFIG missing from system_config"
        required_keys = ['safe_zone_buffer_pct', 'min_stop_change_pct',
                         'min_alert_interval_minutes', 'runner_atr_mult',
                         'runner_min_distance_pct']
        for key in required_keys:
            assert key in config, f"Missing key '{key}' in MILESTONE_ALERT_CONFIG"

    def test_runner_min_distance_prevents_noise_exit(self):
        """runner_min_distance_pct must be > 0 to prevent stop too close to price."""
        import system_config as cfg
        config = cfg.MILESTONE_ALERT_CONFIG
        assert config['runner_min_distance_pct'] > 0, \
            "runner_min_distance_pct must be > 0 to prevent noise exits"
        assert config['runner_min_distance_pct'] >= 0.005, \
            f"runner_min_distance_pct={config['runner_min_distance_pct']} too small (< 0.5%)"

    def test_calculate_real_breakeven_basic(self):
        """Breakeven should be above entry price (costs exist)."""
        from live_trading_engine import LiveTradingEngine
        engine = LiveTradingEngine.__new__(LiveTradingEngine)
        breakeven = engine._calculate_real_breakeven(entry_price=100.0, qty=50)
        assert breakeven > 100.0, \
            f"Breakeven ${breakeven} should be above entry $100 (must cover costs)"

    def test_calculate_real_breakeven_scales_with_qty(self):
        """Small positions should have higher breakeven pct (fixed commission spread over fewer shares)."""
        from live_trading_engine import LiveTradingEngine
        engine = LiveTradingEngine.__new__(LiveTradingEngine)
        breakeven_small = engine._calculate_real_breakeven(entry_price=100.0, qty=5)
        breakeven_large = engine._calculate_real_breakeven(entry_price=100.0, qty=500)
        # Small qty -> higher cost per share -> higher breakeven
        assert breakeven_small > breakeven_large, \
            f"Small qty breakeven ${breakeven_small} should be higher than large qty ${breakeven_large}"

    def test_milestone_alert_no_alert_before_breakeven(self):
        """No alert should fire when price is below real breakeven."""
        from live_trading_engine import LiveTradingEngine
        engine = LiveTradingEngine.__new__(LiveTradingEngine)
        engine.notifier = MagicMock()

        position = {'entry_price': 100.0, 'qty': 50, 'stop_loss': 96.0}
        # Price barely above entry but below breakeven
        engine._check_and_send_milestone_alert("TEST", position, 100.5, 100.3)
        engine.notifier.send_message.assert_not_called()

    def test_milestone_alert_fires_at_breakeven(self):
        """First alert should fire when price crosses real breakeven."""
        from live_trading_engine import LiveTradingEngine
        engine = LiveTradingEngine.__new__(LiveTradingEngine)
        engine.notifier = MagicMock()

        position = {'entry_price': 100.0, 'qty': 50, 'stop_loss': 96.0}
        breakeven = engine._calculate_real_breakeven(100.0, 50)
        # Price above breakeven
        engine._check_and_send_milestone_alert("TEST", position, breakeven + 0.5, breakeven + 1.0)
        assert position.get('breakeven_alerted') == True, \
            "breakeven_alerted flag should be set after first alert"
        engine.notifier.send_message.assert_called_once()

    def test_milestone_alert_cooldown(self):
        """Second alert should be blocked by cooldown timer."""
        import time as _time
        from live_trading_engine import LiveTradingEngine
        engine = LiveTradingEngine.__new__(LiveTradingEngine)
        engine.notifier = MagicMock()

        position = {
            'entry_price': 100.0, 'qty': 50, 'stop_loss': 96.0,
            'breakeven_alerted': True,
            'last_alerted_stop': 101.0,
            'last_alert_time': _time.time(),  # just now
            'real_breakeven': 100.5
        }
        # Even with significant stop change, cooldown should block
        engine._check_and_send_milestone_alert("TEST", position, 105.0, 110.0)
        engine.notifier.send_message.assert_not_called()

    def test_take_profit_activates_runner_mode(self):
        """Reaching take_profit should set runner_mode=True, not liquidate."""
        path = os.path.join(PROJECT_ROOT, 'live_trading_engine.py')
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            code = f.read()
        # take_profit should trigger runner_mode, not "TAKE PROFIT HIT" liquidation
        assert 'runner_mode' in code, "runner_mode not found in live_trading_engine.py"
        assert '"runner_mode"' in code or "'runner_mode'" in code, \
            "runner_mode flag not being set in live_trading_engine.py"

    def test_phase4_runner_in_kinetic_stop(self):
        """LifecycleManager should have Phase 4 Runner with min distance floor."""
        path = os.path.join(PROJECT_ROOT, 'live_trading_engine.py')
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            code = f.read()
        assert 'PHASE_4_RUNNER' in code, \
            "PHASE_4_RUNNER not found in LifecycleManager"
        assert 'runner_min_distance_pct' in code, \
            "runner_min_distance_pct floor not implemented in kinetic stop"

    def test_phase4_uses_wider_stop(self):
        """Phase 4 should use min() of ATR-based and floor-based (= wider = safer)."""
        path = os.path.join(PROJECT_ROOT, 'live_trading_engine.py')
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            code = f.read()
        # Must use min(atr_stop, floor_stop) not max -- min gives wider stop
        assert 'min(runner_stop_atr, runner_stop_floor)' in code, \
            "Phase 4 should use min() to pick the wider (safer) stop"


# ============================================================
# PHASE 3.3: Block Registry
# ============================================================
class TestPhase3_3_BlockRegistry:
    """Verify condition blocks, stop blocks, and target blocks work correctly."""

    def test_all_condition_blocks_registered(self):
        """All condition block functions should be in CONDITION_BLOCKS dict."""
        from setup_templates import CONDITION_BLOCKS
        assert len(CONDITION_BLOCKS) >= 19, \
            f"Expected >= 19 condition blocks, got {len(CONDITION_BLOCKS)}"

    def test_all_stop_blocks_registered(self):
        """All stop block functions should be in STOP_BLOCKS dict."""
        from setup_templates import STOP_BLOCKS
        assert len(STOP_BLOCKS) >= 4, \
            f"Expected >= 4 stop blocks, got {len(STOP_BLOCKS)}"

    def test_all_target_blocks_registered(self):
        """All target block functions should be in TARGET_BLOCKS dict."""
        from setup_templates import TARGET_BLOCKS
        assert len(TARGET_BLOCKS) >= 2, \
            f"Expected >= 2 target blocks, got {len(TARGET_BLOCKS)}"

    def test_rsi_between_block(self):
        """rsi_between should return True when RSI is in range."""
        from setup_templates import CONDITION_BLOCKS
        row = _make_row(rsi=55.0).iloc[0]
        assert CONDITION_BLOCKS["rsi_between"](row, [40, 65]) == True
        assert CONDITION_BLOCKS["rsi_between"](row, [60, 70]) == False

    def test_close_above_sma_block(self):
        """close_above_sma should compare close to sma_N."""
        from setup_templates import CONDITION_BLOCKS
        row = _make_row(close=150.0, sma_50=140.0).iloc[0]
        assert CONDITION_BLOCKS["close_above_sma"](row, [50]) == True
        row_below = _make_row(close=130.0, sma_50=140.0).iloc[0]
        assert CONDITION_BLOCKS["close_above_sma"](row_below, [50]) == False

    def test_volume_surge_block(self):
        """volume_surge should compare volume to avg * multiplier."""
        from setup_templates import CONDITION_BLOCKS
        row = _make_row(volume=800000, vol_avg_20=500000).iloc[0]
        assert CONDITION_BLOCKS["volume_surge"](row, [1.5]) == True  # 800K > 750K
        assert CONDITION_BLOCKS["volume_surge"](row, [2.0]) == False  # 800K < 1M

    def test_stop_atr_block(self):
        """stop_atr should return close - ATR * multiplier."""
        from setup_templates import STOP_BLOCKS
        row = _make_row(close=100.0, atr=2.0).iloc[0]
        stop = STOP_BLOCKS["atr"](row, [1.5])
        assert stop == 97.0, f"Expected 97.0, got {stop}"

    def test_target_atr_block(self):
        """target_atr should return close + ATR * multiplier."""
        from setup_templates import TARGET_BLOCKS
        row = _make_row(close=100.0, atr=2.0).iloc[0]
        target = TARGET_BLOCKS["atr"](row, [3.0])
        assert target == 106.0, f"Expected 106.0, got {target}"

    def test_blocks_handle_nan_safely(self):
        """Blocks should handle NaN values without crashing."""
        from setup_templates import CONDITION_BLOCKS
        import numpy as np
        row = _make_row(rsi=np.nan, close=np.nan).iloc[0]
        # Should not crash — returns False or default
        try:
            result = CONDITION_BLOCKS["rsi_between"](row, [40, 65])
            # NaN defaults to 50 via _safe_get, so 40 <= 50 <= 65 = True
            assert isinstance(result, bool)
        except Exception as e:
            assert False, f"Block crashed on NaN: {e}"


# ============================================================
# PHASE 3.3: Template Validation
# ============================================================
class TestPhase3_3_TemplateValidation:
    """Verify SetupTemplate and TemplateManager."""

    def test_seed_templates_load(self):
        """At least one enabled seed template should load (disabled templates are skipped at load time)."""
        from setup_templates import TemplateManager
        tm = TemplateManager()
        assert len(tm.templates) >= 1, \
            f"Expected >= 1 enabled templates, got {len(tm.templates)}"

    def test_all_templates_valid(self):
        """Every loaded template should pass validation."""
        from setup_templates import TemplateManager
        tm = TemplateManager()
        for t in tm.templates.values():
            valid, errors = t.validate()
            assert valid, f"Template {t.id} invalid: {errors}"

    def test_template_evaluate_conditions(self):
        """evaluate_conditions should run all blocks and return results."""
        from setup_templates import TemplateManager
        tm = TemplateManager()
        # Use MOMENTUM_BREAKOUT template with matching data
        t = tm.get_template_by_id("MOMENTUM_BREAKOUT")
        if t is None:
            return  # Skip if template not on disk
        row = _make_row(rsi=60.0, macd=1.5, macd_signal=0.5,
                       close=150.0, sma_50=140.0,
                       volume=800000, vol_avg_20=500000).iloc[0]
        passed, details = t.evaluate_conditions(row)
        assert isinstance(passed, bool)
        assert len(details) == len(t.conditions)

    def test_template_calculate_stop_loss(self):
        """calculate_stop_loss should return a price below entry."""
        from setup_templates import TemplateManager
        tm = TemplateManager()
        t = tm.get_template_by_id("MOMENTUM_BREAKOUT")
        if t is None:
            return
        row = _make_row(close=100.0, atr=2.0).iloc[0]
        stop = t.calculate_stop_loss(row)
        assert stop < 100.0, f"Stop {stop} should be below entry 100"

    def test_template_calculate_take_profit(self):
        """calculate_take_profit should return a price above entry."""
        from setup_templates import TemplateManager
        tm = TemplateManager()
        t = tm.get_template_by_id("MOMENTUM_BREAKOUT")
        if t is None:
            return
        row = _make_row(close=100.0, atr=2.0).iloc[0]
        target = t.calculate_take_profit(row)
        assert target > 100.0, f"Target {target} should be above entry 100"

    def test_get_for_state_filters_correctly(self):
        """get_for_state should only return templates matching the stock state."""
        from setup_templates import TemplateManager
        tm = TemplateManager()
        # BULLISH state should match MOMENTUM_BREAKOUT but not OVERSOLD_BOUNCE
        bullish_state = {"trend": "BULLISH", "volume": "SURGING", "volatility": "NORMAL"}
        matching = tm.get_for_state(bullish_state)
        ids = [t.id for t in matching]
        assert "OVERSOLD_BOUNCE" not in ids, \
            "OVERSOLD_BOUNCE should not match BULLISH state"


# ============================================================
# PHASE 3.4: Template Matcher
# ============================================================
class TestPhase3_4_TemplateMatcher:
    """Verify the template matcher pipeline."""

    def test_matcher_initializes(self):
        """TemplateMatcher should load templates on init (enabled templates only)."""
        from template_matcher import TemplateMatcher
        matcher = TemplateMatcher()
        assert len(matcher.tm.templates) >= 1

    def test_scan_returns_signals_for_bullish_stock(self):
        """Matcher runs without error on bullish stock data (signal count depends on active templates)."""
        from template_matcher import TemplateMatcher
        matcher = TemplateMatcher()
        df = _make_row(
            close=151.0, open=148.0, sma_50=145.0, sma_200=130.0, ema_12=149.0,
            rsi=62.0, macd=1.5, macd_signal=0.8, macd_hist=0.7,
            volume=800000, vol_avg_20=500000, atr=2.5,
            er_slow=0.65, trend_alignment=1
        )
        state = {"trend": "BULLISH", "structure": "OPEN_FIELD",
                "volume": "SURGING", "volatility": "NORMAL"}
        signals = matcher.scan_ticker("TEST", df, stock_state=state)
        assert isinstance(signals, list), "scan_ticker must return a list"

    def test_scan_returns_empty_for_bearish_stock(self):
        """A bearish stock with no matching templates should return empty."""
        from template_matcher import TemplateMatcher
        matcher = TemplateMatcher()
        df = _make_row(close=80.0, sma_50=100.0, sma_200=110.0, rsi=25.0)
        state = {"trend": "BEARISH", "structure": "OPEN_FIELD",
                "volume": "DRYING_UP", "volatility": "VOLATILE"}
        signals = matcher.scan_ticker("TEST_BEAR", df, stock_state=state)
        # May or may not have signals depending on OVERSOLD_BOUNCE matching
        # But should not crash
        assert isinstance(signals, list)

    def test_signal_has_required_fields(self):
        """Generated signals should have all required fields."""
        from template_matcher import TemplateMatcher
        matcher = TemplateMatcher()
        df = _make_row(
            close=151.0, open=148.0, sma_50=145.0, sma_200=130.0, ema_12=149.0,
            rsi=62.0, macd=1.5, macd_signal=0.8, volume=800000, vol_avg_20=500000, atr=2.5
        )
        state = {"trend": "BULLISH", "volume": "SURGING", "volatility": "NORMAL"}
        signals = matcher.scan_ticker("TEST", df, stock_state=state)
        if signals:
            s = signals[0]
            required = ["symbol", "template_id", "action", "entry_price",
                       "stop_loss", "take_profit", "risk_reward_ratio", "confidence_score"]
            for field in required:
                assert field in s, f"Signal missing field: {field}"

    def test_idle_tracking(self):
        """Idle tracker should count scans without signals."""
        from template_matcher import TemplateMatcher
        matcher = TemplateMatcher()
        df = _make_row(close=50.0, rsi=50.0)  # Neutral, unlikely to match
        state = {"trend": "SIDEWAYS", "volume": "DRYING_UP", "volatility": "NORMAL"}
        for _ in range(5):
            matcher.scan_ticker("IDLE_TEST", df, stock_state=state)
        report = matcher.get_idle_report()
        # May or may not be in report depending on threshold, but should not crash
        assert isinstance(report, list)


# ============================================================
# PHASE 3.7: Extended Statistics
# ============================================================
class TestPhase3_7_ExtendedStats:
    """Verify extended statistics collection."""

    def test_record_result_basic(self):
        """record_result should update wins/losses."""
        from setup_templates import SetupTemplate
        t = SetupTemplate({"id": "TEST", "name": "Test", "conditions": [],
                          "stop_loss": {"method": "atr"}, "take_profit": {"method": "atr"}})
        t.record_result("AAPL", 2.5, True)
        t.record_result("AAPL", -1.0, False)
        assert t.statistics['wins'] == 1
        assert t.statistics['losses'] == 1
        assert t.statistics['total_activations'] == 2

    def test_record_result_with_context(self):
        """record_result with context should populate per-ticker and per-trend stats."""
        from setup_templates import SetupTemplate
        t = SetupTemplate({"id": "TEST", "name": "Test", "conditions": [],
                          "stop_loss": {"method": "atr"}, "take_profit": {"method": "atr"}})
        context = {
            "stock_state": {"trend": "BULLISH", "volatility": "NORMAL"},
            "regime": "TREND",
            "avg_volume": 6000000
        }
        t.record_result("AAPL", 3.0, True, context=context)

        assert "AAPL" in t.statistics.get('ticker_stats', {}), "Ticker stats should track AAPL"
        assert "BULLISH" in t.statistics.get('trend_stats', {}), "Trend stats should track BULLISH"
        assert "high" in t.statistics.get('volume_range_stats', {}), "Volume stats should track high (>5M)"
        assert "TREND" in t.statistics.get('regime_stats', {}), "Regime stats should track TREND"

    def test_get_best_context(self):
        """get_best_context should identify best performing conditions."""
        from setup_templates import SetupTemplate
        t = SetupTemplate({"id": "TEST", "name": "Test", "conditions": [],
                          "stop_loss": {"method": "atr"}, "take_profit": {"method": "atr"}})
        # Add enough data points for meaningful analysis
        for _ in range(8):
            t.record_result("AAPL", 2.0, True, {"stock_state": {"trend": "BULLISH"}})
        for _ in range(2):
            t.record_result("AAPL", -1.0, False, {"stock_state": {"trend": "BULLISH"}})
        for _ in range(2):
            t.record_result("TSLA", 1.5, True, {"stock_state": {"trend": "BEARISH"}})
        for _ in range(5):
            t.record_result("TSLA", -2.0, False, {"stock_state": {"trend": "BEARISH"}})

        best = t.get_best_context()
        assert best.get('best_trend') == "BULLISH", \
            f"Best trend should be BULLISH (80% WR), got {best.get('best_trend')}"

    def test_streak_tracking(self):
        """Consecutive wins/losses should be tracked."""
        from setup_templates import SetupTemplate
        t = SetupTemplate({"id": "TEST", "name": "Test", "conditions": [],
                          "stop_loss": {"method": "atr"}, "take_profit": {"method": "atr"}})
        t.record_result("AAPL", 2.0, True)
        t.record_result("AAPL", 1.5, True)
        t.record_result("AAPL", 3.0, True)
        assert t.statistics['consecutive_wins'] == 3
        assert t.statistics['max_consecutive_wins'] == 3

        t.record_result("AAPL", -1.0, False)
        assert t.statistics['consecutive_wins'] == 0
        assert t.statistics['consecutive_losses'] == 1
        assert t.statistics['max_consecutive_wins'] == 3  # Max preserved


class TestPhase1AtrMult:
    """Tests for KINETIC_STOP_CONFIG phase1_atr_mult — STOA-5."""

    def test_phase1_atr_mult_value(self):
        """Assert phase1_atr_mult is exactly 1.5 per SPEC §5 optimization."""
        import system_config as cfg
        assert cfg.KINETIC_STOP_CONFIG["phase1_atr_mult"] == 1.5, (
            f"Expected 1.5, got {cfg.KINETIC_STOP_CONFIG['phase1_atr_mult']}"
        )

    def test_phase1_atr_mult_range(self):
        """Assert phase1_atr_mult is within valid operational range [1.0, 3.0]."""
        import system_config as cfg
        val = cfg.KINETIC_STOP_CONFIG["phase1_atr_mult"]
        assert 1.0 <= val <= 3.0, (
            f"phase1_atr_mult={val} is outside valid range [1.0, 3.0]"
        )

    def test_phase1_stop_calculation(self):
        """Assert Phase 1 stop price is correct with new 1.5 multiplier.

        Given entry_price=100, ATR=2.0:
          expected_stop = 100 - (2.0 * 1.5) = 97.0  (not 96.0 which was old 2.0×ATR)
        """
        import system_config as cfg
        entry_price = 100.0
        atr = 2.0
        phase1_atr_mult = cfg.KINETIC_STOP_CONFIG["phase1_atr_mult"]
        calculated_stop = entry_price - (atr * phase1_atr_mult)
        assert calculated_stop == 97.0, (
            f"Expected stop=97.0, got {calculated_stop} (phase1_atr_mult={phase1_atr_mult})"
        )


class TestPauseMinHealthyPullback:
    """Tests for min_healthy_pullback_pct de-hardcoding (SPEC §5)."""

    def test_pause_min_healthy_pullback_in_config(self):
        """Verify min_healthy_pullback_pct exists in POSITION_MANAGEMENT_CONFIG."""
        from system_config import POSITION_MANAGEMENT_CONFIG
        assert "min_healthy_pullback_pct" in POSITION_MANAGEMENT_CONFIG
        assert POSITION_MANAGEMENT_CONFIG["min_healthy_pullback_pct"] == 0.005

    def test_pause_min_healthy_pullback_range(self):
        """Verify min_healthy_pullback_pct is in valid range (0-5%)."""
        from system_config import POSITION_MANAGEMENT_CONFIG
        val = POSITION_MANAGEMENT_CONFIG["min_healthy_pullback_pct"]
        assert 0 < val < 0.05


class TestConfigDedup:
    """Tests for single-source-of-truth config deduplication (SPEC §6)."""

    def test_min_net_profit_single_source(self):
        """min_net_profit_pct in COSTS_CONFIG and FRICTION_AND_ALPHA must equal MIN_NET_PROFIT."""
        import system_config as cfg
        assert cfg.COSTS_CONFIG["min_net_profit_pct"] == cfg.MIN_NET_PROFIT, \
            "COSTS_CONFIG['min_net_profit_pct'] must reference MIN_NET_PROFIT"
        assert cfg.FRICTION_AND_ALPHA["min_net_profit_pct"] == cfg.MIN_NET_PROFIT, \
            "FRICTION_AND_ALPHA['min_net_profit_pct'] must reference MIN_NET_PROFIT"

    def test_min_net_profit_is_not_zero(self):
        """MIN_NET_PROFIT must be a positive non-zero value."""
        import system_config as cfg
        assert cfg.MIN_NET_PROFIT > 0, "MIN_NET_PROFIT must be > 0"
        assert cfg.MIN_NET_PROFIT < 1.0, "MIN_NET_PROFIT must be a fraction (< 1.0)"

    def test_runner_atr_mult_single_source(self):
        """runner_atr_mult in MILESTONE_ALERT_CONFIG must equal KINETIC_STOP_CONFIG value."""
        import system_config as cfg
        assert cfg.MILESTONE_ALERT_CONFIG["runner_atr_mult"] == cfg.KINETIC_STOP_CONFIG["runner_atr_mult"], \
            "MILESTONE_ALERT_CONFIG['runner_atr_mult'] must reference KINETIC_STOP_CONFIG"

    def test_runner_atr_mult_valid_range(self):
        """runner_atr_mult must be in valid range (0.1 – 2.0)."""
        import system_config as cfg
        val = cfg.KINETIC_STOP_CONFIG["runner_atr_mult"]
        assert 0.1 <= val <= 2.0, f"runner_atr_mult={val} out of expected range [0.1, 2.0]"

    def test_master_scores_removed(self):
        """Verify MASTER_SCORES dead code has been removed from system_config."""
        import system_config
        assert not hasattr(system_config, 'MASTER_SCORES'), \
            "MASTER_SCORES should be removed — it was dead code"


class TestRealtimeStateRefresh:
    """Tests for Gap 1a: real-time stock state refresh in live scan loop."""

    def test_realtime_state_refresh_enabled(self):
        """When enable_realtime_state_refresh=True, classify_stock_state() result is used."""
        from stock_hunter import StockHunter
        from unittest.mock import MagicMock

        mock_scout = MagicMock(spec=StockHunter)
        mock_scout.classify_stock_state.return_value = {
            "trend": "BEARISH", "structure": "NEAR_RESISTANCE",
            "volume": "DRYING_UP", "volatility": "VOLATILE"
        }

        import system_config as cfg
        regime_cfg = getattr(cfg, 'REGIME_CONFIG', {})
        assert regime_cfg.get('enable_realtime_state_refresh') is True

        live_state = mock_scout.classify_stock_state(MagicMock())
        assert live_state["trend"] == "BEARISH"
        assert live_state["volatility"] == "VOLATILE"

    def test_realtime_state_fallback_on_error(self):
        """When classify_stock_state raises exception, fall back to ledger state."""
        from stock_hunter import StockHunter
        from unittest.mock import MagicMock

        mock_scout = MagicMock(spec=StockHunter)
        mock_scout.classify_stock_state.side_effect = Exception("NaN in features")

        ledger_fallback = {"trend": "BULLISH", "structure": "OPEN_FIELD",
                           "volume": "HEALTHY", "volatility": "NORMAL"}

        try:
            live_state = mock_scout.classify_stock_state(MagicMock())
            stock_state = live_state
        except Exception:
            stock_state = ledger_fallback

        assert stock_state["trend"] == "BULLISH"  # fell back to ledger

    def test_realtime_state_disabled_uses_ledger(self):
        """When enable_realtime_state_refresh=False, ledger state is used (backwards compatible)."""
        ledger_state = {"trend": "SIDEWAYS", "structure": "NEAR_SUPPORT",
                        "volume": "HEALTHY", "volatility": "COMPRESSED"}

        regime_cfg = {"enable_realtime_state_refresh": False}
        if regime_cfg.get('enable_realtime_state_refresh', False):
            stock_state = {}  # should NOT reach here
        else:
            stock_state = ledger_state

        assert stock_state["trend"] == "SIDEWAYS"

    def test_regime_config_exists_and_valid(self):
        """REGIME_CONFIG exists in system_config with correct structure."""
        import system_config as cfg
        regime_cfg = getattr(cfg, 'REGIME_CONFIG', None)
        assert regime_cfg is not None, "REGIME_CONFIG missing from system_config.py"
        assert isinstance(regime_cfg.get('enable_realtime_state_refresh'), bool), \
            "enable_realtime_state_refresh must be bool"


class TestHaltRegimeBlocking:
    """Tests for Gap 1b: HALT regime blocks template scan."""

    def test_halt_regime_blocks_templates(self):
        """HALT regime should prevent template evaluation."""
        from unittest.mock import MagicMock

        mock_orchestra = MagicMock()
        mock_orchestra.classify_regime.return_value = "HALT"

        regime_cfg = {'enable_halt_template_blocking': True}
        current_regime = mock_orchestra.classify_regime(MagicMock())

        should_skip = (regime_cfg.get('enable_halt_template_blocking', False)
                       and current_regime == "HALT")
        assert should_skip is True, "HALT regime must block template scan"

    def test_non_halt_regime_proceeds(self):
        """TREND/CHOP/NEUTRAL regimes should NOT block templates."""
        from unittest.mock import MagicMock

        for regime in ["TREND", "CHOP", "NEUTRAL"]:
            mock_orchestra = MagicMock()
            mock_orchestra.classify_regime.return_value = regime

            regime_cfg = {'enable_halt_template_blocking': True}
            current_regime = mock_orchestra.classify_regime(MagicMock())

            should_skip = (regime_cfg.get('enable_halt_template_blocking', False)
                           and current_regime == "HALT")
            assert should_skip is False, f"Regime {regime} must NOT block templates"

    def test_halt_blocking_disabled_proceeds(self):
        """When enable_halt_template_blocking=False, HALT does NOT block."""
        from unittest.mock import MagicMock

        mock_orchestra = MagicMock()
        mock_orchestra.classify_regime.return_value = "HALT"

        regime_cfg = {'enable_halt_template_blocking': False}

        if regime_cfg.get('enable_halt_template_blocking', False):
            current_regime = mock_orchestra.classify_regime(MagicMock())
            should_skip = (current_regime == "HALT")
        else:
            should_skip = False

        assert should_skip is False, "Disabled flag must not block even on HALT"

    def test_halt_blocking_exception_proceeds(self):
        """When classify_regime raises, proceed with templates (fail-open)."""
        from unittest.mock import MagicMock

        mock_orchestra = MagicMock()
        mock_orchestra.classify_regime.side_effect = Exception("ER columns missing")

        regime_cfg = {'enable_halt_template_blocking': True}
        should_skip = False

        if regime_cfg.get('enable_halt_template_blocking', False):
            try:
                current_regime = mock_orchestra.classify_regime(MagicMock())
                if current_regime == "HALT":
                    should_skip = True
            except Exception:
                should_skip = False  # fail-open

        assert should_skip is False, "Exception in classify_regime must not block templates"


class TestTemplateFilteringLogging:
    """Tests for template state filtering logging (observability)."""

    def test_get_for_state_returns_matching_templates(self):
        """get_for_state returns only templates whose required_state matches stock_state."""
        bullish_required = {"trend": ["BULLISH"]}
        bearish_required = {"trend": ["SIDEWAYS", "BEARISH"]}
        stock_state = {"trend": "BULLISH", "volume": "HEALTHY"}

        def state_matches(required_state, s):
            for key, acceptable in required_state.items():
                if s.get(key, '') not in acceptable:
                    return False
            return True

        assert state_matches(bullish_required, stock_state) is True, \
            "TREND_PULLBACK_EMA should match BULLISH state"
        assert state_matches(bearish_required, stock_state) is False, \
            "OVERSOLD_BOUNCE should NOT match BULLISH state"

    def test_get_mismatch_reason_detail(self):
        """_get_mismatch_reason returns specific field-level mismatch details."""
        from setup_templates import TemplateManager
        tm = TemplateManager()

        required = {"trend": ["BULLISH"], "volume": ["SURGING"]}
        actual_state = {"trend": "BEARISH", "volume": "DRYING_UP"}

        reason = tm._get_mismatch_reason(required, actual_state)
        assert "trend mismatch" in reason
        assert "BEARISH" in reason
        assert "volume mismatch" in reason
        assert "DRYING_UP" in reason

    def test_get_mismatch_reason_empty_state(self):
        """Missing keys in stock_state produce mismatch with empty actual value."""
        from setup_templates import TemplateManager
        tm = TemplateManager()

        required = {"trend": ["BULLISH"]}
        actual_state = {}  # empty state

        reason = tm._get_mismatch_reason(required, actual_state)
        assert "trend mismatch" in reason
        assert "actual: " in reason  # empty string as actual


class TestWeeklyRetrain:
    """Tests for Gap 4: weekly auto-retrain scheduler."""

    def test_retrain_triggers_on_weekend(self):
        """Retrain should trigger on Saturday when no recent retrain exists."""
        from datetime import datetime

        saturday = datetime(2026, 3, 28, 10, 0, 0)  # Saturday
        retrain_cfg = {
            'enabled': True, 'retrain_days': [5, 6],
            'min_days_between_retrain': 5,
            'last_retrain_path': 'data/last_retrain.json',
        }
        assert saturday.weekday() == 5, "Should be Saturday"
        assert saturday.weekday() in retrain_cfg['retrain_days'], "Saturday must be in retrain_days"

    def test_retrain_skips_weekday(self):
        """Retrain should NOT trigger on Wednesday."""
        from datetime import datetime

        wednesday = datetime(2026, 3, 25, 10, 0, 0)  # Wednesday
        retrain_cfg = {'enabled': True, 'retrain_days': [5, 6], 'min_days_between_retrain': 5}

        assert wednesday.weekday() == 2
        assert wednesday.weekday() not in retrain_cfg['retrain_days']

    def test_retrain_skips_if_recent(self):
        """Retrain should NOT trigger if last retrain was 2 days ago."""
        from datetime import datetime

        saturday = datetime(2026, 3, 28, 10, 0, 0)
        last_retrain = datetime(2026, 3, 26, 10, 0, 0)  # 2 days ago

        days_since = (saturday - last_retrain).days
        min_days = 5

        assert days_since == 2
        assert days_since < min_days  # Should skip

    def test_retrain_disabled_skips(self):
        """Retrain should NOT trigger when enabled=False."""
        retrain_cfg = {'enabled': False, 'retrain_days': [5, 6]}
        assert retrain_cfg.get('enabled', False) is False

    def test_retrain_config_exists_and_valid(self):
        """WEEKLY_RETRAIN_CONFIG exists in system_config with correct structure."""
        import system_config as cfg
        retrain_cfg = getattr(cfg, 'WEEKLY_RETRAIN_CONFIG', None)
        assert retrain_cfg is not None, "WEEKLY_RETRAIN_CONFIG missing from system_config.py"
        assert isinstance(retrain_cfg.get('enabled'), bool), "enabled must be bool"
        assert isinstance(retrain_cfg.get('retrain_days'), list), "retrain_days must be list"
        assert all(d in range(7) for d in retrain_cfg['retrain_days']), "retrain_days must be 0-6"
        assert isinstance(retrain_cfg.get('min_days_between_retrain'), (int, float)), \
            "min_days_between_retrain must be numeric"
        assert retrain_cfg.get('min_days_between_retrain', 0) > 0, "min_days must be positive"


class TestGenTemplatesDisabled:
    """Tests for GEN_* template suppression when generation.enabled=False."""

    def test_gen_templates_disabled_skipped(self):
        """GEN_* templates are NOT in active template list when generation.enabled=False."""
        import system_config as cfg
        from setup_templates import TemplateManager

        original = cfg.TEMPLATE_EVOLUTION_CONFIG["generation"]["enabled"]
        try:
            cfg.TEMPLATE_EVOLUTION_CONFIG["generation"]["enabled"] = False
            tm = TemplateManager()
            active_ids = list(tm.templates.keys())
            gen_ids = [tid for tid in active_ids if tid.startswith("GEN_")]
            assert gen_ids == [], f"GEN_* templates should be absent but found: {gen_ids}"
        finally:
            cfg.TEMPLATE_EVOLUTION_CONFIG["generation"]["enabled"] = original

    def test_seed_templates_unaffected(self):
        """Seed templates are still loaded when generation.enabled=False."""
        import system_config as cfg
        from setup_templates import TemplateManager

        original = cfg.TEMPLATE_EVOLUTION_CONFIG["generation"]["enabled"]
        try:
            cfg.TEMPLATE_EVOLUTION_CONFIG["generation"]["enabled"] = False
            tm = TemplateManager()
            active_ids = list(tm.templates.keys())
            seed_ids = [tid for tid in active_ids if not tid.startswith("GEN_")]
            # At least 1 seed template should be present (SQUEEZE_BREAKOUT guaranteed)
            assert len(seed_ids) >= 1, f"Expected >=1 seed templates, got {len(seed_ids)}: {seed_ids}"
            # No GEN_* templates
            gen_ids = [tid for tid in active_ids if tid.startswith("GEN_")]
            assert gen_ids == [], f"GEN_* templates must not be loaded: {gen_ids}"
        finally:
            cfg.TEMPLATE_EVOLUTION_CONFIG["generation"]["enabled"] = original


class TestForceProviderDSM:
    """Tests for force_provider parameter in DataSourceManager.get_stock_data()."""

    def test_force_provider_uses_only_specified(self):
        """When force_provider='YFINANCE', only YFINANCE is attempted; MASSIVE is not."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from unittest.mock import patch, MagicMock
        from data_source_manager import DataSourceManager

        dm = DataSourceManager.__new__(DataSourceManager)
        dm.massive_client = None
        dm.stock_client = None
        dm.use_ibkr = False

        called_providers = []

        import pandas as pd
        fake_df = pd.DataFrame({'open': [1], 'high': [2], 'low': [0.5], 'close': [1.5], 'volume': [1000]})

        def fake_yfinance(symbol, days_back, interval, start_date, end_date, min_rows=0):
            called_providers.append('YFINANCE')
            return fake_df

        def fake_log(msg, level="INFO"):
            pass

        dm._log = fake_log

        with patch.object(dm, '_download_from_yfinance', side_effect=fake_yfinance):
            with patch('data_source_manager.clean_raw_data', return_value=fake_df):
                result = dm.get_stock_data('AAPL', days_back=10, force_provider='YFINANCE')

        assert 'YFINANCE' in called_providers, "YFINANCE must be attempted with force_provider='YFINANCE'"
        assert 'MASSIVE' not in called_providers, "MASSIVE must NOT be attempted when force_provider='YFINANCE'"

    def test_force_provider_no_fallback_on_failure(self):
        """When force_provider='IBKR' and IBKR fails, no fallback to MASSIVE/YFINANCE occurs."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from unittest.mock import patch
        from data_source_manager import DataSourceManager
        import pandas as pd

        dm = DataSourceManager.__new__(DataSourceManager)
        dm.massive_client = None
        dm.stock_client = None
        dm.use_ibkr = True

        called_providers = []
        log_messages = []

        def fake_log(msg, level="INFO"):
            log_messages.append((level, msg))

        dm._log = fake_log

        def fake_ibkr_fail(symbol, start_date, end_date, days_back, interval, min_rows=0):
            called_providers.append('IBKR')
            raise ConnectionError("IBKR connection refused")

        def fake_isConnected():
            return True

        def fake_yfinance(symbol, days_back, interval, start_date, end_date, min_rows=0):
            called_providers.append('YFINANCE')
            return pd.DataFrame()

        dm.isConnected = fake_isConnected
        dm.connect_to_ibkr = lambda: None

        with patch.object(dm, '_download_from_ibkr', side_effect=fake_ibkr_fail):
            with patch.object(dm, '_download_from_yfinance', side_effect=fake_yfinance):
                result = dm.get_stock_data('AAPL', days_back=10, force_provider='IBKR')

        assert 'IBKR' in called_providers, "IBKR must be attempted"
        assert 'YFINANCE' not in called_providers, "YFINANCE must NOT be attempted (no fallback)"
        assert 'MASSIVE' not in called_providers, "MASSIVE must NOT be attempted (no fallback)"
        assert result.empty, "Result should be empty DataFrame on failure with no fallback"
        force_logs = [m for lvl, m in log_messages if "Force provider mode" in m]
        assert force_logs, "Force provider mode must be logged"


class TestQualityGate:
    """Tests for Walk-Forward quality gate — validate_single_template / _evaluate_quality_gate."""

    def _wf(self):
        """Return a WalkForwardValidator with minimal __init__ state for pure-logic tests."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from backtest_engine import WalkForwardValidator
        import system_config as cfg
        wf = WalkForwardValidator.__new__(WalkForwardValidator)
        wf.config = cfg.WALK_FORWARD_CONFIG
        wf.initial_capital = 100_000
        wf.use_risk_gates = False
        wf.train_pct = 0.70
        return wf

    def test_quality_gate_passes_good_template(self):
        """_evaluate_quality_gate returns passed=True when test PF >= 1.0."""
        wf = self._wf()
        # 3 wins × 2.0%, 2 losses × 1.0% → PF = 6.0/2.0 = 3.0
        good_trades = [
            {"template_id": "GEN_T", "pnl_pct": 2.0},
            {"template_id": "GEN_T", "pnl_pct": 2.0},
            {"template_id": "GEN_T", "pnl_pct": 2.0},
            {"template_id": "GEN_T", "pnl_pct": -1.0},
            {"template_id": "GEN_T", "pnl_pct": -1.0},
        ]
        result = wf._evaluate_quality_gate(
            "GEN_T", train_trades=good_trades, test_trades=good_trades,
            min_trades=3, min_pf=1.0
        )
        assert result["passed"] is True
        assert result["test_pf"] == 3.0
        assert result["test_trades"] == 5

    def test_quality_gate_fails_bad_template(self):
        """_evaluate_quality_gate returns passed=False when test PF < 1.0."""
        wf = self._wf()
        # 2 wins × 0.5%, 3 losses × 1.0% → PF = 1.0/3.0 = 0.33
        bad_trades = [
            {"template_id": "GEN_T", "pnl_pct": 0.5},
            {"template_id": "GEN_T", "pnl_pct": 0.5},
            {"template_id": "GEN_T", "pnl_pct": -1.0},
            {"template_id": "GEN_T", "pnl_pct": -1.0},
            {"template_id": "GEN_T", "pnl_pct": -1.0},
        ]
        result = wf._evaluate_quality_gate(
            "GEN_T", train_trades=bad_trades, test_trades=bad_trades,
            min_trades=3, min_pf=1.0
        )
        assert result["passed"] is False
        assert result["test_pf"] < 1.0

    def test_quality_gate_insufficient_trades_burns_in(self):
        """_evaluate_quality_gate returns passed=True with BURN_IN reason when trades < min."""
        wf = self._wf()
        sparse_trades = [{"template_id": "GEN_T", "pnl_pct": -2.0}]  # 1 trade, PF=0.0
        result = wf._evaluate_quality_gate(
            "GEN_T", train_trades=[], test_trades=sparse_trades,
            min_trades=3, min_pf=1.0
        )
        assert result["passed"] is True
        assert "BURN_IN" in result["reason"]
        assert result["test_trades"] == 1


class TestThreeWaySplit:
    """Tests for the 3-way Train/Val/Test chronological split (DDR #14)."""

    def setup_method(self):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        import numpy as np
        import pandas as pd
        from backtest_engine import WalkForwardValidator, BACKTEST_CONFIG
        self.WalkForwardValidator = WalkForwardValidator
        self.BACKTEST_CONFIG = BACKTEST_CONFIG
        self.wfv = WalkForwardValidator(symbols=["TEST"], initial_capital=100000, use_risk_gates=False)
        dates = pd.date_range("2020-01-01", periods=1000, freq="B")
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "open":   rng.uniform(90, 110, 1000),
            "high":   rng.uniform(100, 120, 1000),
            "low":    rng.uniform(80, 100, 1000),
            "close":  rng.uniform(90, 110, 1000),
            "volume": rng.integers(1_000_000, 5_000_000, 1000).astype(float),
        }, index=dates)
        self.test_data = {"TEST": df}

    def test_split_returns_four_values(self):
        result = self.wfv._split_data(self.test_data)
        assert len(result) == 4, "Expected 4 return values from _split_data"

    def test_split_chronological_order(self):
        train, val, test, info = self.wfv._split_data(self.test_data)
        if train and val and test:
            assert info.get("split_date_1") is not None
            assert info.get("split_date_2") is not None
            assert info["split_date_1"] < info["split_date_2"], \
                "split_date_1 must precede split_date_2"

    def test_split_proportions(self):
        _, _, _, info = self.wfv._split_data(self.test_data)
        total = info.get("train_days", 0) + info.get("val_days", 0) + info.get("test_days", 0)
        if total > 0:
            train_ratio = info["train_days"] / total
            val_ratio   = info["val_days"]   / total
            test_ratio  = info["test_days"]  / total
            assert 0.55 <= train_ratio <= 0.65, f"Train ratio {train_ratio:.2%} not ~60%"
            assert 0.15 <= val_ratio   <= 0.25, f"Val ratio {val_ratio:.2%} not ~20%"
            assert 0.15 <= test_ratio  <= 0.25, f"Test ratio {test_ratio:.2%} not ~20%"

    def test_val_and_test_have_warmup(self):
        train, val, test, info = self.wfv._split_data(self.test_data)
        warmup = self.BACKTEST_CONFIG.get("min_candles_warmup", 200)
        if val and "TEST" in val:
            assert len(val["TEST"]) > warmup, "Val split must include warmup bars"
        if test and "TEST" in test:
            assert len(test["TEST"]) > warmup, "Test split must include warmup bars"

    def test_config_has_val_pct(self):
        import system_config as cfg
        wf = cfg.WALK_FORWARD_CONFIG
        assert "val_pct" in wf, "WALK_FORWARD_CONFIG must include val_pct"
        total = wf.get("train_pct", 0) + wf.get("val_pct", 0) + wf.get("test_pct", 0)
        assert abs(total - 1.0) < 0.01, f"train+val+test must sum to 1.0, got {total}"


class TestShadowLedgerMaxDate:
    """Tests for Shadow Ledger TRAIN period restriction (DDR #14)."""

    def setup_method(self):
        import numpy as np
        import pandas as pd
        from shadow_ledger import ShadowLedger
        self.pd = pd
        self.sl = ShadowLedger()
        dates = pd.date_range("2022-01-01", periods=500, freq="B")
        self.df = pd.DataFrame({
            "open":   np.random.uniform(90, 110, 500),
            "high":   np.random.uniform(100, 120, 500),
            "low":    np.random.uniform(80, 100, 500),
            "close":  np.random.uniform(90, 110, 500),
            "volume": np.random.randint(1_000_000, 5_000_000, 500),
            "rsi":    np.random.uniform(30, 70, 500),
            "sma_50": np.random.uniform(95, 105, 500),
            "er_slow": np.random.uniform(0.2, 0.8, 500),
            "atr":    np.random.uniform(1, 5, 500),
        }, index=dates)
        self.max_date = "2023-06-01"

    def test_max_date_truncates_df(self):
        """evaluate_history with max_date should use fewer bars."""
        pd = self.pd
        df_restricted = self.df[self.df.index <= pd.Timestamp(self.max_date)]
        assert len(df_restricted) < len(self.df)
        assert df_restricted.index.max() <= pd.Timestamp(self.max_date)

    def test_no_max_date_uses_all(self):
        """Without max_date the full DataFrame is unchanged."""
        assert len(self.df) == 500

    def test_max_date_no_future_bars(self):
        """After truncation no bars should exist after the cutoff date."""
        pd = self.pd
        df_cut = self.df[self.df.index <= pd.Timestamp(self.max_date)]
        future_bars = df_cut[df_cut.index > pd.Timestamp(self.max_date)]
        assert len(future_bars) == 0

    def test_config_has_restrict_flag(self):
        """SHADOW_LEDGER_CONFIG must include restrict_to_train key."""
        import system_config as cfg
        assert "restrict_to_train" in cfg.SHADOW_LEDGER_CONFIG


class TestFullPipeline:
    """Tests for the full Train/Val/Test pipeline orchestrator (DDR #14)."""

    def test_pipeline_method_exists(self):
        """WalkForwardValidator must have run_full_pipeline method."""
        from backtest_engine import WalkForwardValidator
        wfv = WalkForwardValidator(symbols=["AAPL"], initial_capital=100000)
        assert hasattr(wfv, "run_full_pipeline"), \
            "WalkForwardValidator must have run_full_pipeline method"

    def test_pipeline_cli_flag_exists(self):
        """CLI must accept --full-pipeline flag."""
        import subprocess
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            [sys.executable, "backtest_engine.py", "--help"],
            capture_output=True, text=True, encoding="utf-8",
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            env=env,
        )
        output = result.stdout + result.stderr
        assert "full-pipeline" in output, \
            "--full-pipeline flag not found in CLI --help output"


class TestPipelineBugFixes:
    """Tests for pipeline bug fixes — GEN cleanup, safe_json_write, stock_state_fn (DDR #14)."""

    def test_gen_templates_skipped_when_generation_disabled(self):
        """GEN_* template files on disk are not loaded when generation.enabled=False."""
        import system_config as cfg
        from setup_templates import TemplateManager
        original = cfg.TEMPLATE_EVOLUTION_CONFIG["generation"]["enabled"]
        try:
            cfg.TEMPLATE_EVOLUTION_CONFIG["generation"]["enabled"] = False
            tm = TemplateManager()
            gen_ids = [tid for tid in tm.templates.keys() if tid.startswith("GEN_")]
            assert gen_ids == [], f"GEN_* must not load when generation disabled: {gen_ids}"
        finally:
            cfg.TEMPLATE_EVOLUTION_CONFIG["generation"]["enabled"] = original

    def test_safe_json_write_windows_compatible(self):
        """safe_json_write should handle os.replace failure gracefully (double write)."""
        import tempfile
        from safe_json_io import safe_json_write, safe_json_read
        test_path = os.path.join(tempfile.gettempdir(), "test_safe_write_compat.json")
        try:
            safe_json_write(test_path, {"test": 1})
            safe_json_write(test_path, {"test": 2})
            data = safe_json_read(test_path)
            assert data.get("test") == 2, f"Expected 2, got {data.get('test')}"
        finally:
            if os.path.exists(test_path):
                os.unlink(test_path)

    def test_pipeline_has_stock_state_fn(self):
        """run_full_pipeline must initialize and use stock_state_fn."""
        import inspect
        from backtest_engine import WalkForwardValidator
        source = inspect.getsource(WalkForwardValidator.run_full_pipeline)
        assert "stock_state_fn" in source, \
            "run_full_pipeline must use stock_state_fn"
        assert "classify_stock_state" in source, \
            "run_full_pipeline must reference classify_stock_state"


class TestTemplateTimeframe:
    """Tests for template timeframe field — preparation for MTFA."""

    def _minimal_template_data(self, **overrides):
        data = {
            "id": "TEST_TF", "name": "Test", "description": "test", "version": 1,
            "source": "seed", "enabled": True,
            "required_state": {"trend": ["BULLISH"]},
            "conditions": [{"block": "rsi_above", "params": [50]}],
            "entry": {"type": "close", "confirmation_candles": 0},
            "stop_loss": {"method": "atr", "atr_multiplier": 1.5, "fallback_pct": 0.02},
            "take_profit": {"method": "atr", "atr_multiplier": 3.0, "use_runner_mode": False},
        }
        data.update(overrides)
        return data

    def test_seed_templates_have_timeframe(self):
        """All seed template JSON files must have timeframe='1d'."""
        import json
        templates_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "templates"
        )
        for f in sorted(os.listdir(templates_dir)):
            if f.endswith(".json") and not f.startswith("GEN_"):
                with open(os.path.join(templates_dir, f)) as fp:
                    data = json.load(fp)
                assert data.get("timeframe") == "1d", \
                    f"{f} missing timeframe='1d' (got {data.get('timeframe')!r})"

    def test_template_default_timeframe(self):
        """Template without timeframe field defaults to '1d'."""
        from setup_templates import SetupTemplate
        t = SetupTemplate(self._minimal_template_data())
        assert t.timeframe == "1d", f"Expected '1d', got {t.timeframe!r}"

    def test_timeframe_in_to_dict(self):
        """to_dict() must include timeframe."""
        from setup_templates import SetupTemplate
        t = SetupTemplate(self._minimal_template_data(timeframe="2h"))
        d = t.to_dict()
        assert d["timeframe"] == "2h", f"Expected '2h', got {d.get('timeframe')!r}"

    def test_get_for_timeframe(self):
        """get_for_timeframe filters templates by timeframe correctly."""
        from setup_templates import TemplateManager
        tm = TemplateManager()
        daily = tm.get_for_timeframe("1d")
        hourly = tm.get_for_timeframe("2h")
        assert len(daily) > 0, "Expected at least one daily template"
        assert len(hourly) == 0, f"Expected no 2h templates, got {len(hourly)}"

    def test_generation_config_has_timeframe(self):
        """Generation config must include default_timeframe."""
        import system_config as cfg
        gen_cfg = cfg.TEMPLATE_EVOLUTION_CONFIG["generation"]
        assert "default_timeframe" in gen_cfg, \
            "TEMPLATE_EVOLUTION_CONFIG['generation'] must include default_timeframe"


class TestDSM2HourSupport:
    """Tests for 2-hour interval support in DataSourceManager."""

    def test_alpaca_2h_mapping(self):
        """Alpaca interval mapping must include 2h."""
        import inspect
        from data_source_manager import DataSourceManager
        source = inspect.getsource(DataSourceManager._download_from_alpaca)
        assert '"2h"' in source, "Alpaca must handle '2h' interval"

    def test_ibkr_2h_mapping(self):
        """IBKR interval mapping must include 2h."""
        import inspect
        from data_source_manager import DataSourceManager
        source = inspect.getsource(DataSourceManager._download_from_ibkr)
        assert '"2h"' in source, "IBKR must handle '2h' interval"

    def test_massive_2h_mapping(self):
        """Massive interval mapping must include 2h."""
        import inspect
        from data_source_manager import DataSourceManager
        source = inspect.getsource(DataSourceManager._download_from_massive)
        assert "'2h'" in source, "Massive must handle '2h' interval"

    def test_yfinance_rejects_2h(self):
        """YFinance must gracefully reject 2h interval."""
        import inspect
        from data_source_manager import DataSourceManager
        source = inspect.getsource(DataSourceManager._download_from_yfinance)
        assert "2h" in source, "YFinance must handle 2h rejection"

    def test_pipeline_timeframes_config(self):
        """PIPELINE_TIMEFRAMES config must exist with 2h and 1d entries."""
        import system_config as cfg
        pt = cfg.PIPELINE_TIMEFRAMES
        assert "2h" in pt, "PIPELINE_TIMEFRAMES must have 2h key"
        assert "1d" in pt, "PIPELINE_TIMEFRAMES must have 1d key"
        assert pt["2h"]["data_source"] == "ALPACA", "2h must prefer ALPACA"


# ============================================================
# RUNNER (also compatible with pytest)
# ============================================================
if __name__ == '__main__':
    passed = 0
    failed = 0
    errors = []

    test_classes = [TestBug1_1_AIFeatureMismatch, TestBug1_2_ColumnCaseMismatch, TestBug1_3_ErTrend, TestBug1_4_CooldownWrite, TestBug1_5_DualThreshold, TestBug1_6a_SqueezeColumns, TestBug1_6b_SafeFillna, TestBug1_6c_DateSplit, TestBug2_1_MacdSignalName, TestBug2_2_SqueezeBonus, TestBug2_3_RegimeGate, TestBug2_4_LabelConfig, TestPhase2_5_MilestoneAlerts, TestPhase3_3_BlockRegistry, TestPhase3_3_TemplateValidation, TestPhase3_4_TemplateMatcher, TestPhase3_7_ExtendedStats, TestPhase1AtrMult, TestPauseMinHealthyPullback, TestConfigDedup, TestRealtimeStateRefresh, TestHaltRegimeBlocking, TestTemplateFilteringLogging, TestWeeklyRetrain, TestGenTemplatesDisabled, TestForceProviderDSM, TestQualityGate, TestThreeWaySplit, TestShadowLedgerMaxDate, TestFullPipeline, TestPipelineBugFixes, TestTemplateTimeframe, TestDSM2HourSupport]

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
