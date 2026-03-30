# tests/test_feature_engine.py
"""
TDD v1.1 §4 — Feature Engine Tests
====================================
20 unit tests covering:
  VG-01→08: check_veto_gates() — all three gate paths
  CN-01→05: _reduce_candle_noise() — family grouping + pass-through
  RC-01→07: Regime classification columns from calculate_features()

Tests are self-contained: no DB, no network, no file I/O.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_ohlcv(n=250, base_close=100.0):
    """Return a minimal OHLCV DataFrame with n rows and a DatetimeIndex."""
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    closes = base_close + np.cumsum(np.random.default_rng(42).normal(0, 0.5, n))
    closes = np.maximum(closes, 1.0)
    df = pd.DataFrame({
        "open":   closes * 0.998,
        "high":   closes * 1.005,
        "low":    closes * 0.993,
        "close":  closes,
        "volume": np.full(n, 500_000, dtype=float),
    }, index=idx)
    return df


def _make_fe():
    """Return a FeatureEngine instance (patched so no external calls needed)."""
    from feature_engine import FeatureEngine
    return FeatureEngine()


# ── VG-01: Empty DataFrame is vetoed ───────────────────────────────────────────

class TestVetoGateEmpty:
    """VG-01"""
    def test_empty_df_is_vetoed(self):
        fe = _make_fe()
        vetoed, reason = fe.check_veto_gates(pd.DataFrame())
        assert vetoed is True
        assert reason  # non-empty string


# ── VG-02: None DataFrame is vetoed ────────────────────────────────────────────

class TestVetoGateNone:
    """VG-02"""
    def test_none_df_is_vetoed(self):
        fe = _make_fe()
        vetoed, reason = fe.check_veto_gates(None)
        assert vetoed is True


# ── VG-03: Volume < 1 triggers Gate 1 ─────────────────────────────────────────

class TestVetoGateVolumeLow:
    """VG-03"""
    def test_zero_volume_vetoed(self):
        fe = _make_fe()
        df = _make_ohlcv(10)
        df.at[df.index[-1], "volume"] = 0.0
        vetoed, reason = fe.check_veto_gates(df)
        assert vetoed is True
        assert "Volume" in reason or "volume" in reason.lower()

    def test_negative_volume_vetoed(self):
        fe = _make_fe()
        df = _make_ohlcv(10)
        df.at[df.index[-1], "volume"] = -5.0
        vetoed, reason = fe.check_veto_gates(df)
        assert vetoed is True

    def test_nan_volume_vetoed(self):
        fe = _make_fe()
        df = _make_ohlcv(10)
        df.at[df.index[-1], "volume"] = float("nan")
        vetoed, reason = fe.check_veto_gates(df)
        assert vetoed is True


# ── VG-04: Good volume passes Gate 1 ──────────────────────────────────────────

class TestVetoGateVolumeOk:
    """VG-04"""
    def test_normal_volume_passes(self):
        fe = _make_fe()
        df = _make_ohlcv(5)
        # No death_cross / vsa_squat_bar columns → gates 2 & 3 default to False
        vetoed, _ = fe.check_veto_gates(df)
        assert vetoed is False


# ── VG-05: Death Cross on last row triggers Gate 2 ────────────────────────────

class TestVetoGateDeathCross:
    """VG-05"""
    def test_death_cross_true_vetoed(self):
        fe = _make_fe()
        df = _make_ohlcv(5)
        df["death_cross"] = False
        df.at[df.index[-1], "death_cross"] = True
        vetoed, reason = fe.check_veto_gates(df)
        assert vetoed is True
        assert "Death Cross" in reason

    def test_death_cross_false_passes(self):
        fe = _make_fe()
        df = _make_ohlcv(5)
        df["death_cross"] = False
        vetoed, _ = fe.check_veto_gates(df)
        assert vetoed is False


# ── VG-06: VSA Squat Bar on last row triggers Gate 3 ──────────────────────────

class TestVetoGateVsaSquat:
    """VG-06"""
    def test_vsa_squat_true_vetoed(self):
        fe = _make_fe()
        df = _make_ohlcv(5)
        df["vsa_squat_bar"] = False
        df.at[df.index[-1], "vsa_squat_bar"] = True
        vetoed, reason = fe.check_veto_gates(df)
        assert vetoed is True
        assert "VSA" in reason or "Squat" in reason

    def test_vsa_squat_false_passes(self):
        fe = _make_fe()
        df = _make_ohlcv(5)
        df["vsa_squat_bar"] = False
        vetoed, _ = fe.check_veto_gates(df)
        assert vetoed is False


# ── VG-07: Death cross NOT on last row (only historical) is not vetoed ─────────

class TestVetoGateDeathCrossHistorical:
    """VG-07 — death_cross=True only on a historical candle, last row is False"""
    def test_historical_death_cross_not_vetoed(self):
        fe = _make_fe()
        df = _make_ohlcv(5)
        df["death_cross"] = False
        df.at[df.index[2], "death_cross"] = True   # Historical candle
        vetoed, _ = fe.check_veto_gates(df)
        assert vetoed is False


# ── VG-08: Return type contract ───────────────────────────────────────────────

class TestVetoGateReturnType:
    """VG-08 — always returns (bool, str)"""
    def test_returns_bool_str_on_pass(self):
        fe = _make_fe()
        df = _make_ohlcv(5)
        result = fe.check_veto_gates(df)
        assert isinstance(result, tuple) and len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)

    def test_returns_bool_str_on_veto(self):
        fe = _make_fe()
        df = _make_ohlcv(5)
        df.at[df.index[-1], "volume"] = 0
        result = fe.check_veto_gates(df)
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)


# ── CN-01: Doji family → CANDLE_INDECISION ────────────────────────────────────

class TestCandleNoiseIndecision:
    """CN-01"""
    def test_doji_maps_to_indecision(self):
        fe = _make_fe()
        reduced = fe._reduce_candle_noise(["DOJI", "SPINNINGTOP"])
        assert "CANDLE_INDECISION" in reduced
        # Raw names must be absorbed — not passed through
        assert "DOJI" not in reduced
        assert "SPINNINGTOP" not in reduced


# ── CN-02: Hammer/Engulf → CANDLE_BULLISH_REVERSAL ────────────────────────────

class TestCandleNoiseBullish:
    """CN-02"""
    def test_hammer_maps_to_bullish(self):
        fe = _make_fe()
        reduced = fe._reduce_candle_noise(["HAMMER"])
        assert "CANDLE_BULLISH_REVERSAL" in reduced
        assert "HAMMER" not in reduced

    def test_engulfing_bull_maps_to_bullish(self):
        fe = _make_fe()
        reduced = fe._reduce_candle_noise(["ENGULFING_BULL"])
        assert "CANDLE_BULLISH_REVERSAL" in reduced


# ── CN-03: ShootingStar/Evening → CANDLE_BEARISH_REVERSAL ────────────────────

class TestCandleNoiseBearish:
    """CN-03"""
    def test_shootingstar_maps_to_bearish(self):
        fe = _make_fe()
        reduced = fe._reduce_candle_noise(["SHOOTINGSTAR"])
        assert "CANDLE_BEARISH_REVERSAL" in reduced
        assert "SHOOTINGSTAR" not in reduced

    def test_eveningstar_maps_to_bearish(self):
        fe = _make_fe()
        reduced = fe._reduce_candle_noise(["EVENINGSTAR"])
        assert "CANDLE_BEARISH_REVERSAL" in reduced


# ── CN-04: Unknown / structural patterns pass through unchanged ───────────────

class TestCandleNoisePassthrough:
    """CN-04"""
    def test_unknown_pattern_retained(self):
        fe = _make_fe()
        reduced = fe._reduce_candle_noise(["MOMENTUM_BREAKOUT", "SOME_CUSTOM"])
        assert "MOMENTUM_BREAKOUT" in reduced
        assert "SOME_CUSTOM" in reduced

    def test_empty_input_returns_empty(self):
        fe = _make_fe()
        reduced = fe._reduce_candle_noise([])
        assert reduced == []


# ── CN-05: Mixed families produce all three canonical labels ──────────────────

class TestCandleNoiseMixed:
    """CN-05"""
    def test_mixed_families_produce_all_labels(self):
        fe = _make_fe()
        raw = ["DOJI", "HAMMER", "SHOOTINGSTAR", "STRUCTURAL_PATTERN"]
        reduced = fe._reduce_candle_noise(raw)
        assert "CANDLE_INDECISION" in reduced
        assert "CANDLE_BULLISH_REVERSAL" in reduced
        assert "CANDLE_BEARISH_REVERSAL" in reduced
        assert "STRUCTURAL_PATTERN" in reduced
        # None of the raw family names survive
        for raw_name in ["DOJI", "HAMMER", "SHOOTINGSTAR"]:
            assert raw_name not in reduced


# ── RC-01: calculate_features produces sma_50 and sma_200 ────────────────────

class TestRegimeSMAColumns:
    """RC-01"""
    def test_sma_columns_present(self):
        fe = _make_fe()
        df = _make_ohlcv(260)
        result = fe.calculate_features(df)
        assert "sma_50" in result.columns, "sma_50 missing from calculate_features output"
        assert "sma_200" in result.columns, "sma_200 missing from calculate_features output"


# ── RC-02: death_cross and golden_cross columns produced ─────────────────────

class TestRegimeCrossColumns:
    """RC-02"""
    def test_cross_columns_present(self):
        fe = _make_fe()
        df = _make_ohlcv(260)
        result = fe.calculate_features(df)
        assert "death_cross" in result.columns
        assert "golden_cross" in result.columns

    def test_cross_columns_are_boolean_dtype(self):
        fe = _make_fe()
        df = _make_ohlcv(260)
        result = fe.calculate_features(df)
        # After NaN fill, values should be 0/False/True (boolean or numeric 0/1)
        unique_vals = set(result["death_cross"].dropna().unique())
        assert unique_vals.issubset({True, False, 0, 1, 0.0, 1.0})


# ── RC-03: death_cross is only True on the crossing candle ───────────────────

class TestDeathCrossSingleCandle:
    """RC-03 — crossing column should be sparse (not persistent)"""
    def test_death_cross_sparse(self):
        fe = _make_fe()
        # Build declining price so SMA50 crosses below SMA200 exactly once
        n = 260
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        # Start high, then drop sharply after halfway
        prices = np.concatenate([
            np.linspace(200, 210, 130),
            np.linspace(208, 80, 130),
        ])
        df = pd.DataFrame({
            "open": prices * 0.998, "high": prices * 1.005,
            "low": prices * 0.993, "close": prices,
            "volume": np.full(n, 500_000.0),
        }, index=idx)
        result = fe.calculate_features(df)
        # At most a handful of candles should show death_cross=True (not the entire tail)
        dc_count = (result["death_cross"] == True).sum()
        assert dc_count < 10, (
            f"death_cross appears True on {dc_count} candles — expected sparse single-cross"
        )


# ── RC-04: ADX column is produced ────────────────────────────────────────────

class TestRegimeADX:
    """RC-04"""
    def test_adx_column_present(self):
        fe = _make_fe()
        df = _make_ohlcv(260)
        result = fe.calculate_features(df)
        assert "adx" in result.columns, "adx missing — needed for regime strength assessment"

    def test_adx_values_in_range(self):
        fe = _make_fe()
        df = _make_ohlcv(260)
        result = fe.calculate_features(df)
        adx_vals = result["adx"].dropna()
        if len(adx_vals) > 0:
            assert adx_vals.min() >= 0
            assert adx_vals.max() <= 100


# ── RC-05: RSI column is produced ────────────────────────────────────────────

class TestRegimeRSI:
    """RC-05"""
    def test_rsi_column_present(self):
        fe = _make_fe()
        df = _make_ohlcv(260)
        result = fe.calculate_features(df)
        assert "rsi" in result.columns

    def test_rsi_values_in_range(self):
        fe = _make_fe()
        df = _make_ohlcv(260)
        result = fe.calculate_features(df)
        rsi_vals = result["rsi"].dropna()
        if len(rsi_vals) > 0:
            assert rsi_vals.min() >= 0
            assert rsi_vals.max() <= 100


# ── RC-06: vsa_squat_bar column produced by calculate_features ───────────────

class TestRegimeVsaSquat:
    """RC-06"""
    def test_vsa_squat_bar_column_present(self):
        fe = _make_fe()
        df = _make_ohlcv(260)
        result = fe.calculate_features(df)
        assert "vsa_squat_bar" in result.columns


# ── RC-07: calculate_features is idempotent (re-running doesn't crash) ────────

class TestCalculateFeaturesIdempotent:
    """RC-07"""
    def test_double_call_no_crash(self):
        fe = _make_fe()
        df = _make_ohlcv(260)
        result1 = fe.calculate_features(df.copy())
        result2 = fe.calculate_features(result1.copy())
        assert "sma_50" in result2.columns
        assert "death_cross" in result2.columns
        assert len(result2) == len(result1)
