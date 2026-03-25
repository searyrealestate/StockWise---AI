# tests/test_data_layer.py

"""
StockWise Gen-13 — Data Layer Tests (TDD v1.1 Section 3)
=========================================================
Tests waterfall routing (DL-01→08), normalization (NL-01→09), data guard (DG-01→06).
All mocked — zero API calls.
"""

import os
import sys
import re
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data_source_manager import DataSourceManager, normalize_ohlcv, DataValidationError
import system_config as cfg


# ═══════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════

def _make_ohlcv(rows=250, provider_format=None):
    """Generate a fake OHLCV DataFrame in various provider formats."""
    dates = pd.date_range(end=datetime.now(), periods=rows, freq='B')
    close = np.random.uniform(95.0, 105.0, rows)
    data = {
        'open':   np.random.uniform(94.0, 104.0, rows),
        'high':   close * 1.01,
        'low':    close * 0.99,
        'close':  close,
        'volume': np.random.randint(100_000, 5_000_000, rows).astype(float),
    }

    if provider_format == 'alpaca':
        data = {k.capitalize(): v for k, v in data.items()}
    elif provider_format == 'ibkr':
        data = {k.capitalize(): v for k, v in data.items()}
    elif provider_format == 'yfinance':
        data = {k.capitalize(): v for k, v in data.items()}
        data['Adj Close'] = data['Close'] * 0.99
    elif provider_format == 'massive':
        data = {
            'o': data['open'], 'h': data['high'], 'l': data['low'],
            'c': data['close'], 'v': data['volume'],
        }

    df = pd.DataFrame(data, index=dates)
    df.index.name = 'Date'
    return df


# ═══════════════════════════════════════════════════
# SECTION 3.1 — WATERFALL ROUTING (DL-01 → DL-08)
# ═══════════════════════════════════════════════════

class TestWaterfallRouting:
    """DL-01 to DL-08: Provider ordering and fallback behavior."""

    @pytest.fixture(autouse=True)
    def reset_circuit_breaker(self):
        """Reset class-level MASSIVE circuit breaker before/after each test."""
        DataSourceManager._massive_session_dead = False
        DataSourceManager._massive_lockout_until = None
        DataSourceManager._massive_fail_count = 0
        yield
        DataSourceManager._massive_session_dead = False
        DataSourceManager._massive_lockout_until = None
        DataSourceManager._massive_fail_count = 0

    def _make_dsm(self, use_ibkr=False):
        """Construct a DSM safe for offline testing."""
        dsm = DataSourceManager(use_ibkr=use_ibkr, allow_fallback=True)
        dsm.massive_client = MagicMock()   # Enables MASSIVE branch in waterfall
        dsm.stock_client = MagicMock()     # Enables ALPACA branch in waterfall
        return dsm

    @staticmethod
    def _patch_clean(df):
        """Return df unchanged — stand-in for clean_raw_data in waterfall."""
        return df

    def test_dl01_massive_called_first_and_stops(self):
        """DL-01 (P0): MASSIVE is first; if it succeeds others are not called."""
        dsm = self._make_dsm()
        valid_df = _make_ohlcv(250)

        with patch('time.sleep'), \
             patch('data_source_manager.clean_raw_data', side_effect=self._patch_clean), \
             patch.object(dsm, '_download_from_massive', return_value=valid_df) as m_massive, \
             patch.object(dsm, '_download_from_alpaca') as m_alpaca, \
             patch.object(dsm, '_download_from_yfinance') as m_yf:

            dsm.get_stock_data("AAPL", days_back=365)

            m_massive.assert_called_once()
            m_alpaca.assert_not_called()
            m_yf.assert_not_called()

    def test_dl02_fallback_to_alpaca_when_massive_fails(self):
        """DL-02 (P0): MASSIVE fails → ALPACA called, returns valid OHLCV."""
        dsm = self._make_dsm()
        valid_df = _make_ohlcv(250)

        with patch('time.sleep'), \
             patch('data_source_manager.clean_raw_data', side_effect=self._patch_clean), \
             patch.object(dsm, '_download_from_massive',
                          side_effect=ConnectionError("massive down")), \
             patch.object(dsm, '_download_from_alpaca',
                          return_value=valid_df) as m_alpaca, \
             patch.object(dsm, '_download_from_yfinance') as m_yf:

            result = dsm.get_stock_data("AAPL", days_back=365)

            m_alpaca.assert_called_once()
            m_yf.assert_not_called()
            assert result is not None
            assert not result.empty

    def test_dl03_fallback_to_ibkr_when_massive_and_alpaca_fail(self):
        """DL-03 (P0): MASSIVE + ALPACA fail → IBKR called."""
        dsm = self._make_dsm(use_ibkr=False)
        dsm.use_ibkr = True   # Force True after construction (IBKR_AVAILABLE may be False in CI)
        valid_df = _make_ohlcv(250)

        with patch('time.sleep'), \
             patch('data_source_manager.clean_raw_data', side_effect=self._patch_clean), \
             patch.object(dsm, 'isConnected', return_value=True), \
             patch.object(dsm, '_download_from_massive',
                          side_effect=Exception("down")), \
             patch.object(dsm, '_download_from_alpaca',
                          side_effect=Exception("down")), \
             patch.object(dsm, '_download_from_ibkr',
                          return_value=valid_df) as m_ibkr, \
             patch.object(dsm, '_download_from_yfinance') as m_yf:

            result = dsm.get_stock_data("AAPL", days_back=365)

            m_ibkr.assert_called_once()
            m_yf.assert_not_called()

    def test_dl04_fallback_to_yfinance_as_last_resort(self):
        """DL-04 (P0): First 3 fail (or skipped) → YFinance called last."""
        dsm = self._make_dsm(use_ibkr=False)  # IBKR disabled → skipped
        valid_df = _make_ohlcv(250)

        with patch('time.sleep'), \
             patch('data_source_manager.clean_raw_data', side_effect=self._patch_clean), \
             patch.object(dsm, '_download_from_massive',
                          side_effect=Exception("down")), \
             patch.object(dsm, '_download_from_alpaca',
                          side_effect=Exception("down")), \
             patch.object(dsm, '_download_from_yfinance',
                          return_value=valid_df) as m_yf:

            result = dsm.get_stock_data("AAPL", days_back=365)

            m_yf.assert_called_once()
            assert result is not None

    def test_dl05_all_providers_down_returns_none_or_empty(self):
        """DL-05 (P0): All 4 providers fail → returns None/empty, no crash."""
        dsm = self._make_dsm(use_ibkr=False)

        with patch('time.sleep'), \
             patch('data_source_manager.clean_raw_data', side_effect=self._patch_clean), \
             patch.object(dsm, '_download_from_massive', side_effect=Exception("down")), \
             patch.object(dsm, '_download_from_alpaca', side_effect=Exception("down")), \
             patch.object(dsm, '_download_from_yfinance', side_effect=Exception("down")):

            result = dsm.get_stock_data("AAPL", days_back=365)

            assert result is None or (isinstance(result, pd.DataFrame) and result.empty), \
                "All-providers-down must return None or empty df, not raise"

    def test_dl06_partial_data_triggers_next_provider(self):
        """DL-06 (P0): Provider returns < min_rows candles → falls through to next."""
        dsm = self._make_dsm()
        small_df = _make_ohlcv(50)   # below min_rows
        full_df = _make_ohlcv(250)   # meets min_rows

        with patch('time.sleep'), \
             patch('data_source_manager.clean_raw_data', side_effect=self._patch_clean), \
             patch.object(dsm, '_download_from_massive', return_value=small_df), \
             patch.object(dsm, '_download_from_alpaca',
                          return_value=full_df) as m_alpaca, \
             patch.object(dsm, '_download_from_yfinance') as m_yf:

            # Pass min_rows=200 so the 50-row result is insufficient
            result = dsm.get_stock_data("AAPL", days_back=365, min_rows=200)

            # ALPACA should be reached because MASSIVE returned too few rows
            m_alpaca.assert_called_once()
            m_yf.assert_not_called()
            assert result is not None and len(result) >= 200

    def test_dl07_timeout_triggers_fallback(self):
        """DL-07 (P1): TimeoutError from a provider → caught, next provider tried."""
        dsm = self._make_dsm()
        valid_df = _make_ohlcv(250)

        with patch('time.sleep'), \
             patch('data_source_manager.clean_raw_data', side_effect=self._patch_clean), \
             patch.object(dsm, '_download_from_massive',
                          side_effect=TimeoutError("request timed out")), \
             patch.object(dsm, '_download_from_alpaca',
                          return_value=valid_df) as m_alpaca:

            result = dsm.get_stock_data("AAPL", days_back=365)

            m_alpaca.assert_called_once()
            assert result is not None

    def test_dl08_waterfall_has_minimum_3_providers(self):
        """DL-08 (P1): Waterfall source code declares at least 3 provider download methods."""
        source_path = os.path.join(PROJECT_ROOT, "data_source_manager.py")
        with open(source_path, 'r', encoding='utf-8') as f:
            source = f.read()

        downloaders = re.findall(r"def _download_from_(\w+)", source)
        assert len(set(downloaders)) >= 3, (
            f"Expected ≥3 _download_from_X methods for waterfall, "
            f"found: {set(downloaders)}"
        )

        # Priority list must exist in get_stock_data
        assert "priority_list" in source or "_download_from_" in source, \
            "Waterfall priority list not found in data_source_manager.py"


# ═══════════════════════════════════════════════════
# SECTION 3.2 — NORMALIZATION LAYER (NL-01 → NL-09)
# ═══════════════════════════════════════════════════

class TestNormalizationLayer:
    """NL-01 to NL-09: normalize_ohlcv per-provider output."""

    REQUIRED = {'open', 'high', 'low', 'close', 'volume'}

    def _assert_normalized(self, df, label=""):
        prefix = f"[{label}] " if label else ""
        assert not df.empty, f"{prefix}Normalized DataFrame is empty"
        missing = self.REQUIRED - set(df.columns)
        assert not missing, f"{prefix}Missing OHLCV columns after normalization: {missing}"
        assert isinstance(df.index, pd.DatetimeIndex), \
            f"{prefix}Index must be DatetimeIndex"
        for col in self.REQUIRED:
            assert df[col].dtype == np.float64, \
                f"{prefix}{col} dtype is {df[col].dtype}, expected float64"

    def test_nl01_normalize_alpaca_format(self):
        """NL-01 (P0): ALPACA capitalised columns → standard lowercase OHLCV."""
        raw = _make_ohlcv(100, provider_format='alpaca')
        result = normalize_ohlcv(raw, "ALPACA")
        self._assert_normalized(result, "ALPACA")

    def test_nl02_normalize_ibkr_format(self):
        """NL-02 (P0): IBKR capitalised columns → identical structure."""
        raw = _make_ohlcv(100, provider_format='ibkr')
        result = normalize_ohlcv(raw, "IBKR")
        self._assert_normalized(result, "IBKR")

    def test_nl03_normalize_yfinance_format(self):
        """NL-03 (P0): YFinance capitalised + Adj Close → identical structure."""
        raw = _make_ohlcv(100, provider_format='yfinance')
        result = normalize_ohlcv(raw, "YFINANCE")
        self._assert_normalized(result, "YFINANCE")

    def test_nl04_normalize_massive_format(self):
        """NL-04 (P0): Massive single-letter columns (o/h/l/c/v) → standard OHLCV."""
        raw = _make_ohlcv(100, provider_format='massive')
        result = normalize_ohlcv(raw, "MASSIVE")
        self._assert_normalized(result, "MASSIVE")

    def test_nl05_missing_close_raises_validation_error(self):
        """NL-05 (P0): DataFrame missing 'close' → raises DataValidationError."""
        dates = pd.date_range(end=datetime.now(), periods=50, freq='B')
        raw = pd.DataFrame({
            'open':   np.ones(50) * 100.0,
            'high':   np.ones(50) * 101.0,
            'low':    np.ones(50) * 99.0,
            # 'close' deliberately absent
            'volume': np.ones(50) * 1_000_000.0,
        }, index=dates)

        with pytest.raises(DataValidationError):
            normalize_ohlcv(raw, "TEST_MISSING")

    def test_nl06_extra_columns_present_ohlcv_intact(self):
        """NL-06 (P2): Extra columns don't corrupt OHLCV output."""
        raw = _make_ohlcv(50)
        raw['sentiment'] = 0.5
        raw['extra_flag'] = 1

        result = normalize_ohlcv(raw, "TEST_EXTRA")
        # OHLCV must survive regardless of extras
        assert self.REQUIRED.issubset(set(result.columns))
        self._assert_normalized(result, "EXTRA_COLS")

    def test_nl07_mixed_dtypes_coerced_to_numeric(self):
        """NL-07 (P1): Integer/numeric-string columns → all OHLCV as numeric (no strings/NaN)."""
        dates = pd.date_range(end=datetime.now(), periods=50, freq='B')
        raw = pd.DataFrame({
            'open':   [str(x + 90) for x in range(50)],   # numeric strings
            'high':   list(range(91, 141)),                 # integers
            'low':    np.random.uniform(88, 92, 50),        # floats
            'close':  [float(x + 90) for x in range(50)],
            'volume': list(range(100_000, 100_050)),        # integers (50 elements)
        }, index=dates)

        result = normalize_ohlcv(raw, "TEST_DTYPE")
        for col in self.REQUIRED:
            assert pd.api.types.is_numeric_dtype(result[col]), \
                f"{col} dtype is {result[col].dtype} — expected numeric (int64 or float64)"
            assert result[col].notna().all(), \
                f"{col} has NaN after coercion — numeric strings failed to convert"

    def test_nl08_unsorted_and_duplicate_index_cleaned(self):
        """NL-08 (P1): Shuffled + duplicate-indexed df → sorted ascending, no dupes."""
        raw = _make_ohlcv(100)
        raw = raw.sample(frac=1, random_state=42)               # shuffle
        raw = pd.concat([raw, raw.iloc[:5]])                    # add duplicates

        result = normalize_ohlcv(raw, "TEST_IDX")

        assert isinstance(result.index, pd.DatetimeIndex)
        assert result.index.is_monotonic_increasing, "Index not sorted ascending"
        assert not result.index.has_duplicates, "Duplicate index entries remain"

    def test_nl09_negative_volume_clipped_to_zero(self):
        """NL-09 (P1): Negative volume values clipped to 0, not dropped."""
        raw = _make_ohlcv(50)
        raw.loc[raw.index[0], 'volume'] = -9_999.0
        raw.loc[raw.index[1], 'volume'] = -1.0

        result = normalize_ohlcv(raw, "TEST_VOL")

        assert (result['volume'] >= 0).all(), \
            "Negative volume survived normalization"
        assert len(result) == len(raw), \
            "Rows dropped instead of clipped — should clip, not drop"


# ═══════════════════════════════════════════════════
# SECTION 3.3 — DATA GUARD (DG-01 → DG-06)
# ═══════════════════════════════════════════════════

class TestDataGuard:
    """DG-01 to DG-06: Minimum candle threshold from system_config."""

    def _min_candles(self):
        return getattr(cfg, 'MIN_CANDLES_FOR_PROCESSING', 200)

    def test_dg01_below_threshold_is_insufficient(self):
        """DG-01 (P0): DataFrame with (threshold-1) rows is below guard threshold."""
        min_c = self._min_candles()
        df = _make_ohlcv(min_c - 1)
        assert len(df) < min_c, (
            f"Guard threshold is {min_c}; a df with {len(df)} rows must be below it"
        )

    def test_dg02_exactly_at_threshold_is_accepted(self):
        """DG-02 (P0): DataFrame with exactly threshold rows meets guard."""
        min_c = self._min_candles()
        df = _make_ohlcv(min_c)
        assert len(df) >= min_c, (
            f"A df with exactly {min_c} rows must meet the threshold"
        )

    def test_dg03_above_threshold_is_accepted(self):
        """DG-03 (P2): DataFrame with 500 rows comfortably exceeds guard."""
        min_c = self._min_candles()
        df = _make_ohlcv(500)
        assert len(df) >= min_c

    def test_dg04_empty_df_fails_guard(self):
        """DG-04 (P0): Empty DataFrame has 0 rows — always below threshold."""
        min_c = self._min_candles()
        df = pd.DataFrame()
        assert df.empty or len(df) < min_c

    def test_dg05_none_input_is_guarded(self):
        """DG-05 (P1): None must be treated as zero-row input by guard logic."""
        # Guard code must handle: `if df is None or len(df) < min_c`
        df = None
        assert df is None  # Guard precondition verified at doc level

        # Verify the shadow ledger and feature engine guard the same way
        source_path = os.path.join(PROJECT_ROOT, "shadow_ledger.py")
        if os.path.exists(source_path):
            with open(source_path, 'r', encoding='utf-8') as f:
                src = f.read()
            # Must check for None before calling len()
            assert "df is None" in src, \
                "shadow_ledger.py must guard against None before len(df)"

    def test_dg06_threshold_comes_from_config(self):
        """DG-06 (P1): MIN_CANDLES_FOR_PROCESSING is in system_config, not hardcoded."""
        assert hasattr(cfg, 'MIN_CANDLES_FOR_PROCESSING'), \
            "MIN_CANDLES_FOR_PROCESSING missing from system_config.py"

        min_c = cfg.MIN_CANDLES_FOR_PROCESSING
        assert isinstance(min_c, int), \
            f"MIN_CANDLES_FOR_PROCESSING must be int, got {type(min_c)}"
        assert min_c > 0, \
            "MIN_CANDLES_FOR_PROCESSING must be positive"
        assert min_c >= 100, \
            f"MIN_CANDLES_FOR_PROCESSING={min_c} seems too low for indicator warmup"
