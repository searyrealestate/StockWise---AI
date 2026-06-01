"""TDD test suite for micha7.data.

All tests are OFFLINE — yfinance is monkeypatched; no network calls.
Written BEFORE implementation; expect RED on first run.
"""

import math

import numpy as np
import pandas as pd
import pytest

from micha7.data import (
    BaseDataProvider,
    DataAdapter,
    DataFetchError,
    DataValidationError,
    MarketData,
    YFinanceProvider,
    compute_atr,
    compute_returns,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _loader_with_defaults(tmp_path):
    """Return a minimal ConfigLoader pointing at a tmp config.json."""
    import json

    from micha7.config import ConfigLoader

    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {
                "meta": {"name": "micha7_analyzer", "config_version": "1.0.0"},
                "logging": {
                    "level": "INFO",
                    "format": "json",
                    "directory": str(tmp_path / "logs"),
                    "console": False,
                },
                "data": {
                    "atr_period": 14,
                    "min_rows": 30,
                    "max_gap_days": 5,
                    "yfinance": {
                        "retry_count": 3,
                        "retry_backoff_seconds": 0.0,
                        "auto_adjust": True,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    loader = ConfigLoader(
        config_path=cfg, local_path=tmp_path / "config.local.json"
    )
    loader.load()
    return loader


# ---------------------------------------------------------------------------
# Provider contract
# ---------------------------------------------------------------------------


def test_base_provider_is_abstract():
    """Instantiating BaseDataProvider directly must raise TypeError."""
    with pytest.raises(TypeError):
        BaseDataProvider()


def test_yfinance_retries_then_succeeds(tmp_path, sample_ohlcv, monkeypatch):
    """yfinance raises twice, then returns data; success on 3rd call."""
    loader = _loader_with_defaults(tmp_path)
    call_count = {"n": 0}

    def _fake_download(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise RuntimeError("network blip")
        return sample_ohlcv.copy()

    monkeypatch.setattr("yfinance.download", _fake_download)
    provider = YFinanceProvider(loader=loader)
    result = provider.get_ohlcv("AAPL", "2025-01-02", "2025-03-31")
    assert result is not None
    assert call_count["n"] == 3


def test_yfinance_exhausts_retries_raises(tmp_path, monkeypatch):
    """yfinance always raises; must raise DataFetchError after all retries."""
    loader = _loader_with_defaults(tmp_path)

    def _always_fail(*args, **kwargs):
        raise RuntimeError("always down")

    monkeypatch.setattr("yfinance.download", _always_fail)
    provider = YFinanceProvider(loader=loader)
    with pytest.raises(DataFetchError):
        provider.get_ohlcv("AAPL", "2025-01-02", "2025-03-31")


# ---------------------------------------------------------------------------
# Adapter — validate()
# ---------------------------------------------------------------------------


def test_validate_rejects_missing_column(tmp_path, sample_ohlcv):
    loader = _loader_with_defaults(tmp_path)
    adapter = DataAdapter(provider=None, loader=loader)
    df = sample_ohlcv.drop(columns=["volume"])
    with pytest.raises(DataValidationError):
        adapter.validate(df)


def test_validate_rejects_nan(tmp_path, sample_ohlcv):
    loader = _loader_with_defaults(tmp_path)
    adapter = DataAdapter(provider=None, loader=loader)
    df = sample_ohlcv.copy()
    df.iloc[5, df.columns.get_loc("close")] = float("nan")
    with pytest.raises(DataValidationError):
        adapter.validate(df)


def test_validate_rejects_negative_price(tmp_path, sample_ohlcv):
    loader = _loader_with_defaults(tmp_path)
    adapter = DataAdapter(provider=None, loader=loader)
    df = sample_ohlcv.copy()
    df.iloc[0, df.columns.get_loc("close")] = -1.0
    with pytest.raises(DataValidationError):
        adapter.validate(df)


def test_validate_rejects_high_below_low(tmp_path, sample_ohlcv):
    loader = _loader_with_defaults(tmp_path)
    adapter = DataAdapter(provider=None, loader=loader)
    df = sample_ohlcv.copy()
    df.iloc[0, df.columns.get_loc("high")] = 50.0
    df.iloc[0, df.columns.get_loc("low")] = 80.0
    with pytest.raises(DataValidationError):
        adapter.validate(df)


def test_validate_rejects_duplicate_index(tmp_path, sample_ohlcv):
    loader = _loader_with_defaults(tmp_path)
    adapter = DataAdapter(provider=None, loader=loader)
    df = pd.concat([sample_ohlcv, sample_ohlcv.iloc[[0]]])
    with pytest.raises(DataValidationError):
        adapter.validate(df)


def test_validate_rejects_unsorted_index(tmp_path, sample_ohlcv):
    loader = _loader_with_defaults(tmp_path)
    adapter = DataAdapter(provider=None, loader=loader)
    df = sample_ohlcv.iloc[::-1].copy()
    with pytest.raises(DataValidationError):
        adapter.validate(df)


def test_validate_rejects_too_few_rows(tmp_path, sample_ohlcv):
    loader = _loader_with_defaults(tmp_path)
    adapter = DataAdapter(provider=None, loader=loader)
    df = sample_ohlcv.iloc[:5]  # 5 rows < min_rows=30
    with pytest.raises(DataValidationError):
        adapter.validate(df)


def test_validate_passes_clean_data(tmp_path, sample_ohlcv):
    loader = _loader_with_defaults(tmp_path)
    adapter = DataAdapter(provider=None, loader=loader)
    # Must not raise
    adapter.validate(sample_ohlcv)


# ---------------------------------------------------------------------------
# Adapter — normalize()
# ---------------------------------------------------------------------------


def test_normalize_lowercases_columns(tmp_path, sample_ohlcv):
    loader = _loader_with_defaults(tmp_path)
    adapter = DataAdapter(provider=None, loader=loader)
    df_upper = sample_ohlcv.copy()
    df_upper.columns = [c.upper() for c in df_upper.columns]
    result = adapter.normalize(df_upper)
    assert all(c.islower() for c in result.columns if c != result.index.name)


def test_normalize_sorts_index_ascending(tmp_path, sample_ohlcv):
    loader = _loader_with_defaults(tmp_path)
    adapter = DataAdapter(provider=None, loader=loader)
    df_rev = sample_ohlcv.iloc[::-1].copy()
    result = adapter.normalize(df_rev)
    assert result.index.is_monotonic_increasing


# ---------------------------------------------------------------------------
# Adapter — detect_gaps()
# ---------------------------------------------------------------------------


def test_detect_gaps_finds_large_gap(tmp_path, sample_ohlcv_with_gap):
    loader = _loader_with_defaults(tmp_path)
    adapter = DataAdapter(provider=None, loader=loader)
    gaps = adapter.detect_gaps(sample_ohlcv_with_gap)
    assert len(gaps) >= 1
    assert all("from" in g and "to" in g and "gap_days" in g for g in gaps)
    assert any(g["gap_days"] > 5 for g in gaps)


def test_detect_gaps_none_on_continuous(tmp_path, sample_ohlcv):
    loader = _loader_with_defaults(tmp_path)
    adapter = DataAdapter(provider=None, loader=loader)
    gaps = adapter.detect_gaps(sample_ohlcv)
    assert gaps == []


# ---------------------------------------------------------------------------
# Shared metrics
# ---------------------------------------------------------------------------


def test_compute_atr_small_known_series():
    """Period=3; first valid ATR == mean of first 3 True Ranges."""
    dates = pd.date_range("2025-01-01", periods=6, freq="D")
    df = pd.DataFrame(
        {
            "open":  [10.0, 11.0, 12.0, 11.0, 10.0, 11.0],
            "high":  [12.0, 13.0, 14.0, 13.0, 12.0, 13.0],
            "low":   [ 9.0, 10.0, 11.0, 10.0,  9.0, 10.0],
            "close": [11.0, 12.0, 13.0, 12.0, 11.0, 12.0],
            "volume":[1000, 1000, 1000, 1000, 1000, 1000],
        },
        index=dates,
    )
    atr = compute_atr(df, period=3)
    # Warmup: first value (no prev_close) is NaN
    assert math.isnan(atr.iloc[0])
    # True ranges for rows 1-3 (period=3):
    # row 1: TR = max(13-10, |13-11|, |10-11|) = max(3,2,1) = 3
    # row 2: TR = max(14-11, |14-12|, |11-12|) = max(3,2,1) = 3
    # row 3: TR = max(13-11, |13-13|, |10-13|) = max(2,0,3) = 3
    # mean of first 3 TRs = 3.0
    first_valid = atr.dropna().iloc[0]
    assert abs(first_valid - 3.0) < 1e-9


def test_compute_atr_deterministic(sample_ohlcv):
    """Same input twice → identical Series."""
    a = compute_atr(sample_ohlcv, period=14)
    b = compute_atr(sample_ohlcv, period=14)
    pd.testing.assert_series_equal(a, b)


def test_compute_returns_first_is_nan_and_values_correct(sample_ohlcv):
    """First return is NaN; subsequent values are simple close-to-close returns."""
    ret = compute_returns(sample_ohlcv)
    assert math.isnan(ret.iloc[0])
    closes = sample_ohlcv["close"]
    expected_r1 = (closes.iloc[1] - closes.iloc[0]) / closes.iloc[0]
    assert abs(ret.iloc[1] - expected_r1) < 1e-12


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


def test_fetch_returns_marketdata(tmp_path, mock_provider, sample_ohlcv):
    """DataAdapter.fetch with mock_provider returns a valid MarketData object."""
    loader = _loader_with_defaults(tmp_path)
    adapter = DataAdapter(provider=mock_provider, loader=loader)
    md = adapter.fetch("AAPL", "2025-01-02", "2025-03-31")
    assert isinstance(md, MarketData)
    assert md.symbol == "AAPL"
    assert md.row_count == len(sample_ohlcv)
    assert len(md.atr) == md.row_count
    assert len(md.returns) == md.row_count
    assert md.gaps == []
