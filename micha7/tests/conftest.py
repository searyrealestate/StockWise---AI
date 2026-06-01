"""Shared pytest fixtures for micha7 tests."""

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# OHLCV helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(dates: list[str]) -> pd.DataFrame:
    """Build a deterministic OHLCV DataFrame for the given date strings.

    Prices are fixed and satisfy all OHLC constraints (high >= open/close,
    low <= open/close, high >= low, all positive). Volume is positive.
    """
    n = len(dates)
    # Use a repeating 5-row price pattern so any length works cleanly.
    pattern = [
        {"open": 100.0, "high": 105.0, "low": 98.0, "close": 103.0, "volume": 1_000_000},
        {"open": 103.0, "high": 107.0, "low": 101.0, "close": 106.0, "volume": 1_200_000},
        {"open": 106.0, "high": 108.0, "low": 103.0, "close": 104.0, "volume": 900_000},
        {"open": 104.0, "high": 106.0, "low": 99.0, "close": 100.0, "volume": 1_100_000},
        {"open": 100.0, "high": 102.0, "low": 97.0, "close": 101.0, "volume": 800_000},
    ]
    rows = [pattern[i % len(pattern)] for i in range(n)]
    idx = pd.DatetimeIndex(dates, name="Date")
    return pd.DataFrame(rows, index=idx)


@pytest.fixture()
def sample_ohlcv() -> pd.DataFrame:
    """Deterministic 40-row OHLCV DataFrame on consecutive business days.

    No network, no files. All OHLC relationships valid, prices positive.
    """
    dates = pd.bdate_range(start="2025-01-02", periods=40).strftime("%Y-%m-%d").tolist()
    return _make_ohlcv(dates)


@pytest.fixture()
def sample_ohlcv_with_gap(sample_ohlcv) -> pd.DataFrame:
    """Like sample_ohlcv but with a >5-business-day gap in the middle.

    Achieved by removing 7 consecutive rows from the middle so the jump
    between surrounding dates exceeds max_gap_days=5.
    """
    df = sample_ohlcv.copy()
    # Drop rows 18-24 (7 rows) → gap of ~10 calendar days
    drop_positions = list(range(18, 25))
    df = df.drop(index=df.index[drop_positions])
    return df


# ---------------------------------------------------------------------------
# Mock provider
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_provider(sample_ohlcv):
    """A BaseDataProvider subclass that returns sample_ohlcv for any symbol."""
    from micha7.data import BaseDataProvider

    class _MockProvider(BaseDataProvider):
        def get_ohlcv(self, symbol: str, start: str, end: str) -> pd.DataFrame:
            return sample_ohlcv.copy()

    return _MockProvider()
