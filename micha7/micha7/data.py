"""Data layer: provider contract, yfinance implementation, validation,
normalization, gap detection, and shared metrics (ATR, returns).

All data flows through DataAdapter.fetch() which returns a validated,
normalized MarketData object. Providers are swappable via BaseDataProvider
(ADR-015): yfinance now, IBKR/Alpaca later, with no change to the adapter.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DataError(Exception):
    """Base for all data-layer errors."""


class DataValidationError(DataError):
    """Raised when OHLCV data fails validation (Fail-Loud)."""


class DataFetchError(DataError):
    """Raised when a provider exhausts all retries without success."""


# ---------------------------------------------------------------------------
# MarketData
# ---------------------------------------------------------------------------


@dataclass
class MarketData:
    """Validated, normalized OHLCV with pre-computed shared metrics."""

    symbol: str
    df: pd.DataFrame
    atr: pd.Series
    returns: pd.Series
    gaps: list[dict] = field(default_factory=list)
    row_count: int = 0

    def __post_init__(self) -> None:
        if self.row_count == 0:
            self.row_count = len(self.df)


# ---------------------------------------------------------------------------
# Provider contract
# ---------------------------------------------------------------------------


class BaseDataProvider(ABC):
    """Abstract contract for OHLCV data sources (ADR-015).

    Concrete providers must implement get_ohlcv and return a DataFrame
    with columns: open, high, low, close, volume (case-insensitive; the
    adapter normalises before validation).
    """

    @abstractmethod
    def get_ohlcv(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Fetch raw OHLCV for *symbol* over [*start*, *end*].

        Args:
            symbol: Ticker string (e.g. "AAPL").
            start:  ISO 8601 date string, inclusive.
            end:    ISO 8601 date string, inclusive.

        Returns:
            Raw DataFrame (not yet normalized or validated).
        """


# ---------------------------------------------------------------------------
# YFinance provider
# ---------------------------------------------------------------------------

_DEFAULT_RETRY_COUNT = 3
_DEFAULT_RETRY_BACKOFF = 1.0
_DEFAULT_AUTO_ADJUST = True


class YFinanceProvider(BaseDataProvider):
    """Concrete provider backed by yfinance with configurable retry/backoff."""

    def __init__(self, loader: Any = None, logger: Any = None) -> None:
        if loader is not None:
            self._retry_count = loader.get(
                "data.yfinance.retry_count", default=_DEFAULT_RETRY_COUNT, expected_type=int
            )
            self._backoff = loader.get(
                "data.yfinance.retry_backoff_seconds",
                default=_DEFAULT_RETRY_BACKOFF,
            )
            self._auto_adjust = loader.get(
                "data.yfinance.auto_adjust", default=_DEFAULT_AUTO_ADJUST
            )
        else:
            self._retry_count = _DEFAULT_RETRY_COUNT
            self._backoff = _DEFAULT_RETRY_BACKOFF
            self._auto_adjust = _DEFAULT_AUTO_ADJUST
        self._logger = logger

    def get_ohlcv(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Download OHLCV from yfinance with exponential-ish retry.

        Raises DataFetchError if all retries are exhausted.
        """
        import yfinance as yf

        last_exc: Exception | None = None
        for attempt in range(self._retry_count):
            try:
                df = yf.download(
                    symbol,
                    start=start,
                    end=end,
                    auto_adjust=self._auto_adjust,
                    progress=False,
                )
                return df
            except Exception as exc:
                last_exc = exc
                if attempt < self._retry_count - 1:
                    time.sleep(self._backoff)

        raise DataFetchError(
            f"yfinance failed for {symbol!r} after {self._retry_count} attempts: {last_exc}"
        ) from last_exc


# ---------------------------------------------------------------------------
# Shared metrics (pure functions — ADR-012)
# ---------------------------------------------------------------------------

_REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Compute Average True Range (Wilder SMA over *period* bars).

    True Range = max(high-low, |high - prev_close|, |low - prev_close|).
    The first row has no prev_close so its TR is NaN; warmup rows up to
    (and including) row *period* - 1 are NaN.

    Returns a Series aligned with df.index.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # First TR is NaN (no prev_close); use SMA for the initial period seed,
    # then Wilder smoothing (equivalent to EMA with alpha=1/period).
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    return atr


def compute_returns(df: pd.DataFrame) -> pd.Series:
    """Compute simple period-over-period close returns.

    Returns = (close_t - close_{t-1}) / close_{t-1}.
    First value is NaN (no prior period).
    """
    return df["close"].pct_change()


# ---------------------------------------------------------------------------
# DataAdapter
# ---------------------------------------------------------------------------

_DEFAULT_ATR_PERIOD = 14
_DEFAULT_MIN_ROWS = 30
_DEFAULT_MAX_GAP_DAYS = 5


class DataAdapter:
    """Orchestrates fetch → normalize → validate → gap-detect → metrics.

    The adapter is the single entry point for the pipeline's data stage.
    It wraps any BaseDataProvider and always returns a MarketData object.
    """

    def __init__(
        self,
        provider: BaseDataProvider | None,
        loader: Any = None,
        logger: Any = None,
    ) -> None:
        self._provider = provider
        self._logger = logger

        if loader is not None:
            self._atr_period = loader.get(
                "data.atr_period", default=_DEFAULT_ATR_PERIOD, expected_type=int
            )
            self._min_rows = loader.get(
                "data.min_rows", default=_DEFAULT_MIN_ROWS, expected_type=int
            )
            self._max_gap_days = loader.get(
                "data.max_gap_days", default=_DEFAULT_MAX_GAP_DAYS, expected_type=int
            )
        else:
            self._atr_period = _DEFAULT_ATR_PERIOD
            self._min_rows = _DEFAULT_MIN_ROWS
            self._max_gap_days = _DEFAULT_MAX_GAP_DAYS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(self, symbol: str, start: str, end: str) -> MarketData:
        """Full pipeline: provider → normalize → validate → gaps → metrics.

        Returns a validated MarketData. Logs event "data_fetch".
        Raises DataFetchError or DataValidationError on failure.
        """
        raw = self._provider.get_ohlcv(symbol, start, end)
        df = self.normalize(raw)
        self.validate(df)
        gaps = self.detect_gaps(df)
        atr = compute_atr(df, self._atr_period)
        returns = compute_returns(df)

        self._log("INFO", "data_fetch", f"Fetched {len(df)} rows for {symbol}", {})

        return MarketData(
            symbol=symbol,
            df=df,
            atr=atr,
            returns=returns,
            gaps=gaps,
            row_count=len(df),
        )

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of *df* with lowercase columns, ascending DatetimeIndex.

        Keeps only [open, high, low, close, volume]; sorts ascending.
        """
        out = df.copy()
        out.columns = [str(c).lower() for c in out.columns]

        # If columns are a MultiIndex (yfinance multi-ticker), flatten
        if isinstance(out.columns, pd.MultiIndex):
            out.columns = ["_".join(str(s) for s in c if s).lower() for c in out.columns]

        # Keep only the five required columns (if present)
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in out.columns]
        out = out[keep]

        # Ensure DatetimeIndex
        if not isinstance(out.index, pd.DatetimeIndex):
            out.index = pd.to_datetime(out.index)

        # Sort ascending
        out = out.sort_index(ascending=True)

        return out

    def validate(self, df: pd.DataFrame) -> None:
        """Fail-Loud validation of a normalized OHLCV DataFrame.

        Raises DataValidationError with a descriptive message on the first
        failure found. Logs "data_validation_failed" before raising.
        """

        def _fail(reason: str) -> None:
            self._log("WARNING", "data_validation_failed", reason, {})
            raise DataValidationError(reason)

        # Required columns
        missing = _REQUIRED_COLUMNS - set(df.columns)
        if missing:
            _fail(f"Missing required columns: {sorted(missing)}")

        # NaN check
        if df[list(_REQUIRED_COLUMNS)].isnull().any().any():
            _fail("NaN values found in OHLCV columns")

        # Positive prices
        for col in ("open", "high", "low", "close"):
            if (df[col] <= 0).any():
                _fail(f"Non-positive values in column '{col}'")

        # OHLC relationships
        if (df["high"] < df["low"]).any():
            _fail("high < low detected")
        if (df["high"] < df["open"]).any():
            _fail("high < open detected")
        if (df["high"] < df["close"]).any():
            _fail("high < close detected")
        if (df["low"] > df["open"]).any():
            _fail("low > open detected")
        if (df["low"] > df["close"]).any():
            _fail("low > close detected")

        # Index integrity
        if not isinstance(df.index, pd.DatetimeIndex):
            _fail("Index is not a DatetimeIndex")
        if df.index.duplicated().any():
            _fail("Duplicate index entries detected")
        if not df.index.is_monotonic_increasing:
            _fail("Index is not monotonically increasing (must be sorted ascending)")

        # Minimum rows
        if len(df) < self._min_rows:
            _fail(
                f"Too few rows: {len(df)} < min_rows={self._min_rows}"
            )

    def detect_gaps(self, df: pd.DataFrame) -> list[dict]:
        """Return a list of calendar gaps exceeding max_gap_days.

        Each gap: {"from": date_str, "to": date_str, "gap_days": int}.
        Non-fatal — caller receives the list and decides how to handle.
        """
        if len(df) < 2:
            return []

        gaps: list[dict] = []
        dates = df.index.normalize()  # strip time component
        for i in range(1, len(dates)):
            delta = (dates[i] - dates[i - 1]).days
            if delta > self._max_gap_days:
                gap = {
                    "from": str(dates[i - 1].date()),
                    "to": str(dates[i].date()),
                    "gap_days": delta,
                }
                gaps.append(gap)
                self._log(
                    "WARNING",
                    "data_gap_detected",
                    f"Gap of {delta} days between {gap['from']} and {gap['to']}",
                    gap,
                )
        return gaps

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _log(self, level: str, event: str, message: str, context: dict) -> None:
        if self._logger is not None:
            log_fn = getattr(self._logger, level.lower(), self._logger.info)
            log_fn(message, extra={"event": event, "context": context})
