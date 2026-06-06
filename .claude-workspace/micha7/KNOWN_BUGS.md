# micha7 — Known Bugs and Technical Debt

> Version 1.0.0 — 2026-06-02T22:52:37Z
> Tracking discovered issues and deferred items. Not all are blockers.
> Updated as implementation progresses.

---

## Active Issues (Phase 1 Blockers or Near-Blockers)

| ID | Severity | File | Description | Decision | Status |
|----|----------|------|-------------|----------|--------|
| **B-02** | 🟡 Medium | `data.py:108` | `auto_adjust=True` alters historical OHLC prices (adjusts for splits/dividends). This makes prices differ from raw broker data. | Accepted per D-18: matches TradingView display. Document clearly; no fix. | **Accepted** |
| **B-07** | 🔴 High | `n/a` | State persistence not yet implemented. All state is in-memory only. Without StateManager + WAL, crashes lose all ARMED/TRIGGERED state. | Planned: Prompt 4.1. | **PLANNED — Prompt 4.1** |

---

## Test Coverage Gaps

| ID | Test File | Description | Plan |
|----|-----------|-------------|------|
| **B-08** | `tests/conftest.py` | `sample_ohlcv` uses an unrealistic repeating 5-bar price pattern. Real OHLCV has trend, drift, and realistic HL relationships. Candle pattern tests need realistic Hammer and Engulfing scenarios. | Add `realistic_ohlcv`, `hammer_ohlcv`, `engulfing_ohlcv` fixtures in Prompt 3.2 |
| **B-09** | `tests/conftest.py` | No price-gap fixture. F5 tests need a DataFrame with a known unfilled gap above current price. | Add `ohlcv_with_price_gap` fixture in Prompt 3.6 |
| **B-10** | `tests/conftest.py` | No Hammer candle fixture. Cannot test F1 Hammer condition without a DataFrame ending on a Hammer bar. | Add `hammer_candle_ohlcv` fixture in Prompt 3.2 |

---

## Tech Debt (Non-Blocking)

| ID | Severity | File | Description | Plan |
|----|----------|------|-------------|------|
| **B-01** | 🟢 Low | `data.py` | `compute_atr` uses a Python loop (not vectorized). For large backtests with many symbols, this will be slow. | Benchmark in Prompt 6.2; vectorize only if 90-day run >5 seconds (premature optimization otherwise). |
| **B-04** | 🟢 Low | `data.py` docstring | Historical docstring claimed ATR was used for stop sizing. Corrected by D-19 (ATR for F4 distance only). No code change needed; already resolved in 71882a9. | Resolved by D-19 documentation. |
| **B-06** | 🟢 Low | `data.py` | `compute_returns` is coupled to `auto_adjust=True`. If raw prices are used, returns will be incorrect around split dates. | Acknowledged; auto_adjust=True makes this a non-issue for current usage (D-18). No fix needed. |
| **B-13** | 🟢 Low | `data.py:YFinanceProvider` | Retry backoff is linear (constant sleep `retry_backoff_seconds`), not exponential. Docstring says "retry with backoff" which implies exponential. | Fix docstring to say "linear backoff". Consider exponential backoff option in config. Cosmetic. |

---

## Resolved Bugs

| ID | Description | Resolution | Commit |
|----|-------------|------------|--------|
| **B-00** | `compute_atr` used `ewm(span=period)` (alpha=2/(period+1)) instead of canonical Wilder (alpha=1/period, SMA seed). Self-confirming test masked the error. | Rewritten to canonical Wilder ATR. Test now uses hand-computed literal values. | 71882a9 |
| **B-03** | `detect_gaps()` / `MarketData.gaps` created naming collision with F5 price gaps. | Renamed to `detect_calendar_gaps()` / `calendar_gaps`; log event renamed `calendar_gap_detected` (D-17). | 22edc9d |
| **B-05** | `min_rows=30` insufficient for S/R detection (need ~100 bars). | Raised to 100 in `config.json` + `_DEFAULT_MIN_ROWS`; Fail-Loud range guard `[1, 10000]` added (D-16). | 22edc9d |
| **B-11** | `validate()` did not check for `NaT` in `DatetimeIndex`; corrupted index would pass. | Added `df.index.isna().any()` check before duplicate check; raises `DataValidationError`. | 22edc9d |
| **B-12** | `DataAdapter._log()` silently dropped events when `self._logger is None`. | Falls back to `logging.getLogger("micha7.data")` module logger — identical pattern to features.py. Resolved early (was planned for 4.x). | FIX-002 |

---

## Bug Triage Process

Severity definitions:
- 🔴 **High** — blocks correct behavior; must fix before dependent code
- 🟡 **Medium** — degrades quality or causes future confusion; fix before Phase 1 DoD
- 🟢 **Low** — cosmetic, performance, or future concern; fix when convenient

When a bug is resolved:
1. Move row to "Resolved Bugs" table
2. Add commit hash
3. Update CHANGELOG.md
