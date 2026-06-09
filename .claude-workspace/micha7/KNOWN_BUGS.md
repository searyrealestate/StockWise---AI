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

## Discovered Day 6-7 (B-14..B-37)

### 🔴 Critical — Block F6 (B-14..B-17)

| ID | Severity | File | Description | Decision / Resolution | Status |
|----|----------|------|-------------|----------------------|--------|
| **B-14** | 🔴 Critical | `features.py F6` | Lookahead bias in pivot detection — F6 spec must add "validated at T+N" rule | Pivot at T counted only at T+N; F6 receives `current_bar_index` (D-27) | Open→3.7 |
| **B-15** | 🔴 Critical | `trade.py EntryPlanner` | Targets contract unclear (D-12) — what when T2/T3 missing? | `targets: list[float]` (1-3 elements) instead of fixed tuple | Open→5.1 |
| **B-16** | 🔴 Critical | `features.py F1` | F1 Hammer upper wick not defined in spec — needs `upper_wick_max_ratio` | web_search for canonical definition + config param (default 0.3) | Open→3.2 |
| **B-17** | 🔴 Critical | `features.py F5` | F5 gap definition ambiguous (strict vs open; fill check by low or close) | web_search + 2 config flags: `gap_strict_mode`, `gap_fill_check` | Open→3.6 |

### 🟡 High — Break in Production (B-18..B-22)

| ID | Severity | File | Description | Decision / Resolution | Status |
|----|----------|------|-------------|----------------------|--------|
| **B-18** | 🟡 High | `data.py` | Trading calendar not handled (holidays → MA20 inaccurate) | `pandas_market_calendars` or `business_days_only` config | Open→6.1 |
| **B-19** | 🟡 High | `config.json, data.py` | min_rows=100 + N=5 = pivot last validated at T-5 only | D-28: raise min_rows to 200 | Resolved→2.9b |
| **B-20** | 🟡 High | `data.py` | Data freshness not checked | `validate_freshness(max_staleness_days=3)` in validate() | Open→2.9b |
| **B-21** | 🟡 High | `features.py F6` | Float comparison in clustering → nondeterminism cross-machine | `np.isclose(atol=1e-9)` + round to cents (D-29) | Open→3.7 |
| **B-22** | 🟡 High | `SKILL_RULES_SUMMARY.md` | `eyal-dev-standards` skill never seen by Claude Code | Eyal uploads SKILL.md; Claude summarizes in `.claude-workspace/micha7/SKILL_RULES_SUMMARY.md` | Resolved |

### 🟢 Medium — Not Blocking (B-23..B-25)

| ID | Severity | File | Description | Decision / Resolution | Status |
|----|----------|------|-------------|----------------------|--------|
| **B-23** | 🟢 Medium | `backtest.py` | Symbol delisting / mergers / IPO < 100d | try/except + log + skip in 6.2 | Open→6.2 |
| **B-24** | 🟢 Medium | `analyzer.py` | No liquidity threshold → F3 noise on small stocks | `min_avg_volume_20d` gate in 6.1 (default 1M) | Open→6.1 |
| **B-25** | 🟢 Medium | `features.py F5` | Earnings gaps counted as pivots | Phase 2: yfinance earnings + flag | Deferred→Phase 2 |

### 🔵 Architectural (B-26..B-27)

| ID | Severity | File | Description | Decision / Resolution | Status |
|----|----------|------|-------------|----------------------|--------|
| **B-26** | 🔵 Arch | `features.py` | Missing unified raw contract for all F (only F4 and F6 defined) | ADR-019 — full raw contract for F1..F7 | Resolved→ADR-019 |
| **B-27** | 🔵 Arch | `tests/conftest.py` | Test fixtures not realistic — needed for F6 too, not just F1 | Move B-08/B-09/B-10 fix from Prompt 3.2 to Prompt 3.7 | Open→3.7 |

### Day 7 Additional Bugs (B-28..B-37)

| ID | Severity | File | Description | Decision / Resolution | Status |
|----|----------|------|-------------|----------------------|--------|
| **B-28** | 🔴 High | `features.py F6` | "Find closest pair" tie-break undefined → non-determinism | Tie-break price asc then bar_index asc (D-29) | Open→3.7 |
| **B-29** | 🔴 High | `features.py F6` | max([])/min([]) crash when a side has no level | Guard None | Open→3.7 |
| **B-30** | 🟡 Med | `features.py F6` | strength uses ATR[touch] but clustering uses ATR[current] | Single ATR[current] (D-33) | Open→3.7 |
| **B-31** | 🟡 Med | `features.py F6` | Summing strengths double-counts shared touch bars | Recompute strength vs centroid (unique bars) | Open→3.7 |
| **B-32** | 🟡 Med | `chart.py, viewer` | Full-width S/R line hides T+N validation | valid_from=pivot+N, draw from there (D-35) | Open→2.10/3.7 |
| **B-33** | 🟡 Med | `viewer` | v5 ESM+file:// breaks | Vendor standalone UMD | Open→2.10 |
| **B-34** | 🟡 Med | `chart.py` | BEARISH collapsed for scoring but must be shown | render() uses true direction | Open→2.10 |
| **B-35** | 🟡 Med | `features.py F3/F7` | Warmup NaN desyncs sub-pane | Whitespace points, full-length arrays | Open→3.4/3.8 |
| **B-36** | 🟢 Low | `viewer` | v5 markers are a plugin | Verify bundle includes createSeriesMarkers | Open→3.2 |
| **B-37** | 🟢 Low | `features.py F6` | Role-reversal not handled | Phase 1 document; Phase 2 flip | Note |

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
