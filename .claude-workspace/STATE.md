# StockWise AI — Current State
**Last updated (UTC):** 2026-05-16T13:14:03Z

## 🟢 VERIFIED FACTS (evidence-backed only, append-only)
| Verified (UTC) | Fact | Evidence |
|----------------|------|----------|
| 2026-05-16T11:02:04Z | System imports without crash | `python -c "import backtest_engine"` → OK |
| 2026-05-16T11:02:04Z | NumPy/SciPy version conflict exists | feature_engine.py:11 UserWarning: NumPy 2.4.3, SciPy needs <2.3.0 |
| 2026-05-16T11:02:04Z | Last backtest produced ZERO trades | backtest_results.json: total_trades=0, total_return_pct=0.0 |
| 2026-05-16T11:02:04Z | Last backtest run date | backtest_results.json LastWriteTime 2026-04-26 |
| 2026-05-16T11:02:04Z | Data files stale & mixed-format | AVGO.parquet (2026-02-10, data/gold/), AMD_alpaca_history.csv (2026-01-30, data/ root) |
| 2026-05-16T11:02:04Z | /ultrareview command does NOT exist | grep .claude/commands returned empty |
| 2026-05-16T11:02:04Z | Uncommitted WIP in backtest_engine.py | git diff: +88 lines, deterministic_mode/Trust Cache refactor |
| 2026-05-16T13:14:03Z | feature_engine.py is HEALTHY — no missing columns | DIAGNOSIS_missing_columns.md CHECK 1-3: squeeze_on, mom_sqz, vol_avg_20, macd_signal all present and read correctly by block functions |
| 2026-05-16T13:14:03Z | Prior PRIMARY cause (missing columns) was FALSE | DIAGNOSIS_missing_columns.md: prior diagnosis checked wrong column names (squeeze vs squeeze_on etc.) |
| 2026-05-16T13:14:03Z | pandas_ta NOT installed; falls back to pandas_ta_classic v0.3.59 | runtime check in DIAGNOSIS_missing_columns.md |
| 2026-05-16T13:14:03Z | NumPy/SciPy warning is NOT the zero-trades cause | DIAGNOSIS_zero_trades.md CHECK 1: argrelextrema works despite warning |

## ⏳ UNVERIFIED HYPOTHESES (NOT facts — pending diagnosis)
| Logged (UTC) | Hypothesis | Status |
|--------------|------------|--------|
| 2026-05-16T11:02:04Z | H1: NumPy/SciPy conflict → NaN features → zero trades | DIAGNOSIS running |
| 2026-05-16T11:02:04Z | H2: Data too old/incomplete for backtest date range | unverified |
| 2026-05-16T11:02:04Z | H3: All daily templates may be disabled | unverified |
| 2026-05-16T13:14:03Z | H4: zero trades caused by BEARISH market in April test period (RSI=25, close<SMA50). NOT a code bug. | PLAUSIBLE — needs proof: run backtest on a BULLISH period to confirm system generates trades |
Note: a hypothesis becomes a FACT only when DIAGNOSIS_zero_trades.md
confirms it WITH evidence. Then add a new row to the FACTS table.

## ❌ STALE ASSUMPTIONS — DO NOT TRUST (contradicted by facts)
| Logged (UTC) | Stale assumption | Contradicted by |
|--------------|------------------|-----------------|
| 2026-05-16T11:02:04Z | "PF ~2.0, ~20% return, 176-250 trades" (old memory) | actual = 0 trades (verified 2026-05-16T11:02:04Z) |
| 2026-05-16T11:02:04Z | "04_magic_numbers.md says slippage is 50x" | Investigation A proved it is 2x |
| 2026-05-16T13:14:03Z | "PRIMARY root cause = missing feature columns" (DIAGNOSIS_zero_trades.md) | DISPROVEN by DIAGNOSIS_missing_columns.md — columns exist under different names |

## Session Log (append-only)
| Timestamp (UTC) | What was done | Outcome |
|-----------------|---------------|---------|
| 2026-05-16T11:02:04Z | Verified system state via facts; found zero-trades; created workspace docs | Diagnosis + docs prompts issued |
| 2026-05-16T13:14:03Z | Ran missing-columns root-cause diagnosis | feature_engine proven healthy; prior diagnosis was wrong; real cause likely bearish test period (unproven) |

## Next Step
Prove or disprove H4: run a deterministic backtest on a confirmed
BULLISH period (e.g. a date range where AAPL/AVGO trended up). If
trades > 0 → system works, move to baseline. If still 0 → deeper
pipeline bug. DO NOT change feature_engine — it is healthy.
