# StockWise AI — Known Issues & Lessons Learned
**Last updated (UTC):** 2026-05-16T13:14:03Z
**Purpose:** Bugs found, fixes that failed, traps. APPEND-ONLY.
**Rule:** to update an issue, ADD a new row with a new UTC timestamp.
Never edit or delete an old row. Latest timestamp = current status.

## Issue Log (append-only)
| Logged (UTC) | ID | Issue / Update | Status |
|--------------|----|----|--------|
| 2026-05-16T11:02:04Z | KI-1 | Backtest produces zero trades (total_trades=0) | DIAGNOSING |
| 2026-05-16T11:02:04Z | KI-2 | NumPy 2.4.3 incompatible with SciPy (<2.3.0) | OPEN |
| 2026-05-16T11:02:04Z | KI-3 | Data files stale (months old) & mixed format (parquet/csv) | OPEN |
| 2026-05-16T11:02:04Z | KI-4 | Uncommitted +88 lines in backtest_engine.py, origin unknown | OPEN |
| 2026-05-16T11:02:04Z | KI-5 | 04_magic_numbers.md still says slippage "50x" — actually 2x | OPEN — doc fix needed |
| 2026-05-16T13:14:03Z | KI-1 | UPDATE: zero-trades root cause is NOT missing columns. feature_engine healthy. Likely bearish test period. Re-test on bullish range pending. | RE-DIAGNOSED |
| 2026-05-16T13:14:03Z | KI-6 | DIAGNOSIS_zero_trades.md contains a WRONG primary verdict (checked wrong column names). Kept for history; superseded by DIAGNOSIS_missing_columns.md | DOCUMENTED |

## Traps — DO NOT do these
| Logged (UTC) | Trap |
|--------------|------|
| 2026-05-16T11:02:04Z | Do NOT git stash / git checkout backtest_engine.py — destroys WIP |
| 2026-05-16T11:02:04Z | Do NOT trust old memory numbers (PF~2.0) — contradicted by facts |
| 2026-05-16T11:02:04Z | Do NOT buy new market data before data quality is verified |
| 2026-05-16T11:02:04Z | Do NOT change code based on theory — verify with evidence first |

## Lessons Learned (append-only)
| Logged (UTC) | Lesson |
|--------------|--------|
| 2026-05-16T11:02:04Z | "WR is bad" was actually "0 trades" — verify the metric, not the memory of it |
| 2026-05-16T13:14:03Z | A diagnosis can be confidently wrong. DIAGNOSIS_zero_trades.md asserted "missing columns" with evidence, but checked the wrong names. ALWAYS verify a root cause with a second independent check before coding a fix. The second diagnosis saved us from editing healthy code. |
| 2026-05-16T13:14:03Z | When checking if a column is "missing", list ALL actual output columns first and match by meaning, not by assumed name. |
