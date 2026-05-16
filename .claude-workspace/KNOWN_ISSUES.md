# StockWise AI — Known Issues & Lessons Learned
**Last updated (UTC):** 2026-05-16T11:02:04Z
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
