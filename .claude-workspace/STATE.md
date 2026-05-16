# StockWise AI — Current State
**Last updated (UTC):** 2026-05-16T11:02:04Z

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

## ⏳ UNVERIFIED HYPOTHESES (NOT facts — pending diagnosis)
| Logged (UTC) | Hypothesis | Status |
|--------------|------------|--------|
| 2026-05-16T11:02:04Z | H1: NumPy/SciPy conflict → NaN features → zero trades | DIAGNOSIS running |
| 2026-05-16T11:02:04Z | H2: Data too old/incomplete for backtest date range | unverified |
| 2026-05-16T11:02:04Z | H3: All daily templates may be disabled | unverified |
Note: a hypothesis becomes a FACT only when DIAGNOSIS_zero_trades.md
confirms it WITH evidence. Then add a new row to the FACTS table.

## ❌ STALE ASSUMPTIONS — DO NOT TRUST (contradicted by facts)
| Logged (UTC) | Stale assumption | Contradicted by |
|--------------|------------------|-----------------|
| 2026-05-16T11:02:04Z | "PF ~2.0, ~20% return, 176-250 trades" (old memory) | actual = 0 trades (verified 2026-05-16T11:02:04Z) |
| 2026-05-16T11:02:04Z | "04_magic_numbers.md says slippage is 50x" | Investigation A proved it is 2x |

## Session Log (append-only)
| Timestamp (UTC) | What was done | Outcome |
|-----------------|---------------|---------|
| 2026-05-16T11:02:04Z | Verified system state via facts; found zero-trades; created workspace docs | Diagnosis + docs prompts issued |

## Next Step
Await DIAGNOSIS_zero_trades.md → convert H1/H2/H3 to facts → fix P0-1
