# StockWise AI — Mission & Priorities
**Last updated (UTC):** 2026-05-16T11:02:04Z | **Maintained by:** Architect (LLM) + Eyal

## ⚠️ READ THIS FIRST (for Claude Code AND any new chat)
At the start of EVERY session, read these 4 files in order:
1. MISSION.md (this file) — what we're building & why
2. STATE.md — where we are right now
3. TASKS.md — what to do next
4. KNOWN_ISSUES.md — traps to avoid
Never start work without reading all 4.

## ⏱️ Append-Only Discipline (MANDATORY)
- Every entry added to STATE.md, TASKS.md, KNOWN_ISSUES.md MUST carry
  a UTC ISO-8601 timestamp (YYYY-MM-DDTHH:MM:SSZ).
- NEVER delete or overwrite an existing row. To change something,
  ADD a new row with a newer timestamp.
- When two rows conflict, the row with the LATEST timestamp is the
  current truth. Older rows are kept as history.
- This file (MISSION.md) and ultrareview.md may be edited in place,
  but always bump "Last updated (UTC)".

## Vision — 3 Phases
| Phase | Goal | Status |
|-------|------|--------|
| 1. Rule-Based | Deterministic system, clear templates & rules | IN PROGRESS — BLOCKED |
| 2. Pattern Discovery | System finds new patterns from data ("needle in haystack") | NOT STARTED |
| 3. Self-Reflection | System knows its strengths/weaknesses, proposes improvements | NOT STARTED |

## Operating Mode
- Human-in-the-loop: system PROPOSES, Eyal APPROVES before any change
- Pace: correctness over speed — no rushing
- Glass box, not black box: every decision must be human-explainable

## Success Criteria (gates between phases)
- Phase 1 exit: 2 consecutive backtests produce bit-identical hash
  AND total_trades > 0 AND a measurable baseline PF exists
- Phase 2 entry: stable baseline PF >= 2.5 for 30 days
- Phase 3 entry: >= 3 auto-discovered templates survive OOS validation

## Permanently Banned (NEVER recreate)
- RESISTANCE_SQUEEZE — buying near resistance is fundamentally wrong
- OVERSOLD_BOUNCE — chronically losing, conditions must not be loosened

## Return Target
- Calibrated goal: ~3%/month (Track 2).
- 2%/day is mathematically unrealistic — confirmed, not pursued.

## Priority Order (FACTS-BASED — update only with verified evidence)
P0-1: Fix NumPy/SciPy environment conflict
P0-2: Diagnose & fix root cause of zero trades
P1-1: Verify data quality & freshness
P1-2: Audit + commit the WIP Trust Cache fix in backtest_engine.py
P1-3: Define /ultrareview command
P2-1: Establish clean backtest baseline + metrics
P2-2: Fix slippage convention mismatch
P3:   Symbol expansion + losing-template cleanup
P4-5: Pattern discovery automation → self-reflection
(Rationale for each priority lives in TASKS.md)
