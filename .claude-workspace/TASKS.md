# StockWise AI — Task List
**Last updated (UTC):** 2026-05-16T13:14:03Z
**Legend:** ⬜ TODO | 🔵 IN_PROGRESS | ✅ DONE | ⏸️ BLOCKED | ❌ CANCELLED
**Rule:** never delete a task. Change status by ADDING a status-change
row in the Status Change Log with a UTC timestamp.

## P0 — Critical (system produces zero trades)
| ID | Status | Task | Rationale |
|----|--------|------|-----------|
| P0-0 | 🔵 | Run zero-trades diagnosis (DIAGNOSIS_zero_trades.md) | Must know WHY before any fix |
| P0-1 | ⬜ | Fix NumPy/SciPy version conflict | Confirmed warning; likely NaN-feature cause |
| P0-2 | ⏸️ | Fix root cause of zero trades | BLOCKED by P0-0 verdict |

## P1 — Foundation
| ID | Status | Task | Rationale |
|----|--------|------|-----------|
| P1-1 | ⏸️ | Verify data quality & freshness | BLOCKED by P0-0; stale mixed files observed |
| P1-2 | ⬜ | Audit + commit WIP Trust Cache fix in backtest_engine.py | Undocumented +88 lines; verify before commit |
| P1-3 | ⬜ | Define /ultrareview as .claude/commands/ultrareview.md | Required by workflow; created in this task |

## P2 — Accuracy & Baseline
| ID | Status | Task | Rationale |
|----|--------|------|-----------|
| P2-1 | ⏸️ | Establish clean deterministic backtest baseline + metrics | Cannot measure improvement without baseline |
| P2-2 | ⬜ | Fix slippage convention mismatch (0.05 pp vs 0.001 decimal) | Investigation A: 2x friction gap |

## P3 — Expansion (only after stable base)
| ID | Status | Task | Rationale |
|----|--------|------|-----------|
| P3-1 | ⬜ | Symbol expansion 13 → 50 (validate PF at each stage) | Breadth is the binding constraint |
| P3-2 | ⬜ | Cleanup chronically-losing templates | Noise that lowers PF |

## P4–P5 — Self-Improvement (only after deterministic clean base)
| ID | Status | Task | Rationale |
|----|--------|------|-----------|
| P4-1 | ⬜ | Pattern Discovery automation (Phase 2) | Needs deterministic + clean data first |
| P5-1 | ⬜ | Self-Reflection: strengths/weaknesses/failure-mode catalog (Phase 3) | Needs Phase 2 first |

## Status Change Log (append-only)
| Timestamp (UTC) | Task ID | Change | Note |
|-----------------|---------|--------|------|
| 2026-05-16T11:02:04Z | P0-0 | ⬜ → 🔵 | Diagnosis prompt issued |
| 2026-05-16T11:02:04Z | P1-3 | ⬜ → 🔵 | Being created in this setup task |
| 2026-05-16T13:14:03Z | P0-0 | 🔵 → ✅ | Zero-trades diagnosis complete (two passes) |
| 2026-05-16T13:14:03Z | P0-1 | ⬜ → ❌ | CANCELLED — NumPy/SciPy proven NOT the cause; argrelextrema works |
| 2026-05-16T13:14:03Z | P0-2 | ⏸️ → 🔵 | Re-scoped: root cause is bearish test period, not a bug. Next: prove via bullish-period backtest |

## Completed
(none yet)

## Open Questions (must resolve before any fix)
| ID | Question | Why it matters |
|----|----------|----------------|
| OQ-1 | Does the system generate trades in a BULLISH period? | If yes → no bug, move to baseline. If no → real pipeline bug |
| OQ-2 | What exact date range + symbols had confirmed uptrend? | Needed to design the OQ-1 test fairly |
| OQ-3 | Is the WIP in backtest_engine.py (+88 lines) safe to keep? | Still uncommitted, origin unconfirmed (P1-2) |
| OQ-4 | macd_signal=NaN on early bars — does it cause spurious PASS? | Potential silent wrong-signal bug, separate from zero-trades |
