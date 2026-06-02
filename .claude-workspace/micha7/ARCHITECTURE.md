# micha7_analyzer — Architecture Specification (Public)

> **Version:** 2.0.0
> **Created:** 2026-05-21T05:35:00Z
> **Last Modified:** 2026-06-01T08:30:00Z
> **Status:** Architecture Complete — Implementation Pending
> **Maturity Score:** 98.2%

⚠️ **Privacy Notice:** This is the PUBLIC version. Specific interfaces, function signatures, formulas, and implementation details are documented in `implementation_notes.local.md` (private, gitignored).

---

## 1. Executive Summary

**micha7_analyzer** is a deterministic technical analysis module that scores stocks against a multi-parameter checklist, manages entry-timing via a state machine, and emits actionable signals across multiple channels.

### Design Pillars

| Pillar | Description |
|--------|-------------|
| **Deterministic** | No ML in core flow; pure rule-based scoring |
| **Standalone** | Self-contained; uses parent system infrastructure but doesn't modify core |
| **Template Pattern** | Designed as the first of a reusable analyzer family (see `TEMPLATE_ENGINE.md`) |
| **Recoverable** | Atomic state writes + WAL + crash recovery |
| **Safe** | Multi-level circuit breaker prevents runaway losses |
| **Observable** | Audit logs, state history, structured events |

---

## 2. System Boundaries

### What micha7 IS

- A standalone analysis pipeline (works without StockWise)
- A signal generator (not an executor)
- A template for future deterministic analyzers
- A DataProvider-agnostic consumer (uses interface, not concrete provider)

### What micha7 is NOT

- A replacement for parent system's template system
- A modifier of parent core code
- An ML/probabilistic system
- A position management system (delegates to existing infrastructure)

### Integration Surface (Phase 1: Standalone)

| Component | Source | Modification |
|-----------|--------|--------------|
| DataProvider (BaseDataProvider) | micha7 internal | — |
| YFinanceProvider | micha7 internal | — |
| Standalone RiskManager | micha7 internal | — |
| (Future) StockWise DSM | StockWise (optional, Phase 6+) | None — wrapped in adapter |
| (Future) StockWise Portfolio Risk | StockWise (optional, Phase 6+) | None — wrapped in adapter |

**Untouched components (Phase 1):** No StockWise modules. micha7 is fully isolated.

---

## 3. Layered Architecture

The system is organized in 4 layers, each with clear responsibilities.

### Layer 1: Entry Layer
**Responsibility:** Decides *when* to run analysis and *whether* it's safe.

| Component | Purpose | Phase |
|-----------|---------|-------|
| TradingCalendar | Knows when market is open | Phase 2 |
| Scheduler | Triggers analysis at correct times (modes: EOD/Live/Paper/Backtest) | Phase 2 |
| CircuitBreaker | 4-level safety system | Phase 2 |

### Layer 2: Analysis Layer
**Responsibility:** The analysis pipeline from raw data to trade plan.

| Component | Purpose | Phase |
|-----------|---------|-------|
| DataAdapter | Validates, normalizes, computes shared metrics | Phase 1 ✅ |
| FeatureExtractor | Computes features in DAG order | Phase 1 |
| ScoringEngine | Aggregates features → bullish_count score; tracks history | Phase 1 |
| PivotDetector | 5-state machine for entry timing | Phase 1 |
| EntryPlanner | Computes entry/stop/targets | Phase 1 |
| RiskManager | Validates against portfolio constraints | Phase 1 |

### Layer 3: Persistence Layer
**Responsibility:** Durable state with crash safety.

| Component | Purpose | Phase |
|-----------|---------|-------|
| StateManager | Atomic writes, schema versioning, migration, recovery | Phase 1 |
| Namespace Separation | Modes isolated | Phase 1 |

### Layer 4: Output Layer
**Responsibility:** Distribute signals via unified data model.

| Component | Purpose | Phase |
|-----------|---------|-------|
| TradeJournal (local) | JSON + CSV trade log (ADR-018) | Phase 1 |
| ChartSpec Generator | Single Source of Truth for visualizations | Phase 3 |
| HTML Renderer | Local interactive charts | Phase 3 |
| Pine Script Generator | TradingView integration | Phase 3 |
| SignalEmitter | Routes signals to consumers | Phase 2 |
| TradeJournal (Sheets) | Google Sheets sync (ADR-018) | Phase 4+ |

---

## 4. Data Flow (Conceptual)

```
[1] Scheduler triggers (timing verified)
    ↓
[2] Safety checks (circuit breaker, calendar)
    ↓
[3] Data retrieval and validation
    ↓
[4] Feature extraction (DAG-ordered)
    ↓
[5] Score aggregation
    ↓
[6] State machine evaluation
    ↓
[7] If entry conditions: plan generation
    ↓
[8] Risk validation
    ↓
[9] State persistence (atomic)
    ↓
[10] Output distribution
```

---

## 5. Cross-Cutting Concerns

### Determinism

**Guarantee:** Same input → same output, always.

| Mechanism | Status |
|-----------|--------|
| Declarative feature ordering | ✅ Implemented |
| Atomic state transitions | ✅ Implemented |
| No random seeds in core | ✅ Verified |
| Consistent rounding rules | ⚠️ Deferred (no rounding block in config yet; tracked in IMPROVEMENT_ROADMAP.md) |

### Recoverability

**Guarantee:** System always recovers from crash to a consistent state.

| Mechanism | Status |
|-----------|--------|
| Atomic writes | ✅ Implemented |
| Write-Ahead Log | ✅ Implemented |
| Startup recovery | ✅ Implemented |
| State validation on load | ✅ Implemented |

### Forward Compatibility

**Guarantee:** Code changes don't break existing state.

| Mechanism | Status |
|-----------|--------|
| Schema versioning | ✅ Implemented |
| Migration pipeline | ✅ Implemented |
| Pre-migration backups | ✅ Implemented |

### Safety

**Guarantee:** Bad situations stop the system before runaway damage.

| Level | Trigger Type | Action |
|-------|--------------|--------|
| Warning | Mild concern | Log + info notification |
| Suspended | Pattern detected | Pause for cooling-off period |
| Disabled | Serious concern | Manual override required |
| Emergency | Critical condition | Stop + close positions |

*Specific thresholds and durations: see `config_values.local.md`*

---

## 6. Mode Isolation

The system runs in 4 isolated modes:

```
state/micha7/
├── live/       — production trading
├── paper/      — paper trading
├── backtest/   — historical simulations
└── _system/    — global state
```

**Critical Property:** Backtest cannot affect live. Paper cannot affect backtest. Modes are strictly isolated.

---

## 7. Template Reusability

micha7_analyzer is the **first** of a family. See `TEMPLATE_ENGINE.md` for the complete reusability strategy.

### Generic Infrastructure (designed for reuse)

| Component | Reusable? |
|-----------|-----------|
| Scheduler | ✅ Generic |
| CircuitBreaker | ✅ Generic |
| StateManager | ✅ Generic |
| ChartSpec/Visualizer | ✅ Generic |
| Trading Calendar | ✅ Generic |

### micha7-Specific (will be reimplemented per analyzer)

| Component | Why Specific |
|-----------|--------------|
| FeatureExtractor implementations | Each analyzer has its own features |
| ScoringEngine thresholds | Each analyzer has its own logic |
| PivotDetector states | Each analyzer has its own state machine |

---

## 8. Testing Philosophy

**This system follows strict Agile testing protocols.** See `TESTING_PROTOCOL.md` for full details.

### Core Rules
- No code without tests
- No commit without all tests passing
- No skipping the test pyramid
- Test pyramid: 70% unit, 25% integration, 5% E2E

### Mandatory Tests (Every Component)
- Happy path ≥1
- Edge cases ≥2
- Error cases ≥1
- Boundary values ≥1
- Idempotency (where applicable)

### Special Required Tests (Phase 1)
- Determinism (same run twice = identical results)
- Lookahead bias (no future data in past decisions)
- Crash recovery (mid-transition crash recovers cleanly)

---

## 9. Maturity Assessment

> **⚠️ SUPERSEDED:** This maturity score (98.2%) was computed against the pre-2026-06-03
> architecture, which incorrectly described business logic (signed integer scoring, R:R filter,
> etc.). The methodology has been corrected to match Micha's actual transcript (see
> `business_logic.local.md`). Score will be reassessed at end of Phase 1 after backtest
> validation. Treat this table as historical context only.

| Criterion | Weight | Score | Notes |
|-----------|--------|-------|-------|
| Functional Completeness | 15% | 100% | All required parameters covered |
| Determinism | 12% | 100% | DAG + atomic state |
| Failure Handling | 12% | 98% | Circuit breaker + WAL |
| Data Integrity | 10% | 100% | Atomic writes + validation |
| Forward Compatibility | 8% | 100% | Schema versioning |
| Parent System Integration | 10% | 100% | Standalone, no core mods |
| Testability | 8% | 95% | Pure functions, DI ready |
| Observability | 8% | 90% | Logs + audit trail |
| Operational Safety | 10% | 100% | 4-level circuit breaker |
| Code Organization | 7% | 95% | 4 new files, balanced |
| **WEIGHTED TOTAL** | **100%** | **98.2%** | **Ready for implementation** |

**Path to 100%:** See `IMPROVEMENT_ROADMAP.md` — 7 documented improvements bring system from 98.2% to ~99.5%+.

---

## 10. Open Questions & Future Enhancements

These are **not blockers** for Phase 1. See `IMPROVEMENT_ROADMAP.md` for full list.

| ID | Topic | When |
|----|-------|------|
| IMP-001 | Formal mock framework | When tests become painful |
| IMP-002 | Structured logging | When debugging hurts |
| IMP-003 | Health check endpoint | Before Phase 4 |
| IMP-005 | Distributed tracing | Phase 4+ |
| IMP-006 | Metrics export | Phase 5+ |
| IMP-007 | Comprehensive error recovery | Phase 4+ |

---

## 11. Document Maintenance

This document is updated when:
- Architectural decisions change (also recorded in `DECISIONS.md`)
- Components are added/removed (also in `PROJECT_STRUCTURE.md`)
- Maturity score changes

**Update protocol:**
1. Edit this file
2. Update "Last Modified" timestamp
3. Add entry to `CHANGELOG.md`
4. If a decision: add ADR entry to `DECISIONS.md`
5. **Do not include:** specific values, formulas, code, or interfaces (those go in `.local.md` files)

---

## 12. Reading Order for New Team Members

1. `README.md` — Navigation hub
2. `ARCHITECTURE.md` — This file (system overview)
3. `GLOSSARY.md` — Domain terms
4. `PHASES.md` — Project roadmap
5. `DECISIONS.md` — Why things are the way they are
6. `TEMPLATE_ENGINE.md` — Future reusability
7. `TESTING_PROTOCOL.md` — How we work
8. `SECURITY.md` — What's public vs private
9. `IMPROVEMENT_ROADMAP.md` — Where we're going
10. `PROJECT_STRUCTURE.md` — Where everything lives
11. `CHANGELOG.md` — What happened when

**Then** request access to `.local.md` files for implementation details.

---

## 13. State Machine Reference

See `business_logic.local.md §5` for the full state machine specification.

**Brief summary:** 5 states: WAITING → ARMED → TRIGGERED → IN_POSITION → EXITED.

| State | Trigger in | Trigger out |
|-------|-----------|-------------|
| WAITING | Default / after EXITED / invalidation | Score ≥ 6/7 → ARMED |
| ARMED | Score ≥ 6/7 | Pivot conditions met → TRIGGERED; invalidation → WAITING |
| TRIGGERED | All 3 pivot conditions (D-10) | Entry filled → IN_POSITION; timeout → ARMED |
| IN_POSITION | Entry order filled | Target or stop hit → EXITED |
| EXITED | Target or stop | Symbol returns to WAITING for next setup |

Persistence: all transitions are atomically persisted via StateManager + WAL (ADR-005, ADR-006).
