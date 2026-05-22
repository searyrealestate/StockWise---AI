# micha7_analyzer — Phased Rollout Plan

> **Version:** 1.0.0
> **Last Modified:** 2026-05-21T05:35:00Z
> **Current Phase:** 0 (Architecture Complete) → Ready for Phase 1

---

## Phase Overview

| Phase | Name | Status | Validation Gate |
|-------|------|--------|-----------------|
| **0** | Architecture & Documentation | ✅ COMPLETE | Maturity ≥ 95% |
| **1** | Backtest Engine | ⏳ NEXT | PF ≥ 1.5, WR ≥ 55% |
| **2** | EOD Scheduler + Calendar | ⏸️ PENDING | False positive rate < 30% |
| **3** | Visualizer (HTML + Pine Script) | ⏸️ PENDING | UX validation |
| **4** | Paper Trading (Alpaca) | ⏸️ PENDING | 30 days paper, PF ≥ 2.0 |
| **5** | Live Trading | ⏸️ PENDING | 60 days successful paper |
| **6+** | Enhancements (E1-E8) | ⏸️ FUTURE | Per-enhancement criteria |

---

## Phase 0: Architecture & Documentation ✅ COMPLETE

### Deliverables
- ✅ System architecture defined (4 layers, 12+ components)
- ✅ 47 architectural issues identified and resolved
- ✅ Maturity score: 98.2%
- ✅ File structure planned (4 new, 3 modified)
- ✅ Documentation written (this workspace)

### Exit Criteria Met
- ✅ Architecture maturity ≥ 95%
- ✅ All cross-cutting concerns addressed
- ✅ No critical issues open
- ✅ Documentation in place

---

## Phase 1: Backtest Engine ⏳ NEXT

### Goal
Build the core analysis pipeline and validate it against historical data.

### Scope (IN)
- DataAdapter (StockWise DSM wrapper + validation + ATR)
- FeatureExtractor (DAG-based, all 7 features)
- ScoringEngine (aggregation + history)
- PivotDetector (state machine, in-memory for backtest)
- EntryPlanner (3 targets + stop + R:R)
- RiskManager (uses portfolio_risk)
- StateManager (atomic + WAL + schema versioning)
- BacktestRunner (orchestrates above)
- Unit tests for each component
- Integration test for full pipeline

### Scope (OUT — deferred to later phases)
- ❌ Scheduler in production mode (Phase 2)
- ❌ Trading calendar integration (Phase 2)
- ❌ Circuit breaker live activation (Phase 2)
- ❌ Telegram notifications (Phase 2)
- ❌ HTML visualizer (Phase 3)
- ❌ Pine Script generator (Phase 3)
- ❌ Paper trading (Phase 4)
- ❌ Live trading (Phase 5)

### Validation Gate (Exit Criteria)
| Metric | Target | How Measured |
|--------|--------|--------------|
| Unit test pass rate | 100% | Run unit_tests.py |
| Integration test pass | 100% | Run test_micha7_integration.py |
| Backtest determinism | Bit-identical | Run twice, compare outputs |
| Backtest on 13 symbols | PF ≥ 1.5, WR ≥ 55% | 90-day historical |
| No breakage of StockWise | All existing tests pass | Run master_validator |
| Code review pass | 100% | Manual review against ARCHITECTURE.md |

### Estimated Effort
- Architecture-to-code: ~20-30 hours of focused development
- Testing: ~10-15 hours
- Validation: ~5 hours
- **Total: ~35-50 hours**

### Risks
| Risk | Mitigation |
|------|-----------|
| S/R detection algorithm doesn't match Micha's visual approach | Test against hand-annotated examples; tune thresholds |
| Backtest results poor on small symbol set | Phase 1 success = working pipeline, not necessarily profitable strategy |
| Integration with existing StockWise breaks something | Run full master_validator after each commit |

### Deliverables
- 4 new Python files
- 3 modified Python files
- Test suite expansion
- `CHANGELOG.md` entries per component
- `IMPLEMENTATION_LOG.local.md` with details
- Updated maturity assessment

---

## Phase 2: EOD Scheduler ⏸️ PENDING

### Goal
Activate automated end-of-day analysis with safety controls.

### Scope
- TradingCalendar (NYSE schedule)
- Scheduler (EOD mode)
- CircuitBreaker (4-level)
- Notification routing (Telegram alerts)

### Validation Gate
- 30 days of EOD runs without errors
- Circuit breaker triggered correctly on test scenarios
- False positive rate (alerts that wouldn't lead to trades) < 30%

---

## Phase 3: Visualizer ⏸️ PENDING

### Goal
Visual analysis output for both local viewing and TradingView integration.

### Scope
- ChartSpec generator
- HTML renderer (Lightweight Charts)
- Pine Script generator
- Output storage and retention

### Validation Gate
- HTML matches expected layout per spec
- Pine Script imports cleanly to TradingView
- 100% match between HTML and Pine for same input

---

## Phase 4: Paper Trading ⏸️ PENDING

### Goal
Connect to Alpaca paper trading and validate end-to-end signals.

### Scope
- Paper mode state namespace
- Order submission via Alpaca API
- Trade tracking and outcome recording
- Daily reconciliation reports

### Validation Gate
- 30 days of paper trading
- PF ≥ 2.0
- WR ≥ 60%
- Max drawdown ≤ 10%

---

## Phase 5: Live Trading ⏸️ PENDING

### Goal
Production deployment with real capital.

### Pre-Requisites
- 60 successful days of paper trading
- Manual review of every code path
- Disaster recovery plan tested
- Manual override procedures documented and rehearsed

### Validation Gate
- Continuous monitoring for 90 days
- Performance matches paper trading within 20% variance
- No critical incidents

---

## Phase 6+: Enhancements ⏸️ FUTURE

| ID | Enhancement | Trigger |
|----|-------------|---------|
| E1 | Distributed tracing | Multiple symbols slow down debugging |
| E2 | Metrics export (Prometheus) | Need real-time dashboards |
| E3 | Formal mock framework | Test maintenance becomes painful |
| E4 | Split analyzer.py | File exceeds 800 LOC |
| E5 | Parallel backtest | Backtest time > 5 minutes |
| E6 | Advanced Pine alerts | User requests TradingView push alerts |
| E7 | A/B testing framework | Want to compare config variants |
| E8 | Intraday support | Daily resolution insufficient |

---

## Current Status Snapshot

```
┌──────────────────────────────────────────┐
│  micha7_analyzer — Project Status         │
├──────────────────────────────────────────┤
│  Phase 0: ████████████████████ 100% ✅   │
│  Phase 1: ░░░░░░░░░░░░░░░░░░░░   0% ⏳   │
│  Phase 2: ░░░░░░░░░░░░░░░░░░░░   0% ⏸️   │
│  Phase 3: ░░░░░░░░░░░░░░░░░░░░   0% ⏸️   │
│  Phase 4: ░░░░░░░░░░░░░░░░░░░░   0% ⏸️   │
│  Phase 5: ░░░░░░░░░░░░░░░░░░░░   0% ⏸️   │
├──────────────────────────────────────────┤
│  Architectural Maturity:  98.2% ✅       │
│  Documentation Coverage: 100% ✅         │
│  Ready for next phase:   YES ✅          │
└──────────────────────────────────────────┘
```

---

## Phase Transition Protocol

To move from one phase to the next:

1. ✅ All exit criteria met for current phase
2. ✅ All deliverables complete
3. ✅ `CHANGELOG.md` updated with phase completion entry
4. ✅ `ARCHITECTURE.md` maturity reassessed
5. ✅ Any new decisions added to `DECISIONS.md`
6. ✅ User approval to proceed to next phase
7. ✅ This file (`PHASES.md`) updated with status

---

## Update Log

| Date | Phase | Change |
|------|-------|--------|
| 2026-05-21T05:35:00Z | 0 | Phase 0 complete, ready for Phase 1 |
