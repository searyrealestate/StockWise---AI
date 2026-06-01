# micha7_analyzer — Architecture Decision Records (ADR)

> **Version:** 1.0.0
> **Last Modified:** 2026-05-21T05:35:00Z

This document records *why* architectural choices were made. Each decision includes context, options considered, and rationale.

---

## ADR-001: Standalone Module on StockWise Infrastructure
**Date:** 2026-05-21
**Status:** Superseded by ADR-014

### Context
Three options for integrating micha7 with StockWise AI:
1. Template inside existing StockWise template system
2. Separate engine running parallel to StockWise
3. Standalone module that uses StockWise infrastructure

### Decision
**Option 3 — Standalone module.**

### Rationale
- StockWise template system targets a different abstraction (probabilistic templates with ML)
- A parallel engine duplicates infrastructure (data, risk, notifications)
- Standalone module reuses infrastructure but keeps logic isolated
- Allows micha7 to evolve independently without affecting StockWise core

### Consequences
- ✅ Clear ownership boundaries
- ✅ Easy to test in isolation
- ✅ No risk of breaking existing templates
- ⚠️ Some duplication of utility patterns (mitigated by reusing StockWise utilities)

  ### Superseded By
  ADR-014 (2026-06-01) — Standalone-First Architecture

---

## ADR-002: Deterministic (Phase A) — No ML in Core
**Date:** 2026-05-21
**Status:** Accepted

### Context
StockWise has both Phase A (deterministic) and Phase B (ML). The 7-parameter checklist from Micha's methodology is rule-based.

### Decision
**micha7 is pure Phase A — no ML in the core analysis pipeline.**

### Rationale
- The methodology being implemented is deterministic by nature
- Determinism enables exact reproducibility (critical for backtesting)
- Removes ML complexity (model versioning, drift, training pipelines)
- Faster to implement and validate

### Consequences
- ✅ Reproducible results (same input → same output)
- ✅ Easier debugging (no probabilistic surprises)
- ⚠️ Cannot adapt to changing market regimes automatically
- ⚠️ Future ML enhancement possible but kept out of v1

---

## ADR-003: Topological DAG for Feature Ordering
**Date:** 2026-05-21
**Status:** Accepted

### Context
Features have dependencies:
- Candle pattern context needs S/R levels
- Volume signal needs trend direction

Naive ordering risks silent failures or circular dependencies.

### Decision
**Use Directed Acyclic Graph (DAG) with declarative dependencies and topological sort.**

### Rationale
- Self-documenting (deps are declared per feature)
- Validates at startup (rejects circular references)
- Enables parallel computation of independent features (Level 1)
- Adding new features = adding nodes, not refactoring order

### Alternatives Considered
- **Manual ordering in code:** Brittle, easy to break
- **Single pass with all features:** Loses context-awareness
- **Event-driven:** Over-engineering for this scope

### Consequences
- ✅ Compile-time-like validation of dependency graph
- ✅ Scales to N features without code restructuring
- ⚠️ Slightly more complex than naive sequential code

---

## ADR-004: Single Source of Truth for Visualizations (ChartSpec)
**Date:** 2026-05-21
**Status:** Accepted

### Context
Two visualization targets required:
- HTML for local viewing
- TradingView Pine Script for charting integration

Naive approach: two separate renderers with duplicate logic → drift risk.

### Decision
**Generate intermediate `ChartSpec` (JSON). Renderers consume the spec.**

### Rationale
- Logic in one place; presentation in many
- Adding new renderer (PDF, PNG, mobile) = one file
- Tests focus on spec correctness, not rendering details
- Spec acts as contract between analysis and output

### Consequences
- ✅ Zero drift between HTML and Pine Script
- ✅ Easy to extend with new output formats
- ⚠️ One extra layer of indirection

---

## ADR-005: Atomic Writes + Write-Ahead Log for State
**Date:** 2026-05-21
**Status:** Accepted

### Context
State transitions are critical (e.g., circuit breaker state, position tracking). A crash mid-write can corrupt state or bypass safety checks.

### Decision
**All state writes use atomic rename pattern + WAL for transitions.**

### Rationale
- POSIX rename is atomic on same filesystem (well-known guarantee)
- WAL allows replay on startup if crash occurs
- Standard pattern in databases (PostgreSQL, SQLite use similar)
- Cost is minimal (one extra write per transition)

### Implementation
1. Write to `{file}.tmp`
2. fsync to force disk write
3. Atomic rename to `{file}`
4. Log transition in WAL
5. On startup: scan for orphan `.tmp` files, replay incomplete WAL entries

### Consequences
- ✅ No data loss from crashes
- ✅ Self-healing on startup
- ⚠️ Slightly slower than naive writes (acceptable for low-frequency state changes)

---

## ADR-006: Schema Versioning for Forward Compatibility
**Date:** 2026-05-21
**Status:** Accepted

### Context
State files persist across code updates. Adding a field or changing format breaks old files.

### Decision
**Every state file includes `schema_version`. Auto-migration on load.**

### Rationale
- Zero downtime during deployments
- Old state files automatically upgraded
- Migration history preserved in logs
- Each migration is testable in isolation

### Consequences
- ✅ Smooth upgrades, no manual cleanup
- ✅ Auditable migration history
- ⚠️ Discipline required: every breaking change needs a migration handler

---

## ADR-007: Multi-Level Circuit Breaker (Kill Switch)
**Date:** 2026-05-21
**Status:** Accepted

### Context
Automated trading systems can cause unbounded losses if a bug or market shift triggers consistent bad trades. Knight Capital (2012) lost $440M in 45 minutes — no circuit breaker.

### Decision
**Implement 4-level circuit breaker: Warning → Suspended → Disabled → Emergency.**

### Rationale
- Each level adds friction proportional to severity
- Manual override required for serious states (prevents auto-resume of bad system)
- Multiple triggers (consecutive losses, weekly losses, drawdown) catch different failure modes
- Industry standard for production trading systems

### Consequences
- ✅ Bounded losses by design
- ✅ Time for human review on serious issues
- ⚠️ Risk of false positives suspending good system (mitigated by manual override)

---

## ADR-008: Namespace Separation for Modes (Live/Paper/Backtest)
**Date:** 2026-05-21
**Status:** Accepted

### Context
StockWise had a known issue: trust cache bleed between backtest and live runs. Same data structures used for different purposes caused drift.

### Decision
**Each mode (live/paper/backtest) gets its own state directory. No shared state files.**

### Rationale
- Backtest results must not be affected by live state, ever
- Paper trading should mirror production behavior but stay isolated
- Different update frequencies (live=continuous, backtest=batch) need different caching

### Consequences
- ✅ Guaranteed isolation between modes
- ✅ Backtest reproducibility (no contamination)
- ⚠️ Slightly more disk usage (negligible)

---

## ADR-009: Dual-Track Visualization (Local + TradingView)
**Date:** 2026-05-21
**Status:** Accepted

### Context
User explicitly requested TradingView integration. Need local visualization for development too.

### Decision
**Both: HTML (Lightweight Charts) + Pine Script for TradingView.**

### Rationale
- HTML for fast local iteration during development
- Pine Script for integration with user's TradingView workflow
- Both consume same ChartSpec (no drift)

### Consequences
- ✅ Best of both worlds
- ✅ Single source of truth prevents divergence
- ⚠️ Two renderers to maintain (mitigated by shared ChartSpec)

---

## ADR-010: Merged Pivot Detector with Analyzer (No Separate File)
**Date:** 2026-05-21
**Status:** Accepted

### Context
Originally planned `micha7_pivot_detector.py` as a separate file. Re-evaluation showed PivotDetector calls FeatureExtractor heavily, creating tight coupling.

### Decision
**Keep PivotDetector inside `micha7_analyzer.py`.**

### Rationale
- Tight coupling between PivotDetector and FeatureExtractor
- Separate file would create circular import risk
- Single file with focused classes is preferable to artificially split files

### Consequences
- ✅ Simpler import graph
- ✅ Co-located code that's tightly related
- ⚠️ Larger single file (mitigated by clear class boundaries; will split if > 800 LOC)

---

## ADR-011: Phase 1 Excludes Live Execution and Visualizer
**Date:** 2026-05-21
**Status:** Accepted

### Context
Tempting to build everything at once. Risk: integration complexity, harder to validate each part.

### Decision
**Phase 1 scope:** DataAdapter + FeatureExtractor + ScoringEngine + EntryPlanner + StateManager + Backtest runner ONLY.

**Explicitly out of Phase 1:** Live execution, Telegram notifications, HTML/Pine renderers, Scheduler in production mode.

### Rationale
- Validate the analysis logic before adding execution complexity
- Backtest provides immediate value (historical validation)
- Smaller scope = faster iteration
- Lower risk of cascading bugs across layers

### Consequences
- ✅ Clear, achievable Phase 1 goal
- ✅ Each phase validates before next
- ⚠️ User has to wait for visualization (Phase 3)

---

## ADR-012: Pure Functions in FeatureExtractor
**Date:** 2026-05-21
**Status:** Accepted

### Context
Need to enable rigorous unit testing and reproducibility.

### Decision
**All feature calculation methods are pure functions: same input → same output, no side effects.**

### Rationale
- Trivially testable (no mocks needed for calculation logic)
- Reproducible (no hidden state)
- Parallelizable (no shared mutable state)
- Easier to reason about

### Consequences
- ✅ High test coverage achievable
- ✅ Easier debugging
- ⚠️ State must be threaded explicitly through call chains

---

## ADR-013: Documentation in `.claude-workspace/micha7/`
**Date:** 2026-05-21
**Status:** Accepted

### Context
User requested documentation strategy that allows resuming work in new chat sessions without losing context.

### Decision
**Centralized documentation in `.claude-workspace/micha7/` with split Public/Private files.**

### Rationale
- Aligns with planned StockWise State Sync System
- Easy to upload specific files to new chat sessions
- Clear separation between Git-safe and private content
- Documentation lives with the project (not external wiki)

### Consequences
- ✅ Continuity across chat sessions
- ✅ Security boundaries enforced by file naming convention
- ⚠️ Discipline required to keep `.local.md` files updated

---

## ADR-014: Standalone-First Architecture; StockWise Integration Deferred
**Date:** 2026-06-01
**Status:** Accepted (supersedes ADR-001)

### Context
ADR-001 defined micha7 as a "standalone module on StockWise infrastructure."
Further analysis revealed three issues:
1. Strong coupling to StockWise blocks development when main project unavailable
2. Backtest requires hands-on testing — DSM dependency slows iteration
3. Integration is low-ROI before having a proven system

### Decision
micha7 in Phase 1 = **fully standalone**. No StockWise dependencies.
Optional StockWise integration will be added in a later phase via Adapter Pattern.

### Rationale
- Enables fast, isolated development
- TDD and CI/CD are simpler (no DSM/portfolio_risk mocks needed)
- If micha7 proves itself → integration is easy via adapter

### Alternatives Considered
- **Continue with ADR-001:** Requires duplicating StockWise infrastructure in tests
- **Hybrid:** Too complex to maintain

### Consequences
- ✅ Fast development, fewer dependencies
- ✅ micha7 runs on any machine without StockWise
- ⚠️ Need standalone implementations of: data fetching, risk validation
- ⚠️ Future integration = additional work (Phase 6+)

---

## ADR-015: DataProvider Interface; yfinance as Phase 1 Implementation
**Date:** 2026-06-01
**Status:** Accepted

### Context
Phase 1 requires a single data source. We considered: Alpaca, IBKR, yfinance, StockWise DSM.
Decision: IBKR for future real trading; Phase 1 = yfinance.

### Decision
Implementation via **Abstract Interface**:
- `BaseDataProvider` (ABC) — defines contract
- `YFinanceProvider` — active implementation in Phase 1
- Future implementations: `IBKRProvider`, `StockWiseDSMProvider`, `AlpacaProvider`

### Rationale
- Allows future data source replacement without changing Analyzer
- Tests are simple — `MockDataProvider` for unit tests
- yfinance is strong enough for historical backtest data

### Alternatives Considered
- **Direct yfinance calls:** Tight coupling, hard to replace
- **Multi-provider from day 1:** Over-engineering for Phase 1

### Consequences
- ✅ Phase 1 develops fast (yfinance: pip install + import)
- ✅ Phase 4 (paper) / Phase 5 (live) — add IBKRProvider without touching analysis
- ⚠️ yfinance can be unstable — add retry logic in YFinanceProvider

---

## Decision Template (for future ADRs)

```markdown
## ADR-NNN: [Short Title]
**Date:** YYYY-MM-DD
**Status:** [Proposed | Accepted | Deprecated | Superseded by ADR-XXX]

### Context
[What's the situation that requires a decision?]

### Decision
[What did we decide?]

### Rationale
[Why this decision over alternatives?]

### Alternatives Considered
[What else did we look at?]

### Consequences
[What are the pros/cons we're accepting?]
```
