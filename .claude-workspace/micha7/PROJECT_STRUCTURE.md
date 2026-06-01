# micha7_analyzer — Project Structure

> **Version:** 2.0.0
> **Last Modified:** 2026-06-01T08:30:00Z

---

## 1. File Layout — Standalone Project

### Project Root

```
StockWise - AI/
└── micha7/                         ← Standalone project root
    ├── pyproject.toml              ← Independent dependencies
    ├── config.json                 ← All parameters (no hardcoded)
    ├── README.md
    ├── .gitignore
    │
    ├── micha7/                     ← Python package
    │   ├── __init__.py
    │   ├── analyzer.py             ← Micha7Analyzer (facade)
    │   ├── config_loader.py
    │   ├── logger.py
    │   │
    │   ├── data/
    │   │   ├── __init__.py
    │   │   ├── base_provider.py    ← BaseDataProvider (ABC)
    │   │   ├── yfinance_provider.py
    │   │   ├── data_adapter.py
    │   │   └── shared_metrics.py   ← ATR, returns, etc.
    │   │
    │   ├── features/
    │   │   ├── __init__.py
    │   │   ├── base_feature.py
    │   │   ├── feature_dag.py
    │   │   ├── f1_candle.py
    │   │   ├── f2_trend.py
    │   │   ├── f3_volume.py
    │   │   ├── f4_ma_distance.py
    │   │   ├── f5_gaps.py
    │   │   ├── f6_sr_levels.py
    │   │   ├── f7_cci.py
    │   │   └── scoring.py          ← ScoringEngine
    │   │
    │   ├── state/
    │   │   ├── __init__.py
    │   │   ├── state_manager.py    ← Atomic + schema versioning
    │   │   ├── wal.py              ← Write-Ahead Log
    │   │   └── pivot_detector.py   ← State machine
    │   │
    │   ├── trade/
    │   │   ├── __init__.py
    │   │   ├── entry_planner.py
    │   │   └── risk_manager.py     ← Standalone (no portfolio_risk)
    │   │
    │   └── backtest/
    │       ├── __init__.py
    │       ├── runner.py
    │       └── report.py
    │
    ├── tests/
    │   ├── conftest.py             ← Shared fixtures (mock provider, sample data)
    │   ├── unit/                   ← Per-module unit tests
    │   ├── integration/            ← E2E flows
    │   └── fixtures/               ← Deterministic test data (sample_ohlcv.csv)
    │
    ├── state/                      ← Runtime (gitignored)
    │   ├── live/
    │   ├── paper/
    │   ├── backtest/
    │   └── _system/
    │
    └── outputs/                    ← Runtime (gitignored)
        ├── charts/
        └── reports/
```

### Key Differences from Previous Layout (ADR-001 → ADR-014)

| Previous | Current |
|----------|---------|
| `micha7_*.py` files in StockWise root | All code under `micha7/` subfolder |
| Modified `feature_engine.py`, `system_config.py` | No StockWise file modifications |
| Used StockWise DSM, portfolio_risk | Standalone DataProvider + RiskManager |
| Tests in `unit_tests.py` | Dedicated `tests/` directory with pytest |

---

## 2. Module Responsibilities

### `micha7_analyzer.py`
**Contains:**
- DataAdapter class
- FeatureExtractor class (with DAG)
- ScoringEngine class
- PivotDetector class
- EntryPlanner class
- RiskManager class
- Micha7Analyzer (facade)

**Why one file?** All classes share the analysis context heavily; splitting would create circular imports. Threshold for splitting: 800 LOC.

**Dependencies:**
- `feature_engine` (shared calculations)
- `data_source_manager` (data)
- `portfolio_risk` (risk validation)
- `micha7_state_manager` (state I/O)
- `system_config` (parameters)

### `micha7_state_manager.py`
**Contains:**
- StateManager class
- WriteAheadLog class
- SchemaMigrator class
- StateRecovery class

**Why separate?** State management is generic infrastructure — should be reusable for future analyzers. Strong separation of concerns.

**Dependencies:**
- `safe_json_io`
- `system_config`

### `micha7_visualizer.py`
**Contains:**
- ChartSpec class (data model)
- ChartSpecGenerator class
- HTMLRenderer class
- PineScriptGenerator class

**Why separate?** Output is orthogonal to analysis. Allows adding new renderers without touching analysis code.

**Dependencies:**
- `system_config`
- `safe_json_io`

### `micha7_scheduler.py`
**Contains:**
- TradingCalendar class
- Scheduler class
- CircuitBreaker class
- ModeManager class (live/paper/backtest)

**Why separate?** Entry/control logic is distinct from analysis. Also reusable for future analyzers.

**Dependencies:**
- `pandas_market_calendars`
- `micha7_analyzer` (orchestrates)
- `notification_manager` (for circuit breaker alerts)

---

## 3. Class Hierarchy & Interactions

```
Micha7Analyzer (facade)
    │
    ├── uses: DataAdapter
    │            └── uses: data_source_manager
    │
    ├── uses: FeatureExtractor
    │            └── uses: feature_engine
    │
    ├── uses: ScoringEngine
    │
    ├── uses: PivotDetector
    │            └── uses: StateManager (via state_manager module)
    │
    ├── uses: EntryPlanner
    │
    └── uses: RiskManager
                 └── uses: portfolio_risk

Scheduler (orchestrator)
    │
    ├── uses: TradingCalendar
    │
    ├── uses: CircuitBreaker
    │            └── uses: StateManager
    │
    └── uses: Micha7Analyzer  ← runs the analysis

SignalEmitter (Phase 1: inside scheduler)
    │
    ├── uses: notification_manager (Telegram)
    │
    └── uses: ChartSpecGenerator
                 ├── feeds: HTMLRenderer
                 └── feeds: PineScriptGenerator
```

---

## 4. File Naming Conventions

| Pattern | Usage |
|---------|-------|
| `micha7_*.py` | All module files (consistent prefix) |
| `test_micha7_*.py` | Test files |
| `*.local.md` | Private documentation (gitignored) |
| `_wal.log` | WAL file (underscore prefix = system file) |
| `_system/` | System-level state directory |

---

## 5. Import Graph (No Circular Dependencies)

```
                    system_config
                          ↑
                          │ (all read from)
        ┌─────────────────┼─────────────────┐
        │                 │                 │
   state_manager     visualizer        scheduler
        ↑                                   │
        │                                   │ uses
        └───────────── analyzer ←───────────┘
                          ↑
                          │ uses
                feature_engine, data_source_manager,
                portfolio_risk, notification_manager
                (all StockWise infrastructure)
```

**Verified:** No circular imports. All dependencies flow downward.

---

## 6. Lines of Code Estimates

| File | Estimated LOC | Justification |
|------|---------------|---------------|
| `micha7_analyzer.py` | 500-700 | Many classes but small each |
| `micha7_state_manager.py` | 200-300 | Focused responsibility |
| `micha7_visualizer.py` | 300-450 | ChartSpec + 2 renderers |
| `micha7_scheduler.py` | 250-350 | Calendar + scheduler + breaker |
| `test_micha7_integration.py` | 300-500 | E2E + recovery tests |
| **TOTAL NEW CODE** | **1550-2300** | |

**Updates to existing files:** ~200-400 LOC additional.

---

## 7. Configuration Boundaries

All configurable parameters live in `system_config.py` under `MICHA7_CONFIG`.

**Categories:**

| Category | Examples | Sensitivity |
|----------|----------|-------------|
| Indicator periods | MA period, CCI period | Public |
| Score thresholds | min_score_for_long | **Private** (config_values.local.md) |
| Risk parameters | max_position_size_pct, stop_buffer | **Private** |
| Operational | retry counts, timeouts, paths | Public |
| Mode flags | enable_telegram, enable_viz | Public |
| Circuit breaker | loss thresholds, suspension durations | **Private** |

**Implementation:** Schema in `ARCHITECTURE.md`, **values** in `config_values.local.md`.

---

## 8. Test File Organization

```
tests/
├── test_micha7_integration.py     ← New: E2E flows
└── (additions to unit_tests.py):
    ├── TestMicha7DataAdapter
    ├── TestMicha7FeatureExtractor  ← per-feature tests
    ├── TestMicha7ScoringEngine
    ├── TestMicha7PivotDetector     ← state machine tests
    ├── TestMicha7EntryPlanner
    ├── TestMicha7StateManager      ← atomic + recovery tests
    ├── TestMicha7Visualizer
    ├── TestMicha7Scheduler
    └── TestMicha7CircuitBreaker
```

**Test naming convention:** `test_{component}_{scenario}_{expected_outcome}`

Example: `test_pivot_detector_armed_to_triggered_when_all_conditions_met`

---

## 9. .gitignore Additions

The following entries will be added to project root `.gitignore`:

```
# micha7 — private documentation
.claude-workspace/micha7/*.local.md
.claude-workspace/micha7/credentials*
.claude-workspace/micha7/business_logic*
.claude-workspace/micha7/config_values*
.claude-workspace/micha7/implementation_notes*

# micha7 — runtime state (machine-specific)
state/micha7/
outputs/micha7/

# micha7 — all source code (private per security policy)
# NOTE: Per SECURITY.md, all .py implementation is private.
# Uncomment when ready to enforce:
# micha7_*.py
# tests/test_micha7_*.py
```

⚠️ **Important:** Last block is **commented out** until you decide. Per your statement "כל הקוד פרטי" — uncomment when ready.

---

## 10. Update Protocol

When adding/removing/renaming files:
1. Update this document
2. Update `ARCHITECTURE.md` if responsibilities changed
3. Add entry to `CHANGELOG.md`
4. If structural decision: add to `DECISIONS.md`
