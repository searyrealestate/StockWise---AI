# micha7_analyzer — Project Structure

> **Version:** 2.0.0
> **Last Modified:** 2026-06-01T08:50:00Z

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
    │   │   └── risk_manager.py     ← Standalone (no StockWise dependency)
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
| Modified StockWise core files | No StockWise file modifications |
| Used StockWise data + risk modules | Standalone DataProvider + RiskManager |
| Tests in `unit_tests.py` | Dedicated `tests/` directory with pytest |

---

## 2. Module Responsibilities

### `micha7/analyzer.py` — Micha7Analyzer (Facade)
**Contains:** `Micha7Analyzer` class — single entry point that orchestrates the full pipeline: data → features → scoring → pivot → entry → risk.

**Why:** All subpackages have well-defined interfaces; the facade composes them without owning their logic. Keeps orchestration separate from domain logic.

**Dependencies:** `micha7.data`, `micha7.features`, `micha7.state`, `micha7.trade`, `micha7.config_loader`, `micha7.logger`

---

### `micha7/config_loader.py`
**Contains:** `ConfigLoader` — loads `config.json`, optionally merges `config.local.json` overrides, validates schema.

**Why:** All modules receive a validated config object; no module does its own file I/O for config.

**Dependencies:** standard library only (`json`, `pathlib`)

---

### `micha7/logger.py`
**Contains:** Structured JSON logger setup used by all modules.

**Why:** Consistent log format; enables log parsing and alerting in later phases.

**Dependencies:** standard library only (`logging`)

---

### `micha7/data/` subpackage

| Module | Contains | Notes |
|--------|----------|-------|
| `base_provider.py` | `BaseDataProvider` (ABC) | Defines `fetch_ohlcv(symbol, start, end)` contract |
| `yfinance_provider.py` | `YFinanceProvider` | Phase 1 implementation; includes retry logic |
| `data_adapter.py` | `DataAdapter` | Validates + normalizes provider output; rejects bad data |
| `shared_metrics.py` | Pre-calculated ATR, returns, volume ratios | Shared across all features to avoid duplication |

**Why separate:** `BaseDataProvider` is the main seam for future data source replacement (ADR-015). Changing from yfinance to IBKR = swap one class, touch nothing else.

**Dependencies:** `yfinance`, `pandas`, `numpy`; `micha7.config_loader`, `micha7.logger`

---

### `micha7/features/` subpackage

| Module | Contains | Notes |
|--------|----------|-------|
| `base_feature.py` | `BaseFeature` ABC | Declares `compute(ohlcv, context) → FeatureResult` contract |
| `feature_dag.py` | `FeatureDAG` | Topological sort; validates no cycles; executes in order |
| `f1_candle.py` | Candle pattern scoring | |
| `f2_trend.py` | Monthly trend direction | |
| `f3_volume.py` | Volume momentum | |
| `f4_ma_distance.py` | Distance from MA20 | |
| `f5_gaps.py` | Gap detection above/below | |
| `f6_sr_levels.py` | Support/Resistance levels | |
| `f7_cci.py` | CCI(14) scoring | |
| `scoring.py` | `ScoringEngine` | Aggregates feature scores; maintains score history |

**Why separate:** Each feature is independently testable as a pure function. Adding a new feature = add a node, no refactoring required (ADR-003).

**Dependencies:** `pandas`, `numpy`; `micha7.data.shared_metrics`, `micha7.config_loader`

---

### `micha7/state/` subpackage

| Module | Contains | Notes |
|--------|----------|-------|
| `state_manager.py` | `StateManager` | Atomic writes, schema versioning, migration, startup recovery |
| `wal.py` | `WriteAheadLog` | Logs transitions; enables replay after crash |
| `pivot_detector.py` | `PivotDetector` | State machine: Idle → Armed → Triggered → Invalidated |

**Why separate:** Persistence infrastructure is generic and reusable for future analyzers. Strong separation from domain logic (ADR-005, ADR-006).

**Dependencies:** standard library (`json`, `pathlib`); `micha7.config_loader`, `micha7.logger`

---

### `micha7/trade/` subpackage

| Module | Contains | Notes |
|--------|----------|-------|
| `entry_planner.py` | `EntryPlanner` | Computes entry price, stop loss, 3 targets, R:R ratio |
| `risk_manager.py` | `RiskManager` (standalone) | Position sizing against config limits — no StockWise dependency |

**Why separate:** Trade logic is distinct from analysis; the standalone RiskManager can later be replaced by a StockWise adapter (Phase 6+) without touching the analysis pipeline.

**Dependencies:** `pandas`, `numpy`; `micha7.config_loader`

---

### `micha7/backtest/` subpackage

| Module | Contains | Notes |
|--------|----------|-------|
| `runner.py` | `BacktestRunner` | Loops over historical data bar-by-bar; calls `Micha7Analyzer` |
| `report.py` | `BacktestReport` | Assembles metrics: PF, WR, max drawdown, trade log |

**Why separate:** Backtest is an application layer over the analysis pipeline. Analysis modules have no knowledge of backtesting.

**Dependencies:** `pandas`, `numpy`; `micha7.analyzer`, `micha7.data`, `micha7.config_loader`

---

## 3. Class Hierarchy & Interactions

```
BacktestRunner  (application layer — micha7/backtest/runner.py)
    │
    └── drives: Micha7Analyzer  (facade — micha7/analyzer.py)
                    │
                    ├── uses: DataAdapter  (micha7/data/data_adapter.py)
                    │            └── uses: BaseDataProvider  (ABC)
                    │                         └── impl: YFinanceProvider
                    │
                    ├── uses: FeatureDAG  (micha7/features/feature_dag.py)
                    │            ├── executes: F1_Candle … F7_CCI
                    │            └── reads: shared_metrics (ATR, returns)
                    │
                    ├── uses: ScoringEngine  (micha7/features/scoring.py)
                    │
                    ├── uses: PivotDetector  (micha7/state/pivot_detector.py)
                    │            └── uses: StateManager  (micha7/state/state_manager.py)
                    │                         └── uses: WriteAheadLog  (micha7/state/wal.py)
                    │
                    ├── uses: EntryPlanner  (micha7/trade/entry_planner.py)
                    │
                    └── uses: RiskManager  (micha7/trade/risk_manager.py)
                                 — standalone, no StockWise dependency

Third-party libraries: yfinance (data only), pandas, numpy.
No StockWise modules anywhere in this diagram.
```

---

## 4. File Naming Conventions

| Pattern | Usage |
|---------|-------|
| `micha7/<subpackage>/<module>.py` | All package modules |
| `tests/unit/test_<module>.py` | Unit tests — one file per module |
| `tests/integration/test_<flow>.py` | Integration / E2E tests |
| `tests/fixtures/<name>.csv` | Deterministic test data |
| `config.json` | Public config schema (committed) |
| `config.local.json` | Private values (gitignored) |
| `*.local.md` | Private documentation (gitignored) |
| `_wal.log` | WAL file (underscore prefix = system file) |
| `_system/` | System-level state directory |

---

## 5. Import Graph (No Circular Dependencies)

```
          config_loader    logger
                ↑              ↑
                │  (all modules read from both)
     ┌──────────┼──────────────┼──────────────┐
     │          │              │              │
   data/    features/       state/         trade/
     ↑          ↑              ↑              ↑
     │          └──────────────┴──────────────┘
     │                         │
     └──────── analyzer ────────┘
                    ↑
                    │
             backtest/runner

Third-party (external): yfinance → data/ only; pandas, numpy → data/, features/, trade/, backtest/
```

**Verified:** No circular imports. No StockWise imports anywhere in this graph. All dependencies flow downward from `backtest/runner` to leaf modules.

---

## 6. Lines of Code Estimates

| Module | Estimated LOC | Justification |
|--------|---------------|---------------|
| `micha7/analyzer.py` | 150–250 | Thin facade; delegates to subpackages |
| `micha7/config_loader.py` | 60–100 | Load + validate config.json |
| `micha7/logger.py` | 40–60 | Logger setup |
| `micha7/data/base_provider.py` | 30–50 | ABC only |
| `micha7/data/yfinance_provider.py` | 100–150 | Fetch + retry logic |
| `micha7/data/data_adapter.py` | 100–150 | Validation + normalization |
| `micha7/data/shared_metrics.py` | 80–120 | ATR, returns, volume ratios |
| `micha7/features/base_feature.py` | 30–50 | ABC only |
| `micha7/features/feature_dag.py` | 80–120 | Topological sort + executor |
| `micha7/features/f1–f7.py` (×7) | 60–100 each | ~490–700 combined |
| `micha7/features/scoring.py` | 100–150 | Aggregation + score history |
| `micha7/state/state_manager.py` | 150–200 | Atomic writes + migration |
| `micha7/state/wal.py` | 80–120 | WAL read/write/replay |
| `micha7/state/pivot_detector.py` | 120–180 | State machine |
| `micha7/trade/entry_planner.py` | 100–150 | Targets + stop + R:R |
| `micha7/trade/risk_manager.py` | 80–120 | Standalone position sizing |
| `micha7/backtest/runner.py` | 150–250 | Backtest orchestration |
| `micha7/backtest/report.py` | 100–150 | Metrics assembly |
| **TOTAL** | **~1840–2870** | All standalone — no StockWise files modified |

---

## 7. Configuration Boundaries

All configurable parameters live in `config.json` at the standalone project root (`micha7/config.json`). No hardcoded values in source files.

**Public/Private split:**
- `config.json` — committed; contains keys with safe defaults (no sensitive thresholds)
- `config.local.json` — gitignored; overrides sensitive values for your machine

`config_loader.py` merges both at startup: local values override schema defaults.

| Category | Examples | File |
|----------|----------|------|
| Indicator periods | MA period, CCI period | `config.json` (public) |
| Score thresholds | min_score_for_long | `config.local.json` (**private**) |
| Risk parameters | max_position_size_pct, stop_buffer | `config.local.json` (**private**) |
| Operational | retry counts, timeouts, paths | `config.json` (public) |
| Mode flags | enable_viz, log_level | `config.json` (public) |
| Circuit breaker | loss thresholds, suspension durations | `config.local.json` (**private**) |

**Implementation:** Schema documented in `ARCHITECTURE.md`; actual values in `config_values.local.md` (this workspace, gitignored).

---

## 8. Test File Organization

```
micha7/
└── tests/
    ├── conftest.py                     ← MockDataProvider fixture, sample_ohlcv loader
    ├── fixtures/
    │   └── sample_ohlcv.csv            ← Deterministic 90-day OHLCV (committed)
    │
    ├── unit/
    │   ├── test_config_loader.py
    │   ├── test_data_adapter.py
    │   ├── test_yfinance_provider.py
    │   ├── test_shared_metrics.py
    │   ├── test_feature_dag.py
    │   ├── test_f1_candle.py           ← One file per feature
    │   ├── test_f2_trend.py
    │   ├── test_f3_volume.py
    │   ├── test_f4_ma_distance.py
    │   ├── test_f5_gaps.py
    │   ├── test_f6_sr_levels.py
    │   ├── test_f7_cci.py
    │   ├── test_scoring.py
    │   ├── test_state_manager.py       ← Atomic + recovery tests
    │   ├── test_wal.py
    │   ├── test_pivot_detector.py      ← State machine tests
    │   ├── test_entry_planner.py
    │   ├── test_risk_manager.py
    │   ├── test_backtest_runner.py
    │   └── test_backtest_report.py
    │
    └── integration/
        ├── test_full_pipeline.py       ← DataAdapter → Score → EntryPlan
        ├── test_backtest_e2e.py        ← Full backtest on fixtures
        └── test_state_recovery.py     ← Crash → restart → consistent state
```

**Test naming convention:** `test_{component}_{scenario}_{expected_outcome}`

Example: `test_pivot_detector_armed_to_triggered_when_all_conditions_met`

---

## 9. .gitignore

**Inside `micha7/` (standalone project root — `micha7/.gitignore`):**

```gitignore
# Runtime state and outputs (machine-specific)
state/
outputs/

# Private config values
config.local.json

# Python runtime
__pycache__/
*.py[cod]
*.pyo
*.egg-info/
dist/
build/
.venv/
.env

# Logs and temp files
*.log
*.tmp
*.tmp.*

# IDE
.idea/
.vscode/
```

**Inside StockWise root (workspace docs — append to root `.gitignore`):**

```gitignore
# micha7 — private documentation
.claude-workspace/micha7/*.local.md
.claude-workspace/micha7/credentials*
.claude-workspace/micha7/business_logic*
.claude-workspace/micha7/config_values*
.claude-workspace/micha7/implementation_notes*
```

Note: Source code privacy policy (`micha7/**/*.py` entries) is documented in `SECURITY.md` — configure when ready to enforce.

---

## 10. Update Protocol

When adding/removing/renaming files:
1. Update this document
2. Update `ARCHITECTURE.md` if responsibilities changed
3. Add entry to `CHANGELOG.md`
4. If structural decision: add to `DECISIONS.md`
