# micha7_analyzer — Project Structure

> **Version:** 2.1.0
> **Last Modified:** 2026-06-01T09:40:00Z

---

## 1. File Layout — Minimal Standalone Project

**Design principle:** one source file per pipeline layer. 9 source files total
(was 26 in the previous draft). Files are flat under `micha7/micha7/` — no
subpackage directories. Rationale: minimize file count per project policy and
ADR-010 (avoid artificial file splits). Split trigger: any file exceeding
800 LOC is split per IMP-004 (IMPROVEMENT_ROADMAP.md).

```
StockWise - AI/
└── micha7/                         ← Standalone project root
    ├── pyproject.toml              ← Dependencies + pytest config
    ├── config.json                 ← All parameters (no hardcoded values)
    ├── README.md
    ├── .gitignore
    │
    ├── micha7/                     ← Python package (flat)
    │   ├── __init__.py             ← __version__
    │   ├── __main__.py             ← CLI (--version)
    │   ├── config.py               ← ConfigLoader + Logger
    │   ├── data.py                 ← DataProvider(s) + DataAdapter + metrics
    │   ├── features.py             ← DAG + F1–F7 + ScoringEngine
    │   ├── state.py                ← StateManager + WAL + PivotDetector
    │   ├── trade.py                ← EntryPlanner + RiskManager
    │   ├── analyzer.py             ← Micha7Analyzer (facade)
    │   └── backtest.py             ← BacktestRunner + BacktestReport
    │
    ├── tests/                      ← One test module per source file
    │   ├── __init__.py
    │   ├── conftest.py             ← Shared fixtures (mock provider, sample OHLCV)
    │   ├── test_smoke.py
    │   ├── test_config.py
    │   ├── test_data.py
    │   ├── test_features.py
    │   ├── test_state.py
    │   ├── test_trade.py
    │   └── test_integration.py     ← Full pipeline + backtest E2E
    │
    ├── state/                      ← Runtime (gitignored)
    │   ├── live/  paper/  backtest/  _system/
    │
    └── outputs/                    ← Runtime (gitignored)
        ├── charts/  reports/
```

### File count: 9 source + ~8 test modules. Previous draft: 26 source files.

---

## 2. Module Responsibilities

### `micha7/__init__.py`
**Contains:** `__version__` string.
**Why:** Standard Python package marker. Single source of truth for version.
**Dependencies:** none

---

### `micha7/__main__.py` — CLI
**Contains:** `main()` — `--version` argument; exits cleanly.
**Why:** Enables `python -m micha7` invocation without an installer.
**Dependencies:** `micha7` (self); standard library (`argparse`, `sys`)

---

### `micha7/config.py` — ConfigLoader + Logger
**Contains:** `ConfigLoader` (loads `config.json`, merges `config.local.json` overrides, validates schema); structured JSON logger used by all other modules.
**Why:** All modules receive a single validated config object; consistent log format enables log parsing and alerting in later phases.
**Dependencies:** standard library only (`json`, `pathlib`, `logging`)

---

### `micha7/data.py` — Data Layer
**Contains:** `BaseDataProvider` (ABC — `fetch_ohlcv(symbol, start, end)` contract), `YFinanceProvider` (Phase 1 implementation with retry logic), `DataAdapter` (validates + normalizes provider output), shared metrics (ATR, returns, volume ratios pre-calculated once for all features).
**Why:** `BaseDataProvider` is the main seam for future data source replacement (ADR-015). Changing from yfinance to IBKR = swap one class, touch nothing else. All data concerns cohesive in one file.
**Dependencies:** `yfinance`, `pandas`, `numpy`; `micha7.config`

---

### `micha7/features.py` — Feature Pipeline
**Contains:** `BaseFeature` ABC, `FeatureDAG` (topological sort + executor, ADR-003), F1_Candle through F7_CCI (7 concrete features — pure functions), `ScoringEngine` (aggregates scores, maintains history).
**Why:** All feature logic cohesive in one file; DAG enforces ordering and validates no cycles at startup. **Split trigger: 800 LOC** (IMP-004 — split into `features/` subpackage when reached).
**Dependencies:** `pandas`, `numpy`; `micha7.data`, `micha7.config`

---

### `micha7/state.py` — Persistence Layer
**Contains:** `StateManager` (atomic writes via rename pattern, schema versioning, migration, startup recovery), `WriteAheadLog` (transition log; enables replay after crash), `PivotDetector` (state machine: Idle → Armed → Triggered → Invalidated).
**Why:** Persistence infrastructure is generic and reusable for future analyzers. Kept separate from domain logic (ADR-005, ADR-006).
**Dependencies:** standard library (`json`, `pathlib`); `micha7.config`

---

### `micha7/trade.py` — Trade Layer
**Contains:** `EntryPlanner` (computes entry price, stop loss, 3 targets, R:R ratio), `RiskManager` (standalone position sizing validated against config limits — no StockWise dependency).
**Why:** Trade logic is distinct from analysis. Standalone `RiskManager` can later be replaced by a StockWise adapter (Phase 6+) without touching the analysis pipeline.
**Dependencies:** `pandas`, `numpy`; `micha7.config`

---

### `micha7/analyzer.py` — Micha7Analyzer (Facade)
**Contains:** `Micha7Analyzer` — single entry point orchestrating the full pipeline: data → features → scoring → pivot detection → entry planning → risk validation.
**Why:** Facade pattern (ADR-010); composes all layers without owning their logic. Thin by design (~150 LOC).
**Dependencies:** `micha7.data`, `micha7.features`, `micha7.state`, `micha7.trade`, `micha7.config`

---

### `micha7/backtest.py` — Backtest Engine
**Contains:** `BacktestRunner` (loops over historical data bar-by-bar, calls `Micha7Analyzer`), `BacktestReport` (assembles PF, WR, max drawdown, trade log).
**Why:** Application layer over the analysis pipeline. Analysis modules have no knowledge of backtesting — clean separation.
**Dependencies:** `pandas`, `numpy`; `micha7.analyzer`, `micha7.data`, `micha7.config`

---

## 3. Class Hierarchy & Interactions

```
BacktestRunner  (backtest.py)
    │
    └── drives: Micha7Analyzer  (analyzer.py)
                    │
                    ├── uses: DataAdapter  (data.py)
                    │            └── uses: BaseDataProvider (ABC)
                    │                         └── impl: YFinanceProvider (data.py)
                    │
                    ├── uses: FeatureDAG  (features.py)
                    │            ├── executes: F1_Candle … F7_CCI (features.py)
                    │            └── reads: shared_metrics (data.py)
                    │
                    ├── uses: ScoringEngine  (features.py)
                    │
                    ├── uses: PivotDetector  (state.py)
                    │            └── uses: StateManager (state.py)
                    │                         └── uses: WriteAheadLog (state.py)
                    │
                    ├── uses: EntryPlanner  (trade.py)
                    │
                    └── uses: RiskManager  (trade.py)
                                 — standalone, no StockWise dependency

Third-party: yfinance (data.py only), pandas, numpy.
No StockWise modules anywhere in this diagram.
```

---

## 4. File Naming Conventions

| Pattern | Usage |
|---------|-------|
| `micha7/<module>.py` | All package source files (flat — no subpackages) |
| `tests/test_<module>.py` | One test file per source module |
| `tests/test_integration.py` | Full pipeline + backtest E2E tests |
| `config.json` | Public config schema (committed) |
| `config.local.json` | Private values (gitignored) |
| `*.local.md` | Private documentation (gitignored) |
| `_wal.log` | WAL file (underscore prefix = system file) |
| `_system/` | System-level state directory |

---

## 5. Import Graph (No Circular Dependencies)

```
              config.py
                  ↑
                  │ (all modules import config)
     ┌────────────┼────────────┬────────────┐
     │            │            │            │
  data.py    features.py   state.py     trade.py
     ↑            ↑
     │            │ (features imports data for shared_metrics)
     └────────────┘
                  ↑
             analyzer.py
                  ↑
             backtest.py

Third-party: yfinance → data.py only; pandas, numpy → data.py, features.py, trade.py, backtest.py
```

**Verified:** No circular imports. No StockWise imports anywhere in this graph. All dependencies flow downward from `backtest.py` to `config.py`.

---

## 6. Lines of Code Estimates

| File | Estimated LOC | Justification |
|------|---------------|---------------|
| `micha7/__init__.py` | ~10 | Version only |
| `micha7/__main__.py` | ~30 | CLI entry point |
| `micha7/config.py` | ~150 | ConfigLoader + Logger |
| `micha7/data.py` | ~300 | Provider(s) + DataAdapter + shared metrics |
| `micha7/features.py` | ~650 | DAG + 7 features + ScoringEngine; **split trigger: 800 LOC (IMP-004)** |
| `micha7/state.py` | ~350 | StateManager + WAL + PivotDetector |
| `micha7/trade.py` | ~200 | EntryPlanner + RiskManager |
| `micha7/analyzer.py` | ~150 | Thin facade |
| `micha7/backtest.py` | ~250 | BacktestRunner + BacktestReport |
| **TOTAL** | **~2090** | All standalone — no StockWise files modified |

---

## 7. Configuration Boundaries

All configurable parameters live in `config.json` at the standalone project root (`micha7/config.json`). No hardcoded values in source files.

**Public/Private split:**
- `config.json` — committed; contains keys with safe defaults (no sensitive thresholds)
- `config.local.json` — gitignored; overrides sensitive values for your machine

`config.py` (ConfigLoader) merges both at startup: local values override schema defaults.

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
    ├── __init__.py
    ├── conftest.py             ← MockDataProvider fixture, deterministic OHLCV data
    ├── test_smoke.py           ← Package imports, version format (green baseline)
    ├── test_config.py          ← ConfigLoader, Logger
    ├── test_data.py            ← DataAdapter, YFinanceProvider, shared metrics
    ├── test_features.py        ← FeatureDAG, F1–F7, ScoringEngine
    ├── test_state.py           ← StateManager, WAL, PivotDetector
    ├── test_trade.py           ← EntryPlanner, RiskManager
    └── test_integration.py    ← Full pipeline + backtest E2E + state recovery
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
