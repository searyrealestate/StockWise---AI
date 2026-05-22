# micha7_analyzer — Template Engine Specification

> **Version:** 1.0.0
> **Created:** 2026-05-21T05:55:00Z
> **Last Modified:** 2026-05-21T05:55:00Z
> **Status:** Forward-looking design for analyzer family

This document defines the Template Engine pattern: micha7_analyzer is the **first** of a family of analyzers. Future analyzers (micha8, micha9, etc.) will reuse infrastructure.

---

## 1. Vision

```
┌──────────────────────────────────────────────────────┐
│  Analyzer Family (Future)                            │
├──────────────────────────────────────────────────────┤
│                                                      │
│  ┌─────────────────────────────────────────────┐    │
│  │  Shared Base (Template Engine)              │    │
│  │  - BaseAnalyzer (abstract)                  │    │
│  │  - BaseFeatureExtractor (DAG framework)     │    │
│  │  - BaseScoringEngine                        │    │
│  │  - BasePivotDetector (state machine)        │    │
│  │  - StateManager (shared infrastructure)     │    │
│  │  - CircuitBreaker (shared infrastructure)   │    │
│  │  - Scheduler (shared infrastructure)        │    │
│  │  - ChartSpec (shared infrastructure)        │    │
│  └─────────────────────────────────────────────┘    │
│                       ↑                              │
│         ┌─────────────┼─────────────┐               │
│         │             │             │               │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│   │ micha7   │  │ micha8   │  │ michaN   │         │
│   │ (current)│  │ (future) │  │ (future) │         │
│   │          │  │          │  │          │         │
│   │ 7 params │  │ X params │  │ ...      │         │
│   │ checklist│  │ ...      │  │ ...      │         │
│   └──────────┘  └──────────┘  └──────────┘         │
│                                                      │
└──────────────────────────────────────────────────────┘
```

---

## 2. Current State (Phase 1)

**micha7_analyzer is built as a standalone module first.** This is intentional.

### Why Not Build the Base Now?

| Reason | Detail |
|--------|--------|
| **YAGNI** | Building abstract base before second use case is premature |
| **Discovery** | Real needs only emerge after second analyzer exists |
| **Speed** | Faster to ship Phase 1 without abstraction overhead |
| **Risk** | Wrong abstraction is harder to fix than no abstraction |

### What We Do Now

✅ **Build micha7 with CLEAN INTERFACES** that allow future extraction.
✅ **Document patterns** that will become base class methods.
✅ **Avoid micha7-specific coupling** in infrastructure components.

---

## 3. Extraction Roadmap

### Phase 1: micha7 Standalone
**Status:** Current
- micha7 built as standalone module
- Clean interfaces (no tight coupling)
- Infrastructure components designed for reuse

### Phase 2-3: Operational micha7
**Goal:** Validate the pattern works.
- Run micha7 in production
- Document pain points and patterns
- Identify true reusable components vs micha7-specific

### Phase 4+: Build Second Analyzer
**Goal:** First real test of reusability.
- Identify a second methodology to implement
- Build micha8_analyzer (or similar)
- **Now we have 2 use cases** — abstraction becomes possible

### Phase 5: Extract Base (Refactor)
**Goal:** Create base classes from real patterns.
- Extract common code to `base_analyzer/` package
- micha7 inherits from base
- micha8 inherits from base
- **Reuse is real, not theoretical**

---

## 4. Design Principles for Future-Proofing

### Principle 1: Composition Over Inheritance (For Now)

```python
# CURRENT (Phase 1) — composition
class Micha7Analyzer:
    def __init__(self):
        self.data_adapter = DataAdapter()
        self.feature_extractor = FeatureExtractor()
        self.scoring_engine = ScoringEngine()
        # ...

# FUTURE (Phase 5+) — inheritance via base class
class Micha7Analyzer(BaseAnalyzer):
    FEATURES = ["f1_candle", "f2_trend", ...]  # config
    
    def __init__(self):
        super().__init__()
        # micha7-specific initialization
```

### Principle 2: Interface Stability

**Public interfaces of components should remain stable** even when internals change.

Example:
```python
# This signature should be the same in Phase 1 and Phase 5:
def extract_features(ohlcv: pd.DataFrame) -> Dict[str, FeatureResult]:
    ...
```

When base class arrives, it can require this signature without breaking micha7.

### Principle 3: Generic Infrastructure

These components are **generic from day 1** (not micha7-specific):

| Component | How It Stays Generic |
|-----------|---------------------|
| StateManager | Accepts any JSON-serializable state |
| CircuitBreaker | Operates on metrics, not specific to analyzer |
| Scheduler | Accepts any callable (analyzer_func) |
| ChartSpec | Accepts any spec structure |
| Trading Calendar | Just provides dates, not analyzer-specific |

**Rule:** If a component name starts with `micha7_`, it's analyzer-specific.
**Rule:** If a component is in `tests/`, infrastructure, or future `base/`, it's generic.

---

## 5. Patterns That Will Be Extracted

These patterns in micha7 are candidates for future base class methods:

### 5.1 DAG-Based Feature Extraction
**Currently:** Custom DAG in `FeatureExtractor`
**Future:** `BaseFeatureExtractor` with DAG support; each analyzer declares its own DAG.

### 5.2 State Machine Pattern
**Currently:** PivotDetector states (WAITING/ARMED/TRIGGERED/etc.)
**Future:** `BaseStateMachine` with declarative transitions; each analyzer defines its own states.

### 5.3 Scoring Aggregation
**Currently:** Sum of feature scores with thresholds
**Future:** `BaseScoringEngine` with pluggable aggregation strategies (sum, weighted, ML).

### 5.4 Mode Isolation
**Currently:** Live/Paper/Backtest namespaces
**Future:** `ModeManager` provides isolation for any analyzer.

### 5.5 Atomic State Persistence
**Currently:** WAL + atomic writes in StateManager
**Future:** Same code, reused by all analyzers.

---

## 6. Anti-Patterns to Avoid

These would make future extraction painful — **don't do them**:

❌ **Hardcoded "micha7" strings everywhere**
```python
# BAD
log_path = "logs/micha7_analyzer.log"

# GOOD
log_path = f"logs/{self.analyzer_name}.log"
```

❌ **Direct access to micha7 config from infrastructure**
```python
# BAD (in StateManager)
namespace = config["MICHA7_CONFIG"]["namespace"]

# GOOD
namespace = self.namespace  # set by analyzer at init
```

❌ **Coupling infrastructure to feature names**
```python
# BAD (in ChartSpec)
if features.has("f1_candle"):
    add_candle_marker()

# GOOD
for feature_name, result in features.items():
    add_marker_per_spec(feature_name, result)
```

❌ **Hard-coded 7 in core flow**
```python
# BAD
confidence = score / 7

# GOOD
confidence = score / self.max_possible_score
```

---

## 7. Interface Contracts (Will Become ABCs)

These are the interfaces micha7 implements that **will become abstract base classes** in Phase 5:

### `BaseAnalyzer` (Future)
```
ATTRIBUTES (required by subclass):
  - analyzer_name: str           # "micha7"
  - version: str                 # "1.0.0"
  - features: List[str]          # feature IDs
  - max_possible_score: int      # 7 for micha7
  - score_threshold: int         # 5 for micha7

METHODS (must implement):
  - analyze(symbol, date) -> AnalysisResult
  - get_state(symbol) -> State
  - transition_state(symbol, from_state, to_state) -> None
```

### `BaseFeatureExtractor` (Future)
```
ATTRIBUTES (required by subclass):
  - FEATURE_GRAPH: Dict[str, FeatureSpec]
  - feature_implementations: Dict[str, Callable]

METHODS (provided by base):
  - extract(ohlcv) -> Dict[str, FeatureResult]
  - validate_dag()
  - topological_sort()
```

### `BaseScoringEngine` (Future)
```
ATTRIBUTES (required by subclass):
  - threshold: int
  - weights: Dict[str, float]  # optional

METHODS (must implement or use default):
  - aggregate(features) -> int
  - decide(score) -> Direction
```

### `BasePivotDetector` (Future)
```
ATTRIBUTES (required by subclass):
  - STATES: Set[str]
  - TRANSITIONS: Dict[Tuple[str, str], Callable]
  - INITIAL_STATE: str

METHODS (provided by base):
  - check_transition(current_state, context) -> Optional[str]
  - persist_state(symbol, new_state) -> None
```

---

## 8. Naming Conventions for Future Analyzers

When the next analyzer is built (post-Phase 1), follow these conventions:

| Pattern | Example |
|---------|---------|
| Module name | `{name}_analyzer.py` (e.g., `micha8_analyzer.py`) |
| Class name | `{Name}Analyzer` (e.g., `Micha8Analyzer`) |
| Config block | `{NAME}_CONFIG` (e.g., `MICHA8_CONFIG`) |
| State directory | `state/{name}/` (e.g., `state/micha8/`) |
| Output directory | `outputs/{name}/` (e.g., `outputs/micha8/`) |
| Tests | `tests/unit/test_{name}_*.py` |

**Goal:** When the next analyzer is added, `grep` for `micha7` should NOT find infrastructure code that should be generic.

---

## 9. Validation Checklist

Before declaring "Phase 1 complete," verify:

- [ ] No infrastructure module hardcodes "micha7"
- [ ] StateManager accepts `namespace` parameter
- [ ] CircuitBreaker accepts `metric_provider` parameter
- [ ] Scheduler accepts `analyzer_callable` parameter
- [ ] ChartSpec format is analyzer-agnostic
- [ ] All "magic numbers" (like 7) are in config or parameters

This ensures future extraction is feasible.

---

## 10. Migration Plan to Base Classes

When Phase 5 arrives:

### Step 1: Create base package
```
StockWise - AI/
├── base_analyzer/
│   ├── __init__.py
│   ├── base.py              ← BaseAnalyzer ABC
│   ├── features.py          ← BaseFeatureExtractor
│   ├── scoring.py           ← BaseScoringEngine
│   ├── pivot.py             ← BasePivotDetector
│   └── infrastructure.py    ← shared utilities
```

### Step 2: Refactor micha7 to inherit
```python
# Before
class Micha7Analyzer:
    ...

# After
from base_analyzer import BaseAnalyzer

class Micha7Analyzer(BaseAnalyzer):
    analyzer_name = "micha7"
    features = ["f1_candle", "f2_trend", ...]
    # micha7-specific overrides
```

### Step 3: Run regression tests
- All micha7 tests must still pass
- Bit-identical results for historical backtest
- No behavior change, only structure

### Step 4: Build second analyzer using base
```python
class Micha8Analyzer(BaseAnalyzer):
    analyzer_name = "micha8"
    features = ["g1_xxx", "g2_yyy", ...]
```

---

## 11. Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Wrong abstraction (extracted too early/wrong) | Wait for second analyzer to inform design |
| micha7 becomes coupled to micha7-specific patterns | Code review against this document |
| Future analyzers diverge architecturally | This document serves as guidance |
| Base classes break micha7 during extraction | Comprehensive tests + bit-identical regression |

---

## 12. Update Protocol

This document is updated when:
- New analyzer is planned (add to family roadmap)
- Patterns emerge during micha7 development that should be future-base
- Base extraction happens (update to reflect reality)
- Naming conventions evolve

**Always log updates in CHANGELOG.md with `[ARCH]` tag.**
