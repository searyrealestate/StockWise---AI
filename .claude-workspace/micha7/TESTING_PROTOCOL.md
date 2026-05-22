# micha7_analyzer — Testing Protocol (Agile)

> **Version:** 1.0.0
> **Created:** 2026-05-21T05:55:00Z
> **Last Modified:** 2026-05-21T05:55:00Z
> **Status:** Mandatory for all code changes

This document defines the testing methodology for every code change in micha7_analyzer.

**Core Principle:** No code is "done" without passing tests. No commit without verification.

---

## 1. Agile Testing Philosophy

### The Iron Rule
```
┌─────────────────────────────────────────────┐
│  No commit without:                         │
│   ✅ Tests written                          │
│   ✅ Tests passing                          │
│   ✅ Master validator passing               │
│   ✅ Manual smoke test (where applicable)   │
└─────────────────────────────────────────────┘
```

### Agile Cycle Per Component

```
┌─────────────────────────────────────────────────────────┐
│  AGILE CYCLE — Per Component Implementation             │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. SPEC      — Define interface + behavior            │
│       ↓                                                 │
│  2. TEST      — Write failing tests (TDD)              │
│       ↓                                                 │
│  3. CODE      — Minimum code to pass tests             │
│       ↓                                                 │
│  4. VERIFY    — Run all tests (unit + integration)     │
│       ↓                                                 │
│  5. REFACTOR  — Clean code, tests still pass           │
│       ↓                                                 │
│  6. REGRESS   — Run master_validator (no breakage)     │
│       ↓                                                 │
│  7. COMMIT    — Git commit with descriptive message    │
│       ↓                                                 │
│  8. LOG       — Update CHANGELOG.md                    │
│                                                         │
│  Repeat for next component.                            │
└─────────────────────────────────────────────────────────┘
```

**No skipping steps.** If a step fails, do not proceed to next.

---

## 2. Test Pyramid

```
              /\
             /  \   E2E Tests (5%)
            /    \  - Full pipeline scenarios
           /------\ - Slowest, fewest
          /        \
         / Integration\ Tests (25%)
        /  Tests       \ - Component interactions
       /----------------\ - State persistence
      /                  \ - Crash recovery
     /                    \
    /     Unit Tests       \ (70%)
   /        - Pure functions \ - Fast, many
  /         - Edge cases      \
 /__________ - All branches ____\
```

### Distribution Target

| Layer | % of Total | Speed | Purpose |
|-------|-----------|-------|---------|
| Unit | 70% | < 100ms each | Logic correctness |
| Integration | 25% | < 5s each | Component cooperation |
| E2E | 5% | < 60s each | Full pipeline validation |

---

## 3. Mandatory Tests Per Component

### For EVERY new class/function:

| Test Type | Count | Required |
|-----------|-------|----------|
| Happy path | ≥1 | ✅ Mandatory |
| Edge case | ≥2 | ✅ Mandatory |
| Error case | ≥1 | ✅ Mandatory |
| Boundary values | ≥1 | ✅ Mandatory |
| Idempotency check | If applicable | ✅ Mandatory |

### For EVERY state transition:

| Test Type | Required |
|-----------|----------|
| Valid transition | ✅ |
| Invalid transition rejected | ✅ |
| State persisted correctly | ✅ |
| Recovery from crash mid-transition | ✅ |

### For EVERY feature in FeatureExtractor (f1-f7):

| Test Type | Required |
|-----------|----------|
| Computes correctly on known input | ✅ |
| Handles NaN/missing data | ✅ |
| Handles edge values (0, negative, infinity) | ✅ |
| Pure function (no side effects) | ✅ |
| Deterministic (same input → same output) | ✅ |
| Respects DAG dependencies | ✅ |

---

## 4. Test Files Organization

```
tests/
├── unit/
│   ├── test_micha7_data_adapter.py
│   ├── test_micha7_feature_extractor.py     ← per-feature tests
│   ├── test_micha7_scoring_engine.py
│   ├── test_micha7_pivot_detector.py        ← state machine
│   ├── test_micha7_entry_planner.py
│   ├── test_micha7_risk_manager.py
│   ├── test_micha7_state_manager.py         ← atomic I/O
│   ├── test_micha7_visualizer.py
│   └── test_micha7_scheduler.py
│
├── integration/
│   ├── test_micha7_full_pipeline.py
│   ├── test_micha7_state_persistence.py
│   ├── test_micha7_crash_recovery.py
│   ├── test_micha7_circuit_breaker.py
│   └── test_micha7_namespace_isolation.py
│
└── e2e/
    ├── test_micha7_backtest_e2e.py
    ├── test_micha7_determinism.py            ← same run twice = same result
    └── test_micha7_regression.py             ← StockWise master_validator
```

**Naming Convention:**
```python
def test_{component}_{scenario}_{expected_outcome}():
    # Example:
    # test_pivot_detector_armed_to_triggered_when_all_conditions_met
    # test_state_manager_recovers_from_orphan_temp_file
    # test_feature_extractor_handles_nan_in_volume
```

---

## 5. Required Test Scenarios (Phase 1)

### DataAdapter
- [ ] Returns valid OHLCV for known symbol
- [ ] Validates completeness (no missing days)
- [ ] Handles symbol with <60 days history (rejects)
- [ ] Returns adjusted data only
- [ ] Pre-calculates ATR consistently
- [ ] Cache respects TTL
- [ ] Timezone is UTC throughout

### FeatureExtractor (per feature: 7 sets)
- [ ] Computes correctly on hand-verified example
- [ ] Returns NEUTRAL on insufficient data
- [ ] Handles NaN inputs without crash
- [ ] DAG order respected (f1, f3 wait for deps)
- [ ] No lookahead bias (verified via slicing test)
- [ ] Pure function (no global state mutation)

### ScoringEngine
- [ ] Sums features correctly
- [ ] Returns LONG when score ≥ threshold
- [ ] Returns NEUTRAL when score < threshold
- [ ] Confidence in [0.0, 1.0]
- [ ] Score history persisted

### PivotDetector (state machine)
- [ ] WAITING → ARMED on score ≥ 5
- [ ] ARMED → TRIGGERED when all 4 pivot conditions met
- [ ] ARMED → WAITING on timeout
- [ ] TRIGGERED → IN_POSITION on entry
- [ ] IN_POSITION → TARGET_HIT on target reached
- [ ] IN_POSITION → STOP_HIT on stop hit
- [ ] IN_POSITION → THESIS_BROKEN on score drop
- [ ] State persisted atomically
- [ ] Recovery from crash mid-transition

### EntryPlanner
- [ ] Computes 3 targets in ascending order
- [ ] Stop below entry by config buffer
- [ ] Rejects trade if R:R < min_rr_ratio
- [ ] Fallback to ATR multiples if <3 resistances
- [ ] Min stop distance enforced

### RiskManager
- [ ] Validates against portfolio_risk
- [ ] Rejects if conflict with existing template position
- [ ] Computes position size correctly
- [ ] Respects max_position_size_pct

### StateManager
- [ ] Atomic write (verified by killing process mid-write)
- [ ] WAL entry created before state write
- [ ] Recovery cleans orphan .tmp files
- [ ] Schema migration v1.0 → v1.1 works
- [ ] Backup created before migration
- [ ] Namespace separation enforced

### CircuitBreaker
- [ ] WARNING after 2 consecutive losses
- [ ] SUSPENDED after 3 consecutive losses
- [ ] DISABLED after 5 losses in week
- [ ] EMERGENCY on drawdown breach
- [ ] Manual override works
- [ ] Auto-resume after suspension period

---

## 6. Test Execution Order

When running tests, execute in this order:

```bash
# 1. Fast unit tests first (catch obvious bugs)
pytest tests/unit/ -v --tb=short

# 2. Integration tests (catch interaction bugs)
pytest tests/integration/ -v --tb=short

# 3. E2E tests (catch system-level bugs)
pytest tests/e2e/ -v --tb=short

# 4. StockWise regression (catch breakage)
python master_validator.py

# 5. Manual smoke test if any code changed
python -m micha7_analyzer --smoke-test
```

**Rule:** If step N fails, do not proceed to step N+1.

---

## 7. Pre-Commit Checklist

Before EVERY `git commit`:

```
☐ All unit tests pass
☐ All integration tests pass
☐ All E2E tests pass (if changes affect pipeline)
☐ master_validator.py passes (no regression)
☐ No new lint warnings
☐ CHANGELOG.md updated
☐ No .local.md files staged
☐ No source .py files staged (per security policy)
☐ Commit message follows convention
```

### Commit Message Convention

```
[Component] Action description (max 72 chars)

Why: Brief rationale (optional, but encouraged)
Tests: tests/unit/test_X.py, tests/integration/test_Y.py
Validation: master_validator passed
Refs: ADR-NNN if relevant

Example:
[FeatureExtractor] Add candle pattern detection (f1)

Why: Required for Phase 1 score calculation
Tests: tests/unit/test_micha7_feature_extractor.py::test_f1_*
Validation: master_validator passed, 12 new unit tests added
Refs: ADR-003 (DAG ordering)
```

---

## 8. Test Data Strategy

### Synthetic Data (Default)
Use programmatically-generated OHLCV for predictable tests:
```python
def make_test_ohlcv(pattern="bullish_engulfing", days=90):
    """Generate synthetic OHLCV with known properties."""
    # Returns DataFrame with verified features
```

### Hand-Annotated Examples
For S/R detection validation, maintain a set of hand-annotated charts:
```
tests/fixtures/
├── sr_examples/
│   ├── AAPL_2024_strong_support.json    ← human-tagged
│   ├── TSLA_2024_breakdown.json
│   └── ...
```

### Real Data Snapshots
For regression: frozen snapshots of real market data.
```
tests/fixtures/snapshots/
├── 2026-01-15_AAPL_90d.csv
└── ...
```

**Rule:** Never modify snapshot files. They are the regression baseline.

---

## 9. Coverage Targets

| Metric | Target | Tool |
|--------|--------|------|
| Line coverage | ≥ 85% | pytest-cov |
| Branch coverage | ≥ 80% | pytest-cov |
| Critical paths | 100% | manual review |

### Critical Paths (MUST be 100%)
- State transitions
- Atomic write/recovery
- Schema migrations
- Circuit breaker triggers
- Risk validation

---

## 10. Continuous Validation

### Daily (during development)
- Run full unit suite
- Run master_validator
- Review CHANGELOG entries

### Weekly
- Run full E2E suite
- Review coverage report
- Update test fixtures if needed

### Per Phase
- Full validation against PHASES.md exit criteria
- Performance benchmarks
- Manual UX review (for Phase 3+)

---

## 11. Test Failure Protocol

When a test fails:

### Immediate
1. **Stop** — Don't commit, don't proceed
2. **Read** the failure carefully
3. **Reproduce** locally to confirm

### Diagnose
1. Is it a test bug or code bug?
2. Was it passing before? (check git log)
3. Is it environment-specific?

### Fix
1. Fix the underlying issue (not the test)
2. Add a regression test if it's a new bug
3. Re-run full test suite

### Document
1. Update CHANGELOG.md with `[FIX]` entry
2. If it reveals a design issue, add to DECISIONS.md

**Never:**
- Comment out failing tests
- Skip tests with `@pytest.skip` without justification
- Modify tests to match buggy behavior

---

## 12. Performance Benchmarks

Track these metrics over time:

| Operation | Target | Alert Threshold |
|-----------|--------|-----------------|
| Single symbol analysis | < 500ms | > 2s |
| Backtest 1 symbol × 90 days | < 5s | > 30s |
| Backtest 13 symbols × 365 days | < 60s | > 5min |
| Full unit test suite | < 30s | > 2min |
| Full integration suite | < 5min | > 15min |

Benchmarks tracked in `tests/benchmarks/results.json` (timestamped).

---

## 13. Mock & Fixture Strategy

### What to Mock
- External APIs (Alpaca, IBKR, Telegram)
- File system (in integration tests, use tmp dirs)
- Time (`datetime.now()` mocked for determinism)
- Random (no random in core, but for any test that uses it)

### What NOT to Mock
- Pure functions in FeatureExtractor (use real inputs)
- Pandas operations (use real DataFrames)
- Math/calculations (verify against hand-calculated)

### Fixture Files
- Real OHLCV snapshots: `tests/fixtures/snapshots/`
- Synthetic patterns: `tests/fixtures/patterns/`
- Expected outputs: `tests/fixtures/expected/`

---

## 14. Special Tests for Phase 1

### Determinism Test (MANDATORY)
```python
def test_backtest_determinism():
    """Same backtest run twice must produce bit-identical results."""
    result1 = run_backtest(symbols=["AAPL"], days=90, seed=42)
    result2 = run_backtest(symbols=["AAPL"], days=90, seed=42)
    assert result1 == result2  # exact equality
```

### Lookahead Bias Test (MANDATORY)
```python
def test_no_lookahead_bias():
    """Features computed at day T must not change when day T+1 data is added."""
    df_short = data[:100]
    df_long = data[:110]
    
    features_short = extract_features(df_short, target_day=99)
    features_long = extract_features(df_long, target_day=99)
    
    assert features_short == features_long
```

### Crash Recovery Test (MANDATORY)
```python
def test_recovery_from_crash_during_transition():
    """System recovers correctly if crashed mid-state-transition."""
    # Simulate crash by creating .tmp file
    create_orphan_tmp_file("AAPL", state="ARMED")
    
    # Startup recovery should clean it
    recover()
    
    # State should be the last known clean state
    assert load_state("AAPL")["state"] == "WAITING"  # or whatever was before
```

---

## 15. Definition of "Done"

A component is **DONE** when:

```
✅ Spec written (interface + behavior)
✅ Tests written (unit + integration)
✅ Tests passing (all green)
✅ Code committed
✅ CHANGELOG.md updated
✅ Documentation updated (if interface changed)
✅ master_validator passing (no regression)
✅ Coverage targets met
✅ Code reviewed (against ARCHITECTURE.md)
✅ No known bugs (TODOs explicit and tracked)
```

**Anything less = NOT done. Do not commit.**

---

## 16. Tools & Commands Reference

```bash
# Run specific test file
pytest tests/unit/test_micha7_feature_extractor.py -v

# Run specific test
pytest tests/unit/test_micha7_feature_extractor.py::test_f1_bullish_engulfing -v

# Run with coverage
pytest --cov=micha7_analyzer --cov-report=html

# Run only failed tests
pytest --lf

# Run with timing
pytest --durations=10

# Stop at first failure
pytest -x

# Verbose output
pytest -vv

# Master validator (StockWise regression)
python master_validator.py

# Smoke test
python -m micha7_analyzer --smoke-test --symbol=AAPL
```

---

## 17. Update Protocol

This document is updated when:
- New testing requirements emerge
- Test failures reveal gaps in protocol
- Phase transitions require new test categories
- After every retrospective

**Always log updates in CHANGELOG.md with `[TEST]` tag.**
