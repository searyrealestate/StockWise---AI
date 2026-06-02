# micha7 — Implementation Plan Phase 1

> Version 1.0.0 — 2026-06-02T22:52:37Z
> ⚠️ GITIGNORED — DO NOT COMMIT (except force-add for local tracking)

---

## 1. Prompt Sequence

Each row is one Claude Code implementation prompt. Dependencies must be green before starting.

| Prompt # | Module | What Gets Built | Dependencies | Min Acceptance |
|----------|--------|-----------------|--------------|----------------|
| **3.1** | features.py | BaseFeature ABC + FeatureDAG (topological sort, cycle detection, 2-level execution) | config, data | 35+ tests; DAG cycle test; Level 1 parallel, Level 2 sequential |
| **3.2** | features.py | F1 Candle (Engulfing, Hammer, Harami, Doji, Shooting Star) | 3.1, F6 context stub | 5+ tests per pattern; Hammer wick-ratio parameterized |
| **3.3** | features.py | F2 Trend (MA20, slope, configurable lookback) | 3.1 | 5+ tests; slope direction test; SMA not EMA confirmed |
| **3.4** | features.py | F3 Volume (volume_ratio, decay detection) | 3.1, 3.3 (F2 context) | 5+ tests; decay test; volume_ratio threshold parameterized |
| **3.5** | features.py | F4 Distance (ATR-normalized, raw_distance returned) | 3.1 | 5+ tests; raw_distance sign verified; D-08 return structure |
| **3.6** | features.py | F5 Price gaps (unfilled gap detection; distinct from calendar gap) | 3.1 | 5+ tests; D-17 gap naming; no-gap fixture + gap fixture |
| **3.7** | features.py | F6 S/R (swing pivot detection, level clustering, raw_levels returned) | 3.1 | 5+ tests; D-09 return structure; proximity test |
| **3.8** | features.py | F7 CCI(14) (Lambert formula, ±100 thresholds) | 3.1 | 5+ tests; ±100 zone; period=14 not 20 |
| **3.9** | features.py | ScoringEngine (bullish/bearish/empty aggregation, traffic light, D-06/D-07) | 3.2–3.8 | Score fraction + pct; traffic light mapping; bearish_count logged |
| **4.1** | state.py | StateManager + WriteAheadLog (atomic writes, schema versioning, recovery) | features | Atomic test; WAL replay test; schema migration test |
| **4.2** | state.py | PivotDetector (5-state machine; D-04, D-10 composite trigger) | 4.1, features | All 5 states tested; pivot composite condition test; invalidation test |
| **5.1** | trade.py | EntryPlanner (entry@close, stop below support D-11, 3 targets D-12) | features (F5, F6), state | 3-target test; stop 1% below support; R:R computed not filtered (D-13) |
| **5.2** | trade.py | RiskManager standalone (position size gates, no StockWise dep) | trade | Gates verified; config-driven limits; fail-loud on breach |
| **6.1** | analyzer.py | Micha7Analyzer facade (full pipeline orchestration) | all above | E2E test with mock provider; all components wired; signal emitted |
| **6.2** | backtest.py | BacktestRunner + BacktestReport (PF, WR, max DD, trade log) | analyzer | PF/WR computed correctly; trade log entries correct; determinism test |
| **6.3** | (new file) journal.py | Local-First TradeJournal (JSON + CSV output; D-20) | backtest | JSON parseable; CSV importable; recommendations logged |

**Total prompts:** 16
**Note:** Prompt numbering resumes from 3.x to match the existing 2.x (config, data) series.

---

## 2. Per-Prompt Template

Every Claude Code implementation prompt MUST include these sections in order:

```markdown
# Prompt N.N — <Module>: <What Gets Built>

## LANGUAGE RULE
ALL output English only.

## CONTEXT
<one paragraph: what's done, what this builds, why it matters>

## TIMESTAMP DISCIPLINE
Fixed timestamps — do not generate.
| Field | Value |
|-------|-------|
| CHANGELOG entry | YYYY-MM-DDTHH:MM:SSZ |
| Git commit (--date) | YYYY-MM-DDTHH:MM:SSZ |

## EXPLAIN-BEFORE-ACT RULE
🎯 GOAL / ⚙️ ACTION before each file.

## TDD ORDER (MANDATORY)
1. Write test file (RED)
2. Run pytest → confirm RED
3. Implement to GREEN
4. Syntax check + CHANGELOG + commit

## 🔍 PRE-FLIGHT
- git status --short (must be clean)
- git log --oneline -1

## 🎯 GOAL
<one sentence>

## ⚙️ ACTION STEPS
<numbered list>

## 🧪 TESTS TO WRITE (write BEFORE implementation)
<test function list with brief description>

## 📝 CHANGELOG ENTRY
<exact markdown for CHANGELOG.md>

## ✅ ACCEPTANCE CRITERIA
<numbered checklist — pytest count, behavioral assertions>

## 🚫 CONSTRAINTS
- No hardcoded values; all from config
- No StockWise imports
- No network calls in unit tests
- All tests offline
```

---

## 3. Critical Decisions Cross-Reference

| Decision | Applies to Prompt(s) |
|----------|---------------------|
| D-01: 0-100% + X/7 | 3.9 (ScoringEngine) |
| D-02: Traffic light thresholds | 3.9 |
| D-03: Score ≥ 6/7 → ARMED | 4.2 (PivotDetector) |
| D-04: 5-state machine | 4.2 |
| D-05: Long only | 3.9, 4.2, 5.1, 6.1 |
| D-06: Bearish = empty | 3.9 |
| D-07: Log bearish_count | 3.9 |
| D-08: F4 raw_distance | 3.5 |
| D-09: F6 raw_levels | 3.7 |
| D-10: Composite pivot | 4.2 |
| D-11: Stop = 1% below support | 5.1 |
| D-12: 3 structural targets | 5.1 |
| D-13: R:R logged, not filtered | 5.1, 5.2 |
| D-14: EOD only | 6.1, 6.2 |
| D-15: Trend lookback 5-22 | 3.3 |
| D-16: min_rows = 100 | 2.1 (data.py fix — B-05) |
| D-17: Gap naming | 3.6 (F5), data.py fix |
| D-18: auto_adjust = True | data.py (current) |
| D-19: ATR for F4 only | 3.5, 5.1 |
| D-20: Local journal | 6.3 |

---

## 4. Risk Register

| Risk ID | Description | Probability | Impact | Mitigation |
|---------|-------------|-------------|--------|------------|
| R-01 | F6 S/R algorithm doesn't match Micha's visual lines | High | High | Test against hand-annotated AAPL chart; parameterize N and cluster threshold; add to acceptance criteria |
| R-02 | F1 wick-ratio thresholds wrong | Medium | Medium | `hammer_wick_ratio` in config.json; backtest sweep 1.5x, 2x, 2.5x |
| R-03 | Stop too tight (1% may not work for volatile symbols) | Medium | Medium | ATR-based fallback as config option; flag in KNOWN_BUGS.md after backtest |
| R-04 | features.py exceeds 800 LOC split trigger | Medium | Low | Monitor LOC at each feature prompt; split if F6 or ScoringEngine push it over; IMP-004 |
| R-05 | compute_atr Python loop is slow for large backtests | Low | Medium | Benchmark in Prompt 6.2; vectorize if 90-day run > 5 seconds (B-01) |
| R-06 | PivotDetector composite condition too strict — no trades | Medium | High | Tune D-10 thresholds in backtest; log condition-by-condition breakdown |
| R-07 | CCI period 14 too sensitive vs standard 20 | Low | Low | Parameterize; compare 14 vs 20 in backtest report |

---

## 5. Definition of Done (Phase 1)

**All 16 prompts (3.1 through 6.3) completed.**

Technical gates:
- [ ] 100+ tests passing, 0 failures
- [ ] `python -m py_compile` on all .py files → no errors
- [ ] `python -m micha7 --version` works
- [ ] Bit-identical backtest reproducibility (run twice, diff = empty)

Quality gates:
- [ ] PF ≥ 1.5 on 13-symbol backtest (90-day)
- [ ] WR ≥ 55% on same backtest
- [ ] No hardcoded values anywhere (all in config.json or config.local.json)
- [ ] All D-XX decisions verified in code
- [ ] KNOWN_BUGS.md updated with any new issues found during implementation
- [ ] TradeJournal outputs valid JSON and CSV

Documentation gates:
- [ ] CHANGELOG.md has entry for each prompt
- [ ] business_logic.local.md updated if any Q1-Q4 open questions resolved
- [ ] decisions_registry.local.md updated if new decisions made

---

## 6. Pre-Implementation Mini-Fixes (Before Prompt 3.1)

These small fixes from KNOWN_BUGS.md should be resolved before features.py begins:

| Bug | Fix | Effort |
|-----|-----|--------|
| B-05: min_rows=30 too small | Raise to 100 in config.json (D-16) | 5 min |
| B-03: Gap naming collision | Rename DataAdapter.detect_gaps → detect_calendar_gaps | 15 min |
| B-11: NaT validation gap | Add NaT check to validate() | 10 min |

Suggest bundling into "Prompt 2.5 — Mini-fixes" before 3.1.
