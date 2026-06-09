# micha7 — Decisions Registry (Flat Lookup)

> Version 1.0.0 — 2026-06-02T22:52:37Z
> Source: chat session 2026-06-03; verified against transcript
> ⚠️ GITIGNORED — DO NOT COMMIT (except force-add for local tracking)

D-XX numbers are used throughout business_logic.local.md, IMPLEMENTATION_PLAN.local.md,
and KNOWN_BUGS.md for cross-referencing decisions without repeating rationale.
Mark superseded decisions with ~~strikethrough~~ and reference the replacement.

---

| ID | Domain | Decision | Source | Applies To | Active |
|----|--------|----------|--------|-----------|--------|
| D-01 | Scoring | Score represented as two parallel formats: continuous 0–100% AND X/7 integer fraction (e.g., "5/7" and "71%"). Both stored and logged. | User decision 2026-06-03 | ScoringEngine, BacktestReport | ✅ |
| D-02 | Threshold | Traffic light thresholds: 🔴 0–3/7 (0–43%) = WAITING; 🟡 4–5/7 (57–71%) = WAITING + log breakdown; 🟢 6–7/7 (86–100%) = ARMED trigger | User decision 2026-06-03 | ScoringEngine, PivotDetector | ✅ |
| D-03 | Entry trigger | Score ≥ 6/7 is the ARMED trigger. Exactly derived from D-02 green zone. | Derived from D-02 | PivotDetector | ✅ |
| D-04 | State machine | 5 states: WAITING → ARMED → TRIGGERED → IN_POSITION → EXITED. ARMED can invalidate back to WAITING. | User decision 2026-06-03 | state.py PivotDetector | ✅ |
| D-05 | Direction | Phase 1 is LONG ONLY. No short detection, no short positions. | User decision 2026-06-03 | All components | ✅ |
| D-06 | Scoring model | Phase 1: BEARISH feature output is treated as EMPTY for scoring purposes. Only BULLISH count drives the score and traffic light. | Derived from D-05 | ScoringEngine | ✅ |
| D-07 | Logging | bearish_count is logged separately on every scoring run for diagnostics and future short-mode development. Does not affect score. | Architect decision 2026-06-03 | ScoringEngine, logs | ✅ |
| D-08 | F4 return structure | F4 (Distance from MA20) returns both a score (BULLISH/EMPTY) AND the raw signed distance value (raw_distance in ATR units). Consumers can use raw_distance for further logic. | User decision 2026-06-03 | features.py F4, EntryPlanner | ✅ |
| D-09 | F6 return structure | F6 (S/R levels) returns both a score (BULLISH/BEARISH/EMPTY) AND the raw_levels list [{price, type, strength}]. Consumers use raw_levels for stop/target computation. | User decision 2026-06-03 | features.py F6, EntryPlanner | ✅ |
| D-10 | Pivot trigger | Entry pivot requires ALL THREE conditions simultaneously: (a) price far from MA20 ≥ 2.0 ATR (configurable), (b) price touches nearest support level, (c) reversal candle (F1 BULLISH) OR CCI cross from below -100. | User decision 2026-06-03 | state.py PivotDetector | ✅ |
| D-11 | Stop loss | Stop = nearest_support_price × (1 - stop_buffer_pct). Default stop_buffer_pct = 0.01 (1% below support). Source: transcript line ~77. NOT ATR-based. | Transcript-verified | trade.py EntryPlanner | ✅ |
| D-12 | 3 targets | Target 1 = nearest resistance above entry (from F6). Target 2 = bottom of nearest unfilled price gap above entry (from F5). Target 3 = next resistance above Target 1 (second resistance in F6). Source: transcript line ~65. | Transcript-verified | trade.py EntryPlanner | ✅ |
| D-13 | R:R filter | R:R is computed per trade and logged in BacktestReport, but is NOT used as an entry filter. Trade is taken regardless of R:R value. The old GLOSSARY claim "R:R ≥ 2.0 minimum" is superseded by ADR-017. | User decision 2026-06-03, ADR-017 | trade.py EntryPlanner, BacktestReport | ✅ |
| D-14 | Timeframe | EOD (End of Day) analysis only. Daily bars. No intraday analysis in any phase. Source: transcript line ~27. | Transcript-verified | analyzer.py, scheduler (Phase 2) | ✅ |
| D-15 | Trend lookback | MA20 lookback for trend direction is configurable 5–22 days, default 20 days. Slope detection uses last 5 bars. Source: transcript line ~29. | Transcript-verified | features.py F2 | ✅ |
| D-16 | Min rows | data.min_rows raised from 30 to 100. S/R detection (F6) needs ~100 bars of price history to identify meaningful pivot levels. Less data produces noisy or empty results. | Audit (B-05 resolution) | config.json, data.py | ✅ |
| D-17 | Gap naming | "Gap (Calendar)" = missing trading day in OHLCV — detected by data.py, non-feature. "Gap (Price)" = close-to-open discontinuity — F5 feature. Two completely different concepts. data.py method renamed detect_calendar_gaps(). | Audit (B-03 resolution) | data.py, features.py F5 | ✅ |
| D-18 | auto_adjust | YFinanceProvider uses auto_adjust=True. This adjusts prices for splits and dividends, matching TradingView's default display. Accepted as intentional behavior (B-02). | Audit decision | data.py YFinanceProvider | ✅ |
| D-19 | ATR usage scope | ATR (Wilder, period 14) is used ONLY in F4 for normalizing distance from MA20. ATR is NOT used for stop sizing. Stop is structural (below support, D-11). | Audit decision | features.py F4, trade.py EntryPlanner | ✅ |
| D-20 | Trade journal | Phase 1: Local-First trade journal — JSON + CSV files in outputs/. Google Sheets sync deferred to Phase 4+ via adapter pattern (ADR-018). | User decision 2026-06-03 | journal.py (Prompt 6.3) | ✅ |
| D-21 | CoT workflow gate | Chain-of-Thought (CoT) gates steps 2–7 of the 8-step development workflow. Each step's reasoning must align with Phase 1 acceptance criteria (PF ≥ 1.5, WR ≥ 55%). Steps 1 and 8 (problem statement + commit) are human gates. | Architect ratification 2026-06-06 | All implementation prompts | ✅ |
| D-22 | F6 before F1 | F6 (Support/Resistance) must be implemented before F1 (Candle pattern) — enforced by the FeatureDAG (F1 depends on F6 per ADR-003). In the prompt sequence this means Prompt 3.7 (F6) before Prompt 3.2 (F1). F6 is also Risk R-01 (highest implementation risk). | Architect ratification 2026-06-06 | features.py, IMPLEMENTATION_PLAN.local.md | ✅ |
| D-23 | Config seed before impl | Q1–Q4 open questions (hammer_wick_ratio, cci_thresholds, sr_lookback_N + sr_cluster_atr, gap_max_age_bars) must be seeded as config keys in config.json BEFORE any feature implementation prompt begins. Zero hardcoded values from day 1. | Architect ratification 2026-06-06 | config.json, features.py F1/F6/F5/F7 | ✅ |
| D-24 | Skill pairing constraint | micha7-coder pairs ONLY with eyal-dev-standards. Do NOT pair with stockwise-coder, QA, Validator, or Compliance Auditor — those target StockWise core and conflict on path, config-hub, JSON-safety, and test-paths. Identified empirically from 4 concrete conflicts (Day 4 skill analysis). | Empirical audit 2026-06-06 | .claude/skills/micha7-coder/SKILL.md | ✅ |
| D-25 | Feature DoD (5 conditions) | Definition of Done per feature: (1) Tests ≥5 AND count grows monotonically, (2) Config-driven (zero hardcoded), (3) Deps declared as class attribute, (4) Structured logging via _log(), (5) py_compile PASS on all changed .py files. CHANGELOG update is enforced at commit level, not as a feature-DoD condition. | Architect ratification 2026-06-06 | All feature prompts (3.2–3.8) | ✅ |
| D-26 | Workflow | Three efficiency rules adopted: (a) File Manifest Upfront — every step declares EDIT/CREATE/READ files in a table before action; (b) Read Delegation to CC — local file inspection (structure verification, signature checks) delegated to Claude Code's pre-flight rather than uploaded to claude.ai; source-of-truth design docs (business_logic.local.md, decisions_registry, IMPLEMENTATION_PLAN) still uploaded; (c) Rigor Scaled to Risk — full 8-step + CoT for R-01/R-02 high-risk items, full 8-step (compressed CoT) for medium, 4-step (Problem→Solution→Tests/Verify→Prompt) for low-risk doc/config items. | Architect ratification 2026-06-08 | All implementation prompts | ✅ |
| D-27 | Lookahead | Pivot at bar T validated only when current_bar_index >= T + lookback_n; F6 receives current_bar_index and never reads bars beyond it. Prevents backtest lookahead bias. | B-14, Day 6 | features.py F6, BacktestRunner | ✅ |
| D-28 | Data | data.min_rows raised 100 -> 200 (config edit in 2.9b). At 100 with lookback_n=5 only T-5 is validated; 200 gives ample validated-pivot history. | B-19, Day 6 | config.json, data.py | ✅ |
| D-29 | Determinism | Level math float policy: equality via abs(a-b) < threshold + 1e-9; cluster prices round(2) cents; clustering merge tie-break = price ascending then bar_index ascending. Cross-machine bit-identical. | B-21, B-28, Day 6-7 | features.py F6 | ✅ |
| D-30 | F6 algorithm | F6 S/R (Q1-Q6 closed): pivots on High/Low; strength=touch count within cluster_atr*ATR; agglomerative single-linkage clustering; cluster price=simple mean (rounded, D-29); top_k=proximity-filtered then by strength; no levels->EMPTY. Hand-rolled, no sklearn (D-24). | Industry std, Day 6 | features.py F6 | ✅ |
| D-33 | F6 algorithm | Single reference ATR: strength counting AND clustering use ATR[current_bar_index], not per-touch-bar ATR. One yardstick -> consistent and deterministic. | B-30, Day 7 | features.py F6 | ✅ |
| D-34 | Timeframe | Real-time check (EOD-faithful): analysis computed through the last COMPLETE daily bar only; current forming bar shown as partial/live, never fed to F1-F7. Clarifies D-14: intraday checks allowed, intraday-bar analysis not. | User 2026-06-09, ADR-021 | analyzer, viewer | ✅ |
| D-35 | Visualization | S/R lines drawn only from valid_from = pivot_bar_index + lookback_n (point-in-time faithful). Candles always drawn in full; only derived levels are time-gated. | User 2026-06-09, B-32, ADR-020 | chart.py, viewer | ✅ |
| D-36 | Validation | TradingView used as external oracle: micha7 hand-rolled indicators must match TradingView values on identical data (recorded as test fixtures). Replaces self-confirming tests (B-00 trap). VALUES only, never TradingView's repaint/draw behavior (B-32). | User 2026-06-09 | tests, features.py F2/F7 | ✅ |
| D-37 | F7 algorithm | F7 CCI canonical (TradingView built-in): source=hlc3; length=20; cci = (src - SMA(src,len)) / (0.015 * mean_abs_dev(src,len)); deviation is MEAN ABSOLUTE deviation, NOT stdev. Config cci.length=20, cci.source="hlc3" seeded in 2.9b. | TradingView CCI v6, User 2026-06-09 | features.py F7, config.json | ✅ |
| D-38 | Workflow | Every CC prompt ends with a REVIEW & GOAL-CONFORMANCE GATE: CC restates the GOAL, verifies each acceptance criterion with ACTUAL command output (never from memory), confirms every File-Manifest file changed and nothing else did, explains how the artifacts meet the goal, STOPS on any FAIL/skip. The gate checks the FIXED manifest copied verbatim from the prompt — CC may not narrow it. Architect independently re-reviews. | User 2026-06-09 | All prompts, IMPL_PLAN section 2 | ✅ |

---

## Superseded Decisions

| Superseded Claim | Replacement | Source |
|-----------------|-------------|--------|
| ~~"R:R ≥ 2.0 minimum" as hard gate~~ (GLOSSARY.md:202) | D-13 + ADR-017: R:R logged, not filtered | 2026-06-03 audit |
| ~~Score range -7 to +7 (signed integer)~~ (GLOSSARY.md:57, pre-2026-06-03) | D-01 + ADR-016: bullish_count/7 model | 2026-06-03 audit |

---

## Adding New Decisions

When a new implementation decision is made during coding:
1. Add row to this table with next D-XX number
2. Add full ADR to DECISIONS.md if architectural scope
3. Cross-reference in business_logic.local.md §9
4. Update IMPLEMENTATION_PLAN.local.md §3 if it affects a prompt
