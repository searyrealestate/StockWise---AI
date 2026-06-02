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
