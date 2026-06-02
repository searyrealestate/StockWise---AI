# micha7 — Business Logic Specification (PRIVATE)

> Source of truth for all scoring rules, state machine, entry/exit logic.
> Based on Micha's video transcript (2026-06-03 extraction).
> Version 1.0.0 — 2026-06-02T22:52:37Z
> ⚠️ GITIGNORED — DO NOT COMMIT

---

## 1. Methodology Origin

**Source:** Micha's video transcript "checklist 7" (videoplayback_chacklist7.txt)
**Extraction date:** 2026-06-03
**Confidence level:** 100% — direct quote-by-quote extraction from Hebrew transcript
**System type:** Technical analysis, rule-based, deterministic checklist

> **Critical disclaimer:** This document supersedes ALL previous business logic
> speculation in public docs. Any content in GLOSSARY.md, ARCHITECTURE.md, or
> PHASES.md that conflicts with this document is INCORRECT and should defer
> to this specification.

**What Micha's methodology is NOT:**
- Not value investing
- Not earnings-based selection
- Not ML or probabilistic — purely rule-based
- Not intraday — EOD analysis only (transcript line 27)

---

## 2. The 7-Parameter Checklist

Each parameter (F1–F7) outputs one of three states:
- ✓ **BULLISH** — condition favors a long entry
- ✗ **BEARISH** — condition is unfavorable
- — **EMPTY** — neutral or inconclusive

### F1: Candle Pattern (Last Bar)

**What it checks:** The most recent closed candle's body and wick geometry.
Assessed on the daily timeframe at EOD.

| Result | Pattern | Definition |
|--------|---------|------------|
| BULLISH ✓ | Bullish Engulfing | Green candle body engulfs previous red candle body |
| BULLISH ✓ | Bullish Hammer | Small body at top, lower wick ≥ 2× body length (at support) |
| BULLISH ✓ | Bullish Harami | Small green body contained within previous larger red body |
| BEARISH ✗ | Bearish Engulfing | Red candle body engulfs previous green candle body |
| BEARISH ✗ | Shooting Star | Long upper wick ≥ 2× body, small body at bottom (at resistance) |
| BEARISH ✗ | Bearish Harami | Small red body contained within previous larger green body |
| EMPTY — | Doji | Open ≈ Close within ≤ 5% of day's range |
| EMPTY — | No signal | Any other candle not matching above |

**Implementation notes:**
- Last bar ONLY (no rolling window)
- Body = |close - open|
- Wick threshold for Hammer/Shooting Star: lower/upper wick ≥ 2× body length
  (Open question Q1: exact ratio threshold — parameterize in config)
- Engulfing check: current body size > previous body size AND color differs
- Harami check: current body fits within previous body range AND color differs
- F1 **depends on F6 (S/R levels)**: Hammer only scores BULLISH when price is
  near a support level (per DAG dependency ADR-003)

### F2: Trend Direction (20-Day SMA)

**What it checks:** Whether price is above or below its 20-day Simple Moving Average,
and the slope direction.

| Result | Condition |
|--------|-----------|
| BULLISH ✓ | Close > MA20 AND MA20 slope is positive (rising) |
| BEARISH ✗ | Close < MA20 AND MA20 slope is negative (declining) |
| EMPTY — | Close > MA20 but slope flat, or price crossing MA20 |

**Implementation notes:**
- MA20 = SMA(close, 20) — NOT EMA (explicitly confirmed in GLOSSARY.md:138)
- Lookback for slope: 5 bars for slope detection (configurable D-15)
- "Slope positive" = MA20[today] > MA20[5 bars ago]
- Trend lookback configurable 5–22 days, default 20 (transcript line 29, D-15)
- F2 provides context to F3 (Volume momentum): F3 depends on F2 output

### F3: Volume Momentum

**What it checks:** Whether volume supports the price movement direction.

| Result | Condition |
|--------|-----------|
| BULLISH ✓ | Volume on up-days > volume on down-days over 5-bar window |
| BULLISH ✓ | Volume increasing as price rises toward MA20 |
| BEARISH ✗ | Volume Decay — decreasing volume on up-days while price rises |
| EMPTY — | Volume neutral or mixed signal |

**Implementation notes:**
- Compare average volume on green days vs red days over last 5 bars
- Volume Decay: 3 consecutive decreasing-volume up-days = BEARISH
- Uses F2 trend direction as context (DAG dependency: F3 depends on F2)
- Raw metric: volume_ratio = avg_up_volume / avg_down_volume (last 5 bars)
- Threshold: volume_ratio > 1.5 → BULLISH; < 0.67 → BEARISH; else EMPTY
  (Open question Q1: exact threshold — parameterize in config)

### F4: Distance from MA20 (in ATR Units)

**What it checks:** How far price is from MA20, normalized by ATR.

| Result | Condition |
|--------|-----------|
| BULLISH ✓ | Distance ≥ threshold_atr AND price below MA20 (stretched down — mean reversion) |
| EMPTY — | Distance < threshold_atr (price near MA20 — no stretched condition) |
| EMPTY — | Distance ≥ threshold_atr AND price above MA20 (already extended up — not entry) |

**Return structure (D-08):**
```
{
  "score": "BULLISH" | "EMPTY",
  "raw_distance": float,        # (close - MA20) / ATR(14)
  "raw_distance_abs": float,    # abs(close - MA20) / ATR(14)
  "ma20": float,
  "atr14": float
}
```

**Implementation notes:**
- ATR = Wilder ATR(14) — canonical, SMA-seeded (confirmed fixed in commit 71882a9)
- raw_distance = (close - MA20) / ATR(14) [signed — negative means price below MA20]
- F4 scores BULLISH when price is **below** MA20 by ≥ threshold (stretched, mean-reversion setup)
- BULLISH threshold: raw_distance < -2.0 ATR (configurable; D-19 confirms ATR for F4 only)
  (Open question Q2: exact threshold — parameterize in config)
- NOT used for stop calculation (D-19: ATR for F4 only, not stop sizing)

### F5: Price Gap Above Head

**What it checks:** Whether there is an unfilled price gap above the current price
that acts as a magnetic target.

| Result | Condition |
|--------|-----------|
| BULLISH ✓ | Open unfilled gap exists above current price |
| EMPTY — | No qualifying gap above current price |

**Definition (GLOSSARY.md:102, D-17):**
- Gap (Price) = previous_close → next_open discontinuity where next_open > previous_close
- "Gap above head" = an unfilled gap in a price range ABOVE current close
- "Unfilled" = price has not traded through the gap zone since it formed
- Statistical basis: "80% of gaps eventually close" (GLOSSARY.md:106)

**Implementation notes:**
- DISTINCT from DataAdapter.detect_gaps() which detects calendar gaps (D-17)
- Scan last 60 bars (configurable) for unfilled upside gaps
- Gap qualifies if: gap_top < current_price * 1.50 (not too far out of reach)
- Gap qualifies if: gap_bottom > current_price (gap is genuinely above)
- Returns BULLISH if ≥ 1 qualifying gap found; EMPTY otherwise
- Use as Target 2 in EntryPlanner (D-12)
  (Open question: max age of gap — parameterize in config)

### F6: Support/Resistance Levels

**What it checks:** Whether significant S/R levels exist near current price,
and whether price is near a support level (entry timing).

| Result | Condition |
|--------|-----------|
| BULLISH ✓ | Price is within proximity threshold of a known support level |
| BEARISH ✗ | Price is within proximity threshold of a known resistance level (pressing into resistance) |
| EMPTY — | Price is between S/R levels (mid-range) |

**Return structure (D-09):**
```
{
  "score": "BULLISH" | "BEARISH" | "EMPTY",
  "raw_levels": [
    {"price": float, "type": "support" | "resistance", "strength": int}
    ...
  ],
  "nearest_support": float | None,
  "nearest_resistance": float | None
}
```

**Detection algorithm (swing pivot method):**
- Lookback: 100 bars (min_rows raised to 100, D-16, B-05)
- A swing high is a bar whose high > N bars on each side (N=5, configurable)
- A swing low is a bar whose low < N bars on each side (N=5, configurable)
- Cluster levels within 0.5 ATR of each other into a single level
- Strength = number of touches
- Keep top 5 support levels + top 5 resistance levels
- Proximity threshold: |price - level| < 0.5 ATR → BULLISH if support, BEARISH if resistance
  (Open question Q3: exact algorithm — must test against Micha's hand-drawn lines)

**Note:** F6 provides S/R context to F1 (Candle). F6 must compute before F1 (DAG dependency, ADR-003).

### F7: CCI(14)

**What it checks:** Commodity Channel Index with period 14 (Micha's non-standard choice).

| Result | Condition |
|--------|-----------|
| BULLISH ✓ | CCI < -100 (oversold — mean-reversion opportunity) |
| BEARISH ✗ | CCI > +100 (overbought — price extended up) |
| EMPTY — | -100 ≤ CCI ≤ +100 (neutral zone) |

**CCI formula:**
```
TypicalPrice = (High + Low + Close) / 3
CCI = (TypicalPrice - SMA(TypicalPrice, 14)) / (0.015 × MeanDeviation)
MeanDeviation = mean(|TypicalPrice - SMA(TypicalPrice, 14)|) over 14 bars
```

**Implementation notes:**
- Period: 14 (vs standard 20 — Micha's explicit choice, GLOSSARY.md:44)
- Constant: 0.015 (Lambert's original constant)
- Score boundaries: ±100 (Open question Q2: transcript implies ±100 but not numeric)
- Long-only: BULLISH when oversold (<-100) matches mean-reversion logic
- No multi-level scoring (no ±200 extra score in Phase 1)

---

## 3. Scoring Model

### Phase 1 Model (Long-Only, ADR-016)

Each F1-F7 outputs: BULLISH | BEARISH | EMPTY

```
bullish_count = count(features where result == BULLISH)
bearish_count = count(features where result == BEARISH)  # logged only
score_fraction = bullish_count / 7
score_pct = score_fraction × 100
```

**Phase 1 rule (D-06):** BEARISH is treated as EMPTY for scoring purposes.
Only BULLISH count drives the traffic light.

**Rationale:** Long-only mode; a bearish signal is "not bullish" but does not
penalize the score. The bearish_count is logged separately (D-07) for future
diagnostics and short-mode development.

### Score Representation

Two parallel representations (D-01):
1. **Fraction:** `X/7` (e.g., "5/7")
2. **Percentage:** `X/7 × 100%` (e.g., "71%")

Both are computed and stored. The traffic light uses the fraction.

---

## 4. Traffic Light Thresholds

**Source:** User decision, verified conceptually against transcript tone (D-02)

| Traffic Light | Score Range | Fraction | Action |
|---------------|-------------|----------|--------|
| 🔴 RED | 0–43% | 0/7 – 3/7 | WAITING — no action, log score |
| 🟡 YELLOW | 57–71% | 4/7 – 5/7 | WAITING — log full breakdown |
| 🟢 GREEN | 86–100% | 6/7 – 7/7 | ARMED — enter watchlist |

**ARMED trigger (D-03):** Score ≥ 6/7 → transition WAITING → ARMED.

---

## 5. State Machine (5 States)

**States (D-04):**
```
WAITING → ARMED → TRIGGERED → IN_POSITION → EXITED
                ↘ (invalidation)
                  WAITING
```

### WAITING
- Default state for all symbols at startup
- Score < 6/7: stays WAITING
- Score ≥ 6/7: transition to ARMED
- Logs traffic light color and breakdown each EOD run

### ARMED
- Entered when: score ≥ 6/7
- Monitors for: pivot detection conditions (D-10)
- Exit to TRIGGERED when: all 3 pivot conditions met (see §6)
- Exit to WAITING (invalidation) when:
  - Score drops below 4/7 on subsequent EOD run
  - Price closes above MA20 by > 1 ATR (no longer in stretch condition)
  - Max ARMED duration exceeded (configurable, default 10 trading days)
- Persists across EOD runs via StateManager (ADR-005)

### TRIGGERED
- Entered when: all 3 pivot conditions simultaneously satisfied
- Action: EntryPlanner computes entry/stop/targets
- Next EOD: if position confirmed → IN_POSITION
- If entry price not reached: remains TRIGGERED for N days (configurable, default 3)
- If entry price not reached after N days: returns to ARMED

### IN_POSITION
- Active open long position
- Monitors: price vs targets and stop
- Exit to EXITED when: Target 1 reached, or stop hit, or max hold exceeded
- TradeJournal updated on each EOD run

### EXITED
- Terminal state for this trade instance
- Trade record finalized in TradeJournal
- Symbol returns to WAITING after EXITED (ready for next setup)

**Persistence:** All state transitions persisted atomically via StateManager + WAL (ADR-005, ADR-006).

---

## 6. Pivot Detection (The Entry Trigger)

**Definition (D-10):** A composite event requiring ALL THREE conditions simultaneously:

```
condition_a: abs(raw_distance_f4) ≥ pivot_atr_threshold
             (price is far from MA20 — stretched)
             Default: pivot_atr_threshold = 2.0 ATR

condition_b: price touches support level (from F6.nearest_support)
             i.e.: low ≤ nearest_support ≤ close + 0.5 ATR

condition_c: reversal candle present (from F1)
             i.e.: F1 result == BULLISH (Engulfing, Hammer, or Harami)
             OR CCI crosses from below -100 to above -100 (F7 cross)
```

**Pivot trigger logic:**
```python
is_pivot = (
    condition_a  # far from MA20
    and condition_b  # touching support
    and condition_c  # reversal signal
)
```

All three required. If any condition is absent, no trigger.

---

## 7. Entry / Stop / Targets

### Entry Price
- Entry at the **close of the pivot bar** (EOD execution)
- Alternatively: entry at next open (configurable, default = close)
- Phase 1 backtest uses close

### Stop Loss (D-11)
- Stop = nearest_support × (1 - stop_buffer_pct)
- Default: stop_buffer_pct = 0.01 (1% below support)
- Source: transcript line 77 — "below the support level"
- Stop is NOT ATR-based (D-19: ATR for F4 distance only)

### 3 Targets (D-12)
- **Target 1:** nearest resistance above entry (from F6.raw_levels, type="resistance")
- **Target 2:** bottom of nearest unfilled price gap above entry (from F5 detection)
- **Target 3:** next resistance level above Target 1 (second resistance in F6.raw_levels)
- Source: transcript line 65

### R:R Metric (D-13)
```
rr = (target1 - entry) / (entry - stop)
```
- Computed and logged per trade
- NOT used as entry filter (D-13, ADR-017)
- Target: R:R ≥ 2.0 is a quality signal but not a hard gate

### Position Sizing (RiskManager)
- Standalone (ADR-014, no portfolio_risk dependency)
- Specific limits classified in config_values.local.md (not created yet)
- Gates: max_position_pct, max_symbols_active, max_risk_per_trade_pct

---

## 8. Long-Only Constraint (D-05, ADR)

Phase 1 is **LONG ONLY**.

- Only BULLISH signals drive score and triggers
- No short-side detection in Phase 1
- No put/short position management
- Bearish features are logged but do not trigger short trades
- Future: symmetric short detection via config flag `enable_short_side`

---

## 9. The 20 Architectural Decisions (D-01..D-20)

Cross-reference to `decisions_registry.local.md` for full table.

| ID | Domain | One-line summary |
|----|--------|-----------------|
| D-01 | Scoring | Continuous 0-100% + X/7 fraction representation |
| D-02 | Threshold | Traffic light 🔴 0-3, 🟡 4-5, 🟢 6-7 |
| D-03 | Entry trigger | Score ≥ 6/7 → ARMED state |
| D-04 | State machine | 5 states: WAITING→ARMED→TRIGGERED→IN_POSITION→EXITED |
| D-05 | Direction | Long only in Phase 1 |
| D-06 | Scoring model | Bearish = empty for Phase 1 scoring |
| D-07 | Logging | Bearish count logged separately for diagnostics |
| D-08 | F4 structure | Returns score + raw_distance (signed + abs) |
| D-09 | F6 structure | Returns score + raw_levels list |
| D-10 | Pivot | Composite: far from MA20 + support touch + reversal candle |
| D-11 | Stop | Below support, default 1% configurable |
| D-12 | Targets | 3 structural: nearest resistance, gap bottom, next resistance |
| D-13 | R:R | Logged as metric, NOT used as entry filter |
| D-14 | Timeframe | EOD only (daily bars) |
| D-15 | Trend lookback | 5–22 days configurable, default 20 |
| D-16 | Min rows | 100 bars (for robust S/R detection) |
| D-17 | Gap naming | Price gap ≠ Calendar gap (separate concepts/code) |
| D-18 | auto_adjust | True (matches TradingView display; acknowledged as B-02) |
| D-19 | ATR usage | ATR for F4 distance only; NOT for stop sizing |
| D-20 | TradeJournal | Local-First JSON+CSV; Google Sheets deferred to Phase 4+ |

---

## 10. Anti-Patterns (Don't Do This)

Extracted from transcript warnings:

1. **Don't trade without running the full checklist** (transcript lines 91, 119)
   - Micha explicitly says he made his worst trades when he skipped the checklist
   - The Shopify example: "I was sure it would keep going up, didn't check my system"
   - Implication: all 7 features must run; no shortcutting on "obvious" setups

2. **Don't add features beyond the 7** (transcript line 25)
   - "הרבה אנשים שואלים אותי למה לא להוסיף עוד פרמטרים"
     ("Many people ask me why not add more parameters")
   - Micha's answer: the 7 cover the essential dimensions; more = noise + overfitting
   - Implementation constraint: features.py implements exactly F1-F7, no more

3. **Don't override the system based on news/sentiment** (implied by lines 91-119)
   - "אם המערכת אומרת לא — אל תכנס" ("If the system says no — don't enter")
   - No manual override in code; circuit breaker and system gates are the only override path

---

## 11. Open Questions (Deferred to Architect)

These require either additional transcript research or architect decision before implementation:

**Q1: Exact wick ratio for Hammer/Shooting Star**
- Transcript says "wick significantly larger than body" (transcript line ~44)
- Current working assumption: wick ≥ 2× body length
- Risk: too strict misses setups; too loose gives false positives
- Resolution: parameterize as `hammer_wick_ratio` in config.json; sweep in backtest

**Q2: CCI exact threshold**
- Transcript implies ±100 ("when CCI is very low" / "when CCI is very high")
- Working assumption: <-100 = BULLISH, >+100 = BEARISH
- Risk: standard ±100 may need tuning per symbol universe
- Resolution: parameterize as `cci_oversold` and `cci_overbought` in config.json

**Q3: S/R detection algorithm vs Micha's visual lines**
- PHASES.md explicitly flags this as the highest-risk feature (§Risks)
- Transcript shows Micha drawing S/R by eye on TradingView
- Working approach: swing pivot (N=5 bar lookback, cluster within 0.5 ATR)
- Resolution: test against hand-annotated AAPL chart; tune N and cluster threshold

**Q4: Max age of price gap (F5)**
- Transcript does not specify when a gap is "too old"
- Working assumption: scan last 60 bars
- Risk: old gaps may not function as magnets in current market
- Resolution: parameterize as `gap_max_age_bars` in config.json

---

## 12. Validation Plan

### Phase 1 Backtest Acceptance Criteria
- Symbol universe: 13 stocks from StockWise inventory (to be specified)
- Lookback period: 90-day historical
- Minimum threshold (PF ≥ 1.5, WR ≥ 55%): Phase 1 pass
- Stretch target (PF ≥ 2.0, WR ≥ 60%): Aligns with Micha's claim of 60-80% accuracy

### Determinism Check
- Same input → bit-identical output (ADR-002, ADR-012)
- Run backtest twice, diff outputs → must be empty

### Feature Sanity Checks
- F1: test against known hand-labeled Hammer/Engulfing/Doji setups
- F6: test against hand-annotated AAPL S/R chart from TradingView
- ATR: verified against canonical Wilder calculation (commit 71882a9)

---

## 13. Source Transcript References

Key quotes used in this specification (Hebrew → English):

| Line range | Hebrew excerpt | English translation | Used for |
|------------|----------------|---------------------|----------|
| ~25 | "שבעה פרמטרים בדיוק..." | "Exactly seven parameters..." | F1-F7 count |
| ~27 | "סוף יום, לא במהלך היום" | "End of day, not intraday" | D-14 EOD |
| ~29 | "מגמה על 20 ימים" | "Trend over 20 days" | F2 lookback |
| ~29 | "נר מחיר מעל הראש" | "Price gap above the head" | F5 naming |
| ~44 | "פתיל ארוך משמעותית מהגוף" | "Wick significantly larger than body" | Q1 |
| ~65 | "שלושה יעדים: רזיסטנס, גאפ, רזיסטנס" | "Three targets: resistance, gap, resistance" | D-12 |
| ~77 | "סטופ מתחת לסאפורט" | "Stop below support" | D-11 |
| ~91 | "כשלא השתמשתי במערכת..." | "When I didn't use the system..." | Anti-pattern |
| ~107 | "בוליש / בריש / ריק" | "Bullish / Bearish / Empty" | ADR-016 |
| ~117 | "ספירת הבוליש בלבד" | "Only bullish count" | D-06 long-only |
| ~119 | "שופיפיי — הלכתי נגד המערכת" | "Shopify — I went against the system" | Anti-pattern |

> Full Hebrew quotes preserved verbatim in private notes. These line numbers are approximate
> references to the transcript file; exact positions may vary by rendering.
