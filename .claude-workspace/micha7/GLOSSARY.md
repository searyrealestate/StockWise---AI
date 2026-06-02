# micha7_analyzer — Glossary

> **Version:** 1.1.0
> **Last Modified:** 2026-06-02T22:52:37Z

Domain-specific terms used throughout the project. Alphabetical.

---

## A

**ARMED (state)**
PivotDetector state entered when score ≥ 6/7 (D-03, D-04). System awaits price reaching a support level with a reversal candle to form a complete pivot trigger (D-10). ARMED persists across EOD runs until pivot fires or invalidation conditions are met (score drops < 4/7, or max ARMED duration exceeded).

**ATR (Wilder)**
Average True Range using Wilder's smoothing: SMA seed at bar (period−1), then recursive ATR[t] = (ATR[t−1]×(period−1) + TR[t]) / period. Alpha = 1/period (NOT 2/(period+1)). Period = 14. Used exclusively in F4 for normalizing distance from MA20. NOT used for stop loss sizing (D-19).

**Atomic Write**
A file write operation that either completes fully or not at all. Implemented via temp file + POSIX rename pattern.

---

## B

**Backtest**
Running the analysis pipeline against historical data to evaluate performance without risk to real capital.

**Bullish/Bearish Feature**
The output state of each F1–F7 computation. One of three values: BULLISH (✓ favorable), BEARISH (✗ unfavorable), EMPTY (— neutral or insufficient data). Phase 1 (long-only): only BULLISH count drives the score; BEARISH is treated as EMPTY for scoring purposes but logged separately (ADR-016, D-06, D-07). Replaces the previous speculative signed integer model (−1/0/+1).

**Bearish Engulfing**
A candle pattern where a red candle's body completely contains the previous (smaller) green candle's body. Bearish reversal signal.

**Blast Radius**
An impact analysis technique. For any change, asks "what else might break?" Used in StockWise development methodology.

**Bullish Engulfing**
Mirror of Bearish Engulfing. Green candle's body engulfs previous red body. Bullish reversal signal.

**Bullish Hammer**
A candle with a small body at the top and a long lower wick (≥2× body length). Bullish reversal signal when appearing at support.

---

## C

**CCI (Commodity Channel Index)**
Momentum oscillator. Used in micha7 with period 14 (vs standard 20). Values above +100 = overbought, below -100 = oversold.

**ChartSpec**
Single Source of Truth data structure for visualizations. Generated once, consumed by multiple renderers (HTML, Pine Script).

**Checklist (Micha's 7-Parameter)**
The core methodology: scoring each stock on 7 deterministic parameters and aggregating to a single decision score.

**Circuit Breaker**
Multi-level safety system that pauses or stops the system when problematic patterns are detected (consecutive losses, drawdown, etc.).

**Confidence**
Normalized absolute score (`|score|/7`) ranging 0.0-1.0. Higher = stronger signal.

---

## D

**DAG (Directed Acyclic Graph)**
Used for ordering feature calculation based on dependencies. Prevents circular references.

**DataAdapter**
Wraps StockWise `data_source_manager` (DSM) to provide validated, normalized OHLCV for micha7.

**Determinism**
Property that same input always produces same output. Critical for backtesting and reproducibility.

**Doji**
A candle where open ≈ close (very small body). Indicates indecision; not a strong signal alone.

**DSM (Data Source Manager)**
StockWise's existing data routing layer (Massive → Alpaca → IBKR → YFinance).

---

## E

**EOD (End of Day)**
Analysis run after market close (16:00 ET for NYSE). The primary trigger for micha7 in production.

**EntryPlanner**
Component that computes entry price, stop loss, 3 target prices, and risk:reward ratio.

---

## F

**FeatureExtractor**
Component that computes the 7 feature values from raw OHLCV data. Uses DAG-based ordering.

**Forward Trace**
Validation technique. Step through code path to verify expected behavior. Used in StockWise development methodology.

---

## G

**Gap (Calendar)**
A missing trading period in OHLCV data — e.g., a weekend, holiday, or data outage. Detected by `DataAdapter.detect_calendar_gaps()` (renamed from detect_gaps, D-17). Non-fatal; logged as a data quality warning. NOT the same as a price gap. Not a feature input.

**Gap (Price)**
A discontinuity between one bar's close and the next bar's open. "Gap above the head" = an unfilled upward price gap in a price range above the current close (acts as a magnetic target for price action; 80% of gaps historically close). F5 feature (D-17). Distinct from Calendar Gap. See business_logic.local.md §F5.

**Gap Detector**
See **Gap (Calendar)** and **Gap (Price)**. The term "Gap Detector" is deprecated in favor of the two distinct concepts above.

---

## H

**Hammer**
See "Bullish Hammer."

**Half-Day**
Shortened trading session (typically closing at 13:00 ET) on certain holidays (e.g., Black Friday). Requires special scheduler handling.

**Harami**
A candle pattern where current candle's body is small and contained within the previous candle's body. Reversal signal.

---

## I

**IN_POSITION (state)**
PivotDetector state indicating an open long position has been entered following a pivot trigger (D-04). The system monitors price against Target 1, Target 2, Target 3 (D-12) and the stop level (D-11) on each EOD run. Transitions to EXITED when a target is reached or stop is hit.

---

## L

**Lightweight Charts**
TradingView's open-source charting library. Used for HTML rendering in micha7. Already used in StockWise Simulator.

**Live (mode)**
Production trading mode with real money. Strict namespace isolation from paper and backtest.

**Lookahead Bias**
Bug where future data influences past decisions. Critical to avoid in backtesting.

---

## M

**MA20**
Simple Moving Average over 20 periods. Key indicator in Micha's methodology (not EMA).

**Micha**
The trader/educator whose methodology this system implements. The "7" in micha7 refers to his 7-parameter checklist.

**Multi-Pass Extraction**
Computing features in stages (Level 1 independent → Level 2 contextual). Used in micha7 to allow features to inform each other.

---

## N

**Namespace Separation**
Isolating state between modes (live/paper/backtest) via separate directories. Prevents one mode's data from affecting another.

**NYSE**
New York Stock Exchange. Used as the reference calendar for trading days (via `pandas_market_calendars`).

---

## O

**OHLCV**
Open, High, Low, Close, Volume — the standard market data tuple.

**Overbought**
CCI value above +100. Indicates price has moved too far up too fast; may revert.

**Oversold**
CCI value below -100. Indicates price has moved too far down too fast; may revert.

---

## P

**Paper (mode)**
Simulated trading with real-time data but no real money. Uses Alpaca Paper Trading API.

**Pause/Pivot**
The moment when price reverses direction at a support or resistance level. Critical entry timing.

**PF (Profit Factor)**
Total profits / Total losses. Target for production: ≥ 2.0.

**Phase A**
StockWise terminology for deterministic, rule-based components. micha7 is entirely Phase A.

**Phase B**
StockWise terminology for probabilistic, ML-based components. micha7 explicitly excludes Phase B.

**Pine Script**
TradingView's scripting language for custom indicators and strategies. micha7 generates Pine Script for TV integration.

**Pivot (entry trigger)**
A composite entry event requiring ALL THREE conditions simultaneously (D-10): (a) price is far from MA20 ≥ 2.0 ATR, (b) price touches the nearest support level (from F6), (c) a reversal candle pattern is present (F1 BULLISH). When all three conditions hold, PivotDetector transitions ARMED → TRIGGERED and EntryPlanner generates the trade plan.

**PivotDetector**
State machine component that tracks signals through their lifecycle: WAITING → ARMED → TRIGGERED → IN_POSITION → EXITED (D-04). Five states total (corrected from earlier 4-state description).

**Portfolio Risk**
Existing StockWise module that manages position sizing and exposure limits. micha7 reuses it.

---

## R

**R:R (Risk:Reward Ratio)**
(Target distance) / (Stop distance). Trades require R:R ≥ 2.0 minimum.

**Resistance**
Price level where selling pressure historically exceeds buying. Often the upper boundary of a range.

**RiskManager**
Component that validates trades against portfolio constraints before execution.

---

## S

**S/R (Support and Resistance)**
Key price levels identified by historical behavior. Micha's methodology heavily relies on these.

**Schema Versioning**
Including `schema_version` in state files to allow automatic migration when format changes.

**Score**
The aggregate of the 7 feature results, represented as bullish_count/7 (fraction) and bullish_count/7×100% (percentage). Phase 1: 6/7 or 7/7 (86–100%) triggers ARMED state. Previous signed-integer model (-7 to +7) superseded by ADR-016.

**Shooting Star**
A candle with long upper wick (≥2× body) and small body at the bottom. Bearish reversal signal.

**SignalEmitter**
Component that routes generated signals to consumers (Telegram, Visualizer, etc.).

**SUSPENDED (state)**
Circuit breaker state pausing all new trades for 24 hours after 3 consecutive losses.

---

## T

**TradingView (TV)**
Popular charting and trading platform. Target for Pine Script integration.

**Trend**
Direction of price movement over the lookback period. In micha7: window of 20 days.

**Traffic Light Score**
Discretization of the continuous bullish_count/7 score into three operational zones (D-02):
🔴 0–3/7 (0–43%) = WAITING, no action.
🟡 4–5/7 (57–71%) = WAITING, log full breakdown.
🟢 6–7/7 (86–100%) = ARMED trigger.
Used by PivotDetector and ScoringEngine for state transitions.

**TRIGGERED (state)**
PivotDetector state entered when all 3 pivot conditions are simultaneously satisfied (D-10): far from MA20 + support touch + reversal candle. EntryPlanner generates the trade plan. Symbol remains TRIGGERED for up to N trading days awaiting entry price fill before returning to ARMED.

---

## V

**Visualizer**
Component generating visual representations (HTML charts, Pine Scripts) of analysis results.

**Volume Decay**
Decreasing volume over time during a price move. Signals weakening momentum and possible reversal.

---

## W

**WAITING (state)**
Default PivotDetector state. No signal currently detected for this symbol.

**WAL (Write-Ahead Log)**
Database pattern of logging changes before applying them. Enables recovery from crashes.

**Win Rate (WR)**
Percentage of trades that closed profitable. Target: ≥ 60%.

---

## Common Abbreviations

| Abbr | Full |
|------|------|
| ADR | Architecture Decision Record |
| ATR | Average True Range |
| CCI | Commodity Channel Index |
| DAG | Directed Acyclic Graph |
| DSM | Data Source Manager |
| EOD | End Of Day |
| MA | Moving Average |
| OHLCV | Open/High/Low/Close/Volume |
| PF | Profit Factor |
| R:R | Risk:Reward (ratio) |
| S/R | Support/Resistance |
| SMA | Simple Moving Average |
| TV | TradingView |
| WAL | Write-Ahead Log |
| WR | Win Rate |
