# micha7_analyzer — Glossary

> **Version:** 1.0.0
> **Last Modified:** 2026-05-21T05:35:00Z

Domain-specific terms used throughout the project. Alphabetical.

---

## A

**ARMED (state)**
A state in the PivotDetector state machine indicating that a signal has been detected (score ≥ threshold) and the system is now actively watching for entry conditions to be met.

**ATR (Average True Range)**
A volatility indicator. Used in micha7 for normalizing distances (e.g., distance from MA20 in ATR units).

**Atomic Write**
A file write operation that either completes fully or not at all. Implemented via temp file + POSIX rename pattern.

---

## B

**Backtest**
Running the analysis pipeline against historical data to evaluate performance without risk to real capital.

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

**Gap**
Difference between previous day's close and current day's open. A gap "above the head" = current price is below an unfilled gap (acts as magnet pulling price up).

**Gap Detector**
Component identifying open gaps in the price chart that may attract future price action (statistically, 80% of gaps eventually close).

---

## H

**Hammer**
See "Bullish Hammer."

**Half-Day**
Shortened trading session (typically closing at 13:00 ET) on certain holidays (e.g., Black Friday). Requires special scheduler handling.

**Harami**
A candle pattern where current candle's body is small and contained within the previous candle's body. Reversal signal.

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

**PivotDetector**
State machine component that tracks signals through their lifecycle (WAITING → ARMED → TRIGGERED → IN_POSITION).

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
The aggregate of all 7 feature scores, ranging from -7 to +7. ≥+5 = strong long signal.

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

**TRIGGERED (state)**
PivotDetector state when all 4 pivot conditions are met and entry should occur.

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
