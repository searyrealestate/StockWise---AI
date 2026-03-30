# StockWise AI — Master Implementation Specification
## Version 1.0 | March 2026
### For Implementing Agent — Single Source of Truth

---

# Part A: Executive Summary

This document defines the **complete, ordered work plan** for bringing StockWise AI from its current 45-50% implementation state to a production-grade autonomous trading signal platform.

**Current State:** 6,417 lines of Python across 10 modules. Core architecture exists (4-Agent Orchestra, MLFQ Scanner, Feature Engine with 63+ indicators, Kinetic Trailing Stop). However, 3 critical modules are missing, 5 show-stopping bugs prevent any BUY signal from being generated, and the AI/ML core returns a hardcoded neutral score.

**Target State:** A fully integrated, backtested, and paper-validated system capable of generating reliable BUY/SELL signals with measurable net-positive expectancy.

**Guiding Principles:**
1. **DO NOT DELETE OR REWRITE** working code. Fix surgically. Preserve Hebrew comments.
2. Every change must pass `master_validator.py` before moving forward.
3. All column names: **lowercase snake_case** throughout the entire system.
4. All thresholds must live in `system_config.py` — zero hardcoded magic numbers.
5. Every function: INFO log for operational events, DEBUG log for math forensics.

---

# Part B: Phase-by-Phase Implementation Plan

---

## PHASE 1: Critical Bug Fixes (5 Bugs)
**Priority: CRITICAL — Nothing else works until this is done.**
**Estimated Duration: 4-5 days**
**Files Modified: strategy_engine.py, live_trading_engine.py, feature_engine.py**

### Bug 1.1 — AI Feature Mismatch (CRITICAL)

**Location:** `strategy_engine.py` → `TacticalSniper.get_ai_probability()` + `train_model.py`

**Problem:** train_model.py saves feature lists as `['tech_score', 'ai_score', 'master_score', 'regime_val']`. But `get_ai_probability()` loads these feature lists and tries to filter the live DataFrame. The live DataFrame has columns like `rsi`, `sma_50`, `adx` — none match. Result: AI always returns 50.0 (neutral fallback), making Core B of the Dual-Core engine dead.

**Fix:**
- When training models (Phase 4), save the EXACT feature column list used during `model.fit()` as `Trend_Master_Model_features.json` / `Chop_Master_Model_features.json`.
- These feature lists must be the actual technical indicator columns from `feature_engine.py` output (e.g., `['rsi', 'adx', 'sma_50', 'er_slow', 'atr', ...]`), not meta-scores.
- **Temporary fix for Phase 1:** Create dummy `.pkl` model files and matching feature JSON files so the pipeline doesn't crash on missing files. The models should return `random.uniform(40, 60)` until Phase 4 produces real trained models.

**Validation:** After fix, `get_ai_probability()` returns a value != 50.0 for at least 50% of test cases.

---

### Bug 1.2 — Column Name Case Mismatch (CRITICAL)

**Location:** `strategy_engine.py` → `StrategyEngine.apply_checklist_bonus()`

**Problem:** References `row.get('SMA_50')` and `row.get('SMA_200')` (uppercase). But `feature_engine.py` creates `sma_50` and `sma_200` (lowercase). The bonus logic silently returns 0, suppressing score calibration.

**Fix (SURGICAL — exact replacements):**
```
In apply_checklist_bonus():
- 'SMA_50'  → 'sma_50'
- 'SMA_200' → 'sma_200'
- 'BBU_20_2.0' → 'bb_upper'
- 'rsi_14' → 'rsi'
```

**Also check:** `live_trading_engine.py` → `TradeJournal.log_signal()` references `last.get('SMA_50', last['close'])` — change to `sma_50`.

**Validation:** Run `apply_checklist_bonus()` on a DataFrame from `feature_engine.calculate_features()`. Confirm bonus > 0 when trend is aligned.

---

### Bug 1.3 — Non-Existent Column 'er_trend' (CRITICAL)

**Location:** `strategy_engine.py` → `TacticalSniper.analyze()`, Setup 1 (DSP_SUPER_TREND)

**Problem:** Code checks `last.get('er_trend', 0) == 1`. The column `er_trend` is NEVER created by `feature_engine.py`. Feature engine creates `er_slow` and `er_fast`. This means the highest-value setup (35 points, "King of Trends") is NEVER detected.

**Fix:**
Replace in `TacticalSniper.analyze()`:
```python
# OLD (broken):
if last.get('er_trend', 0) == 1 and last.get('trend_alignment', 0) == 1:

# NEW (correct):
er_slow = last.get('er_slow', 0)
threshold = cfg.DSP_CONFIG.get('threshold_coherent_trend', 0.55)
if er_slow >= threshold and last.get('trend_alignment', 0) == 1:
```

**Validation:** Scan NVDA/AAPL during a trending period. Confirm `DSP_SUPER_TREND` appears in `active_setups`.

---

### Bug 1.4 — Cooldown File Never Written (HIGH)

**Location:** `live_trading_engine.py` → `LiveTradingEngine.manage_open_positions()`

**Problem:** When a position is liquidated via STOP_LOSS, the ticker is removed from `self.positions` but NEVER written to `cooldown_list.json`. The wash-trade prevention gate in `strategy_engine.py._is_in_cooldown()` reads this file but finds it empty forever.

**Fix:** Add the following BEFORE `liquidated.append(symbol)` in `manage_open_positions()`:
```python
if reason == "STOP LOSS HIT":
    self._write_cooldown(symbol)
```

Add new method to `LiveTradingEngine`:
```python
def _write_cooldown(self, ticker):
    """Writes a ticker to the cooldown blacklist after a stop-loss hit."""
    cooldown_path = getattr(cfg, 'COOLDOWN_FILE_PATH', 'data/cooldown_list.json')
    try:
        data = {}
        if os.path.exists(cooldown_path):
            with open(cooldown_path, 'r') as f:
                data = json.load(f)
        data[ticker] = {"timestamp": time.time(), "reason": "STOP_LOSS"}
        with open(cooldown_path, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"[{ticker}] Added to 24h cooldown blacklist.")
    except Exception as e:
        logger.error(f"Failed to write cooldown for {ticker}: {e}")
```

**Validation:** Trigger a stop-loss in paper mode. Verify `cooldown_list.json` contains the ticker. Verify next scan skips it.

---

### Bug 1.5 — Dual Threshold Conflict (HIGH)

**Location:** `strategy_engine.py` — `TacticalSniper.analyze()` returns BUY at >60, but `evaluate_ticker()` overrides to WAIT if master_score < 80 (MIN_MASTER_SCORE_APPROVAL).

**Problem:** Combined with Bug 1.2 (broken checklist bonus) and Bug 1.3 (missing DSP_SUPER_TREND), almost no stock reaches 80. The system generates zero BUY signals.

**Fix — TWO OPTIONS (choose one):**

**Option A (Recommended): Lower the approval threshold.**
In `system_config.py`:
```python
MIN_MASTER_SCORE_APPROVAL = 65.0  # Was 80.0
```
Rationale: The Friction-Adjusted Alpha veto already protects against bad trades mathematically. The Master Score threshold is a secondary safety net, not the primary gate.

**Option B: Recalibrate the scoring pipeline.**
Keep threshold at 80 but increase the weight of the checklist bonus and fix the score aggregation formula. This requires Phase 4 (ML Pipeline) to be complete first.

**Validation:** After fixing bugs 1.1-1.5, scan 20 VIP stocks. Confirm at least 2-3 produce BUY signals with master_score > threshold.

---

### Bug 1.6 — Additional Fixes Needed in Phase 1

**1.6a — Missing squeeze columns:**
`TacticalSniper.analyze()` references `last.get('squeeze_on', 0)` and `last.get('mom_sqz', 0)`. These are never calculated in `feature_engine.py`.

**Fix:** Add to `feature_engine.py` → `add_volatility_block()`:
```python
# Bollinger Squeeze Detection (BB inside KC)
if 'bb_lower' in df.columns and 'kc_lower' in df.columns:
    df['squeeze_on'] = ((df['bb_lower'] > df['kc_lower']) & 
                        (df['bb_upper'] < df['kc_upper'])).astype(int)
else:
    df['squeeze_on'] = 0

# Momentum Squeeze (using MACD Histogram as proxy)
if 'macd_hist' in df.columns:
    df['mom_sqz'] = df['macd_hist']
else:
    df['mom_sqz'] = 0
```

**1.6b — NaN → 0 danger in feature_engine:**
`df.fillna(0)` at the end of `calculate_features()` converts missing SMA_200 (which means "not enough data") to 0.0 (which means "price is at zero"). This causes false Death Cross signals and zero-value stop losses.

**Fix:** Replace `df = df.fillna(0)` with:
```python
# Safe NaN handling: fill indicator columns, but NOT price columns
price_cols = ['open', 'high', 'low', 'close', 'volume']
indicator_cols = [c for c in df.columns if c not in price_cols]
df[indicator_cols] = df[indicator_cols].fillna(0)
```

**1.6c — buy_date_clean bug:**
In `live_trading_engine.py` → `manage_open_positions()`:
```python
buy_date_clean = buy_date_raw.split("T") if "T" in buy_date_raw else buy_date_raw
```
This creates a LIST `["2025-02-05", "14:30:00"]`, not a string. Fix:
```python
buy_date_clean = buy_date_raw.split("T")[0] if "T" in buy_date_raw else buy_date_raw
```

---

## PHASE 2: Missing Core Modules
**Priority: CRITICAL**
**Estimated Duration: 7-9 days**
**Files Created: market_intelligence.py, portfolio_manager.py, dag_optimizer.py**

### 2.1 — market_intelligence.py (The Gatekeeper)

**Purpose:** Protect the system from macro-level catastrophic events that no individual stock analysis can detect.

**Required Components:**

**2.1.1 — SPY Circuit Breaker:**
```
Function: check_market_health()
Input: Real-time SPY data (from data_source_manager)
Logic:
  - Fetch SPY intraday change %
  - If SPY drops > cfg.RISK_CONFIG['spy_crash_trigger_pct'] (-1.5%):
    → Return MARKET_HALT
    → Log at INFO: "CIRCUIT BREAKER TRIGGERED: SPY -{x}%"
    → Send Telegram alert
  - If VIX > 30 (add VIX_PANIC_THRESHOLD to system_config):
    → Return MARKET_CAUTION (reduce position sizes by 50%)
  - Else: Return MARKET_CLEAR
```

**2.1.2 — Event Horizon Calendar:**
```
Function: check_event_horizon(symbol)
Input: Ticker symbol
Logic:
  - Fetch next earnings date using yfinance: yf.Ticker(symbol).calendar
  - If earnings within cfg.PORTFOLIO_DEFENSE['event_horizon_buffer_days'] (2 days):
    → Return EVENT_VETO
    → For active positions: trigger force-liquidation
  - Else: Return EVENT_CLEAR
```

**2.1.3 — News Sentiment (Phase 2 Stub → Full in Phase 4):**
```
Function: get_sentiment_score(symbol)
Input: Ticker symbol
Output: float between -1.0 and +1.0
Logic (Stub):
  - Return 0.0 (neutral) for Phase 2
  - Full implementation in Phase 4: API calls to news providers,
    source reliability weighting, recency decay
```

**Integration Point:** Called by `live_trading_engine.py` main loop BEFORE scanning VIP stocks:
```python
market_state = market_intel.check_market_health()
if market_state == "MARKET_HALT":
    logger.info("Circuit Breaker active. Skipping all scans.")
    time.sleep(300)
    continue
```

---

### 2.2 — portfolio_manager.py (The Accountant)

**Purpose:** Track all positions (real + shadow), calculate net PnL, enforce portfolio-level risk limits.

**Required Components:**

**2.2.1 — Shadow Ledger:**
```
Class: ShadowLedger
Storage: data/shadow_ledger.json

Methods:
  - log_signal(ticket): Records EVERY signal (BUY/WAIT) with timestamp,
    scores, price, stop, target. This tracks what the system WANTED to do.
  - log_execution(ticker, exec_price): Records user confirmation.
    Marks signal as EXECUTED with 2x ML training weight.
  - log_outcome(ticker, exit_price, reason): Records trade closure.
    Calculates PnL net of friction.
  - get_prediction_accuracy(): Returns % of signals where
    price hit target before stop within 5 trading days.
```

**2.2.2 — Portfolio Correlation Defense:**
```
Function: check_correlation(new_ticker, open_positions, data_manager)
Logic:
  - For each open position ticker:
    - Fetch 30-day daily returns for both
    - Calculate Pearson correlation
    - If any pair > cfg.PORTFOLIO_DEFENSE['max_covariance_corr'] (0.85):
      → Return CORR_VETO with pair details
  - Return CORR_CLEAR
```

**2.2.3 — Aggregate PnL Calculator:**
```
Function: calculate_daily_pnl(positions, current_prices)
Output: Dict with total_pnl_usd, total_pnl_pct, win_rate, best_trade, worst_trade
Used by: EOD Summary in notification_manager.py
```

**2.2.4 — FIFO Tax Estimator:**
```
Function: estimate_tax(closed_trades)
Logic: Sort by entry date (FIFO). Short-term (< 1 year) taxed at 25%.
       Long-term taxed at 15%. Return estimated_tax_liability.
```

---

### 2.3 — dag_optimizer.py (The Information Theory Engine)

**Purpose:** Find the optimal sequence and combination of technical indicators for each market regime using SHAP values and XGBoost.

**Required Components:**
```
Function: optimize_dag(universal_dataset, regime='TREND')
Logic:
  1. Filter dataset by regime (ER > 0.6 for TREND, ER < 0.3 for CHOP)
  2. Train XGBoost classifier on ground truth labels
  3. Extract SHAP feature importance values
  4. Rank features by |SHAP value|
  5. Select top-K features (K determined by diminishing returns curve)
  6. Save to data/best_params.json:
     {
       "TREND": {"features": [...], "shap_values": {...}, "timestamp": "..."},
       "CHOP": {"features": [...], "shap_values": {...}, "timestamp": "..."}
     }

Schedule: Runs weekly (Saturday night) or on-demand after retraining.
```

---

## PHASE 3: Data Pipeline Hardening
**Priority: HIGH**
**Estimated Duration: 4-5 days**
**Files Modified: data_source_manager.py, system_config.py**

### 3.1 — Single Fetch Resampler

**Purpose:** Download 1H data once, derive Daily/Weekly mathematically. Reduce API calls by 66%.

**Implementation in data_source_manager.py:**
```
Function: resample_intraday_to_daily(df_hourly)
Logic:
  - Filter to Regular Trading Hours only (09:30-16:00 ET)
  - Group by calendar date
  - For each date:
    Daily.Open   = first hourly Open
    Daily.High   = max(all hourly Highs)
    Daily.Low    = min(all hourly Lows)
    Daily.Close  = last hourly Close
    Daily.Volume = sum(all hourly Volumes)
  - Return daily DataFrame

Function: resample_daily_to_weekly(df_daily)
Logic:
  - Group by ISO week
  - Same OHLCV aggregation as above
```

**Config Addition to system_config.py:**
```python
SESSION_BOUNDARIES = {
    "market_open": "09:30",
    "market_close": "16:00",
    "timezone": "US/Eastern"
}
ENABLE_SINGLE_FETCH = True  # Feature flag
```

**Integration:** When `ENABLE_SINGLE_FETCH = True`, `get_stock_data()` fetches 1H bars and calls `resample_intraday_to_daily()` internally.

---

### 3.2 — Pre-Market Validator

**Purpose:** Re-evaluate all VIP BUY signals at 09:25 AM using pre-market data to detect overnight GAP destruction.

**Implementation in live_trading_engine.py:**
```
Function: pre_market_gap_check(vip_buy_signals, data_manager)
Schedule: Fires once at 09:25 AM ET
Logic:
  For each pending BUY signal:
    1. Fetch current pre-market price
    2. Calculate gap_pct = (pre_market_price - yesterday_close) / yesterday_close
    3. Recalculate expected_rise with the new entry price
    4. If expected_rise - friction < cfg.MIN_NET_PROFIT:
       → VETO the signal
       → Send Telegram: "GAP VETO: {symbol} gapped {gap_pct}%. Edge destroyed."
    5. Else: Confirm signal for execution at market open
```

**Config Addition:**
```python
PRE_MARKET_CHECK_TIME = "09:25"  # ET
GAP_VETO_ENABLED = True
```

---

### 3.3 — Data Pipeline Cleanup

**Task 3.3a:** Remove duplicated `clean_raw_data()` — exists in BOTH `data_source_manager.py` AND `system_config.py`. Keep the one in `system_config.py` (centralized), import from there.

**Task 3.3b:** Verify Massive API import. Current code: `from massive import RESTClient`. If the actual package is `polygon`, change to:
```python
try:
    from polygon import RESTClient
    MASSIVE_AVAILABLE = True
except ImportError:
    MASSIVE_AVAILABLE = False
```

**Task 3.3c:** Add ML_FEATURES alignment. `system_config.py` defines `ML_FEATURES` list with `'rsi_14'` and `'wt1'`, `'wt2'`. But feature_engine creates `'rsi'` (not `'rsi_14'`) and never creates `'wt1'`/`'wt2'`. Fix ML_FEATURES to match actual output:
```python
ML_FEATURES = [
    'close', 'volume', 'daily_return',
    'rsi', 'adx', 'er_slow', 'er_fast', 'supertrend_direction',
    'vsa_squat_bar', 'rvol', 'atr',
    'sma_20', 'sma_50', 'sma_200',
    'macd', 'macd_hist', 'stoch_k',
    'bb_width', 'squeeze_on',
    'trend_alignment', 'golden_cross', 'death_cross'
]
```

Or better — generate this list dynamically from `feature_engine.calculate_features()` output columns.

---

## PHASE 4: ML Pipeline Rebuild
**Priority: HIGH**
**Estimated Duration: 7-9 days**
**Files Modified: train_model.py (major rewrite), strategy_engine.py**
**Files Created: None (train_model.py exists)**

### 4.1 — Universal Dataset Builder

```
Function: build_universal_dataset(data_manager, tickers, days_back=730)
Logic:
  1. For each ticker (minimum 50 S&P 500 stocks):
     a. Fetch 2 years daily OHLCV via data_source_manager
     b. Run feature_engine.calculate_features() with all indicators
     c. Calculate ground truth label:
        label = 1 if max(close[t+1:t+5]) >= close[t] * 1.03 (3% target)
                    AND min(close[t+1:t+5]) > close[t] * 0.975 (2.5% stop not hit)
        label = 0 otherwise
     d. Append to universal DataFrame with 'symbol' column
  2. Save to data/universal_training_data.parquet
```

### 4.2 — Regime-Segregated Training

```
Function: train_regime_models(universal_data)
Logic:
  1. Split data by ER:
     trend_data = universal_data[universal_data['er_slow'] > 0.55]
     chop_data  = universal_data[universal_data['er_slow'] < 0.30]

  2. Feature Masking:
     trend_features = [drop 'rsi', 'stoch_k', 'stoch_d', 'willr']  # Oscillators lie in trends
     chop_features  = [drop 'sma_50', 'sma_200', 'trend_alignment']  # Trend followers fail in chop

  3. Train:
     trend_model = RandomForestClassifier(n_estimators=200, max_depth=8)
     trend_model.fit(trend_data[trend_features], trend_data['label'])

     chop_model = RandomForestClassifier(n_estimators=200, max_depth=8)
     chop_model.fit(chop_data[chop_features], chop_data['label'])

  4. Save:
     joblib.dump(trend_model, 'models/Trend_Master_Model.pkl')
     json.dump(trend_features, 'models/Trend_Master_Model_features.json')
     joblib.dump(chop_model, 'models/Chop_Master_Model.pkl')
     json.dump(chop_features, 'models/Chop_Master_Model_features.json')
```

### 4.3 — Walk-Forward Validation

```
Split: Train on months 1-18, validate on months 19-21, test on months 22-24
Metrics to report:
  - Out-of-sample accuracy (target: > 58%)
  - Precision (% of predicted BUYs that were profitable)
  - Recall (% of profitable opportunities detected)
  - F1 Score
  - SHAP feature importance plots (save as PNG for documentation)
```

### 4.4 — Model Drift Monitor

Add to `strategy_engine.py`:
```python
class ModelDriftMonitor:
    def __init__(self):
        self.predictions = []  # List of (predicted, actual) tuples
        self.window = 20       # Rolling window

    def record(self, predicted_label, actual_outcome):
        self.predictions.append((predicted_label, actual_outcome))
        if len(self.predictions) > self.window:
            self.predictions.pop(0)

    def get_accuracy(self):
        if len(self.predictions) < self.window:
            return None
        correct = sum(1 for p, a in self.predictions if p == a)
        return correct / len(self.predictions)

    def needs_retraining(self):
        acc = self.get_accuracy()
        if acc is not None and acc < 0.55:
            return True
        return False
```

---

## PHASE 5: Backtesting & Calibration
**Priority: HIGH**
**Estimated Duration: 6-8 days**
**Files Created: backtest_engine.py**

### 5.1 — Walk-Forward Backtester

```
Class: BacktestEngine
Input: 2 years historical data, parameter set
Logic:
  For each trading day in test period:
    1. Run stock_hunter scan logic (using historical data up to that day)
    2. Run strategy_engine.evaluate_ticker() on VIP list
    3. Simulate entry at signal price + slippage
    4. Simulate Agent 4 kinetic stop management day-by-day
    5. Record outcome: hit target, hit stop, or timed out

Output:
  - Signal accuracy (% of BUYs that hit target)
  - Net expectancy (avg return per trade after friction)
  - Sharpe ratio (annualized)
  - Max drawdown
  - Equity curve (cumulative returns over time)
  - Win/Loss distribution histogram
```

### 5.2 — Parameter Calibration Targets

```
Parameter to tune:
  MIN_MASTER_SCORE_APPROVAL: Test range [55, 60, 65, 70, 75, 80]
  KINETIC_STOP phase1_atr_mult: Test range [1.5, 2.0, 2.5, 3.0]
  KINETIC_STOP phase2_breakeven_trigger_pct: Test range [0.01, 0.015, 0.02]
  FRICTION_AND_ALPHA min_net_profit_pct: Test range [0.005, 0.01, 0.013, 0.02]
  DSP_CONFIG threshold_coherent_trend: Test range [0.45, 0.50, 0.55, 0.60, 0.65]

Optimization method: Grid search over parameter combinations
Objective: Maximize Sharpe Ratio subject to:
  - Win rate > 55%
  - Max drawdown < 10%
  - Min 50 trades in test period (statistical significance)
```

### 5.3 — Success Criteria

```
The system is validated when ALL of the following are met:
  ✅ Backtested Signal Accuracy > 55% on out-of-sample data
  ✅ Net Expectancy > 0.5% per trade after all friction
  ✅ Sharpe Ratio > 1.5 (annualized)
  ✅ Max Drawdown < 10%
  ✅ Minimum 100 trades in 2-year backtest
```

---

## PHASE 6: Simulation Dashboard Rebuild (stockwise_simulation_v2.py)
**Priority: HIGH**
**Estimated Duration: 5-7 days**
**File: stockwise_simulation_v2.py (full rebuild, preserve log parser)**

### 6.1 — Architecture

The simulation dashboard is a Streamlit application with **3 tabs:**

**Tab 1: Historical Backtester (NEW)**
```
User Input:
  - Ticker symbol (text input)
  - Date range (start/end date pickers)
  - Data provider dropdown (ALPACA/YFINANCE/IBKR)

Output:
  - Interactive candlestick chart (using plotly) with:
    → BUY arrows (green ▲) at signal entry points
    → SELL arrows (red ▼) at exit points
    → Stop-loss lines (dashed red horizontal)
    → Take-profit lines (dashed green horizontal)
    → SMA 50/200 overlays
    → Volume bars below
  - Score breakdown panel:
    → Tech Score gauge (0-100)
    → AI Score gauge (0-100)
    → Master Score gauge (0-100)
    → Regime badge (TREND/CHOP/HALT)
    → Active setups list
  - Performance summary:
    → Total signals generated
    → Win rate
    → Average PnL per trade
    → Max drawdown
    → Equity curve chart
```

**Tab 2: Log Analyzer (PRESERVE existing logic)**
```
User Input:
  - Upload .txt log file OR select from local logs

Output (already works, preserve):
  - VIP Targets table
  - Detected Setups table
  - Trade Veto Analysis
  - Error summary with aggregation

Enhancement:
  - Add chart visualization of logged signals
  - Parse BUY/SELL signals from log and overlay on price chart
  - Show timeline of system decisions
```

**Tab 3: System Health (NEW)**
```
Output:
  - Model accuracy (from drift monitor)
  - Last training date
  - Data provider status (connected/disconnected)
  - Open positions summary
  - Cooldown list
  - Shadow Ledger statistics
```

### 6.2 — Chart Implementation Notes

```
Library: plotly (already available via Streamlit)
Chart type: Candlestick with overlays

Key implementation:
  import plotly.graph_objects as go
  from plotly.subplots import make_subplots

  fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                      vertical_spacing=0.03, row_heights=[0.7, 0.3])

  # Candlestick
  fig.add_trace(go.Candlestick(x=df.index, open=df.open, high=df.high,
                                low=df.low, close=df.close), row=1, col=1)

  # SMA overlays
  fig.add_trace(go.Scatter(x=df.index, y=df.sma_50, name='SMA50'), row=1, col=1)

  # BUY/SELL markers
  fig.add_trace(go.Scatter(x=buy_dates, y=buy_prices, mode='markers',
                           marker=dict(symbol='triangle-up', size=12, color='green'),
                           name='BUY'), row=1, col=1)

  # Volume
  fig.add_trace(go.Bar(x=df.index, y=df.volume, name='Volume'), row=2, col=1)
```

---

## PHASE 7: Integration Testing & Hardening
**Priority: MEDIUM**
**Estimated Duration: 3-4 days**
**File Modified: master_validator.py**

### 7.1 — New Test Cases Required

```
Test Suite A: End-to-End Pipeline Test
  - Fetch data for 5 tickers
  - Run full feature calculation
  - Run regime classification
  - Run strategy evaluation
  - Verify: at least 1 BUY signal generated
  - Verify: stop_loss < entry_price < take_profit for all BUY signals

Test Suite B: Market Intelligence Integration
  - Mock SPY at -2% → verify circuit breaker halts all scans
  - Mock earnings in 1 day → verify Event Horizon veto fires
  - Verify: Telegram alert sent for both

Test Suite C: Portfolio Defense
  - Create 2 open positions in same sector
  - Attempt to open 3rd correlated position → verify correlation veto

Test Suite D: Lifecycle Management
  - Create position at $100, stop at $95
  - Simulate price at $103 → verify Phase 2 breakeven triggered
  - Simulate price at $106 → verify Phase 3 parabolic choke
  - Simulate regime change → verify zombie protocol initiates
  - Simulate 73 hours → verify force liquidation

Test Suite E: Notification Flow
  - Verify BUY signal sends Telegram message
  - Verify SELL signal sends Telegram with PnL
  - Verify EOD summary contains correct counts
  - Verify /confirm command updates shadow ledger
```

---

## PHASE 8: Live Paper Trading Validation
**Priority: MEDIUM**
**Estimated Duration: 5 trading days (1 week)**

### 8.1 — Paper Trading Protocol

```
Duration: 5 consecutive trading days
Mode: PAPER (cfg.MODE = "PAPER")
Broker: Alpaca Paper Trading API

Daily checklist:
  □ 20:00 ET: Nightly scan runs automatically
  □ 07:00 IST: Verify IB Gateway health check fires
  □ 09:25 ET: Pre-market gap check runs
  □ 09:30 ET: Live engine processes VIP signals
  □ Throughout day: Monitor kinetic stops on open positions
  □ 23:00 IST: EOD summary sent via Telegram

Success criteria:
  - System runs 5 days without crashes
  - Signals generated are consistent with backtest expectations (±2σ)
  - All Telegram alerts fire correctly
  - Dashboard displays correct decision breakdowns
  - Shadow Ledger captures all signals accurately
```

---

# Part C: File-by-File Change Summary

| File | Phase | Action | Description |
|------|-------|--------|-------------|
| system_config.py | 1, 3 | MODIFY | Lower MIN_MASTER_SCORE_APPROVAL, add SESSION_BOUNDARIES, fix ML_FEATURES, add PRE_MARKET configs |
| feature_engine.py | 1 | MODIFY | Fix NaN handling, add squeeze_on/mom_sqz columns |
| strategy_engine.py | 1 | MODIFY | Fix 5 bugs (column case, er_trend, threshold). Add ModelDriftMonitor |
| live_trading_engine.py | 1, 3 | MODIFY | Fix cooldown write, buy_date_clean, add pre_market_gap_check |
| market_intelligence.py | 2 | CREATE | SPY circuit breaker, Event Horizon, sentiment stub |
| portfolio_manager.py | 2 | CREATE | Shadow Ledger, correlation defense, PnL calculator |
| dag_optimizer.py | 2 | CREATE | SHAP feature ranking, best_params.json output |
| data_source_manager.py | 3 | MODIFY | Add resampler, remove duplicated clean_raw_data, fix Massive import |
| train_model.py | 4 | REWRITE | Regime-segregated training, feature masking, walk-forward validation |
| backtest_engine.py | 5 | CREATE | Full walk-forward backtester with equity curve |
| stockwise_simulation_v2.py | 6 | REBUILD | 3-tab dashboard with charts, log analyzer, system health |
| master_validator.py | 7 | MODIFY | Add integration test suites A-E |
| notification_manager.py | — | NO CHANGE | Working correctly. Minor: parts[0] shadowing is cosmetic. |
| stock_hunter.py | — | MINOR | Add 7-month garbage collector (low priority) |

---

# Part D: Artifacts Needed Before Starting

| Artifact | Description | Status |
|----------|-------------|--------|
| This Spec Document | Complete implementation plan | ✅ DONE |
| All 10 source code files | Current codebase | ✅ DONE |
| All 4 spec documents (v12-v13.3) | Original requirements | ✅ DONE |
| Deep Research Document | Gap analysis | ✅ DONE |
| secrets.toml template | API key placeholders | ❌ NEEDS CREATION |
| Sample JSON test fixtures | trade_journal, scan_ledger, etc. | ❌ NEEDS CREATION |
| Dummy model artifacts | Placeholder .pkl + feature .json | ❌ NEEDS CREATION (Phase 1) |
| Backtest dataset (50 stocks, 2yr) | Pre-downloaded .parquet files | ❌ NEEDS CREATION (Phase 4) |

---

# Part E: Rules for the Implementing Agent

1. **Language:** All code, comments in new functions, and documentation must be in **English**. Existing Hebrew comments must be **preserved**.
2. **Order:** Execute phases sequentially. Do NOT start Phase 2 until Phase 1 passes all validation criteria.
3. **Testing:** Run `master_validator.py` after EVERY change. Run a manual 10-ticker scan after each bug fix.
4. **Logging:** Every new function must include INFO and DEBUG logging following the existing pattern.
5. **Config:** All new thresholds, paths, and parameters go in `system_config.py`. Zero magic numbers in logic files.
6. **Safety:** Before modifying any function, read the entire function first. Understand what it does. Make surgical edits. Do NOT rewrite functions that already work.
7. **Serialization:** All JSON writes must use the `NumpyEncoder` pattern from `stock_hunter.py`.
8. **Column Names:** Always lowercase snake_case. Verify by printing `df.columns.tolist()` after feature calculation.
9. **Git Discipline:** One commit per bug fix. One commit per new function. Descriptive commit messages.
10. **When Stuck:** Ask. Do not guess. Do not invent solutions that contradict this specification.

---

**END OF SPECIFICATION**

**Document Status: FINAL — Version 1.0**
**Next Action: Begin Phase 1, Bug 1.1**
