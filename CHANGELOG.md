# Changelog

## Phase 1 — Bug Fixes

### Bug 1.2 — Column name case mismatch in `apply_checklist_bonus` (strategy_engine.py)

**Fixed:** Column lookups in `apply_checklist_bonus()` now match the lowercase snake_case names produced by `feature_engine.py`.

| Before | After |
|--------|-------|
| `SMA_50` | `sma_50` |
| `SMA_200` | `sma_200` |
| `rsi_14` | `rsi` |
| `BBU_20_2.0` (with `bb_upper` fallback) | `bb_upper` |

**Impact:** The checklist bonus was silently returning 0 for trend alignment, RSI momentum, and Bollinger Band distance checks because the keys never matched. With this fix, the bonus can now award up to +40 points as intended.

---

### Bug 1.3 — Non-existent `er_trend` column in `TacticalSniper.analyze()` (strategy_engine.py)

**Fixed:** Setup 1 (`DSP_SUPER_TREND`) used `er_trend` which was never created by `feature_engine.py`. Only `er_slow` (continuous 0–1) and `er_fast` exist.

| Before | After |
|--------|-------|
| `if last.get('er_trend', 0) == 1 and ...` | `er_slow_val = last.get('er_slow', 0)` |
| (binary equality check on non-existent column) | `er_threshold = getattr(cfg, 'DSP_CONFIG', {}).get('threshold_coherent_trend', 0.55)` |
| | `if er_slow_val >= er_threshold and ...` |

**Impact:** Setup 1 (`DSP_SUPER_TREND`, +35 pts) was always skipped. Now fires correctly for trending stocks where `er_slow >= 0.55` (from `DSP_CONFIG`).

**Verified:** 5/5 unit tests pass (threshold boundary, above/below, alignment gate, dead-string removal).

---

### Test Infrastructure Added

- **`tests/unit_tests.py`** — Per-bug unit test classes (Bug 1.2: 5 tests, Bug 1.3: 5 tests = 10 total). Uses `sys.modules` stubbing for `pandas_ta` so all tests run fully offline. Targets the correct classes: `StrategyEngine.apply_checklist_bonus` and `TacticalSniper.analyze`.
- **`tests/master_validator.py`** — Cross-system integration checks (16 checks): column name consistency between `feature_engine` and `strategy_engine`, `DSP_CONFIG` integrity, score threshold sanity, dead column reference detection, and `analyze()` return contract validation.

---

### Bug 1.4 — Cooldown file never written on stop-loss (live_trading_engine.py + strategy_engine.py)

**Problem:** `strategy_engine._is_in_cooldown()` reads `data/cooldown_list.json` to prevent re-buying a stopped-out ticker, but nothing in the codebase ever *wrote* to this file. Result: after a stop-loss, the ticker was immediately re-scanned and re-bought → loss loop.

**Files changed:**

| File | Change |
|------|--------|
| `system_config.py` | Added `COOLDOWN_PERIOD_HOURS = 24` |
| `live_trading_engine.py` | Added `_write_cooldown()` method; called it on `STOP LOSS HIT` and `ZOMBIE PROTOCOL` exits in `manage_open_positions()` |
| `strategy_engine.py` | `_is_in_cooldown()` now uses `COOLDOWN_PERIOD_HOURS * 3600` instead of hardcoded `86400` |

**Impact:** Stop-loss and zombie exits now blacklist the ticker for 24 hours (configurable). The write/read round-trip between `LiveTradingEngine` and `StrategyEngine` is verified by tests.

**Tests added:** 4 unit tests (write creates file, appends not overwrites, round-trip read detection, config-driven period) + 2 system checks in master_validator.

**Totals:** 14/14 unit tests pass, 20/20 system checks pass.

---

### Bug 1.5 — Dual threshold conflict: dead zone 60-79 (system_config.py)

**Problem:** `TacticalSniper.analyze()` returns `BUY` when `master_score > 60`, but `evaluate_ticker()` immediately overrides to `WAIT` if `master_score < 80` (`MIN_MASTER_SCORE_APPROVAL`). Any score in 60-79 was silently killed — the Friction Alpha veto already filters unprofitable trades, making 80 redundant and unreachable in practice.

**Fix (config-only):**

| Setting | Before | After |
|---------|--------|-------|
| `MIN_MASTER_SCORE_APPROVAL` | `80.0` | `65.0` |

**Impact:** BUY signals scoring 65-79 now survive the approval gate. The Friction Alpha veto remains the real quality filter. Score range 60-64 is still gated (approval sits just above Sniper threshold).

**Tests added:** 4 unit tests (exact value, above Sniper threshold, below unreachable ceiling, BUY survival check). Existing `master_validator` `check_threshold_sanity` passes at 65.0.

**Totals:** 18/18 unit tests pass, 20/20 system checks pass.

---

### Bug 1.6a — Missing `squeeze_on` and `mom_sqz` columns (feature_engine.py)

**Problem:** `strategy_engine.py` Setup 2 (`VOLATILITY_SQUEEZE`, 20-30 pts) reads `squeeze_on` and `mom_sqz` from the feature DataFrame, but `feature_engine.py` never created these columns. Setup 2 was permanently dead.

**Fix:** Added two derived columns after the Keltner Channel block in `feature_engine.add_volatility_block()`:

| Column | Formula |
|--------|---------|
| `squeeze_on` | `(bb_lower > kc_lower) & (bb_upper < kc_upper)` cast to int — classic TTM Squeeze logic |
| `mom_sqz` | `macd_hist` — MACD histogram used as momentum proxy for squeeze firing direction |

**Impact:** Setup 2 `VOLATILITY_SQUEEZE_PREP` (+20 pts) and `SQUEEZE_FIRING_LONG` (+30 pts) can now fire when BB contracts inside KC.

**Tests added:** 5 unit tests (column creation guards, squeeze_prep fires, squeeze_firing fires, wide-band gate). `pending_investigation` set in master_validator pruned: `squeeze_on` and `mom_sqz` removed.

**Totals:** 23/23 unit tests pass, 20/20 system checks pass.

---

### Bug 1.6b — Dangerous `df.fillna(0)` on price columns (feature_engine.py)

**Problem:** `feature_engine.py` applied `df = df.fillna(0)` globally, zero-filling `open`, `high`, `low`, `close`, and `volume` wherever SMA/indicator warmup NaNs existed (first ~200 rows). This caused:
- RSI/SMA calculations to divide by zero close price
- ATR to produce negative values when `high=0`
- RVOL to become infinite when `vol_avg_20=0`

**Fix:** Replaced the blanket `fillna(0)` with a split strategy:

| Column group | Strategy | Reason |
|---|---|---|
| `open`, `high`, `low`, `close`, `volume` | `ffill()` | Last known price is correct; 0 is wrong |
| All indicator columns | `fillna(0)` | Safe for oscillators, flags, and booleans |

**Tests added:** 3 unit tests (no blanket fillna, ffill present, ffill preserves prices while zeroing indicators).

**Totals:** 26/26 unit tests pass, 20/20 system checks pass.

---

### Bug 1.6c — `split("T")` returns list instead of string (live_trading_engine.py)

**Problem:** Line ~482 used `buy_date_raw.split("T")` without `[0]`, producing `["2025-02-05", "14:30:00"]` instead of `"2025-02-05"`. This list was passed to Telegram notifications and CSV logging, corrupting both outputs.

**Fix:** Added `[0]` index to extract only the date portion:

```python
# Before:
buy_date_clean = buy_date_raw.split("T") if "T" in buy_date_raw else buy_date_raw
# After:
buy_date_clean = buy_date_raw.split("T")[0] if "T" in buy_date_raw else buy_date_raw
```

**Impact:** Telegram closed-position messages and trade history CSV now show clean `YYYY-MM-DD` dates.

**Tests added:** 4 unit tests (ISO extraction, no-T passthrough, UNKNOWN passthrough, source code guard).

**Totals:** 30/30 unit tests pass, 20/20 system checks pass.

---

### Bug 1.1 — AI Feature Mismatch: Core B (AI probability) was dead (train_model.py + strategy_engine.py)

**This was the most critical bug.** Three intertwined issues caused the AI to always return 50.0 (neutral):

**Issue 1 — Wrong feature list saved to disk (`train_model.py`)**
`train_and_save()` wrote `["tech_score", "ai_score", "master_score", "regime_val"]` as the feature schema. These are meta-scores computed *after* the AI runs — a circular dependency. The live engine would try to feed these back as model inputs, but they don't exist in the raw feature DataFrame.

**Issue 2 — Training pipeline used Shadow Ledger instead of real features (`train_model.py`)**
`execute_training_pipeline()` called `prepare_training_data()` which pulled from the Shadow Ledger (only 4 meta-columns). Replaced with the correct pipeline: `build_universal_dataset()` → `segregate_by_regime()` → `apply_feature_masking()` → `train_and_save()` using actual technical columns (`sma_50`, `rsi`, `macd_hist`, etc.).

**Issue 3 — XGBRegressor called with `predict_proba()` (`strategy_engine.py`)**
`get_ai_probability()` unconditionally called `predict_proba()`, which only exists on Classifiers. The Regressor always raised an exception → caught silently → returned 50.0.

**Fixes applied:**

| File | Change |
|------|--------|
| `train_model.py` | `XGBRegressor` → `XGBClassifier` (binary target: profit ≥ 2% in 5 days) |
| `train_model.py` | `execute_training_pipeline()` now uses `build_universal_dataset` → real features |
| `train_model.py` | `train_and_save()` now saves `list(X.columns)` — the actual trained feature names |
| `strategy_engine.py` | `get_ai_probability()` checks `hasattr(model, 'predict_proba')` with Regressor fallback |
| `system_config.py` | Added `DEFAULT_TRAINING_SYMBOLS` fallback list (10 liquid US equities) |

**Impact:** Core B (AI probability score) can now produce real 0-100 predictions instead of always returning 50.0. The full dual-core scoring formula `(tech_score * 0.7) + (ai_prob * 0.3)` is live.

**Tests added:** 6 unit tests + 3 system checks in master_validator.

**Totals:** 36/36 unit tests pass, 23/23 system checks pass.

---

### Bug 2.1 — `macdsignal` vs `macd_signal` column name mismatch (strategy_engine.py)

**Problem:** Setup 5 (`MOMENTUM_BREAKOUT`) read `last.get('macdsignal', 0)` but `feature_engine.py` creates the column as `macd_signal`. The default `0` caused `macd > signal` to be `True` any time MACD was positive — generating false-positive momentum signals with no actual crossover.

**Fix:** `strategy_engine.py:284` — `'macdsignal'` → `'macd_signal'` (one character, underscore vs camelCase).

**Impact:** Setup 5 now correctly detects real MACD crossovers. `macdsignal` removed from `pending_investigation` set in master_validator.

**Tests added:** 4 unit tests (column name guard, crossover fires, blocked below signal, no false positive from default).

**Totals:** 40/40 unit tests pass, 23/23 system checks pass.

---

### Bug 2.2 — `is_consolidating` / `BOLLINGER_SQUEEZE` columns don't exist (strategy_engine.py)

**Problem:** `apply_checklist_bonus()` computed `is_squeeze` by reading `'is_consolidating'` and `'BOLLINGER_SQUEEZE'` from the row — neither of which is created by `feature_engine.py`. Both always returned `False`, so the squeeze bonus (+10 pts) never fired.

**Fix:** Replaced with `squeeze_on` (created in Bug 1.6a), which performs the same BB-inside-KC detection:

```python
# Before:
is_squeeze = row.get('is_consolidating', False) or row.get('BOLLINGER_SQUEEZE', False)
# After:
is_squeeze = row.get('squeeze_on', 0) == 1
```

**Impact:** Squeeze bonus (+10 pts) in `apply_checklist_bonus()` can now fire.

**Cleanup:** `pending_investigation` set in master_validator is now empty — all previously tracked column mismatches have been resolved across Bugs 1.6a, 2.1, and 2.2. Column consistency check now runs with zero exclusions.

**Tests added:** 3 unit tests (bonus fires with squeeze_on=1, old column names absent, squeeze_on present in source).

**Totals:** 43/43 unit tests pass, 23/23 system checks pass.

---

### Bug 2.3 — HALT/NEUTRAL Regime Does Not Block Analysis (strategy_engine.py)

**Problem:** `evaluate_ticker()` classified the regime but never acted on HALT or NEUTRAL outcomes. Both regimes passed straight through to `TacticalSniper.analyze()`, allowing the system to generate BUY signals during a crash (HALT) or in a directionless dead zone (NEUTRAL).

**Fix:** Inserted two early-return gates immediately after `classify_regime()` in `evaluate_ticker()`:

```python
# [Bug 2.3 Fix] Block analysis for HALT and NEUTRAL regimes
if regime == "HALT":
    logger.debug(f"[{symbol}] Regime HALT -- velocity divergence detected. Skipping analysis.")
    return {"symbol": symbol, "action": "WAIT", "master_score": 0,
            "reason": "Regime HALT: Velocity Divergence (er_slow/er_fast conflict)"}
if regime == "NEUTRAL":
    logger.debug(f"[{symbol}] Regime NEUTRAL -- dead zone. Skipping analysis.")
    return {"symbol": symbol, "action": "WAIT", "master_score": 0,
            "reason": "Regime NEUTRAL: No clear trend or chop signal"}
```

**Impact:** HALT and NEUTRAL tickers now return `WAIT` immediately. TREND and CHOP still pass through to full sniper analysis as before.

**Tests added:** 4 unit tests (HALT returns WAIT, NEUTRAL returns WAIT, TREND still analyzed, CHOP still analyzed). Tests mock `calculate_features` to pass-through hand-crafted er_slow/er_fast values that define the regime under test. 2 system checks added to master_validator.

**Totals:** 47/47 unit tests pass, 25/25 system checks pass.

---

### Bug 2.4 — `_generate_labels` profit_target Inconsistency + Hardcoded Values (train_model.py + system_config.py)

**Problem:** Two issues in `_generate_labels()`:
1. Default was `profit_target=0.03` (3%) but `build_universal_dataset()` called it with `profit_target=0.02` (2%) — docstring and call site contradicted each other.
2. Both `lookahead=5` and `profit_target=0.02` were hardcoded, violating the zero-hardcoded-values rule.

**Fix:**

| File | Change |
|------|--------|
| `system_config.py` | Added `AI_LABEL_CONFIG = {"lookahead_days": 5, "profit_target_pct": 0.02}` after `DEFAULT_TRAINING_SYMBOLS` |
| `train_model.py` | `_generate_labels()` defaults changed to `None`; reads from `AI_LABEL_CONFIG` at runtime |
| `train_model.py` | `build_universal_dataset()` call changed from `_generate_labels(df, lookahead=5, profit_target=0.02)` to `_generate_labels(df)` |

**Impact:** Label generation parameters are now centrally configurable. Default and explicit calls share the same config source — no more docstring/call-site mismatch.

**Tests added:** 5 unit tests (config exists, values consistent, defaults are None, config-driven labeling, hardcoded call removed). xgboost stubbed in sys.modules so train_model.py can be imported without the package installed.

**Totals:** 52/52 unit tests pass, 25/25 system checks pass.

---

## Phase 2 — Let Winners Run (Milestone Alert System)

### Phase 2.5a — Milestone Alert Infrastructure (live_trading_engine.py + system_config.py)

**Context:** Converting from "sell at fixed take_profit" to "let winners run with milestone-based alerts." This is step 1 of 3 — adds config and helper methods without changing any existing behavior.

**Changes:**

| File | Change |
|------|--------|
| `system_config.py` | Added `MILESTONE_ALERT_CONFIG` after `KINETIC_STOP_CONFIG` |
| `live_trading_engine.py` | Added `_calculate_real_breakeven()` method |
| `live_trading_engine.py` | Added `_check_and_send_milestone_alert()` method |

**MILESTONE_ALERT_CONFIG keys:**

| Key | Value | Purpose |
|-----|-------|---------|
| `safe_zone_buffer_pct` | 0.002 | 0.2% safety margin above true breakeven |
| `min_stop_change_pct` | 0.01 | Alert only if stop moved > 1% of current price |
| `min_alert_interval_minutes` | 15 | Minimum minutes between alerts per ticker |
| `runner_atr_mult` | 0.5 | Runner stop = highest_high - (ATR * 0.5) |
| `runner_min_distance_pct` | 0.008 | Floor: stop never closer than 0.8% from high |

**`_calculate_real_breakeven(entry_price, qty)`:** Computes true breakeven = entry + (commissions + slippage) / (1 - tax_rate) + buffer. Prevents premature "safe zone" alerts before costs are actually covered.

**`_check_and_send_milestone_alert(symbol, position, new_stop, current_price)`:** 4-gate event-driven alert system:
- Gate 1: No alerts before real breakeven is reached
- Gate 2: First alert = "Safe Zone Reached" notification (breakeven covered)
- Gate 3: Cooldown — minimum `min_alert_interval_minutes` between alerts
- Gate 4: Stop must have moved > `min_stop_change_pct` to fire

**Impact:** Infrastructure only — methods added but not yet called. No existing behavior changed. Next: Phase 2.5b wires these into `manage_open_positions` and adds Phase 4 Runner Mode.

**Totals:** 52/52 unit tests pass, 25/25 system checks pass (no new tests — no logic changes).

---

### Phase 2.5b — Runner Mode + Milestone Alert Wiring (live_trading_engine.py)

**Context:** Step 2 of 3 — connects the Phase 2.5a infrastructure into live position management. Winning trades no longer close at `take_profit`; instead they enter Runner Mode and trail with an ultra-tight stop.

**3 changes in `live_trading_engine.py`:**

**Change 1 — `take_profit` activates Runner Mode instead of liquidating:**

```python
# Before:
elif current_price >= position["take_profit"]:
    reason = "TAKE PROFIT HIT"

# After:
elif current_price >= position["take_profit"] and not position.get("runner_mode"):
    position["runner_mode"] = True
    position["runner_activated_at"] = current_price
    position["runner_activated_time"] = time.time()
    # Sends "TARGET REACHED -- RUNNER MODE" Telegram notification
```

**Change 2 — Milestone alert called after every kinetic stop update:**

```python
self._check_and_send_milestone_alert(symbol, position, new_stop, current_price)
```

**Change 3 — Phase 4 Runner added to `LifecycleManager.manage_kinetic_stop()`:**

```
runner_stop = min(highest - ATR*0.5, highest * 0.992)
new_stop    = max(current_stop, runner_stop)
phase       = "PHASE_4_RUNNER"
```

Phase 4 checks `runner_mode` first (before Phase 3 `if`/`elif` chain). Phase 3 header changed from `if` to `elif` to maintain correct mutual exclusion.

**Impact:** Winning trades now run until the trailing stop is hit. No more hard take_profit exit cutting winners short. User receives a Telegram alert at target + subsequent alerts as stop climbs.

**Totals:** 52/52 unit tests pass, 25/25 system checks pass (no new tests — behavioral wiring, not new logic).

---

### Phase 2.5c — Tests for Runner Mode + Milestone Alerts (tests/unit_tests.py + tests/master_validator.py)

**Context:** Step 3 of 3 — adds test coverage for Phase 2.5a and 2.5b infrastructure.

**Unit tests added — `TestPhase2_5_MilestoneAlerts` (10 tests):**

| Test | Validates |
|------|-----------|
| `test_milestone_config_exists` | All 5 required keys present in `MILESTONE_ALERT_CONFIG` |
| `test_runner_min_distance_prevents_noise_exit` | `runner_min_distance_pct >= 0.005` |
| `test_calculate_real_breakeven_basic` | Breakeven > entry (costs exist) |
| `test_calculate_real_breakeven_scales_with_qty` | Small qty has higher breakeven % |
| `test_milestone_alert_no_alert_before_breakeven` | Gate 1 blocks alerts below breakeven |
| `test_milestone_alert_fires_at_breakeven` | Gate 2 fires first alert, sets `breakeven_alerted` flag |
| `test_milestone_alert_cooldown` | Gate 3 blocks alert when called within cooldown window |
| `test_take_profit_activates_runner_mode` | `runner_mode` flag set on take_profit hit |
| `test_phase4_runner_in_kinetic_stop` | `PHASE_4_RUNNER` and `runner_min_distance_pct` in source |
| `test_phase4_uses_wider_stop` | `min(runner_stop_atr, runner_stop_floor)` — wider stop chosen |

**System checks added — `check_milestone_alert_system()` (6 checks):**
`_calculate_real_breakeven` exists, `_check_and_send_milestone_alert` exists, `runner_mode` wired to `take_profit`, `PHASE_4_RUNNER` + floor present, milestone alert called after stop update, `MILESTONE_ALERT_CONFIG` has runner floor key.

**Totals: 62/62 unit tests pass, 31/31 system checks pass.**

---

## Phase 3 — Scanner Upgrade (Mandatory Templates + Priority Tiers)

### Phase 3.1a — Mandatory Scan Templates + Priority Tiers (stock_hunter.py + system_config.py)

**Context:** Upgrading the scanner to classify every stock using 4 mandatory structural templates BEFORE any trading setup analysis. Templates answer "what is the state of this stock" — not "should I buy it." Infrastructure only — not yet wired into the scan loop (next: 3.1b).

**`system_config.py` — 2 new config blocks:**

`MANDATORY_SCAN_CONFIG`: thresholds for all 4 templates (SMA slope, S/R lookback, min volume, BB width bands).

`SCAN_TIER_CONFIG`: score-based scan frequency tiers:
- Tier 1 (VIP): `master_score >= 85` — every 20 min all day
- Tier 2 (Watch): `75-84` — 3x/day at 09:30, 12:30, 15:30 (top 10)
- Tier 3 (Pool): `< 75` — morning/evening full scan only

**`stock_hunter.py` — 6 new methods added after `_save_json`:**

| Method | Returns | Logic |
|--------|---------|-------|
| `_classify_trend_direction(df)` | BULLISH / BEARISH / SIDEWAYS | SMA_50 > SMA_200, close alignment, SMA_50 slope |
| `_classify_structure(df)` | NEAR_SUPPORT / NEAR_RESISTANCE / OPEN_FIELD | Distance to 60-day high/low within 2% |
| `_classify_volume_health(df)` | HEALTHY / SURGING / DRYING_UP / ILLIQUID | 20-day avg vs 5-day avg ratio; min 500K floor |
| `_classify_volatility_state(df)` | COMPRESSED / NORMAL / VOLATILE | BB width vs 0.10 / 0.30 thresholds |
| `classify_stock_state(df)` | `{trend, structure, volume, volatility}` | Calls all 4 templates, returns state dict |
| `assign_tier(master_score)` | 1 / 2 / 3 | Score thresholds from `SCAN_TIER_CONFIG` |

No existing methods modified. Next: Phase 3.1b wires these into the scan loop.

**Totals: 62/62 unit tests pass, 31/31 system checks pass (no new tests — infrastructure only).**

---

### Phase 3.1b — Wire Templates + Tiers into Scan Loop (stock_hunter.py)

**Context:** Step 2 of 2 — connects Phase 3.1a infrastructure into the live scan loop. Every stock scanned now gets a structural state classification and a priority tier.

**Changes in `stock_hunter.py`:**

**1. `run_nightly_scan()` — ledger entry now includes state + tier:**
```python
# New step before ledger update:
stock_state = self.classify_stock_state(df_features)
tier = self.assign_tier(master_score)

# Ledger entry adds:
"state": stock_state,   # {trend, structure, volume, volatility}
"tier": tier,           # 1, 2, or 3
```

**2. `_update_daily_review_list()` — generates `tiered_watchlist.json`:**
- `tier1_vip`: all stocks with `tier == 1` (score >= 85)
- `tier2_watch`: top 10 stocks with `tier == 2` (score 75-84), sorted by master_score
- Saved to `DB_DIR/tiered_watchlist.json`

**3. Leaderboard updated** — now shows TREND column and TIER column:
```
RANK  | SYMBOL | REGIME | TREND    | TECH   | AI     | MASTER  | TIER
#1    | NVDA   | TREND  | BULLISH  | 82.5   | 76.0   | 80.6    | T1
```

**Totals: 62/62 unit tests pass, 31/31 system checks pass (no new tests — wiring only).**

---

### Phase 3.2 — Template Data Model (setup_templates.py + system_config.py)

**Context:** Templates define WHAT to look for (entry conditions), WHERE to set stops/targets, and HOW the template has performed historically. Templates are DATA (JSON), not CODE — adding/modifying templates requires no changes to strategy_engine.py.

**New file: `setup_templates.py`**

**`SetupTemplate` class:**
- Initialized from dict (loaded from JSON)
- Fields: `id`, `name`, `source` (seed/discovered), `enabled`, `required_state`, `conditions`, `entry`, `stop_loss`, `take_profit`, `statistics`
- `validate()` — checks required fields, condition operators, stop/target methods; returns `(bool, errors[])`
- `record_result(ticker, profit_pct, won)` — running average win/loss stats, per-ticker tracking
- `get_win_rate()` — returns win% from statistics block
- `to_dict()` — serializes back to JSON-ready dict

**`TemplateManager` class:**
- Loads all `*.json` from `data/templates/` on init; skips invalid files with warning
- `get_for_state(stock_state)` — filters enabled templates by `required_state` compatibility (each field is a list of acceptable values; missing fields accept any value)
- `add_template(data)` — validates before adding, saves to disk
- `save_all()` / `save_template(t)` — persist to individual JSON files
- `get_statistics_summary()` — sorted leaderboard of all templates by win rate

**`system_config.py`:** Added `TEMPLATES_DIR = os.path.join(DB_DIR, "templates")`, included in auto-mkdir loop.

Infrastructure only — no seed templates created yet. Next: Phase 3.3 (Seed Templates).

**Totals: 62/62 unit tests pass, 31/31 system checks pass (no new tests — data model only).**

---

### Phase 3.3 — Block Registry + 5 Seed Templates (setup_templates.py + data/templates/)

**Context:** "LEGO system" for trading templates. Instead of each template defining its own logic, reusable Building Blocks (functions) are referenced by name + params. New template = JSON list of block names + params only. No code changes needed.

**Block Registry added to `setup_templates.py`:**

| Category | Blocks |
|----------|--------|
| Trend (5) | `close_above_sma`, `sma_above_sma`, `close_above_ema`, `er_slow_above`, `trend_alignment` |
| Momentum (5) | `rsi_between`, `rsi_below`, `rsi_above`, `macd_above_signal`, `macd_histogram_positive` |
| Volume (2) | `volume_surge`, `rvol_above` |
| Volatility (4) | `squeeze_active`, `squeeze_momentum_positive`, `bb_width_below`, `atr_percent_above` |
| Price Action (3) | `bullish_candle`, `close_above_ref`, `close_below_ref` |
| Stop blocks (4) | `atr`, `swing_low`, `fixed_pct`, `sma` |
| Target blocks (2) | `atr`, `fixed_pct` |

**New `SetupTemplate` methods:**
- `evaluate_conditions(row)` — runs all blocks, returns `(all_passed, details[])`
- `calculate_stop_loss(row)` — dispatches to stop block registry with fallback
- `calculate_take_profit(row)` — dispatches to target block registry with fallback
- `validate()` updated to check block names against `CONDITION_BLOCKS` registry

**5 Seed Templates (`data/templates/*.json`):**

| Template | Conditions | Required State |
|----------|-----------|----------------|
| `MOMENTUM_BREAKOUT` | rsi_between + macd_above_signal + close_above_sma + volume_surge | BULLISH trend |
| `TREND_PULLBACK_EMA` | rsi_between + close_above_ema + close_above_sma + sma_above_sma | BULLISH + OPEN_FIELD/NEAR_SUPPORT |
| `SQUEEZE_BREAKOUT` | squeeze_active + squeeze_momentum_positive + bb_width_below + close_above_sma | COMPRESSED volatility |
| `VSA_INSTITUTIONAL` | volume_surge + bullish_candle + close_above_sma + rsi_between | SURGING volume |
| `OVERSOLD_BOUNCE` | rsi_below + bullish_candle + volume_surge | NEAR_SUPPORT + SIDEWAYS/BEARISH |

**Totals: 62/62 unit tests pass, 31/31 system checks pass (no new tests — data model + JSON files).**

---

### Phase 3.4 — Template Matcher (template_matcher.py)

**Context:** The bridge between stock state and trading signals. Called on each ticker every scan cycle to produce actionable BUY signals with entry/stop/target.

**New file: `template_matcher.py` — `TemplateMatcher` class**

**`scan_ticker(symbol, df, stock_state)` pipeline:**
1. Filter templates by `stock_state` (trend/structure/volume/volatility)
2. Evaluate each matching template's condition blocks against latest candle
3. Calculate `entry_price` / `stop_loss` / `take_profit` via block registry
4. Validate: stop < entry, target > entry, R:R >= `FRICTION_AND_ALPHA.min_net_rr`
5. Compute confidence score: `win_rate*0.6 + R:R_quality*0.2 + sample_size*0.2`
   (new templates with <10 trades get baseline 50 + R:R bonus instead of 0%)
6. Return signals sorted by confidence (best first)

**Signal dict keys:** `symbol`, `template_id`, `template_name`, `action`, `entry_price`, `stop_loss`, `take_profit`, `risk_reward_ratio`, `risk_pct`, `reward_pct`, `template_win_rate`, `template_total_trades`, `conditions_detail`, `stock_state`, `confidence_score`, `use_runner_mode`, `confirmation_candles`, `timestamp`

**Anti-Overflow:** `idle_tracker` per symbol counts scans without signal; logs warning at 50+ consecutive idle scans. `get_idle_report()` returns sorted idle summary.

**Verification result with synthetic BULLISH data:**
```
Templates loaded: 5 | Signals generated: 2
  MOMENTUM_BREAKOUT: Entry=$151.0, Stop=$147.25, Target=$158.5, R:R=2.0, Conf=60.0
  VSA_INSTITUTIONAL: Entry=$151.0, Stop=$145.75, Target=$158.5, R:R=1.43, Conf=57.2
```

**Totals: 62/62 unit tests pass, 31/31 system checks pass (no new tests — pipeline connector).**
