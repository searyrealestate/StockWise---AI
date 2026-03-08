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
