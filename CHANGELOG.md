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
