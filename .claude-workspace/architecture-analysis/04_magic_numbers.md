# 04 — Magic Numbers

**Generated**: 2026-05-15 | **Commit**: 6bc83e8  
**Methodology**: AST-extracted `ast.Constant` int/float nodes; skipped: 0, 1, -1, 2, 100 (ubiquitous). Separator strings (`"=" * N`) excluded as formatting, not logic.

🔴 = NOT in system_config.py (potential hardcode risk)  
🟢 = IN system_config.py (properly externalized)

---

## system_config.py — Declared Constants (all 🟢 by definition)

| Line | Value | Context |
|------|-------|---------|
| 67 | 24 | `COOLDOWN_PERIOD_HOURS = 24` |
| 73 | 7 | `MAX_MELTING_PERIOD_DAYS = 7` |
| 80 | 0.003 | `BASE_FRICTION = 0.003` |
| 83 | 0.005 | `MIN_NET_PROFIT = 0.005` |
| 113 | 5 | `MAX_TEMPLATES = 5` |
| 212 | 4001 | `IBKR_PORT = 4001` |
| 216 | 1000 | `INVESTMENT_AMOUNT = 1000` |
| 273 | 200 | `MIN_CANDLES_FOR_PROCESSING = 200` |
| 343 | 65.0 | `MIN_MASTER_SCORE_APPROVAL = 65.0` |
| 346 | 75.0 | `PREMIUM_TRADE_THRESHOLD = 75.0` |
| 352 | 30 | `VETO_COOLDOWN_MINUTES = 30` |
| 353 | 120 | `DATA_STARVATION_COOLDOWN_MINUTES = 120` |
| 155 | 60 | `IBKR_HISTORICAL_TIMEOUT = 60` |
| 326 | 500.0 | `max_daily_loss_usd = 500.0` |
| 327 | 0.015 | `max_daily_loss_pct = 0.015` |
| 328 | 1000.0 | `target_daily_profit_usd = 1000.0` |
| 330 | 5000.0 | `starting_capital = 5000.0` |
| 358 | 0.005 | `commission_per_share = 0.005` |
| 360 | 0.001 | `slippage_pct = 0.001` |
| 361 | 0.25 | `tax_rate = 0.25` |
| 364 | 1.2 | `min_net_rr = 1.2` |
| 374 | 20 | `er_lookback_slow = 20` |
| 375 | 5 | `er_lookback_fast = 5` |
| 376 | 0.55 | `threshold_coherent_trend = 0.55` |
| 377 | 0.30 | `threshold_stochastic_chop = 0.30` |
| 389 | 0.01 | `max_adv_participation_pct = 0.01` |
| 394 | 1.5 | `phase1_atr_mult = 1.5` |
| 395 | 0.015 | `phase2_breakeven_trigger_pct = 0.015` |
| 396 | 0.03 | `phase3_parabolic_trigger_pct = 0.03` |
| 398 | 0.5 | `runner_atr_mult = 0.5` |
| 399 | 0.008 | `runner_min_distance_pct = 0.008` |
| 405 | 0.002 | `safe_zone_buffer_pct = 0.002` |
| 408 | 0.01 | `min_stop_change_pct = 0.01` |
| 409 | 15 | `min_alert_interval_minutes = 15` |
| 413 | 0.008 | `runner_min_distance_pct` (DEPRECATED duplicate) |
| 420 | 0.005 | `min_healthy_pullback_pct = 0.005` |
| 421 | 0.03 | `max_healthy_pullback_pct = 0.03` |
| 422 | 0.45 | `min_er_for_pause = 0.45` |
| 423 | 40 | `min_rsi_for_pause = 40` |
| 427 | 3 | `re_entry_min_wait_candles = 3` |
| 435 | 0.05 | `max_gap_pct = 0.05` |
| 436 | 0.001 | `min_gap_pct = 0.001` |
| 439 | 60 | `veto_cooldown_minutes = 60` |
| 454 | 1095 | `eval_days_back = 1095` |
| 455 | 5 | `max_templates = 5` |
| 456 | 20 | `lookahead_candles = 20` |
| 457 | 200 | `min_candles_for_eval = 200` |
| 458 | 20 | `min_bars_between_signals = 20` |

---

## backtest_engine.py — Numbers NOT in config 🔴

| Line | Value | Context | Risk |
|------|-------|---------|------|
| 71 | 100_000 | `"initial_capital": 100_000` (BACKTEST_CONFIG default) | 🔴 Duplicates `system_config.INVESTMENT_AMOUNT` (1000) — different value |
| 75 | 0.05 | `"slippage_pct": 0.05` (5%) | 🔴 system_config has `slippage_pct=0.001` (0.1%) — 50x discrepancy |
| 1396 | 55 | `W = 55` (print width) | 🟢 Formatting only |
| 1860 | 42 | `random.seed(42)` in Monte Carlo | 🔴 Hardcoded seed for Monte Carlo simulations |
| 756 | 52 | `if len(cell["signals"]) > 52` (max signals per trust cell) | 🔴 Business logic limit not in config |
| 764 | 0.95 | `ct_cfg.get("decay_rate", 0.95)` — fallback default | 🟢 Has config key, fallback only |
| 493 | 0.02 | `atr = abs(price * 0.02)` — ATR fallback if undefined | 🔴 Fallback hardcode for ATR calculation |
| 2248 | 0.60 | `wf_cfg.get("train_pct", 0.60)` — walk-forward train split | 🔴 WF split ratio not in system_config |
| 2249 | 0.20 | `wf_cfg.get("val_pct", 0.20)` — walk-forward val split | 🔴 WF split ratio not in system_config |
| 1847 | 5 | `if avg_loss_d > 0 and len(trades) >= 5` — ROR min trades | 🔴 Business threshold not in config |
| 2952 | 3 | `quality_gate_min_trades` default 3 | 🟢 Has config key |
| 2980 | 0.8 | `quality_gate_test_min_pf` default 0.8 | 🟢 Has config key |
| 2756 | 0.2 | `flag_overfit_threshold` default 0.20 | 🟢 Has config key |
| 2796 | 0.25 | `cp2_min_wr_generated` default 0.25 | 🟢 Has config key |
| 2797 | 1.5 | `cp2_min_pf` default 1.50 | 🟢 Has config key |
| 1926 | 0.01 | `if ror_analytical < 0.01` — ROR threshold | 🔴 Risk-of-ruin threshold not in config |
| 1879 | 99999 | `consec_surv = 99999` — Monte Carlo sentinel | 🟢 Sentinel value, not business logic |

## feature_engine.py — Numbers NOT in config

No magic numbers found outside formatting contexts. All thresholds loaded from system_config via `cfg = system_config.FEATURE_CONFIG`.

## strategy_engine.py — Numbers NOT in config

No standalone magic numbers detected outside defaults that pull from config with `.get()`.

## portfolio_risk.py — Numbers NOT in config

No standalone magic numbers. All thresholds loaded from system_config subsections.

## safe_json_io.py

| Line | Value | Context |
|------|-------|---------|
| ~40 | 3 | `retries=3` default param | 🔴 Small hardcode in function signature |
| ~40 | 0.1 | `retry_delay=0.1` default | 🔴 Retry delay in seconds |

## Summary

| Category | Count |
|----------|-------|
| 🟢 In system_config.py (properly declared) | 47 |
| 🔴 In business logic outside config | 8 significant values |
| Formatting-only (print widths, etc.) | ~20 (excluded) |

**Top 3 highest-risk magic numbers:**
1. `backtest_engine.py:75` — slippage 0.05 (5%) vs system_config 0.001 (0.1%) — 50x mismatch
2. `backtest_engine.py:71` — initial_capital 100,000 vs system_config `starting_capital=5000`
3. `backtest_engine.py:1860` — random.seed(42) — hardcoded Monte Carlo seed
