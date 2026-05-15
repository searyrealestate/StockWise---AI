# 05 — Templates Inventory

**Generated**: 2026-05-15 | **Commit**: 6bc83e8  
**Source**: `data/templates/*.json` (23 files)  
**Note**: `active` field is `MISSING` in all JSON files — the `TemplateManager.load_all()` reads this field but it defaults per code logic. Active state is managed at runtime via `disabled_reason` field and in-memory flags.

---

## Template Inventory

| Template ID | Timeframe | Trend States | Conditions Count | Source | Notes |
|-------------|-----------|-------------|-----------------|--------|-------|
| BEARISH_VOLATILITY_EXPANSION | 1d | BEARISH | 2 | discrimination_v3_auto | atr_pct > 4.95% |
| DISC_BEARISH_COMPRESSED_10D | 1d | BEARISH | 2 | discrimination_v3_auto | BEARISH + COMPRESSED vol |
| DISC_BEARISH_VOLATILE_5D | 1d | BEARISH | 2 | discrimination_v3_auto | BEARISH + VOLATILE vol |
| DISC_BULLISH_COMPRESSED_10D | 1d | BULLISH | 2 | discrimination_v3_auto | BULLISH + COMPRESSED vol |
| DISC_SIDEWAYS_COMPRESSED_10D | 1d | SIDEWAYS | 2 | discrimination_v3_auto | PF=1.59 per memory |
| DISC_SIDEWAYS_NORMAL_5D | 1d | SIDEWAYS | 2 | discrimination_v3_auto | PF=1.66 per memory |
| GEN_2H_BEARISH_BOUNCE | 2h | BEARISH | 3 | recipe_generator | 2h intraday bounce |
| GEN_2H_BREAKOUT_VOLUME | 2h | BULLISH, SIDEWAYS | 4 | recipe_generator | 2h volume breakout |
| GEN_2H_BULLISH_TREND_RIDE | 2h | BULLISH | 4 | recipe_generator | 2h trend following |
| GEN_2H_SIDEWAYS_ACCUMULATION | 2h | SIDEWAYS | 4 | recipe_generator | 2h accumulation |
| GEN_2H_SIDEWAYS_VOLATILE_ACCUMULATION | 2h | SIDEWAYS | 4 | recipe_generator | 2h volatile sideways |
| GEN_2H_SIDEWAYS_VOLATILE_BREAKOUT | 2h | SIDEWAYS | 4 | recipe_generator | 2h volatile breakout |
| GEN_BEARISH_SQUEEZE_BREAK | 1d | BEARISH, SIDEWAYS | 4 | recipe_generator | QG_TEST_PERIOD_FAIL per memory |
| GEN_BULLISH_BREAKOUT_VOLUME | 1d | BULLISH | 4 | recipe_generator | Daily vol breakout |
| GEN_BULLISH_TREND_RIDE | 1d | BULLISH | 4 | recipe_generator | All daily BULLISH disabled per memory |
| GEN_SIDEWAYS_ACCUMULATION | 1d | SIDEWAYS | 4 | recipe_generator | Disabled: overfitting PF=0.61 full data |
| GEN_SIDEWAYS_BREAKOUT_CLEAN | 1d | SIDEWAYS | 4 | recipe_generator | |
| GEN_TREND_EXHAUSTION_BOUNCE | 1d | BEARISH | 3 | recipe_generator | Disabled: PF=0.83 per memory |
| MOMENTUM_BREAKOUT | 1d | BULLISH | 4 | recipe_generator | All daily BULLISH disabled per memory |
| OVERSOLD_BOUNCE | 1d | SIDEWAYS, BEARISH | 3 | recipe_generator | |
| SQUEEZE_BREAKOUT | 1d | BULLISH, SIDEWAYS | 4 | recipe_generator | |
| TREND_PULLBACK_EMA | 1d | BULLISH | 4 | recipe_generator | All daily BULLISH disabled per memory |
| VSA_INSTITUTIONAL | 1d | BULLISH, SIDEWAYS | 4 | recipe_generator | Volume spread analysis |

---

## Template Categories

| Category | Count | Template IDs |
|----------|-------|-------------|
| Daily (1d) DISC (data-driven) | 6 | BEARISH_VOLATILITY_EXPANSION, DISC_BEARISH_COMPRESSED_10D, DISC_BEARISH_VOLATILE_5D, DISC_BULLISH_COMPRESSED_10D, DISC_SIDEWAYS_COMPRESSED_10D, DISC_SIDEWAYS_NORMAL_5D |
| Daily (1d) GEN (recipe-based) | 11 | GEN_BEARISH_SQUEEZE_BREAK, GEN_BULLISH_BREAKOUT_VOLUME, GEN_BULLISH_TREND_RIDE, GEN_SIDEWAYS_ACCUMULATION, GEN_SIDEWAYS_BREAKOUT_CLEAN, GEN_TREND_EXHAUSTION_BOUNCE, MOMENTUM_BREAKOUT, OVERSOLD_BOUNCE, SQUEEZE_BREAKOUT, TREND_PULLBACK_EMA, VSA_INSTITUTIONAL |
| 2-Hour (2h) | 6 | GEN_2H_BEARISH_BOUNCE, GEN_2H_BREAKOUT_VOLUME, GEN_2H_BULLISH_TREND_RIDE, GEN_2H_SIDEWAYS_ACCUMULATION, GEN_2H_SIDEWAYS_VOLATILE_ACCUMULATION, GEN_2H_SIDEWAYS_VOLATILE_BREAKOUT |

---

## State Coverage Map

| Trend State | Templates Available | Templates Known Disabled |
|-------------|--------------------|-----------------------|
| BULLISH (1d) | 6 | 3+ (per memory: all daily BULLISH disabled) |
| BEARISH (1d) | 5 | GEN_TREND_EXHAUSTION_BOUNCE (PF=0.83) |
| SIDEWAYS (1d) | 6 | GEN_SIDEWAYS_ACCUMULATION (overfitting) |
| BULLISH (2h) | 2 | 0 known |
| BEARISH (2h) | 1 | 0 known |
| SIDEWAYS (2h) | 3 | 0 known |

---

## Known Disabled Templates (from memory.md)

| Template | Reason | Source |
|----------|--------|--------|
| GEN_SIDEWAYS_ACCUMULATION | PF=2.65 training, PF=0.61 full data. Overfitting confirmed. | memory.md 2026-04-11 |
| GEN_TREND_EXHAUSTION_BOUNCE | PF=0.83 on 5 trades. Losing money. | memory.md 2026-04-11 |
| GEN_BEARISH_SQUEEZE_BREAK | QG_TEST_PERIOD_FAIL (WR=18.8% on TEST period) | memory.md 2026-04-08 |
| All daily BULLISH templates | 0 active coverage for BULLISH daily state | memory.md 2026-04-10 |

**Note**: `active` field is `MISSING` from all JSON template files inspected. Runtime active/disabled status is managed by `TemplateManager` which reads a separate `disabled_reason` field. The JSON files do not have an explicit `"active": true/false` boolean — this is an observation, not a recommendation.

---

## Block Functions Available (31 condition blocks + 6 stop/target)

See `03_function_signatures.md` for full list. Block functions defined in `setup_templates.py` lines 1–600 (estimated).

---

## Permanently Banned Templates

Search for "banned" or "do not use" in codebase: **No permanently banned templates found** via grep. Disabled templates are managed via `disabled_reason` field and `TemplateHealthMonitor`.
