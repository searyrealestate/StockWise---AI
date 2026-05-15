# 07 — Config Audit

**Generated**: 2026-05-15 | **Commit**: 6bc83e8  
**Sources**: `system_config.py` (1,757 lines), `config.yaml` (539 chars)

---

## system_config.py — Top-Level Constants (50+)

| Key | Type / Value | File Source | Purpose |
|-----|-------------|-------------|---------|
| `AI_LABEL_CONFIG` | dict | system_config | AI labeling thresholds |
| `ALPACA_BASE_URL` | str | system_config | Alpaca paper trading URL |
| `ALPACA_KEY` | str | env/.env | Alpaca API key |
| `ALPACA_SECRET` | str | env/.env | Alpaca API secret |
| `ANALYTICS_CONFIG` | dict | system_config | Analytics settings |
| `API_TIMEOUTS` | dict | system_config | Per-provider timeouts |
| `ASSET_SPECIFIC_CONFIG` | dict | system_config | Per-symbol overrides |
| `BASE_DIR` | path | system_config | Project root path |
| `BASE_FRICTION` | 0.003 | system_config | Base friction cost |
| `BENCHMARK_TICKER` | str | system_config | SPY (benchmark comparison) |
| `COOLDOWN_FILE_PATH` | path | system_config | Path to cooldown state JSON |
| `COOLDOWN_PERIOD_HOURS` | 24 | system_config | Veto cooldown in hours |
| `COSTS_CONFIG` | dict | system_config | Commission, slippage, tax |
| `DATA_DIR` | path | system_config | Data storage directory |
| `DATA_END_DATE` | date | system_config | Backtest end date |
| `DATA_START_DATE` | date | system_config | Backtest start date |
| `DATA_STARVATION_COOLDOWN_MINUTES` | 120 | system_config | Data gap cooldown |
| `DB_DIR` | path | system_config | Database directory |
| `DEFAULT_TRAINING_SYMBOLS` | list | system_config | Default symbol universe |
| `DISCOVERY_CONFIG` | dict | system_config | Template discovery settings |
| `DISCRIMINATION_BUILDER_CONFIG` | dict | system_config | Discrimination builder params |
| `DSP_CONFIG` | dict | system_config | DSP indicator parameters |
| `EN_ALPACA` | bool | system_config | Alpaca provider enabled |
| `EN_IBKR` | bool | system_config | IBKR provider enabled |
| `EN_MASSIVE` | bool | system_config | Massive API enabled |
| `EN_YFINANCE` | bool | system_config | yfinance provider enabled |
| `FEATURE_CONFIG` | dict | system_config | Feature engine params |
| `FILTER_USAGE_CONFIG` | dict | system_config | Block tracking config |
| `FRICTION_AND_ALPHA` | dict | system_config | Friction + alpha thresholds |
| `HISTORICAL_SOURCE` | str | system_config | Primary data provider |
| `IBKR_CLIENT_ID` | int | system_config | IBKR connection client ID |
| `IBKR_HOST` | str | system_config | IBKR host (127.0.0.1) |
| `IBKR_PORT` | 4001 | system_config | IBKR paper trading port |
| `INDICATOR_PARAMS` | dict | system_config | Technical indicator params |
| `INVESTMENT_AMOUNT` | 1000 | system_config | Per-trade capital |
| `KINETIC_STOP_CONFIG` | dict | system_config | Trailing stop config |
| `LEDGER_PATH` | path | system_config | Shadow ledger JSON path |
| `LOG_DIR_LOCAL` | path | system_config | Local log directory |
| `LOG_FILE_PATH` | path | system_config | Main log file path |
| `LOG_LEVEL` | str | system_config | Overall log level |
| `MAX_MELTING_PERIOD_DAYS` | 7 | system_config | Max decay period |
| `MAX_TEMPLATES` | 5 | system_config | Max templates per symbol/state |
| `MILESTONE_ALERT_CONFIG` | dict | system_config | Alert threshold config |
| `MIN_CANDLES_FOR_PROCESSING` | 200 | system_config | Minimum bars for features |
| `MIN_MASTER_SCORE_APPROVAL` | 65.0 | system_config | Trade approval score floor |
| `MIN_NET_PROFIT` | 0.005 | system_config | Minimum net profit threshold |
| `PAUSE_MECHANISM_CONFIG` | dict | system_config | Kinetic pause config |
| `PREMIUM_TRADE_THRESHOLD` | 75.0 | system_config | Premium signal score |
| `QUALITY_GATE_CONFIG` | dict | system_config | QG thresholds |
| `RISK_GATE_CONFIG` | dict | system_config | Risk gate params |
| `SHADOW_LEDGER_CONFIG` | dict | system_config | Shadow ledger behavior |
| `SUIT_CONFIG` | dict | system_config | Suit assignment params |
| `TEMPLATE_EVOLUTION_CONFIG` | dict | system_config | Template lifecycle params |
| `VETO_COOLDOWN_MINUTES` | 30 | system_config | Veto repeat suppression |

---

## config.yaml — Authentication Only

| Key | Value | Purpose |
|-----|-------|---------|
| `credentials.usernames.emesika.email` | eyalmesika@gmail.com | Streamlit auth user 1 |
| `credentials.usernames.emesika.name` | Eyal Mesika | Display name |
| `credentials.usernames.nroda.email` | rodanis@gmail.com | Streamlit auth user 2 |
| `cookie.expiry_days` | 30 | Auth cookie lifetime |
| `cookie.key` | `a_very_secret_key_12345` | **⚠️ Hardcoded secret in file** |
| `cookie.name` | `stockwise_auth_cookie` | Cookie identifier |

---

## Potential Issues

### Orphan Keys (defined in system_config but referenced nowhere in core modules)
- `EN_ORCHESTRAL` — No reference found in scanned core modules
- `MASSIVE_API_KEY` — Only in data_source_manager.py
- `AI_LABEL_CONFIG` — Only in train_model.py and master_validator.py

### Dual-Path Config Access
Modules access system_config via multiple patterns:
- `import system_config as cfg` → `cfg.KEY` (feature_engine, live_trading_engine)
- `import system_config` → `system_config.KEY` (most modules)
- `from system_config import KEY` (some modules)

This inconsistency means grep for `system_config.X` will miss `cfg.X` references.

### DEPRECATED Key in system_config.py
- `system_config.py:413` — `runner_min_distance_pct` in `KINETIC_STOP_CONFIG` marked `# DEPRECATED — canonical source is now KINETIC_STOP_CONFIG`
  - The key exists twice in the same config section with the same value (0.008). One is labeled deprecated.

### TODO in system_config.py
- `system_config.py:1746` — `# TODO: migrate to safe_json_io (needs ensure_ascii=False support not in safe_json_write)`
  - One location still uses raw `json.dump` instead of `safe_json_write`

### INVESTMENT_AMOUNT vs initial_capital Discrepancy
- `system_config.INVESTMENT_AMOUNT = 1000` (per-trade capital)
- `backtest_engine.py:71` BACKTEST_CONFIG `"initial_capital": 100_000` (backtest portfolio capital)
- These are different concepts but same-sounding names — not a bug but a clarity issue.
