# 02 — Dependency Graph

**Generated**: 2026-05-15 | **Commit**: 6bc83e8

Local-only imports (production modules importing each other). External deps listed separately.

---

## system_config.py
- **Imports (local)**: `safe_json_io`
- **Imported by**: ALL 24 other core modules (god module)
- **External deps**: `os`, `json`, `logging`, `dotenv`, `pandas`, `datetime`, `re`, `io`
- **Data flow**: Reads `.env` + `watchlist.json` → exposes CONFIG dict and logger setup

## safe_json_io.py
- **Imports (local)**: *(none)*
- **Imported by**: system_config, backtest_engine, template_matcher, setup_templates, shadow_ledger, strategy_engine, live_trading_engine, portfolio_manager, notification_manager, stock_hunter, validation_report, master_validator, train_model — **13 modules**
- **External deps**: `json`, `logging`, `os`, `tempfile`, `time`
- **Data flow**: Wraps json.load/dump with atomic file write via tempfile; returns data dicts

## feature_engine.py
- **Imports (local)**: `decision_logger`, `system_config`
- **Imported by**: backtest_engine, data_source_manager, shadow_ledger, strategy_engine, live_trading_engine, data_engineer, stock_hunter, validation_runner, template_discovery, dag_optimizer, master_validator, train_model — **12 modules**
- **External deps**: `pandas`, `numpy`, `pandas_ta`, `pandas_ta_classic`, `scipy`
- **Data flow**: Receives OHLCV DataFrame → returns DataFrame with 60+ indicator columns

## data_source_manager.py
- **Imports (local)**: `feature_engine`, `system_config`
- **Imported by**: backtest_engine, shadow_ledger, live_trading_engine, data_engineer, stock_hunter, validation_runner, template_discovery, dag_optimizer, master_validator, train_model — **10 modules**
- **External deps**: `ibapi`, `alpaca_trade_api`, `yfinance`, `asyncio`, `concurrent`, `massive`
- **Data flow**: Fetches OHLCV from IBKR/Alpaca/yfinance/Massive → normalizes → calls feature_engine → returns enriched DataFrame

## decision_logger.py
- **Imports (local)**: `system_config`
- **Imported by**: feature_engine, template_matcher, strategy_engine, portfolio_risk, pre_market_validator — **5 modules**
- **External deps**: `datetime`, `json`, `logging`, `os`, `time`
- **Data flow**: Receives signal/veto/risk/execution events → writes structured CSV rows

## setup_templates.py
- **Imports (local)**: `safe_json_io`, `system_config`
- **Imported by**: backtest_engine, template_matcher, shadow_ledger, live_trading_engine, template_discovery — **5 modules**
- **External deps**: `datetime`, `json`, `logging`, `math`, `os`
- **Data flow**: Reads template JSON from `data/templates/` → exposes TemplateManager; block functions evaluate indicator rows

## template_matcher.py
- **Imports (local)**: `decision_logger`, `safe_json_io`, `setup_templates`, `shadow_ledger`, `system_config`
- **Imported by**: backtest_engine, shadow_ledger, live_trading_engine — **3 modules**
- **External deps**: `datetime`, `hashlib`, `logging`, `math`
- **Data flow**: Receives ticker DataFrame + stock_state → evaluates block conditions → returns signal dict with trust/suit scores

## shadow_ledger.py
- **Imports (local)**: `data_source_manager`, `feature_engine`, `safe_json_io`, `setup_templates`, `stock_hunter`, `system_config`, `template_matcher`
- **Imported by**: backtest_engine, template_matcher, validation_runner — **3 modules**
- **External deps**: `pandas`, `math`, `logging`, `argparse`, `datetime`
- **Data flow**: Replays historical data per template → records signal outcomes → updates `shadow_ledger.json`

## strategy_engine.py
- **Imports (local)**: `decision_logger`, `feature_engine`, `safe_json_io`, `system_config`
- **Imported by**: live_trading_engine, stock_hunter, master_validator — **3 modules**
- **External deps**: `joblib`, `json`, `numpy`, `os`, `pandas`
- **Data flow**: Receives feature DataFrame → classifies regime → scores AI probability → returns action ticket

## backtest_engine.py
- **Imports (local)**: `data_source_manager`, `feature_engine`, `portfolio_risk`, `safe_json_io`, `setup_templates`, `shadow_ledger`, `stock_hunter`, `system_config`, `template_matcher`, `versioned_save`
- **Imported by**: validation_runner — **1 module**
- **External deps**: `argparse`, `collections`, `datetime`, `hashlib`, `json`, `logging`, `pandas`, `numpy`
- **Data flow**: Orchestrates full backtest; calls data_source_manager → feature_engine → template_matcher → portfolio_risk → records to JSON

## live_trading_engine.py
- **Imports (local)**: `data_source_manager`, `feature_engine`, `notification_manager`, `portfolio_risk`, `pre_market_validator`, `safe_json_io`, `setup_templates`, `stock_hunter`, `strategy_engine`, `system_config`, `template_matcher`, `train_model`
- **Imported by**: master_validator — **1 module**
- **External deps**: `asyncio`, `csv`, `datetime`, `json`, `logging`, `argparse`
- **Data flow**: Fetches live data → runs signal pipeline → executes trades via Alpaca/IBKR; manages open positions

## portfolio_risk.py
- **Imports (local)**: `decision_logger`, `system_config`
- **Imported by**: backtest_engine, live_trading_engine, validation_runner — **3 modules**
- **External deps**: `datetime`, `logging`, `pandas`
- **Data flow**: Receives proposed trade + open positions → runs 4 risk gates → returns PASS/VETO

## portfolio_manager.py
- **Imports (local)**: `safe_json_io`, `system_config`
- **Imported by**: master_validator — **1 module**
- **External deps**: `datetime`, `json`, `logging`, `numpy`, `os`, `pandas`
- **Data flow**: Tracks portfolio state (positions, cash) → calculates commissions/slippage

## stock_hunter.py
- **Imports (local)**: `data_source_manager`, `feature_engine`, `safe_json_io`, `strategy_engine`, `system_config`
- **Imported by**: backtest_engine, shadow_ledger, live_trading_engine, master_validator — **4 modules**
- **External deps**: `json`, `logging`, `numpy`, `os`, `pandas`
- **Data flow**: Receives symbol list → classifies each symbol's market state → returns VIP candidate list

## notification_manager.py
- **Imports (local)**: `safe_json_io`, `system_config`
- **Imported by**: live_trading_engine, master_validator — **2 modules**
- **External deps**: `csv`, `datetime`, `json`, `logging`, `os`, `requests`
- **Data flow**: Receives alert message → deduplicates → sends via Telegram/email

## market_intelligence.py
- **Imports (local)**: `system_config`
- **Imported by**: master_validator — **1 module**
- **External deps**: `datetime`, `logging`, `pandas`, `textblob`, `yfinance`
- **Data flow**: Fetches news + fundamentals → returns sentiment score + Graham number

## train_model.py
- **Imports (local)**: `data_source_manager`, `feature_engine`, `safe_json_io`, `system_config`
- **Imported by**: live_trading_engine, master_validator — **2 modules**
- **External deps**: `joblib`, `json`, `logging`, `numpy`, `os`, `pandas`, `sklearn`
- **Data flow**: Fetches + features data → trains regime classifier → saves model with joblib

## pre_market_validator.py
- **Imports (local)**: `decision_logger`, `system_config`
- **Imported by**: live_trading_engine — **1 module**
- **External deps**: `datetime`, `logging`, `pytz`
- **Data flow**: Checks current time against market hours → returns boolean gate result

## validation_runner.py
- **Imports (local)**: `backtest_engine`, `data_source_manager`, `feature_engine`, `portfolio_risk`, `shadow_ledger`, `system_config`, `versioned_save`
- **Imported by**: *(none — leaf/entry point)*
- **External deps**: `argparse`, `copy`, `datetime`, `json`, `logging`
- **Data flow**: Configures + runs backtest via BacktestEngine → compares metrics → writes versioned results

## validation_report.py
- **Imports (local)**: `safe_json_io`, `versioned_save`
- **Imported by**: *(none — leaf/entry point)*
- **External deps**: `argparse`, `datetime`, `docx`, `os`, `sys`
- **Data flow**: Reads backtest JSON results → generates Word document report

## template_discovery.py
- **Imports (local)**: `data_source_manager`, `feature_engine`, `setup_templates`, `system_config`
- **Imported by**: *(none — leaf/entry point)*
- **External deps**: `datetime`, `itertools`, `json`, `logging`, `numpy`, `os`
- **Data flow**: Brute-forces indicator combos → tests against historical data → suggests new templates

## dag_optimizer.py
- **Imports (local)**: `data_source_manager`, `feature_engine`, `system_config`
- **Imported by**: *(none — leaf/entry point)*
- **External deps**: `datetime`, `json`, `logging`, `numpy`, `os`, `pandas`
- **Data flow**: Computes mutual information between feature pairs → outputs indicator DAG

## data_engineer.py
- **Imports (local)**: `data_source_manager`, `feature_engine`, `system_config`
- **Imported by**: *(none — leaf/entry point)*
- **External deps**: `datetime`, `logging`, `numpy`, `os`, `pandas`
- **Data flow**: Batch-fetches all watchlist symbols → enriches with features → caches to disk

## master_validator.py
- **Imports (local)**: `data_source_manager`, `feature_engine`, `live_trading_engine`, `market_intelligence`, `notification_manager`, `portfolio_manager`, `safe_json_io`, `stock_hunter`, `strategy_engine`, `system_config`, `train_model`
- **Imported by**: *(none — leaf/entry point)*
- **External deps**: `ast`, `asyncio`, `colorama`, `csv`, `datetime`, `glob`, `unittest`
- **Data flow**: Test suite that instantiates all major components and runs integration/unit tests

## versioned_save.py
- **Imports (local)**: *(none)*
- **Imported by**: backtest_engine, validation_runner, validation_report — **3 modules**
- **External deps**: `datetime`, `logging`, `os`, `shutil`, `subprocess`
- **Data flow**: Before overwriting a file, copies current version to `.bak/` with timestamp

---

## Structural Analysis

### Circular Dependencies
**None detected.** Import graph is a DAG. Confirmed by tracing all paths — no module imports another that (directly or transitively) imports it back.

### God Modules (imported by 10+ files)
| Module | Imported By |
|--------|-------------|
| `system_config` | 21 modules |
| `safe_json_io` | 13 modules |
| `feature_engine` | 12 modules |
| `data_source_manager` | 10 modules |

### Leaf Modules (imported by nobody — entry points or dead)
| Module | Status |
|--------|--------|
| `validation_runner` | Entry point (CLI) |
| `validation_report` | Entry point (CLI) |
| `template_discovery` | Entry point (CLI/research) |
| `dag_optimizer` | Entry point (CLI/research) |
| `data_engineer` | Entry point (CLI) |
| `master_validator` | Entry point (test runner) |
