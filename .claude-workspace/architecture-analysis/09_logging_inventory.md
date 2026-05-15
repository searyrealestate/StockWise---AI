# 09 — Logging Inventory

**Generated**: 2026-05-15 | **Commit**: 6bc83e8  
**Methodology**: `re.findall` for `.debug(`, `.info(`, `.warning(`, `.error(` patterns.

---

## Per-Module Log Statement Counts

| File | debug | info | warning | error | Total |
|------|-------|------|---------|-------|-------|
| backtest_engine.py | 7 | 79 | 17 | 8 | **111** 🔴 Noisy |
| setup_templates.py | 12 | 29 | 12 | 6 | **59** |
| live_trading_engine.py | 11 | 34 | 9 | 13 | **67** 🔴 Noisy |
| shadow_ledger.py | 10 | 12 | 19 | 5 | **46** |
| strategy_engine.py | 28 | 6 | 2 | 4 | **40** |
| validation_runner.py | 0 | 28 | 13 | 0 | **41** |
| template_matcher.py | 8 | 14 | 5 | 5 | **32** |
| feature_engine.py | 9 | 6 | 5 | 9 | **29** |
| template_discovery.py | 2 | 21 | 1 | 3 | **27** |
| train_model.py | 5 | 11 | 5 | 5 | **26** |
| dag_optimizer.py | 6 | 5 | 1 | 4 | **16** |
| notification_manager.py | 2 | 9 | 1 | 4 | **16** |
| portfolio_manager.py | 1 | 8 | 4 | 2 | **15** |
| market_intelligence.py | 2 | 4 | 4 | 3 | **13** |
| data_engineer.py | 0 | 8 | 2 | 2 | **12** |
| portfolio_risk.py | 5 | 1 | 2 | 0 | **8** |
| pre_market_validator.py | 3 | 0 | 1 | 0 | **4** |
| system_config.py | 0 | 2 | 1 | 2 | **5** |
| safe_json_io.py | 1 | 0 | 1 | 2 | **4** |
| versioned_save.py | 0 | 1 | 2 | 0 | **3** |
| decision_logger.py | 0 | 0 | 1 | 1 | **2** |
| data_source_manager.py | 4 | 6 | 4 | 6 | **20** |
| stock_hunter.py | 13 | 19 | 6 | 4 | **42** |
| validation_report.py | 0 | 0 | 0 | 0 | **0** 🔵 Silent |
| master_validator.py | 0 | 0 | 0 | 0 | **0** 🔵 Silent |

---

## Analysis

### Silent Modules (zero log statements)

| Module | Lines | Impact |
|--------|-------|--------|
| validation_report.py | 589 | No audit trail for report generation |
| master_validator.py | 3,043 | Test suite uses `print()` instead of logging |

**Note**: `master_validator.py` uses `print()` statements via `ColorfulTestResult` (colorama). Not captured by logger grep. Not considered a production issue — it is a test runner.

### Noisy Modules (>50 log statements)

| Module | Total | Primary Type | Notes |
|--------|-------|-------------|-------|
| backtest_engine.py | 111 | info (79) | Extensive trade lifecycle logging — expected for simulation engine |
| live_trading_engine.py | 67 | info (34) + error (13) | High error count reflects live trading risk-awareness |
| setup_templates.py | 59 | info (29) | Template lifecycle + block evaluation logging |

### Notable Logging Patterns

- `strategy_engine.py`: Heavy on `.debug(28)` — highest debug-to-info ratio. Debug logs likely suppressed in production.
- `shadow_ledger.py`: 19 warnings — highest warning count. Reflects trust lifecycle edge cases.
- `validation_runner.py`: 0 debug, 28 info — clean production-style logging.
- `decision_logger.py`: Only 2 log statements in a 220-line module that IS a logger. Uses CSV writes directly instead.

### Log Format Standards

Structured log events confirmed in `data_source_manager.py`:
- `IBKR_FETCH_START` / `IBKR_FETCH_OK` / `IBKR_FETCH_FAIL` format (per memory.md)
- These use `logger.info(...)` with dict context

### Logger Setup

All loggers instantiated via `system_config.LoggerSetup.setup_logger(name, log_file, level)`.  
Log files written to `system_config.LOG_DIR_LOCAL`.  
`decision_logger.py` writes to separate CSV files (not the main log).
