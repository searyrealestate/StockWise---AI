# 01 — Module Inventory

**Generated**: 2026-05-15 | **Commit**: 6bc83e8 | **Scope**: 25 core .py files (root + backtest/)

---

## Core Production Modules

| File | Lines | Public Classes | Public Fn Count | Responsibility |
|------|-------|---------------|-----------------|----------------|
| system_config.py | 1,757 | 3 (EmojiFilter, LoggerSetup, SystemActionLogger) | 10 | Central config: all constants, logger setup, dynamic watchlist |
| backtest_engine.py | 3,251 | 3 (Position, BacktestEngine, WalkForwardValidator) | 6 | Portfolio backtest simulation, walk-forward validation, deterministic mode |
| setup_templates.py | 2,424 | 6 (SetupTemplate, TemplateManager, TemplateGenerator, DiscriminationTemplateBuilder, TemplateHealthMonitor, TradeOutcomeAnalyzer) | 65 | Template definition, block functions, template lifecycle, auto-generation |
| shadow_ledger.py | 2,070 | 1 (ShadowLedger) | 6 | Historical signal replay, trust/suit learning, lifecycle tracking |
| master_validator.py | 3,043 | 8 (StockWiseMasterValidator, TestGen12Performance, ColorfulTestResult, ColorfulTestRunner, TestGen12Acceptance + 3 test classes) | 187 | Full system validation test suite (unittest-based) |
| template_matcher.py | 1,461 | 2 (FilterUsageTracker, TemplateMatcher) | 15 | Scans ticker bars against templates; computes trust/suit scores |
| data_source_manager.py | 1,408 | 9 (DataValidationError, IBKRDataApp, DataSourceManager, SectorMapper, EClient + subcls) | 22 | Multi-provider data fetching: IBKR, Alpaca, yfinance, Massive |
| live_trading_engine.py | 1,233 | 4 (TradeJournal, Notifier, LifecycleManager, LiveTradingEngine) | 7 | Live trade execution, kinetic stop management, zombie protocol |
| strategy_engine.py | 737 | 4 (RegimeRouter, TacticalSniper, RiskActuary, StrategyEngine) | 9 | Regime classification, AI probability scoring, entry equation |
| validation_runner.py | 736 | 0 | 9 | Orchestrates backtest validation runs, compares results to thresholds |
| stock_hunter.py | 763 | 2 (NumpyEncoder, StockHunter) | 6 | Symbol scanning, state classification, VIP candidate selection |
| feature_engine.py | 704 | 1 (FeatureEngine) | 11 | Technical indicator calculation: trend, momentum, volatility, volume, pattern blocks |
| template_discovery.py | 611 | 1 (TemplateDiscovery) | 7 | Automated discovery of new template opportunities from data |
| validation_report.py | 589 | 1 (ValidationReport) | 1 | Generates Word document validation reports |
| notification_manager.py | 477 | 1 (NotificationManager) | 10 | Telegram + email alerts, alert deduplication |
| train_model.py | 377 | 1 (RegimeModelTrainer) | 8 | ML model training for regime classification (joblib/sklearn) |
| market_intelligence.py | 332 | 1 (MarketIntelligence) | 10 | News sentiment (TextBlob), Graham number, macro health check |
| portfolio_manager.py | 277 | 2 (PortfolioManager, RiskManager) | 8 | Position tracking, commission/slippage calculation, portfolio state |
| portfolio_risk.py | 257 | 1 (PortfolioRiskManager) | 4 | Risk gates: correlation, drawdown, weekly trend, reversal bypass |
| dag_optimizer.py | 231 | 1 (InformationTheoryEngine) | 4 | Information-theoretic indicator DAG optimization |
| decision_logger.py | 220 | 1 (DecisionLogger) | 6 | Structured CSV logging: signals, vetoes, risk, execution, exits |
| safe_json_io.py | 123 | 0 | 2 | Atomic JSON read/write with retry; prevents data corruption |
| versioned_save.py | 118 | 0 | 2 | Versioned file backup before overwrite (git-tag based) |
| data_engineer.py | 114 | 0 | 1 | One-shot data pipeline orchestration (batch fetch + process) |
| pre_market_validator.py | 175 | 1 (PreMarketValidator) | 1 | Pre-market checks: time-of-day gates, timezone handling |

---

## backtest/ Subfolder (Secondary Backtest Package)

| File | Lines | Classes | Responsibility |
|------|-------|---------|----------------|
| backtest/backtester.py | UNKNOWN: not scanned in core pass | UNKNOWN | Subfolder backtest runner |
| backtest/config.py | UNKNOWN | UNKNOWN | Backtest-specific config |
| backtest/data_loader.py | UNKNOWN | UNKNOWN | Data loading for backtest subfolder |
| backtest/pipeline.py | UNKNOWN | UNKNOWN | Backtest pipeline |
| backtest/reporter.py | UNKNOWN | UNKNOWN | Backtest result reporting |
| backtest/run_backtest.py | UNKNOWN | UNKNOWN | Entry point for subfolder backtest |
| backtest/template_optimizer.py | UNKNOWN | UNKNOWN | Template parameter optimization |

---

## Excluded from Core Analysis

| Category | Count | Reason |
|----------|-------|--------|
| archave/*.py | 96 | Archived/historical code, not in active system |
| scripts/*.py | 2 | Research scripts (discrimination_test_v2/v3), not production |
| tests/*.py | 30 | Test files, analyzed separately in 08_test_coverage_map.md |
| Misc root files | 8 | alpace_test_connection.py, IBKR_connection_test.py, debug_yfinance.py, notebooklm_sync.py, stockwise_simulation.py, stockwise_simulation_v2.py, verify_brokers.py, verify_fix.py |
