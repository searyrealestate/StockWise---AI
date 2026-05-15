# 08 — Test Coverage Map

**Generated**: 2026-05-15 | **Commit**: 6bc83e8  
**Total test files**: 30 | **Total test functions**: 852

---

## Test File Inventory

| Test File | Test Count | Modules Tested (by import) | Category |
|-----------|-----------|---------------------------|----------|
| tests/unit_tests.py | 246 | backtest_engine, data_source_manager, feature_engine, live_trading_engine, portfolio_risk, safe_json_io, setup_templates, shadow_ledger, stock_hunter, strategy_engine, system_config, template_matcher, train_model | Unit |
| tests/test_template_system.py | 206 | notification_manager, setup_templates, shadow_ledger, system_config, template_matcher | Unit |
| tests/test_anti_overfitting.py | 41 | setup_templates, system_config | Unit |
| tests/test_feature_engine.py | 31 | feature_engine | Unit |
| tests/test_execution.py | 28 | live_trading_engine, system_config | Integration |
| tests/test_portfolio_risk.py | 25 | portfolio_risk, system_config | Unit |
| tests/test_strategy_engine.py | 24 | strategy_engine, system_config, template_matcher | Unit |
| tests/test_data_layer.py | 23 | data_source_manager, system_config | Integration |
| tests/test_indicator_snapshot.py | 22 | backtest_engine, setup_templates | Unit |
| tests/test_integration.py | 17 | data_source_manager, feature_engine, live_trading_engine, portfolio_risk, setup_templates, stock_hunter, template_matcher | Integration |
| tests/test_block_stats.py | 16 | setup_templates | Unit |
| tests/test_backtest_analytics.py | 15 | backtest_engine | Unit |
| tests/test_regression.py | 15 | system_config | Regression |
| tests/test_versioned_save.py | 13 | (none — tests versioned_save directly) | Unit |
| tests/test_block_evaluations.py | 13 | backtest_engine, system_config | Unit |
| tests/test_vip_scanner.py | 13 | stock_hunter, system_config | Unit |
| tests/test_walk_forward.py | 12 | backtest_engine, system_config | Unit |
| tests/test_notification.py | 11 | notification_manager, safe_json_io, system_config | Unit |
| tests/test_volatility_classification.py | 11 | setup_templates, stock_hunter, system_config | Unit |
| tests/test_integration_pipeline.py | 10 | live_trading_engine, system_config | Integration |
| tests/test_backtest_shadow_feed.py | 10 | backtest_engine, safe_json_io, system_config | Unit |
| tests/test_performance.py | 10 | feature_engine, live_trading_engine, safe_json_io, system_config, template_matcher | Performance |
| tests/test_shadow_ledger.py | 9 | shadow_ledger, system_config | Unit |
| tests/test_shadow_ledger_cli.py | 9 | shadow_ledger, system_config | Unit |
| tests/test_template_conditions_ceiling.py | 9 | setup_templates, system_config | Unit |
| tests/test_gen7.py | 3 | feature_engine, strategy_engine | Unit |
| tests/test_gen7_validation.py | 7 | feature_engine, live_trading_engine, strategy_engine | Regression |
| tests/test_bug_1_3_er_trend.py | 3 | strategy_engine | Regression |
| tests/test_ibkr_data_completeness.py | 0 | data_source_manager | Integration (live, requires IBKR) |
| tests/test_ibkr_determinism.py | 0 | data_source_manager, system_config | Integration (live, requires IBKR) |

---

## Test Count Summary

| Category | Test Files | Test Functions |
|----------|-----------|----------------|
| Unit | 20 | ~700 |
| Integration | 6 | ~107 |
| Performance | 1 | 10 |
| Regression | 3 | 25 |
| Live (requires IBKR) | 2 | 0 (all marked skip) |
| **Total** | **30** | **852** |

---

## Module Coverage

| Module | Tested By | Coverage Level |
|--------|-----------|----------------|
| system_config | 20+ test files | Partial (config values tested, not logger) |
| backtest_engine | unit_tests, test_backtest_analytics, test_block_evaluations, test_indicator_snapshot, test_backtest_shadow_feed, test_walk_forward | Good |
| feature_engine | unit_tests, test_feature_engine, test_gen7, test_performance | Good |
| setup_templates | unit_tests, test_template_system, test_anti_overfitting, test_block_stats, test_indicator_snapshot, test_template_conditions_ceiling, test_volatility_classification | Very Good |
| template_matcher | unit_tests, test_template_system, test_strategy_engine, test_performance, test_integration | Good |
| shadow_ledger | unit_tests, test_template_system, test_shadow_ledger, test_shadow_ledger_cli | Good |
| data_source_manager | unit_tests, test_data_layer, test_integration | Partial |
| live_trading_engine | unit_tests, test_execution, test_integration, test_integration_pipeline, test_gen7_validation, test_performance | Good |
| strategy_engine | unit_tests, test_strategy_engine, test_gen7, test_bug_1_3_er_trend | Good |
| portfolio_risk | unit_tests, test_portfolio_risk, test_integration | Good |
| stock_hunter | unit_tests, test_vip_scanner, test_volatility_classification, test_integration | Good |
| notification_manager | test_template_system, test_notification | Partial |
| portfolio_manager | unit_tests (via master_validator) | Partial |
| safe_json_io | unit_tests, test_backtest_shadow_feed, test_notification, test_performance | Partial |
| train_model | unit_tests | Minimal |
| decision_logger | NONE | **No dedicated tests** |
| market_intelligence | NONE | **No dedicated tests** |
| pre_market_validator | NONE | **No dedicated tests** |
| data_engineer | NONE | **No dedicated tests** |
| dag_optimizer | NONE | **No dedicated tests** |
| template_discovery | NONE | **No dedicated tests** |
| validation_runner | NONE | **No dedicated tests** |
| validation_report | NONE | **No dedicated tests** |
| versioned_save | test_versioned_save | Partial (3rd-party mock) |

---

## Untested Core Modules (zero dedicated test files)

| Module | Lines | Risk Level |
|--------|-------|-----------|
| decision_logger | 220 | Medium — audit trail infrastructure |
| market_intelligence | 332 | Medium — sentiment + fundamentals |
| pre_market_validator | 175 | Low — simple time gate |
| data_engineer | 114 | Low — one-shot batch job |
| dag_optimizer | 231 | Low — research tool |
| template_discovery | 611 | Medium — complex brute-force search |
| validation_runner | 736 | High — core validation pipeline |
| validation_report | 589 | Low — report generation |
