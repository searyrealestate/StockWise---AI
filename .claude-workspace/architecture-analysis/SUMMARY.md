# StockWise AI — Wave 1 Architecture Anatomy: SUMMARY

**Generated**: 2026-05-15T00:00:00  
**Git Commit**: 6bc83e89bd923de2d40812b7f8da3f3d9fb8ed7f  
**Branch**: feature/backtest-validation-pipeline  
**Files Scanned**: 25 core .py | 30 test .py | 23 templates | ~30 configs  

---

## Table of Contents

| Report | File | Focus |
|--------|------|-------|
| SUMMARY | SUMMARY.md | Master index (this file) |
| 01 | 01_module_inventory.md | Every core .py: lines, classes, functions, responsibility |
| 02 | 02_dependency_graph.md | Import adjacency, god modules, leaf modules, circular deps |
| 03 | 03_function_signatures.md | All public functions with args, returns, docstrings |
| 04 | 04_magic_numbers.md | All hardcoded numeric literals: in-config vs not |
| 05 | 05_templates_inventory.md | 23 templates: state, active, conditions |
| 06 | 06_data_flow_diagram.md | Mermaid end-to-end data flow |
| 07 | 07_config_audit.md | system_config.py keys: declared vs used, orphans |
| 08 | 08_test_coverage_map.md | 30 test files, 852 tests, untested modules |
| 09 | 09_logging_inventory.md | Per-file log counts, silent/noisy modules |
| 10 | 10_external_dependencies.md | requirements.txt: 18 packages, pinning status |
| 11 | 11_pain_signals.md | TODO/FIXME/DEPRECATED/dead code |
| 12 | 12_architecture_alignment.md | 7-layer new-architecture vs StockWise coverage |

---

## Top 10 Most-Imported Modules (local imports only)

| Rank | Module | Imported By Count | Importers |
|------|--------|-------------------|-----------|
| 1 | `system_config` | 21 | All core modules |
| 2 | `feature_engine` | 12 | backtest_engine, data_source_manager, shadow_ledger, strategy_engine, live_trading_engine, data_engineer, stock_hunter, validation_runner, template_discovery, dag_optimizer, master_validator, train_model |
| 3 | `safe_json_io` | 13 | system_config, backtest_engine, template_matcher, setup_templates, shadow_ledger, strategy_engine, live_trading_engine, portfolio_manager, notification_manager, stock_hunter, validation_report, master_validator, train_model |
| 4 | `data_source_manager` | 10 | backtest_engine, shadow_ledger, live_trading_engine, data_engineer, stock_hunter, validation_runner, template_discovery, dag_optimizer, master_validator, train_model |
| 5 | `setup_templates` | 5 | backtest_engine, template_matcher, shadow_ledger, live_trading_engine, template_discovery |
| 6 | `decision_logger` | 5 | feature_engine, template_matcher, strategy_engine, portfolio_risk, pre_market_validator |
| 7 | `shadow_ledger` | 3 | backtest_engine, template_matcher, validation_runner |
| 8 | `stock_hunter` | 4 | backtest_engine, shadow_ledger, live_trading_engine, master_validator |
| 9 | `strategy_engine` | 3 | live_trading_engine, stock_hunter, master_validator |
| 10 | `system_config (via safe_json_io)` | 1 | system_config itself imports safe_json_io |

---

## Top 10 Largest Files by Line Count

| Rank | File | Lines |
|------|------|-------|
| 1 | backtest_engine.py | 3,251 |
| 2 | master_validator.py | 3,043 |
| 3 | setup_templates.py | 2,424 |
| 4 | shadow_ledger.py | 2,070 |
| 5 | system_config.py | 1,757 |
| 6 | template_matcher.py | 1,461 |
| 7 | data_source_manager.py | 1,408 |
| 8 | live_trading_engine.py | 1,233 |
| 9 | validation_runner.py | 736 |
| 10 | strategy_engine.py | 737 |

---

## System Signature

```
25 core modules | 53 public classes | ~230 public functions
23 templates on disk (active status stored per-file)
30 test files | 852 test functions
18 external packages in requirements.txt
60+ hardcoded numeric literals (majority in system_config.py)
2 pain signals (1 TODO, 1 DEPRECATED) in production code
```

---

## Discovery Totals

| Category | Count |
|----------|-------|
| Core .py files (root + backtest/) | 25 |
| Test .py files (tests/) | 30 |
| Archive .py files (archave/) | 96 |
| Script .py files (scripts/) | 2 |
| Config files (yaml/json/toml) | ~30 |
| Template JSON files | 23 |
| Public classes (core only) | 53 |
| Public functions (core, unique) | ~230 |
| Magic numbers (non-0/1/-1) | 60+ |
| Total test functions | 852 |
