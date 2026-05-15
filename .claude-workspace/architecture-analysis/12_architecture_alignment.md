# 12 — Architecture Alignment

**Generated**: 2026-05-15 | **Commit**: 6bc83e8  
**Coverage scale**: MISSING / PARTIAL / IMPLEMENTED / ADVANCED  
**Method**: Count of features present in code, not subjective quality rating.

---

## Layer 1: Data Layer

| Capability | StockWise Module(s) | Coverage | Feature Count |
|-----------|---------------------|----------|---------------|
| Multi-provider data ingestion | data_source_manager.py (IBKRDataApp, DataSourceManager) | IMPLEMENTED | 4 providers: IBKR, Alpaca, yfinance, Massive |
| Provider fallback chain | data_source_manager.py:get_stock_data() | IMPLEMENTED | Config-driven priority: HISTORICAL_SOURCE |
| Data normalization | data_source_manager.py:normalize_ohlcv() | IMPLEMENTED | Standardizes OHLCV column names per provider |
| Data validation | data_source_manager.py:DataValidationError, clean_raw_data() | IMPLEMENTED | Schema checks, gap detection |
| Market hours filtering | data_source_manager.py:filter_regular_trading_hours() | IMPLEMENTED | Configurable open/close times |
| Streaming/real-time | data_source_manager.py:stream_data(), get_realtime_quote() | PARTIAL | Method exists; IBKR streaming path not fully tested |
| Batch caching | data_engineer.py:run_data_pipeline() | PARTIAL | Runs batch fetch; cache format UNKNOWN |
| Fundamentals | data_source_manager.py:get_fundamentals() | PARTIAL | FMP API; not used in main signal pipeline |
| Sector/benchmark mapping | data_source_manager.py:SectorMapper | PARTIAL | get_benchmark_symbol() only |

**Layer 1 Summary**: IMPLEMENTED — Full multi-provider ingestion with normalization and validation.

---

## Layer 2: Feature & Pattern Layer

| Capability | StockWise Module(s) | Coverage | Feature Count |
|-----------|---------------------|----------|---------------|
| Trend indicators | feature_engine.py:add_trend_block() | IMPLEMENTED | SMA, EMA, supertrend, golden/death cross, ADX |
| Momentum indicators | feature_engine.py:add_momentum_block() | IMPLEMENTED | RSI, MACD, ROC, Stochastic, CCI |
| Volatility indicators | feature_engine.py:add_volatility_block() | IMPLEMENTED | ATR, BB, KC, BB width, squeeze |
| Volume indicators | feature_engine.py:add_volume_block() | IMPLEMENTED | OBV, CMF, RVOL, VWAP, volume surge |
| Pattern recognition | feature_engine.py:add_pattern_block() | IMPLEMENTED | Double bottom, HS pattern, gap up, Fibonacci |
| Geometry analysis | feature_engine.py:add_geometry_block() | IMPLEMENTED | Donchian, support/resistance levels |
| DSP features | feature_engine.py:add_dsp_block() | IMPLEMENTED | scipy.signal-based features |
| Efficiency Ratio | feature_engine.py | IMPLEMENTED | ER slow (20-day) + ER fast (5-day) |
| Technical score | feature_engine.py:calculate_technical_score() | IMPLEMENTED | Composite score 0–100 |
| Veto gates | feature_engine.py:check_veto_gates() | IMPLEMENTED | Spread, volume, gap checks |
| Block functions (conditions) | setup_templates.py | ADVANCED | 31 block functions + 6 stop/target functions |
| Block evaluation & tracking | setup_templates.py, template_matcher.py:FilterUsageTracker | IMPLEMENTED | Per-block pass/fail stats |

**Layer 2 Summary**: ADVANCED — 60+ indicators, 31 condition blocks, full veto system.

---

## Layer 3: Regime Detection Layer

| Capability | StockWise Module(s) | Coverage | Feature Count |
|-----------|---------------------|----------|---------------|
| Market state classification | stock_hunter.py:StockHunter | IMPLEMENTED | 4-dimensional: trend, structure, volatility, volume |
| State dimensions | stock_hunter.py | IMPLEMENTED | trend (BULLISH/BEARISH/SIDEWAYS), structure (OPEN_FIELD/NEAR_RESISTANCE/NEAR_SUPPORT), volatility (NORMAL/COMPRESSED/VOLATILE), volume (HEALTHY/SURGING) |
| ML regime classification | strategy_engine.py:RegimeRouter, train_model.py:RegimeModelTrainer | IMPLEMENTED | sklearn model (joblib-saved) |
| Efficiency Ratio regime | strategy_engine.py | IMPLEMENTED | ER thresholds from system_config |
| Macro health check | market_intelligence.py:check_macro_health() | PARTIAL | VIX + SPY-based gate |
| Sentiment analysis | market_intelligence.py (TextBlob) | PARTIAL | News sentiment only, no on-bar integration confirmed |
| Weekly trend gate | portfolio_risk.py:check_weekly_trend_gate() | IMPLEMENTED | SMA_40 weekly trend veto |
| Reversal bypass | portfolio_risk.py:check_all_gates(is_reversal) | IMPLEMENTED | Reversal templates bypass weekly gate |

**Layer 3 Summary**: IMPLEMENTED — 4D state space + ML classifier + macro gate.

---

## Layer 4: Signal / Strategy Engine

| Capability | StockWise Module(s) | Coverage | Feature Count |
|-----------|---------------------|----------|---------------|
| Template-based signal generation | template_matcher.py:TemplateMatcher.scan_ticker() | IMPLEMENTED | Evaluates all enabled templates per bar |
| Trust scoring | template_matcher.py:get_trust_score(), shadow_ledger.py | IMPLEMENTED | Per-template + per-symbol win rate with decay |
| Suit assignment | template_matcher.py:assign_suits(), get_suit() | IMPLEMENTED | Symbol-to-template fit scoring |
| Auto-disable | template_matcher.py:evaluate_auto_disable() | IMPLEMENTED | WR < threshold → disable |
| Walk-forward validation | backtest_engine.py:WalkForwardValidator | IMPLEMENTED | Train/val/test split (60/20/20) |
| Quality Gate | backtest_engine.py:validate_single_template() | IMPLEMENTED | VAL + TEST period gates |
| Signal stacking protection | backtest_engine.py | IMPLEMENTED | Per-symbol cooldown (min_bars_after_exit_by_timeframe) |
| AI probability | strategy_engine.py:TacticalSniper.get_ai_probability() | IMPLEMENTED | ML probability overlay |
| Friction-adjusted alpha | strategy_engine.py:evaluate_friction_adjusted_alpha() | IMPLEMENTED | Net R:R after commission + slippage |
| Discrimination-driven templates | setup_templates.py:DiscriminationTemplateBuilder | IMPLEMENTED | Cohen's d mining → auto-generated templates |
| Health monitoring | setup_templates.py:TemplateHealthMonitor | IMPLEMENTED | Auto enable/disable based on PF |
| Trade outcome analysis | setup_templates.py:TradeOutcomeAnalyzer | IMPLEMENTED | Post-hoc indicator analysis |

**Layer 4 Summary**: ADVANCED — Full signal pipeline with trust/suit system, discrimination mining, and health monitoring.

---

## Layer 5: Risk Overlay Layer

| Capability | StockWise Module(s) | Coverage | Feature Count |
|-----------|---------------------|----------|---------------|
| Correlation gate | portfolio_risk.py:check_correlation_gate() | IMPLEMENTED | Blocks correlated positions |
| Drawdown gate | portfolio_risk.py:check_drawdown_gate() | IMPLEMENTED | Portfolio-level drawdown limit |
| Weekly trend gate | portfolio_risk.py:check_weekly_trend_gate() | IMPLEMENTED | SMA_40 veto |
| Reversal bypass | portfolio_risk.py:check_all_gates() | IMPLEMENTED | is_reversal parameter |
| Pre-market time gate | pre_market_validator.py | IMPLEMENTED | Market hours check |
| Position sizing | portfolio_manager.py:RiskManager.calculate_position_size() | IMPLEMENTED | ATR-based sizing |
| Max daily loss | system_config.RISK_PARAMETERS.max_daily_loss_usd | PARTIAL | Defined in config; enforcement in live_trading_engine |
| Commission model | portfolio_manager.py:calculate_commission() | IMPLEMENTED | Per-share IBKR model |
| Slippage model | portfolio_manager.py:apply_slippage() | IMPLEMENTED | Configurable pct |
| Tax calculation | system_config.COSTS_CONFIG.tax_rate | PARTIAL | 25% CGT in config; applied in validation_report |
| Kinetic stop | live_trading_engine.py:LifecycleManager.manage_kinetic_stop() | ADVANCED | 4-phase trailing stop (ATR → breakeven → parabolic → runner) |
| Zombie protocol | live_trading_engine.py:check_zombie_protocol() | IMPLEMENTED | Forced exit on regime flip |
| Audit trail | decision_logger.py:DecisionLogger | IMPLEMENTED | CSV log: signals, vetoes, risk, execution, exits |

**Layer 5 Summary**: ADVANCED — 4-gate risk overlay + 4-phase kinetic stop + full audit trail.

---

## Layer 6: Execution Layer

| Capability | StockWise Module(s) | Coverage | Feature Count |
|-----------|---------------------|----------|---------------|
| Order execution | live_trading_engine.py:execute_ticket() | IMPLEMENTED | Market orders via Alpaca/IBKR |
| Trade journal | live_trading_engine.py:TradeJournal | IMPLEMENTED | CSV logging of all signals + outcomes |
| Position management | live_trading_engine.py:manage_open_positions() | IMPLEMENTED | Per-bar lifecycle management |
| Daily position summary | live_trading_engine.py:send_daily_position_summary() | IMPLEMENTED | Telegram notification |
| Notifications | notification_manager.py:NotificationManager | IMPLEMENTED | Telegram + email with deduplication |
| Versioned saves | versioned_save.py | IMPLEMENTED | Backup before overwrite |

**Layer 6 Summary**: IMPLEMENTED — Full live execution with position management and notifications.

---

## Layer 7: Monitoring & Health Layer

| Capability | StockWise Module(s) | Coverage | Feature Count |
|-----------|---------------------|----------|---------------|
| Backtest validation | validation_runner.py | IMPLEMENTED | Full pipeline validation with threshold checks |
| Report generation | validation_report.py:ValidationReport | IMPLEMENTED | Word document reports |
| Template health monitoring | setup_templates.py:TemplateHealthMonitor | IMPLEMENTED | Auto enable/disable based on live PF |
| Full test suite | master_validator.py:StockWiseMasterValidator | IMPLEMENTED | 852 tests across all modules |
| Opportunity scanner | backtest_engine.py:scan_missed_opportunities() | IMPLEMENTED | Finds missed ≥5% moves |
| Monte Carlo simulation | backtest_engine.py | IMPLEMENTED | Risk-of-ruin analytical + MC simulation |
| Block statistics | template_matcher.py:FilterUsageTracker | IMPLEMENTED | Per-block pass/fail analytics |
| Trade outcome analysis | setup_templates.py:TradeOutcomeAnalyzer | IMPLEMENTED | Cohen's d on wins vs losses |
| Deterministic mode | backtest_engine.py (--deterministic flag) | IMPLEMENTED | Reproducible backtests |
| Walk-forward validation | backtest_engine.py:WalkForwardValidator | IMPLEMENTED | OOS performance tracking |
| Discrimination tests | scripts/discrimination_test_v2.py, v3.py | IMPLEMENTED | Feature quality mining |

**Layer 7 Summary**: ADVANCED — Comprehensive monitoring including opportunity scanner, Monte Carlo, block stats, and template health.

---

## Overall Alignment Matrix

| Layer | Coverage | Key Gaps |
|-------|----------|----------|
| 1. Data Layer | IMPLEMENTED | Streaming not fully tested; cache format UNKNOWN |
| 2. Feature & Pattern Layer | ADVANCED | None identified |
| 3. Regime Detection Layer | IMPLEMENTED | Macro sentiment not integrated into bar-level decisions |
| 4. Signal / Strategy Engine | ADVANCED | None identified |
| 5. Risk Overlay Layer | ADVANCED | Daily loss enforcement not verified in live path |
| 6. Execution Layer | IMPLEMENTED | Broker API path for IBKR live trading not fully traced |
| 7. Monitoring & Health Layer | ADVANCED | No dedicated monitoring dashboard (only master_validator CLI) |
