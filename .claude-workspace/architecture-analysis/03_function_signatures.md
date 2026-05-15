# 03 — Function Signatures

**Generated**: 2026-05-15 | **Commit**: 6bc83e8  
**Scope**: All public (non-`_`) functions/methods in 25 core modules. `self` param omitted.

---

## system_config.py

| Class | Function | Args | Returns | Docstring |
|-------|----------|------|---------|-----------|
| MODULE | `load_dynamic_watchlist` | () | — | Yes |
| MODULE | `validate_template_evolution_config` | () | — | Yes |
| MODULE | `validate_shadow_ledger_config` | () | — | Yes |
| MODULE | `validate_ibkr_timeouts` | () | — | Yes |
| MODULE | `snapshot_configuration` | (logger_instance) | — | Yes |
| EmojiFilter | `filter` | (record) | — | No |
| LoggerSetup | `setup_logger` | (name, log_file, level) | — | Yes |
| LoggerSetup | `read_logs` | (log_file) | — | Yes |
| SystemActionLogger | `get_logger` | (cls) | — | Yes |
| SystemActionLogger | `log_action` | (component, action, details) | — | Yes |

## backtest_engine.py

| Class | Function | Args | Returns | Docstring |
|-------|----------|------|---------|-----------|
| BacktestEngine | `run` | () | dict | No |
| BacktestEngine | `scan_missed_opportunities` | (min_move_pct, lookahead) | — | Yes |
| WalkForwardValidator | `validate` | () | dict | Yes |
| WalkForwardValidator | `run_full_pipeline` | () | — | Yes |
| WalkForwardValidator | `validate_single_template` | (template_id, all_data_dict, config) | — | Yes |
| MODULE | `main` | () | — | No |

## feature_engine.py

| Class | Function | Args | Returns | Docstring |
|-------|----------|------|---------|-----------|
| FeatureEngine | `calculate_features` | (df, strategy_config) | — | Yes |
| FeatureEngine | `add_trend_block` | (df) | — | Yes |
| FeatureEngine | `add_momentum_block` | (df) | — | Yes |
| FeatureEngine | `add_volatility_block` | (df) | — | Yes |
| FeatureEngine | `add_volume_block` | (df) | — | Yes |
| FeatureEngine | `add_pattern_block` | (df) | — | Yes |
| FeatureEngine | `add_geometry_block` | (df) | — | Yes |
| FeatureEngine | `add_dsp_block` | (df) | — | Yes |
| FeatureEngine | `calculate_technical_score` | (df, strategy_config) | — | Yes |
| FeatureEngine | `check_veto_gates` | (df, symbol) | — | Yes |
| FeatureEngine | `safe_sma` | (source, length) | — | No |

## data_source_manager.py

| Class | Function | Args | Returns | Docstring |
|-------|----------|------|---------|-----------|
| MODULE | `clean_raw_data` | (df) | pd.DataFrame | Yes |
| MODULE | `filter_regular_trading_hours` | (df, market_open, market_close) | — | Yes |
| MODULE | `normalize_ohlcv` | (df, provider_name) | — | Yes |
| IBKRDataApp | `error` | (reqId, errorCode, errorString, advancedOrderRejectJson) | — | No |
| IBKRDataApp | `historicalData` | (reqId, bar) | — | Yes |
| IBKRDataApp | `historicalDataEnd` | (reqId, start, end) | — | Yes |
| IBKRDataApp | `contractDetails` | (reqId, contractDetails) | — | Yes |
| IBKRDataApp | `contractDetailsEnd` | (reqId) | — | No |
| IBKRDataApp | `fetch_historical_data` | (contract, durationStr, barSizeSetting, whatToShow, useRTH) | — | Yes |
| DataSourceManager | `get_new_req_id` | () | — | Yes |
| DataSourceManager | `connect_to_ibkr` | () | — | No |
| DataSourceManager | `disconnect` | () | — | No |
| DataSourceManager | `isConnected` | () | — | No |
| DataSourceManager | `get_fundamentals` | (ticker) | — | Yes |
| DataSourceManager | `fetch_data_sequential` | (tickers, days_back) | — | Yes |
| DataSourceManager | `get_stock_data` | (symbol, start_date, end_date, days_back, interval) | — | Yes |
| DataSourceManager | `fetch_and_process` | (symbol, interval) | — | Yes |
| DataSourceManager | `fetch_data` | (symbol, limit, interval) | — | Yes |
| DataSourceManager | `stream_data` | (symbol, queue) | — | Yes |
| DataSourceManager | `get_realtime_quote` | (symbol) | — | Yes |
| DataSourceManager | `regenerate_all_data` | () | — | Yes |
| SectorMapper | `get_benchmark_symbol` | (ticker) | str | No |

## template_matcher.py

| Class | Function | Args | Returns | Docstring |
|-------|----------|------|---------|-----------|
| FilterUsageTracker | `record` | (template_id, symbol, details) | — | No |
| FilterUsageTracker | `reset` | () | — | No |
| FilterUsageTracker | `get_block_stats` | (template_id, block_name) | — | No |
| FilterUsageTracker | `get_symbol_stats` | (template_id, symbol) | — | No |
| FilterUsageTracker | `get_report` | () | — | No |
| FilterUsageTracker | `save` | (path) | — | No |
| TemplateMatcher | `scan_ticker` | (symbol, df, stock_state, timeframe) | — | Yes |
| TemplateMatcher | `get_idle_report` | () | — | Yes |
| TemplateMatcher | `get_scan_statistics` | () | — | Yes |
| TemplateMatcher | `get_template_win_rate` | (template_id, symbol) | — | Yes |
| TemplateMatcher | `evaluate_auto_disable` | (template_id, symbol, stock_state, shadow_stats, notifier) | — | Yes |
| TemplateMatcher | `get_trust_score` | (template_id, symbol, stock_state) | — | Yes |
| TemplateMatcher | `get_suit_sharing_report` | () | — | Yes |
| TemplateMatcher | `get_suit` | (symbol, state_key) | — | Yes |
| TemplateMatcher | `assign_suits` | () | — | Yes |

## setup_templates.py (Block Functions — module level)

| Function | Args | Description |
|----------|------|-------------|
| `block_close_above_sma` | (row, params) | Close > SMA[period] |
| `block_sma_above_sma` | (row, params) | Short SMA > Long SMA |
| `block_close_above_ema` | (row, params) | Close > EMA[period] |
| `block_er_slow_above` | (row, params) | ER_slow > threshold |
| `block_trend_alignment` | (row, params) | Multi-timeframe trend alignment |
| `block_rsi_between` | (row, params) | RSI in [low, high] range |
| `block_rsi_below` | (row, params) | RSI < threshold |
| `block_rsi_above` | (row, params) | RSI > threshold |
| `block_macd_above_signal` | (row, params) | MACD > signal line |
| `block_macd_histogram_positive` | (row, params) | MACD histogram > 0 |
| `block_volume_surge` | (row, params) | Volume > N * avg_volume |
| `block_rvol_above` | (row, params) | Relative volume > threshold |
| `block_squeeze_active` | (row, params) | BB inside KC (TTM Squeeze on) |
| `block_squeeze_momentum_positive` | (row, params) | Squeeze momentum > 0 |
| `block_bb_width_below` | (row, params) | BB width% < threshold |
| `block_atr_percent_above` | (row, params) | ATR% > threshold |
| `block_bullish_candle` | (row, params) | Close > Open |
| `block_close_above_ref` | (row, params) | Close > reference price |
| `block_close_below_ref` | (row, params) | Close < reference price |
| `block_adx_above` | (row, params) | ADX > threshold |
| `block_supertrend_bullish` | (row, params) | Supertrend direction = bullish |
| `block_golden_cross_active` | (row, params) | SMA50 > SMA200 |
| `block_stoch_oversold` | (row, params) | Stochastic %K < threshold |
| `block_cci_between` | (row, params) | CCI in range |
| `block_roc_positive` | (row, params) | Rate of change > 0 |
| `block_obv_rising` | (row, params) | OBV > OBV[N bars ago] |
| `block_cmf_positive` | (row, params) | Chaikin Money Flow > 0 |
| `block_vwap_above` | (row, params) | Close > VWAP |
| `block_gap_up_today` | (row, params) | Open > prev_close * (1 + threshold) |
| `block_fib_near_support` | (row, params) | Price near Fibonacci support |
| `block_double_bottom_active` | (row, params) | Double bottom pattern detected |
| `stop_atr` | (row, params) | Stop = close - N*ATR |
| `stop_swing_low` | (row, params) | Stop = recent swing low |
| `stop_fixed_pct` | (row, params) | Stop = close * (1 - pct) |
| `stop_sma` | (row, params) | Stop = SMA[period] |
| `target_atr` | (row, params) | Target = close + N*ATR |
| `target_fixed_pct` | (row, params) | Target = close * (1 + pct) |

## setup_templates.py (Class Methods)

| Class | Function | Args | Returns |
|-------|----------|------|---------|
| SetupTemplate | `get_category` | () | — |
| SetupTemplate | `validate` | () | — |
| SetupTemplate | `get_win_rate` | () | — |
| SetupTemplate | `get_best_context` | () | — |
| SetupTemplate | `evaluate_conditions` | (row) | — |
| SetupTemplate | `calculate_stop_loss` | (row) | — |
| SetupTemplate | `calculate_take_profit` | (row) | — |
| SetupTemplate | `record_result` | (ticker, profit_pct, won, context) | — |
| SetupTemplate | `record_block_results` | (details, symbol, all_passed, outcome) | — |
| SetupTemplate | `to_dict` | () | — |
| TemplateManager | `load_all` | () | — |
| TemplateManager | `save_template` | (template) | — |
| TemplateManager | `save_all` | () | — |
| TemplateManager | `disable_template` | (template_id) | — |
| TemplateManager | `get_enabled` | () | — |
| TemplateManager | `get_for_timeframe` | (timeframe) | — |
| TemplateManager | `get_for_state` | (stock_state, symbol, timeframe) | — |
| TemplateManager | `get_template_by_id` | (template_id) | — |
| TemplateManager | `add_template` | (template_data) | — |
| TemplateManager | `get_statistics_summary` | () | — |
| TemplateGenerator | `generate_from_gaps` | (coverage_gaps) | — |
| TemplateGenerator | `generate_all_recipes` | () | — |
| TemplateGenerator | `get_generation_report` | () | — |
| DiscriminationTemplateBuilder | `build_from_results` | (results_path) | list |
| TemplateHealthMonitor | `prepare_full_evaluation` | () | list |
| TemplateHealthMonitor | `apply_decisions` | (backtest_results, results_path, temp_enabled) | list |
| TradeOutcomeAnalyzer | `analyze` | (backtest_results, results_path) | list |

## Other Key Modules

| Module | Class | Function | Args | Returns |
|--------|-------|----------|------|---------|
| shadow_ledger | ShadowLedger | `evaluate_history` | (symbol, df, stock_state_fn, max_date, timeframe) | — |
| shadow_ledger | ShadowLedger | `determine_lifecycle` | (signals, decayed_wr, prev_lifecycle) | — |
| shadow_ledger | ShadowLedger | `get_template_stats` | (symbol) | — |
| shadow_ledger | ShadowLedger | `apply_decay` | () | — |
| shadow_ledger | ShadowLedger | `run_full_evaluation` | (data_source_manager, symbols, feature_engine, max_date, stock_state_fn) | — |
| strategy_engine | RegimeRouter | `classify_regime` | (df) | — |
| strategy_engine | TacticalSniper | `get_ai_probability` | (df, regime) | — |
| strategy_engine | TacticalSniper | `evaluate_friction_adjusted_alpha` | (price, stop_loss, target) | — |
| strategy_engine | TacticalSniper | `analyze` | (symbol, df, regime) | — |
| strategy_engine | RiskActuary | `calculate_size` | (price, stop_loss, volume_avg) | — |
| strategy_engine | StrategyEngine | `evaluate_ticker` | (symbol, df_features) | — |
| strategy_engine | StrategyEngine | `decide_action` | (ticker, dataframe, news_context) | — |
| live_trading_engine | LifecycleManager | `manage_kinetic_stop` | (symbol, position, current_price, current_atr) | — |
| live_trading_engine | LifecycleManager | `check_zombie_protocol` | (symbol, position, current_regime) | — |
| live_trading_engine | LiveTradingEngine | `execute_ticket` | (ticket, current_regime) | — |
| live_trading_engine | LiveTradingEngine | `manage_open_positions` | (market_data, agent1_router, feature_engine) | — |
| portfolio_risk | PortfolioRiskManager | `check_all_gates` | (symbol, df, open_positions, market_data, portfolio_value) | — |
| portfolio_risk | PortfolioRiskManager | `check_correlation_gate` | (symbol, open_positions, market_data) | — |
| portfolio_risk | PortfolioRiskManager | `check_drawdown_gate` | (open_positions, portfolio_value) | — |
| portfolio_risk | PortfolioRiskManager | `check_weekly_trend_gate` | (symbol, df) | — |
| decision_logger | DecisionLogger | `log_signal` | (symbol, template_id, confidence, regime) | None |
| decision_logger | DecisionLogger | `log_veto` | (symbol, gate, passed, reason) | None |
| decision_logger | DecisionLogger | `log_risk` | (symbol, gate, passed, reason) | None |
| decision_logger | DecisionLogger | `log_execution` | (symbol, action, price, qty, stop) | None |
| decision_logger | DecisionLogger | `log_exit` | (symbol, exit_price, pnl_pct, reason) | None |
| safe_json_io | MODULE | `safe_json_write` | (filepath, data, cls, indent) | — |
| safe_json_io | MODULE | `safe_json_read` | (filepath, default, retries, retry_delay) | — |
| versioned_save | MODULE | (2 functions) | — | — |
| data_engineer | MODULE | `run_data_pipeline` | () | — |
