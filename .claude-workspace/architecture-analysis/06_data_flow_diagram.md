# 06 — Data Flow Diagram

**Generated**: 2026-05-15 | **Commit**: 6bc83e8  
**Tool**: Mermaid (flowchart). Module names are exact filenames (without .py).

---

## End-to-End Data Flow

```mermaid
flowchart TD
    subgraph DATA_LAYER["Data Layer"]
        EXT["External Sources\n(IBKR / Alpaca / yfinance / Massive)"]
        DSM["data_source_manager\nIBKRDataApp + DataSourceManager"]
        FE["feature_engine\nFeatureEngine"]
        DE["data_engineer\n(batch cache)"]
    end

    subgraph PATTERN_LAYER["Feature & Pattern Layer"]
        ST["setup_templates\nBlock Functions (31 blocks)"]
        TM["template_matcher\nTemplateMatcher"]
        FUT["FilterUsageTracker"]
    end

    subgraph REGIME_LAYER["Regime Detection Layer"]
        SH["stock_hunter\nStockHunter"]
        SE["strategy_engine\nRegimeRouter + TacticalSniper"]
        TRN["train_model\nRegimeModelTrainer"]
        MI["market_intelligence\nMacro sentiment + Graham number"]
    end

    subgraph SIGNAL_LAYER["Signal / Strategy Engine"]
        SL["shadow_ledger\nShadowLedger - trust/suit learning"]
        BE["backtest_engine\nBacktestEngine + WalkForwardValidator"]
        VR["validation_runner\nOrchestrates validation pipeline"]
    end

    subgraph RISK_LAYER["Risk Overlay Layer"]
        PR["portfolio_risk\nPortfolioRiskManager"]
        PMV["pre_market_validator\nTime gates"]
        DL["decision_logger\nStructured audit trail"]
    end

    subgraph EXECUTION_LAYER["Execution Layer"]
        LTE["live_trading_engine\nLifecycleManager + KineticStop"]
        PM["portfolio_manager\nPortfolioManager + RiskManager"]
        NM["notification_manager\nTelegram/Email alerts"]
    end

    subgraph CONFIG_LAYER["Cross-Cutting"]
        SC["system_config\nAll constants + logger setup"]
        SJI["safe_json_io\nAtomic JSON read/write"]
        VS["versioned_save\nFile versioning"]
    end

    %% Data ingestion flow
    EXT -->|OHLCV raw bars| DSM
    DSM -->|normalized OHLCV DataFrame| FE
    FE -->|enriched DataFrame 60+ indicators| DSM
    DE -->|batch fetch + cache| DSM

    %% State classification
    DSM -->|enriched DataFrame| SH
    SH -->|symbol state dict| SE
    SE -->|regime + score| TM
    TRN -->|joblib model| SE
    MI -->|sentiment + fundamentals| SE

    %% Template matching
    FE -->|indicator row| TM
    ST -->|block functions| TM
    SL -->|trust + suit scores| TM
    TM -->|signal ticket| FUT

    %% Shadow ledger / backtest
    TM -->|signal tickets| SL
    SL -->|historical win rates| TM
    DSM -->|historical data| BE
    FE -->|features| BE
    TM -->|matching| BE
    SL -->|trust data| BE
    BE -->|trade results JSON| VR
    VR -->|validation report| VS

    %% Risk gates
    TM -->|signal| PR
    PR -->|PASS/VETO| LTE
    PMV -->|time gate| LTE
    DL -->|audit CSV| LTE

    %% Execution
    PR -->|approved signal| LTE
    LTE -->|order| EXT
    LTE -->|position update| PM
    LTE -->|alert| NM

    %% Config wiring (omitted from main flow for clarity — all modules read SC)
    SC -.->|constants| DSM
    SC -.->|constants| FE
    SC -.->|constants| BE
    SJI -.->|atomic I/O| SL
    SJI -.->|atomic I/O| TM
```

---

## Key Data Objects (passed between modules)

| Object | From | To | Shape |
|--------|------|----|-------|
| OHLCV DataFrame | data_source_manager | feature_engine | pd.DataFrame (N rows × OHLCV cols) |
| Enriched DataFrame | feature_engine | template_matcher, strategy_engine, shadow_ledger | pd.DataFrame (N rows × 60+ indicator cols) |
| Stock State dict | stock_hunter | template_matcher | `{trend, structure, volatility, volume}` |
| Signal ticket | template_matcher | backtest_engine, live_trading_engine | `{symbol, template_id, confidence, stop, target, ...}` |
| Trust score | shadow_ledger | template_matcher | `{win_rate, decayed_wr, lifecycle}` |
| Risk gate result | portfolio_risk | live_trading_engine | `{passed: bool, reason: str, gate: str}` |
| Trade result | backtest_engine | validation_runner | JSON dict with PnL, bars_held, etc. |
| Template JSON | setup_templates (disk) | template_matcher | Loaded via TemplateManager |

---

## Unclear / Unverified Connections `[?]`

- `[?]` live_trading_engine → broker API (Alpaca/IBKR): exact order submission path not traced (asyncio + data_source_manager)
- `[?]` data_engineer → disk cache: cache format (parquet vs CSV vs JSON) not verified in this pass
- `[?]` train_model → live_trading_engine: model file path hardcoded vs config — not verified
