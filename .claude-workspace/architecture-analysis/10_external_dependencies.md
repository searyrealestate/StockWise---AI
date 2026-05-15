# 10 — External Dependencies

**Generated**: 2026-05-15 | **Commit**: 6bc83e8  
**Source**: `requirements.txt` (UTF-16 encoded)

---

## requirements.txt Package List

| Package | Pinned Version | Used In (files) | Critical? | Notes |
|---------|---------------|-----------------|-----------|-------|
| `streamlit` | ❌ Unpinned | stockwise_simulation*.py, archave/ | No (UI only) | Dashboard UI |
| `yfinance` | ❌ Unpinned | data_source_manager.py, market_intelligence.py | Yes | Primary data fallback provider |
| `pandas` | ❌ Unpinned | All core modules | Yes | Core DataFrame type throughout system |
| `numpy` | ❌ Unpinned | feature_engine, backtest_engine, strategy_engine, others | Yes | Numeric computation |
| `pandas-ta==0.4.71b0` | ✅ Pinned (beta) | feature_engine.py | Yes | Technical indicators (primary) |
| `joblib` | ❌ Unpinned | strategy_engine.py, train_model.py | Yes | ML model serialization |
| `scikit-learn` | ❌ Unpinned | train_model.py | Yes | Regime classification model |
| `plotly` | ❌ Unpinned | archave/, stockwise_simulation*.py | No | Visualization (UI only) |
| `shap` | ❌ Unpinned | archave/ | No | Model explainability (archived) |
| `matplotlib` | ❌ Unpinned | archave/ | No | Plotting (archived) |
| `google-cloud-storage` | ❌ Unpinned | archave/ | No | Cloud storage (archived) |
| `streamlit-authenticator` | ❌ Unpinned | stockwise_simulation*.py | No | Auth (UI only) |
| `lightgbm` | ❌ Unpinned | archave/ | No | Boosted tree model (archived) |
| `lxml` | ❌ Unpinned | validation_report.py (via docx) | No | XML parser |
| `financialmodelingprep` | ❌ Unpinned | data_source_manager.py | No | FMP fundamentals API |
| `gymnasium` | ❌ Unpinned | archave/ | No | RL environment (archived) |
| `stable-baselines3[extra]` | ❌ Unpinned | archave/ | No | RL training (archived) |
| `scipy` | ❌ Unpinned (merged with pandas-ta-classic line) | feature_engine.py | Yes | DSP blocks (scipy.signal) |

**File encoding note**: requirements.txt is UTF-16 encoded (has BOM). The last line appears malformed: `scipypandas-ta-classic` is a concatenation artifact.

---

## Not in requirements.txt but used in core

| Package | Used In | Source |
|---------|---------|--------|
| `ibapi` | data_source_manager.py | Must be installed separately from IBKR TWS API (not PyPI canonical) |
| `alpaca_trade_api` | data_source_manager.py | Not in requirements.txt |
| `textblob` | market_intelligence.py | Not in requirements.txt |
| `python-docx` (`docx`) | validation_report.py | Not in requirements.txt |
| `pytz` | pre_market_validator.py | Not in requirements.txt |
| `pandas_ta_classic` | feature_engine.py | Commented out in requirements (Python 3.11 version) |
| `colorama` | master_validator.py | Not in requirements.txt |
| `dotenv` (`python-dotenv`) | system_config.py | Not in requirements.txt |

---

## Flagged Issues

### Unpinned Critical Dependencies
All 4 critical dependencies (`pandas`, `numpy`, `yfinance`, `joblib`) are unpinned. This creates reproducibility risk.

### pandas-ta 0.4.71b0 — Beta Pin
The `pandas-ta` package is pinned to a **beta version** (`b0`). The comment says "for python 3.12" — but the project runs on Python 3.11 per memory.md. This should be `pandas-ta-classic` for Python 3.11 (line is commented out).

### ibapi Version Risk
`ibapi 9.81.1.post1` (community PyPI version) is installed. Official IBKR API is 10.x but has protobuf issues (per memory.md 2026-04-25). Rollback to 9.81 was deliberate.

### scipy Malformed Line
The requirements.txt line `scipypandas-ta-classic` appears to be a merge artifact. `scipy` is used in `feature_engine.py:DSP_block` — it must be installed but the requirements line is broken.

### archave/ Packages
Several packages (`gymnasium`, `stable-baselines3`, `shap`, `lightgbm`) are only used in archived code (archave/). They inflate the dependency footprint of the active system.
