# micha7_analyzer

Standalone deterministic technical analysis — Micha 7-parameter checklist, backtest engine.

## Status

Phase 1 skeleton. Package is importable and CLI works. Business logic implemented in subsequent prompts.

## Architecture

- **Standalone** (ADR-014): no StockWise dependencies in Phase 1.
- **Data source**: yfinance via `BaseDataProvider` interface (ADR-015); IBKR in Phase 5+.
- **Full documentation**: `../.claude-workspace/micha7/` (ARCHITECTURE.md, DECISIONS.md, PHASES.md).

## Requirements

- Python >= 3.10

## Install (development)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -e ".[dev]"
```

## Run

```bash
python -m micha7 --version
# micha7_analyzer 0.1.0
```

## Test

```bash
pytest
```

---

> Last Modified: 2026-06-01T09:40:00Z
