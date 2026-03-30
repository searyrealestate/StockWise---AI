"""
Main orchestrator. Enforces strict train/val/test data separation.
"""

import logging
from pathlib import Path

from backtest.config import MIN_WIN_RATE, MIN_TRADES
from backtest.data_loader import download_all, load_and_split
from backtest.template_optimizer import discover_templates
from backtest.backtester import run_backtest, calc_win_rate, template_val_wr
from backtest.reporter import generate_reports

logger = logging.getLogger("backtest.pipeline")

_HERE       = Path(__file__).parent
RESULTS_DIR = _HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_pipeline(symbols: list, skip_download: bool = False, verbose: bool = False) -> dict:
    """
    Full 4-phase backtest pipeline with strict data separation.

    Phase 0: Data download & split
    Phase 1: Template discovery on TRAIN only
    Phase 2: Validation on VAL only — prune weak templates
    Phase 3: Final test on TEST only — no more changes
    Phase 4: Reports
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    print(f"\n{'='*60}")
    print(f"StockWise Backtest Pipeline — {len(symbols)} symbols")
    print(f"{'='*60}\n")

    # ── Phase 0: Data ─────────────────────────────────────────────────────
    print("Phase 0: Loading data...")
    if not skip_download:
        download_all(symbols)
    train, val, test = load_and_split(symbols)
    print(f"  Data split: {len(train)} train / {len(val)} val / {len(test)} test symbols")

    if not train:
        print("ERROR: No training data available. Check data download.")
        return {}

    # ── Phase 1: Template Discovery (TRAIN only) ──────────────────────────
    print("\nPhase 1: Template discovery (train data only)...")
    templates = discover_templates(train)
    print(f"  Discovered: {len(templates)} templates")

    if not templates:
        print("WARNING: No templates discovered. "
              "Try --symbols 50+ or relax thresholds in config.py")
        return {"templates": 0, "val_wr": 0, "test_wr": 0, "verdict": "FAIL"}

    # ── Phase 2: Validation (VAL only) ────────────────────────────────────
    print("\nPhase 2: Validation (val data only)...")
    if val:
        val_trades, val_daily = run_backtest(val, templates)
        val_wr = calc_win_rate(val_trades)
        print(f"  Trades: {len(val_trades)} | Win rate: {val_wr:.1f}%")
    else:
        print("  WARNING: No validation data available")
        val_trades, val_daily = [], []
        val_wr = 0.0

    # Prune templates with < 60% win rate on validation
    if val_trades:
        surviving = [t for t in templates if template_val_wr(t, val_trades) >= 60.0]
    else:
        surviving = templates  # Keep all if no val data (small symbol count)

    print(f"  Survived validation: {len(surviving)}/{len(templates)} templates")

    if not surviving:
        print("WARNING: All templates failed validation. Using top 3 by train WR.")
        surviving = sorted(templates,
                           key=lambda t: t['statistics'].get('win_rate', 0),
                           reverse=True)[:3]

    # ── Phase 3: Test (TEST only, FINAL) ─────────────────────────────────
    print("\nPhase 3: Final test (unseen test data)...")
    if test:
        test_trades, test_daily = run_backtest(test, surviving)
        test_wr = calc_win_rate(test_trades)
        print(f"  Trades: {len(test_trades)} | Win rate: {test_wr:.1f}%")
    else:
        print("  WARNING: No test data available (dates may be in future)")
        test_trades, test_daily = [], []
        test_wr = 0.0

    # ── Phase 4: Reports ──────────────────────────────────────────────────
    print("\nPhase 4: Generating reports...")
    generate_reports(
        all_templates=templates,
        surviving_templates=surviving,
        val_trades=val_trades,
        test_trades=test_trades,
        val_daily=val_daily,
        test_daily=test_daily,
    )

    # ── Final Verdict ─────────────────────────────────────────────────────
    verdict = "PASS" if test_wr >= MIN_WIN_RATE else "FAIL"
    print(f"\n{'='*60}")
    if verdict == "PASS":
        print(f"PASS — System meets {MIN_WIN_RATE}% win rate target ({test_wr:.1f}%)")
    else:
        print(f"FAIL — {test_wr:.1f}% < {MIN_WIN_RATE}%. See backtest/results/summary_report.txt")
    print(f"{'='*60}\n")

    return {
        "templates_discovered": len(templates),
        "templates_surviving": len(surviving),
        "val_trades": len(val_trades),
        "val_win_rate": val_wr,
        "test_trades": len(test_trades),
        "test_win_rate": test_wr,
        "verdict": verdict,
    }
