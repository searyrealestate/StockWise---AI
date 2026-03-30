"""
Entry point for the StockWise backtest pipeline.

Usage:
  Quick test:  python backtest/run_backtest.py --symbols 5 --verbose
  Medium test: python backtest/run_backtest.py --symbols 50
  Full run:    python backtest/run_backtest.py --symbols 500
  Skip DL:     python backtest/run_backtest.py --skip-download --symbols 500
"""

import sys
import os

# Add project root to path so imports work regardless of CWD
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from backtest.pipeline import run_pipeline
from backtest.data_loader import SP500_SYMBOLS

parser = argparse.ArgumentParser(
    description="StockWise Backtest & Validation Pipeline",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=__doc__,
)
parser.add_argument("--symbols", type=int, default=len(SP500_SYMBOLS),
                    help=f"Number of S&P 500 symbols to use (default: {len(SP500_SYMBOLS)})")
parser.add_argument("--skip-download", action="store_true",
                    help="Skip data download (use cached parquet files only)")
parser.add_argument("--verbose", action="store_true",
                    help="Enable DEBUG-level logging")
parser.add_argument("--train-only", action="store_true",
                    help="Stop after template discovery (skip val/test phases)")
args = parser.parse_args()

symbols = SP500_SYMBOLS[:args.symbols]
print(f"Using {len(symbols)} symbols: {symbols[:5]}{'...' if len(symbols) > 5 else ''}")

result = run_pipeline(
    symbols=symbols,
    skip_download=args.skip_download,
    verbose=args.verbose,
)

sys.exit(0 if result.get("verdict") == "PASS" else 1)
