"""
IBKR Cross-Process Determinism Test (P0.2)

Purpose: Verify that IBKR returns IDENTICAL data when the same query
is issued from two separate processes. Required to qualify IBKR as
the deterministic provider for backtest reproducibility.

Usage:
    python tests/test_ibkr_determinism.py

Requires: IB Gateway running on configured port (system_config.IBKR_PORT).
Output:   data/ibkr_determinism_test/{round_1.json, round_2.json, comparison.json}
Exit:     0 if DETERMINISTIC, 1 otherwise.
"""
import os
import sys
import json
import time
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import system_config as cfg

# Configuration
TEST_SYMBOLS = ["AAPL", "AMD", "NFLX", "SPY", "CRM"]
TEST_OUTPUT_DIR = PROJECT_ROOT / "data" / "ibkr_determinism_test"
INTERVAL = "1d"
DAYS_BACK = 30
WAIT_BETWEEN_RUNS_SEC = 30
MIN_ROWS_REQUIRED = 15
SUBPROCESS_TIMEOUT_SEC = 30   # Was 120; clean fetch takes ~3s, 30s = 10x safety


def fetch_one_symbol(symbol: str) -> dict:
    """Subprocess entry point: fetch one symbol via IBKR, return hash + meta.

    CRITICAL: dsm.disconnect() in finally is mandatory.
    Without it, EClient.run() thread stays alive and the subprocess hangs
    until the parent's timeout kills it (verified 2026-04-26: 61s timeout
    vs 3s clean exit with disconnect).
    """
    from data_source_manager import DataSourceManager

    dsm = DataSourceManager()
    try:
        if not dsm.connect_to_ibkr():
            return {"symbol": symbol, "error": "IBKR connection failed"}

        try:
            df = dsm._download_from_ibkr(
                symbol=symbol,
                start_date=None,
                end_date=None,
                days_back=DAYS_BACK,
                interval=INTERVAL,
                min_rows=MIN_ROWS_REQUIRED
            )
        except Exception as e:
            return {"symbol": symbol, "error": f"Fetch exception: {e}"}

        if df is None or df.empty:
            return {"symbol": symbol, "error": "Empty DataFrame"}

        # Deterministic hash via canonical CSV bytes
        cols_lower = [c.lower() for c in df.columns]
        df_lower = df.copy()
        df_lower.columns = cols_lower

        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required if c not in df_lower.columns]
        if missing:
            return {"symbol": symbol, "error": f"Missing columns: {missing}"}

        canonical = df_lower[required].copy()
        payload = canonical.to_csv(index=True, lineterminator='\n').encode('utf-8')
        md5 = hashlib.md5(payload).hexdigest()

        return {
            "symbol": symbol,
            "rows": len(df),
            "first_date": str(df.index[0]),
            "last_date": str(df.index[-1]),
            "md5": md5,
            "error": None
        }
    finally:
        # Mandatory cleanup — without disconnect, EClient.run() thread
        # keeps the subprocess alive until parent timeout
        try:
            dsm.disconnect()
        except Exception:
            pass


def run_one_round(round_id: int) -> dict:
    """Run all symbols in subprocesses for one round."""
    print(f"\n=== Round {round_id} ===")
    results = {}

    for symbol in TEST_SYMBOLS:
        print(f"  Fetching {symbol}...", end=" ", flush=True)
        cmd = [
            sys.executable, "-c",
            f"import sys, os; sys.path.insert(0, r'{PROJECT_ROOT}'); "
            f"from tests.test_ibkr_determinism import fetch_one_symbol; "
            f"import json; "
            f"print('===RESULT===' + json.dumps(fetch_one_symbol('{symbol}'))); "
            f"sys.stdout.flush(); "
            f"os._exit(0)"
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_SEC
            )
            out = proc.stdout
            marker = "===RESULT==="
            if marker in out:
                result_json = out.split(marker, 1)[1].strip().split('\n')[0]
                result = json.loads(result_json)
            else:
                result = {
                    "symbol": symbol,
                    "error": f"No result marker in output. stdout={out[:200]} stderr={proc.stderr[:200]}"
                }
        except subprocess.TimeoutExpired:
            result = {"symbol": symbol, "error": f"Subprocess timeout ({SUBPROCESS_TIMEOUT_SEC}s)"}
        except Exception as e:
            result = {"symbol": symbol, "error": f"Subprocess failed: {e}"}

        results[symbol] = result
        if result.get("error"):
            print(f"X {result['error'][:80]}")
        else:
            print(f"OK rows={result['rows']} md5={result['md5'][:8]}")

    return {
        "round_id": round_id,
        "timestamp": datetime.utcnow().isoformat(),
        "results": results
    }


def compare_rounds(round1: dict, round2: dict) -> dict:
    """Compare hashes between two rounds."""
    comparison = {"symbols": {}, "overall": "DETERMINISTIC"}

    for symbol in TEST_SYMBOLS:
        r1 = round1["results"].get(symbol, {})
        r2 = round2["results"].get(symbol, {})

        if r1.get("error") or r2.get("error"):
            verdict = "ERROR"
            detail = f"R1={r1.get('error', 'OK')} | R2={r2.get('error', 'OK')}"
        elif r1.get("md5") != r2.get("md5"):
            verdict = "DIVERGENT"
            detail = (
                f"R1: md5={r1.get('md5', 'N/A')[:12]} rows={r1.get('rows')} | "
                f"R2: md5={r2.get('md5', 'N/A')[:12]} rows={r2.get('rows')}"
            )
        else:
            verdict = "STABLE"
            detail = f"md5={r1['md5'][:12]} rows={r1['rows']}"

        comparison["symbols"][symbol] = {"verdict": verdict, "detail": detail}

        if verdict != "STABLE":
            comparison["overall"] = "NOT_DETERMINISTIC"

    return comparison


def main():
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"IBKR Cross-Process Determinism Test")
    print(f"Symbols: {TEST_SYMBOLS}")
    print(f"Days back: {DAYS_BACK} | Interval: {INTERVAL}")
    print(f"Output: {TEST_OUTPUT_DIR}")

    round1 = run_one_round(1)
    with open(TEST_OUTPUT_DIR / "round_1.json", "w", encoding="utf-8") as f:
        json.dump(round1, f, indent=2)

    print(f"\nWaiting {WAIT_BETWEEN_RUNS_SEC}s before round 2 (IBKR rate limit safety)...")
    time.sleep(WAIT_BETWEEN_RUNS_SEC)

    round2 = run_one_round(2)
    with open(TEST_OUTPUT_DIR / "round_2.json", "w", encoding="utf-8") as f:
        json.dump(round2, f, indent=2)

    print(f"\n=== COMPARISON ===")
    comparison = compare_rounds(round1, round2)

    for symbol, result in comparison["symbols"].items():
        marker = {"STABLE": "OK", "DIVERGENT": "FAIL", "ERROR": "ERR"}[result["verdict"]]
        print(f"  [{marker}] {symbol}: {result['verdict']} - {result['detail']}")

    print(f"\n=== VERDICT: {comparison['overall']} ===")

    with open(TEST_OUTPUT_DIR / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    sys.exit(0 if comparison["overall"] == "DETERMINISTIC" else 1)


if __name__ == "__main__":
    main()
