#!/usr/bin/env python3
"""
validation_runner.py — StockWise Gen-13 Automated System Validator
===================================================================
One command → full validation pipeline → data/validation_results.json

Phases:
  0. Environment  — imports, config keys, module load
  1. Data Fetch   — fetch OHLCV for each symbol via DSM waterfall
  2. Features     — calculate indicators via FeatureEngine
  3. Shadow Ledger— candle-by-candle template evaluation (separate ledger)
  4. Risk Gates   — synthetic portfolio risk checks
  5. pytest       — run all test files, collect per-file pass/fail
  6. Backtest     — chronological portfolio backtest (--full flag)

Usage:
    python validation_runner.py              # Full run (~5–15 min with data fetch)
    python validation_runner.py --quick      # Skip shadow ledger phase
    python validation_runner.py --no-pytest  # Skip pytest phase
    python validation_runner.py --symbols AAPL MSFT NVDA
    python validation_runner.py --full       # Include Phase 6 backtest
"""

import argparse
import copy
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

# ── Project root on sys.path ────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ── Constants ────────────────────────────────────────────────────────────────
VALIDATION_LEDGER_PATH = "data/validation_shadow_ledger.json"
OUTPUT_PATH = "data/validation_results.json"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
log = logging.getLogger("ValidationRunner")

# ── Test files (relative to project root) ───────────────────────────────────
TEST_FILES = [
    "tests/test_bug_1_3_er_trend.py",
    "tests/test_data_layer.py",
    "tests/test_execution.py",
    "tests/test_feature_engine.py",
    "tests/test_integration.py",
    "tests/test_integration_pipeline.py",
    "tests/test_notification.py",
    "tests/test_performance.py",
    "tests/test_portfolio_risk.py",
    "tests/test_regression.py",
    "tests/test_shadow_ledger.py",
    "tests/test_strategy_engine.py",
    "tests/test_template_system.py",
    "tests/test_vip_scanner.py",
    "tests/unit_tests.py",
    "tests/master_validator.py",
]

# Skipped — known collection errors from legacy gen7 files
SKIP_TEST_FILES = {"tests/test_gen7.py", "tests/test_gen7_validation.py"}


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _elapsed(start: float) -> float:
    return round(time.perf_counter() - start, 2)


def _safe(fn, default=None, label=""):
    """Call fn(), catch all exceptions, return default on error."""
    try:
        return fn()
    except Exception as exc:
        log.warning(f"[{label}] {exc}")
        return default


# ═══════════════════════════════════════════════════════════════════════════
# Phase 0 — Environment
# ═══════════════════════════════════════════════════════════════════════════

def phase_environment() -> dict:
    """Verify required modules import and key config keys exist."""
    log.info("Phase 0: Environment checks")
    t0 = time.perf_counter()

    REQUIRED_MODULES = [
        "system_config", "data_source_manager", "feature_engine",
        "shadow_ledger", "setup_templates", "portfolio_risk",
        "safe_json_io", "decision_logger",
    ]
    REQUIRED_CONFIG_KEYS = [
        "DEFAULT_TRAINING_SYMBOLS", "SHADOW_LEDGER_CONFIG", "RISK_CONFIG",
        "KINETIC_STOP_CONFIG", "PRE_MARKET_CONFIG", "SCAN_ROUTING_CONFIG",
        "OBSERVABILITY_CONFIG",
    ]

    module_results = {}
    for mod in REQUIRED_MODULES:
        try:
            __import__(mod)
            module_results[mod] = "OK"
        except Exception as exc:
            module_results[mod] = f"FAIL: {exc}"
            log.warning(f"  Module {mod}: {exc}")

    import system_config as cfg
    config_results = {}
    for key in REQUIRED_CONFIG_KEYS:
        config_results[key] = "OK" if hasattr(cfg, key) else "MISSING"

    # Template files
    templates_dir = os.path.join(PROJECT_ROOT, "data", "templates")
    template_files = []
    if os.path.isdir(templates_dir):
        template_files = [f for f in os.listdir(templates_dir) if f.endswith(".json")]

    modules_ok = sum(1 for v in module_results.values() if v == "OK")
    config_ok  = sum(1 for v in config_results.values() if v == "OK")

    return {
        "elapsed_s": _elapsed(t0),
        "modules_checked": len(REQUIRED_MODULES),
        "modules_ok": modules_ok,
        "modules_failed": len(REQUIRED_MODULES) - modules_ok,
        "module_detail": module_results,
        "config_keys_checked": len(REQUIRED_CONFIG_KEYS),
        "config_keys_ok": config_ok,
        "config_keys_missing": len(REQUIRED_CONFIG_KEYS) - config_ok,
        "config_detail": config_results,
        "template_files_found": len(template_files),
        "template_files": template_files,
        "passed": modules_ok == len(REQUIRED_MODULES) and config_ok == len(REQUIRED_CONFIG_KEYS),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1 — Data Fetch
# ═══════════════════════════════════════════════════════════════════════════

def phase_data_fetch(symbols: list, days_back: int) -> dict:
    """Fetch OHLCV data for each symbol via DSM waterfall."""
    log.info(f"Phase 1: Data fetch — {len(symbols)} symbols, {days_back} days back")
    t0 = time.perf_counter()

    from data_source_manager import DataSourceManager

    # use_ibkr=False by default — rely on Alpaca/YFinance waterfall
    try:
        dsm = DataSourceManager(use_ibkr=False, allow_fallback=True)
    except Exception as exc:
        log.warning(f"DSM init failed: {exc}")
        return {"elapsed_s": _elapsed(t0), "error": str(exc), "passed": False,
                "fetched": {}, "fetch_ok": 0, "fetch_failed": len(symbols)}

    fetched = {}   # symbol → df
    results = {}   # symbol → metadata

    for sym in symbols:
        t_sym = time.perf_counter()
        try:
            df = dsm.get_stock_data(sym, days_back=days_back, interval='1d')
            if df is None or df.empty:
                results[sym] = {"status": "empty", "rows": 0, "elapsed_s": _elapsed(t_sym)}
                log.warning(f"  {sym}: empty DataFrame")
            else:
                fetched[sym] = df
                results[sym] = {
                    "status": "ok",
                    "rows": len(df),
                    "start": str(df.index[0].date()) if hasattr(df.index[0], 'date') else str(df.index[0]),
                    "end":   str(df.index[-1].date()) if hasattr(df.index[-1], 'date') else str(df.index[-1]),
                    "elapsed_s": _elapsed(t_sym),
                }
                log.info(f"  {sym}: {len(df)} rows")
        except Exception as exc:
            results[sym] = {"status": "error", "error": str(exc), "elapsed_s": _elapsed(t_sym)}
            log.warning(f"  {sym}: {exc}")

    fetch_ok = sum(1 for v in results.values() if v["status"] == "ok")
    return {
        "elapsed_s": _elapsed(t0),
        "symbols": symbols,
        "fetch_ok": fetch_ok,
        "fetch_failed": len(symbols) - fetch_ok,
        "symbol_detail": results,
        "passed": fetch_ok > 0,
        "_frames": fetched,   # internal — stripped before JSON save
    }


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2 — Feature Engineering
# ═══════════════════════════════════════════════════════════════════════════

def phase_features(fetched_frames: dict) -> dict:
    """Run FeatureEngine.calculate_features on each fetched DataFrame."""
    log.info(f"Phase 2: Feature engineering — {len(fetched_frames)} symbols")
    t0 = time.perf_counter()

    from feature_engine import FeatureEngine
    fe = FeatureEngine()

    feature_frames = {}  # symbol → df_with_features
    results = {}

    for sym, df in fetched_frames.items():
        t_sym = time.perf_counter()
        try:
            df_feat = fe.calculate_features(df.copy())
            if df_feat is None or df_feat.empty:
                results[sym] = {"status": "empty", "columns": 0, "elapsed_s": _elapsed(t_sym)}
                log.warning(f"  {sym}: FeatureEngine returned empty")
            else:
                feature_frames[sym] = df_feat
                results[sym] = {
                    "status": "ok",
                    "rows": len(df_feat),
                    "columns": len(df_feat.columns),
                    "elapsed_s": _elapsed(t_sym),
                }
                log.info(f"  {sym}: {len(df_feat)} rows × {len(df_feat.columns)} features")
        except Exception as exc:
            results[sym] = {"status": "error", "error": str(exc), "elapsed_s": _elapsed(t_sym)}
            log.warning(f"  {sym}: FeatureEngine error — {exc}")

    feat_ok = sum(1 for v in results.values() if v["status"] == "ok")
    return {
        "elapsed_s": _elapsed(t0),
        "feat_ok": feat_ok,
        "feat_failed": len(fetched_frames) - feat_ok,
        "symbol_detail": results,
        "passed": feat_ok > 0,
        "_frames": feature_frames,  # internal
    }


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3 — Shadow Ledger
# ═══════════════════════════════════════════════════════════════════════════

def phase_shadow_ledger(feature_frames: dict) -> dict:
    """
    Run candle-by-candle Shadow Ledger evaluation using a SEPARATE ledger file
    (data/validation_shadow_ledger.json) to avoid touching production data.
    """
    log.info(f"Phase 3: Shadow Ledger — {len(feature_frames)} symbols")
    t0 = time.perf_counter()

    import system_config as cfg

    # Redirect to validation ledger (non-destructive)
    saved_config = copy.deepcopy(dict(cfg.SHADOW_LEDGER_CONFIG))
    cfg.SHADOW_LEDGER_CONFIG['ledger_path'] = VALIDATION_LEDGER_PATH

    try:
        from shadow_ledger import ShadowLedger
        sl = ShadowLedger()
        # Ensure ledger_path points to validation file (double-check after init)
        sl.ledger_path = VALIDATION_LEDGER_PATH
    except Exception as exc:
        cfg.SHADOW_LEDGER_CONFIG.update(saved_config)
        log.warning(f"ShadowLedger init failed: {exc}")
        return {"elapsed_s": _elapsed(t0), "error": str(exc), "passed": False}
    finally:
        # Always restore production config
        cfg.SHADOW_LEDGER_CONFIG.update(saved_config)

    results = {}
    all_template_stats = {}

    for sym, df in feature_frames.items():
        t_sym = time.perf_counter()
        try:
            per_template = sl.evaluate_history(sym, df)
            total_signals = sum(s["signal_count"] for s in per_template.values())
            total_wins    = sum(s["wins"]         for s in per_template.values())
            results[sym] = {
                "status": "ok",
                "templates_evaluated": len(per_template),
                "total_signals": total_signals,
                "total_wins": total_wins,
                "overall_win_rate": round(total_wins / total_signals * 100, 1) if total_signals > 0 else 0.0,
                "elapsed_s": _elapsed(t_sym),
                "template_detail": per_template,
            }
            all_template_stats[sym] = per_template
            log.info(f"  {sym}: {total_signals} signals, {total_wins} wins, "
                     f"{results[sym]['overall_win_rate']}% win rate")
        except Exception as exc:
            results[sym] = {"status": "error", "error": str(exc), "elapsed_s": _elapsed(t_sym)}
            log.warning(f"  {sym}: shadow ledger error — {exc}")

    # Save validation ledger
    try:
        sl._save_ledger()
        log.info(f"  Validation ledger saved to {VALIDATION_LEDGER_PATH}")
    except Exception as exc:
        log.warning(f"  Could not save validation ledger: {exc}")

    # Aggregate cross-symbol template stats
    template_aggregate = {}
    for sym_stats in all_template_stats.values():
        for tid, stats in sym_stats.items():
            if tid not in template_aggregate:
                template_aggregate[tid] = {"signal_count": 0, "wins": 0, "losses": 0, "symbols": 0}
            template_aggregate[tid]["signal_count"] += stats["signal_count"]
            template_aggregate[tid]["wins"]         += stats["wins"]
            template_aggregate[tid]["losses"]       += stats["losses"]
            template_aggregate[tid]["symbols"]      += 1

    for tid, agg in template_aggregate.items():
        sc = agg["signal_count"]
        agg["win_rate"] = round(agg["wins"] / sc * 100, 1) if sc > 0 else 0.0

    ok_count = sum(1 for v in results.values() if v["status"] == "ok")
    return {
        "elapsed_s": _elapsed(t0),
        "symbols_evaluated": ok_count,
        "symbols_failed": len(feature_frames) - ok_count,
        "symbol_detail": results,
        "template_aggregate": template_aggregate,
        "validation_ledger_path": VALIDATION_LEDGER_PATH,
        "passed": ok_count > 0,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4 — Portfolio Risk Gates
# ═══════════════════════════════════════════════════════════════════════════

def phase_risk_gates(feature_frames: dict) -> dict:
    """
    Run synthetic portfolio risk checks against the fetched data.
    Uses realistic test scenarios — no side effects on production state.
    """
    log.info("Phase 4: Portfolio risk gates")
    t0 = time.perf_counter()

    from portfolio_risk import PortfolioRiskManager
    mgr = PortfolioRiskManager()

    checks = []

    # ── Check 1: Correlation gate — 2 tech stocks → 3rd tech blocked ────────
    def _check_correlation_block():
        open_pos = {"AAPL": {"entry_price": 150, "qty": 10},
                    "MSFT": {"entry_price": 300, "qty": 5}}
        ok, reason = mgr.check_correlation_gate("NVDA", open_pos)
        return not ok, "3rd tech stock blocked by correlation gate", reason

    # ── Check 2: Correlation gate — different sector allowed ────────────────
    def _check_correlation_allow():
        open_pos = {"AAPL": {"entry_price": 150, "qty": 10},
                    "MSFT": {"entry_price": 300, "qty": 5}}
        ok, reason = mgr.check_correlation_gate("JNJ", open_pos)
        return ok, "Healthcare allowed alongside 2 tech positions", reason

    # ── Check 3: Drawdown circuit breaker ───────────────────────────────────
    def _check_circuit_breaker():
        pos = {"AAPL": {"entry_price": 150, "qty": 10}}
        mgr.check_drawdown_gate(pos, portfolio_value=100_000)  # set high-water mark
        ok, reason = mgr.check_drawdown_gate(pos, portfolio_value=89_000)  # 11% drawdown
        return not ok, "Circuit breaker fires at 11% drawdown", reason

    # ── Check 4: Zero portfolio blocked ─────────────────────────────────────
    def _check_zero_portfolio():
        ok, reason = mgr.check_drawdown_gate({}, portfolio_value=0)
        return not ok, "Zero portfolio value blocked", reason

    # ── Check 5: Unknown sector allowed ─────────────────────────────────────
    def _check_unknown_sector():
        ok, reason = mgr.check_correlation_gate("ZZZZ", {"AAPL": {"entry_price": 150, "qty": 10}})
        return ok, "Unknown sector stock not blocked", reason

    # ── Check 6: Weekly trend gate on live data ──────────────────────────────
    weekly_df_check = None
    weekly_sym = None
    for sym, df in feature_frames.items():
        if len(df) >= 200:
            weekly_df_check = df
            weekly_sym = sym
            break

    for label, fn in [
        ("correlation_block",  _check_correlation_block),
        ("correlation_allow",  _check_correlation_allow),
        ("circuit_breaker",    _check_circuit_breaker),
        ("zero_portfolio",     _check_zero_portfolio),
        ("unknown_sector",     _check_unknown_sector),
    ]:
        try:
            passed, description, detail = fn()
            checks.append({"check": label, "passed": passed,
                           "description": description, "detail": str(detail)})
            status = "PASS" if passed else "FAIL"
            log.info(f"  {status}: {description}")
        except Exception as exc:
            checks.append({"check": label, "passed": False,
                           "description": label, "error": str(exc)})
            log.warning(f"  ERROR: {label} — {exc}")

    # Weekly trend on live data
    if weekly_df_check is not None:
        try:
            wt_ok, wt_reason = mgr.check_weekly_trend_gate(weekly_sym, weekly_df_check)
            checks.append({
                "check": "weekly_trend_live",
                "passed": True,   # pass = didn't crash; result logged
                "description": f"Weekly trend gate on {weekly_sym}",
                "trend_ok": wt_ok,
                "detail": str(wt_reason),
            })
            log.info(f"  INFO: Weekly trend ({weekly_sym}): ok={wt_ok}, {wt_reason}")
        except Exception as exc:
            checks.append({"check": "weekly_trend_live", "passed": False, "error": str(exc)})

    checks_passed = sum(1 for c in checks if c["passed"])
    return {
        "elapsed_s": _elapsed(t0),
        "checks_total": len(checks),
        "checks_passed": checks_passed,
        "checks_failed": len(checks) - checks_passed,
        "check_detail": checks,
        "passed": checks_passed == len(checks),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Phase 5 — pytest
# ═══════════════════════════════════════════════════════════════════════════

def phase_pytest() -> dict:
    """
    Run each test file via subprocess pytest and parse pass/fail counts.
    Uses --tb=no for speed; captures stdout.
    """
    log.info("Phase 5: pytest")
    t0 = time.perf_counter()

    file_results = {}
    total_passed = 0
    total_failed = 0
    total_errors = 0

    for test_file in TEST_FILES:
        if test_file in SKIP_TEST_FILES:
            file_results[test_file] = {"status": "skipped", "reason": "known collection error"}
            continue

        abs_path = os.path.join(PROJECT_ROOT, test_file)
        if not os.path.exists(abs_path):
            file_results[test_file] = {"status": "not_found"}
            continue

        t_file = time.perf_counter()
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", abs_path, "--tb=no", "-q", "--no-header"],
                capture_output=True, text=True, timeout=120,
                cwd=PROJECT_ROOT,
            )
            stdout = result.stdout.strip()

            # Parse "X passed, Y failed" from pytest summary line
            passed = failed = error_count = 0
            for line in stdout.splitlines():
                line_lower = line.lower()
                if "passed" in line_lower or "failed" in line_lower or "error" in line_lower:
                    import re
                    m_passed = re.search(r'(\d+)\s+passed', line_lower)
                    m_failed = re.search(r'(\d+)\s+failed', line_lower)
                    m_error  = re.search(r'(\d+)\s+error', line_lower)
                    if m_passed: passed      = int(m_passed.group(1))
                    if m_failed: failed      = int(m_failed.group(1))
                    if m_error:  error_count = int(m_error.group(1))

            file_results[test_file] = {
                "status": "ok",
                "passed": passed,
                "failed": failed,
                "errors": error_count,
                "returncode": result.returncode,
                "elapsed_s": _elapsed(t_file),
            }
            total_passed += passed
            total_failed += failed + error_count
            marker = "PASS" if result.returncode == 0 else "FAIL"
            log.info(f"  {marker}: {test_file} — {passed}p {failed}f {error_count}e")

        except subprocess.TimeoutExpired:
            file_results[test_file] = {"status": "timeout", "elapsed_s": _elapsed(t_file)}
            total_errors += 1
            log.warning(f"  TIMEOUT: {test_file}")
        except Exception as exc:
            file_results[test_file] = {"status": "error", "error": str(exc)}
            total_errors += 1
            log.warning(f"  ERROR: {test_file} — {exc}")

    all_pass = total_failed == 0 and total_errors == 0
    return {
        "elapsed_s": _elapsed(t0),
        "files_run": len([v for v in file_results.values() if v.get("status") == "ok"]),
        "files_skipped": len([v for v in file_results.values() if v.get("status") == "skipped"]),
        "total_passed": total_passed,
        "total_failed": total_failed,
        "total_errors": total_errors,
        "file_detail": file_results,
        "passed": all_pass,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def phase_backtest(feature_frames: dict, symbols: list) -> dict:
    """Phase 6: Chronological portfolio backtest using cached feature frames."""
    t0 = time.perf_counter()
    from backtest_engine import BacktestEngine
    engine = BacktestEngine(
        data_cache=feature_frames,
        symbols=symbols,
        use_risk_gates=False,
    )
    bt = engine.run()
    summary = bt.get("summary", {})
    surv = bt.get("survivability", {})
    return {
        "passed": True,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "total_trades": len(bt.get("trades", [])),
        "total_return_pct": summary.get("total_return_pct", 0),
        "win_rate": summary.get("win_rate", 0),
        "profit_factor": summary.get("profit_factor", 0),
        "max_drawdown_pct": summary.get("max_drawdown_pct", 0),
        "survival_verdict": surv.get("survival_verdict", "NO_TRADES"),
        "risk_of_ruin_mc_pct": surv.get("risk_of_ruin_monte_carlo_pct", None),
        "summary": summary,
        "survivability": surv,
        "monthly_returns": bt.get("monthly_returns", []),
        "per_template": bt.get("per_template", {}),
        "per_symbol": bt.get("per_symbol", {}),
    }


def run_validation(symbols: list, days_back: int, quick: bool, no_pytest: bool,
                   run_backtest: bool = False) -> dict:
    run_start = time.perf_counter()
    run_ts = _ts()

    log.info("=" * 60)
    log.info("StockWise Gen-13 Validation Runner")
    log.info(f"Symbols: {symbols}")
    log.info(f"Mode: {'quick (shadow ledger skipped)' if quick else 'full'}")
    log.info("=" * 60)

    results = {
        "run_timestamp": run_ts,
        "symbols": symbols,
        "days_back": days_back,
        "quick_mode": quick,
        "phases": {},
    }

    # ── Phase 0: Environment ─────────────────────────────────────────────────
    results["phases"]["environment"] = _safe(
        phase_environment, default={"passed": False, "error": "exception"},
        label="P0"
    )

    # ── Phase 1: Data Fetch ──────────────────────────────────────────────────
    p1 = _safe(
        lambda: phase_data_fetch(symbols, days_back),
        default={"passed": False, "error": "exception", "_frames": {}},
        label="P1"
    )
    fetched_frames = p1.pop("_frames", {})
    results["phases"]["data_fetch"] = p1

    # ── Phase 2: Features ────────────────────────────────────────────────────
    if fetched_frames:
        p2 = _safe(
            lambda: phase_features(fetched_frames),
            default={"passed": False, "error": "exception", "_frames": {}},
            label="P2"
        )
        feature_frames = p2.pop("_frames", {})
        results["phases"]["features"] = p2
    else:
        feature_frames = {}
        results["phases"]["features"] = {"passed": False, "error": "no data from phase 1"}

    # ── Phase 3: Shadow Ledger ───────────────────────────────────────────────
    if not quick and feature_frames:
        results["phases"]["shadow_ledger"] = _safe(
            lambda: phase_shadow_ledger(feature_frames),
            default={"passed": False, "error": "exception"},
            label="P3"
        )
    else:
        reason = "quick mode" if quick else "no feature frames available"
        results["phases"]["shadow_ledger"] = {"passed": None, "skipped": True, "reason": reason}

    # ── Phase 4: Risk Gates ──────────────────────────────────────────────────
    results["phases"]["risk_gates"] = _safe(
        lambda: phase_risk_gates(feature_frames),
        default={"passed": False, "error": "exception"},
        label="P4"
    )

    # ── Phase 5: pytest ──────────────────────────────────────────────────────
    if not no_pytest:
        results["phases"]["pytest"] = _safe(
            phase_pytest,
            default={"passed": False, "error": "exception"},
            label="P5"
        )
    else:
        results["phases"]["pytest"] = {"passed": None, "skipped": True, "reason": "--no-pytest"}

    # ── Phase 6: Backtest (optional, --full flag) ─────────────────────────────
    if run_backtest and feature_frames:
        results["phases"]["backtest"] = _safe(
            lambda: phase_backtest(feature_frames, symbols),
            default={"passed": False, "error": "exception"},
            label="P6"
        )
    elif run_backtest:
        results["phases"]["backtest"] = {"passed": False, "skipped": True,
                                         "reason": "no feature frames available"}

    # ── Summary ───────────────────────────────────────────────────────────────
    phases_with_result = [
        v for v in results["phases"].values()
        if v.get("passed") is not None and not v.get("skipped")
    ]
    all_passed  = sum(1 for v in phases_with_result if v["passed"])
    all_failed  = len(phases_with_result) - all_passed
    overall_ok  = all_failed == 0

    results["summary"] = {
        "overall_passed": overall_ok,
        "phases_ok":      all_passed,
        "phases_failed":  all_failed,
        "total_elapsed_s": _elapsed(run_start),
        "data_symbols_ok":    results["phases"]["data_fetch"].get("fetch_ok", 0),
        "features_ok":        results["phases"]["features"].get("feat_ok", 0),
        "risk_checks_passed": results["phases"]["risk_gates"].get("checks_passed", 0),
        "pytest_tests_passed": results["phases"]["pytest"].get("total_passed", 0) if not no_pytest else None,
        "pytest_tests_failed": results["phases"]["pytest"].get("total_failed", 0) if not no_pytest else None,
    }

    return results


def _strip_internal(obj):
    """Remove _-prefixed internal keys before JSON serialisation."""
    if isinstance(obj, dict):
        return {k: _strip_internal(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [_strip_internal(i) for i in obj]
    return obj


def main():
    parser = argparse.ArgumentParser(description="StockWise Gen-13 Validation Runner")
    parser.add_argument("--quick",     action="store_true", help="Skip shadow ledger phase")
    parser.add_argument("--no-pytest", action="store_true", help="Skip pytest phase")
    parser.add_argument("--full",      action="store_true", help="Include Phase 6 backtest")
    parser.add_argument("--symbols",   nargs="+",           help="Override symbol list")
    parser.add_argument("--days-back", type=int, default=1095, help="Days of history (default 1095)")
    parser.add_argument("--output",    default=OUTPUT_PATH,  help="Output JSON path")
    args = parser.parse_args()

    import system_config as cfg
    symbols = args.symbols or list(getattr(cfg, "DEFAULT_TRAINING_SYMBOLS", [
        "SPY", "NVDA", "MSFT", "AAPL", "AMZN", "META", "GOOGL",
        "TSLA", "AMD", "NFLX", "BRK-B", "LLY", "AVGO",
    ]))

    results = run_validation(
        symbols=symbols,
        days_back=args.days_back,
        quick=args.quick,
        no_pytest=args.no_pytest,
        run_backtest=args.full,
    )

    # Strip internal keys and save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    clean = _strip_internal(results)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, default=str)

    log.info("=" * 60)
    s = results["summary"]
    log.info(f"OVERALL: {'PASS' if s['overall_passed'] else 'FAIL'}")
    log.info(f"  Phases OK/Failed : {s['phases_ok']}/{s['phases_failed']}")
    log.info(f"  Symbols fetched  : {s['data_symbols_ok']}")
    log.info(f"  Features computed: {s['features_ok']}")
    log.info(f"  Risk checks pass : {s['risk_checks_passed']}")
    if s["pytest_tests_passed"] is not None:
        log.info(f"  pytest pass/fail : {s['pytest_tests_passed']}/{s['pytest_tests_failed']}")
    log.info(f"  Total elapsed    : {s['total_elapsed_s']}s")
    log.info(f"Results saved to   : {args.output}")
    log.info("=" * 60)

    sys.exit(0 if s["overall_passed"] else 1)


if __name__ == "__main__":
    main()
