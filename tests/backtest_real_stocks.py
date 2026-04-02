# tests/backtest_real_stocks.py

"""
StockWise Gen-13 Historical Backtest
====================================
Runs the FULL template pipeline on real historical data.
Reports template performance, signal quality, and system weaknesses.

Usage:
    python tests/backtest_real_stocks.py [--provider YFINANCE] [--days 500] [--symbols AAPL,MSFT,NVDA]

Requires network access for data fetching.
"""

import sys
import os
import argparse
import logging
import time
import types
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Stubs
for mod in ['xgboost', 'xgboost.sklearn', 'xgboost.core']:
    if mod not in sys.modules:
        sys.modules[mod] = types.ModuleType(mod)
try:
    import pandas_ta
except ImportError:
    try:
        import pandas_ta_classic
        sys.modules['pandas_ta'] = pandas_ta_classic
    except ImportError:
        sys.modules['pandas_ta'] = types.ModuleType('pandas_ta')

import system_config as cfg
from data_source_manager import DataSourceManager
from feature_engine import FeatureEngine
from stock_hunter import StockHunter
from template_matcher import TemplateMatcher
from portfolio_risk import PortfolioRiskManager
from shadow_ledger import ShadowLedger  # CP-1: evolution engine verification
from safe_json_io import safe_json_read, safe_json_write  # CP-1

logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("Backtest")
logger.setLevel(logging.DEBUG)


def run_backtest(symbols, days_back=500, provider='YFINANCE'):
    """
    Run the full pipeline on each day of historical data.
    Simulates what would have happened if the system was running.
    """
    dm = DataSourceManager()
    fe = FeatureEngine()
    hunter = StockHunter(dm)
    matcher = TemplateMatcher()

    # Results tracking
    results = {
        "total_days_scanned": 0,
        "total_signals": 0,
        "signals_by_template": {},
        "trades": [],
        "days_with_signals": 0,
        "days_without_signals": 0,
    }

    for symbol in symbols:
        logger.info(f"\n{'='*60}")
        logger.info(f"BACKTESTING: {symbol} ({days_back} days)")
        logger.info(f"{'='*60}")

        try:
            # Fetch full history
            df_raw = dm.get_stock_data(symbol, days_back=days_back)

            if df_raw is None or len(df_raw) < 200:
                logger.warning(f"[{symbol}] Insufficient data: {len(df_raw) if df_raw is not None else 0} rows")
                continue

            logger.info(f"[{symbol}] Fetched {len(df_raw)} rows")

            # Calculate features on full dataset
            df_features = fe.calculate_features(df_raw)

            # Walk forward: scan each day from day 200 onward
            start_idx = 200  # Need at least 200 days for indicators

            for i in range(start_idx, len(df_features)):
                results["total_days_scanned"] += 1

                # Slice: all data up to day i (simulate having only past data)
                df_slice = df_features.iloc[:i+1]

                # Classify stock state
                state = hunter.classify_stock_state(df_slice)

                # Run template matcher
                signals = matcher.scan_ticker(symbol, df_slice, stock_state=state)

                # Detailed debug: log ALL template evaluations, not just matches
                if not signals:
                    state_matched = matcher.tm.get_for_state(state)
                    all_enabled = matcher.tm.get_enabled()

                    if len(state_matched) == 0:
                        logger.debug(f"[{symbol}][Day {i}] STATE BLOCK: {state} matched 0/{len(all_enabled)} templates")
                    else:
                        for tmpl in state_matched:
                            passed, details = tmpl.evaluate_conditions(df_slice.iloc[-1])
                            failed_blocks = [d['block'] for d in details if not d.get('passed')]
                            if failed_blocks:
                                logger.debug(f"[{symbol}][Day {i}] {tmpl.id}: BLOCKS FAILED: {failed_blocks}")

                if signals:
                    results["days_with_signals"] += 1
                    results["total_signals"] += len(signals)

                    for signal in signals:
                        template_id = signal['template_id']
                        if template_id not in results["signals_by_template"]:
                            results["signals_by_template"][template_id] = {
                                "count": 0, "wins": 0, "losses": 0, "draws": 0,
                                "total_profit": 0, "total_loss": 0
                            }
                        results["signals_by_template"][template_id]["count"] += 1

                        # Simulate: check what happened in next 5 days
                        entry_price = signal['entry_price']
                        stop_price = signal['stop_loss']
                        target_price = signal['take_profit']

                        if i + 5 < len(df_features):
                            future = df_features.iloc[i+1:i+6]
                            max_high = future['high'].max()
                            min_low = future['low'].min()
                            exit_close = future.iloc[-1]['close']

                            # Check stop hit first (worst case)
                            if min_low <= stop_price:
                                pnl = ((stop_price - entry_price) / entry_price) * 100
                                results["signals_by_template"][template_id]["losses"] += 1
                                results["signals_by_template"][template_id]["total_loss"] += abs(pnl)
                            elif max_high >= target_price:
                                pnl = ((target_price - entry_price) / entry_price) * 100
                                results["signals_by_template"][template_id]["wins"] += 1
                                results["signals_by_template"][template_id]["total_profit"] += pnl
                            else:
                                pnl = ((exit_close - entry_price) / entry_price) * 100
                                if pnl > 0:
                                    results["signals_by_template"][template_id]["wins"] += 1
                                    results["signals_by_template"][template_id]["total_profit"] += pnl
                                else:
                                    results["signals_by_template"][template_id]["losses"] += 1
                                    results["signals_by_template"][template_id]["total_loss"] += abs(pnl)

                            date_str = (df_features.index[i].strftime('%Y-%m-%d')
                                        if hasattr(df_features.index[i], 'strftime')
                                        else str(df_features.index[i]))
                            results["trades"].append({
                                "symbol": symbol,
                                "date": date_str,
                                "template": template_id,
                                "entry": entry_price,
                                "stop": stop_price,
                                "target": target_price,
                                "pnl_pct": round(pnl, 2),
                                "won": pnl > 0,
                                "state": state,
                            })
                else:
                    results["days_without_signals"] += 1

            time.sleep(1)  # API throttle between stocks

        except Exception as e:
            logger.error(f"[{symbol}] Backtest failed: {e}")
            import traceback
            traceback.print_exc()

    return results


def print_report(results):
    """Print a formatted backtest report."""
    print(f"\n{'='*70}")
    print(f"  STOCKWISE BACKTEST REPORT")
    print(f"{'='*70}")
    print(f"  Days Scanned:    {results['total_days_scanned']}")
    print(f"  Total Signals:   {results['total_signals']}")
    print(f"  Days w/ Signal:  {results['days_with_signals']}")
    print(f"  Days w/o Signal: {results['days_without_signals']}")

    idle_pct = (results['days_without_signals'] / max(results['total_days_scanned'], 1)) * 100
    print(f"  Idle Rate:       {idle_pct:.1f}%")

    if results["signals_by_template"]:
        print(f"\n--- Template Performance ---")
        print(f"  {'Template':<25} {'Signals':>8} {'Wins':>6} {'Losses':>6} {'WR%':>6} {'AvgP':>7} {'AvgL':>7} {'PF':>6}")
        print(f"  {'-'*68}")

        for tid, stats in sorted(results["signals_by_template"].items(),
                                  key=lambda x: x[1]['count'], reverse=True):
            total = stats['wins'] + stats['losses']
            wr = (stats['wins'] / total * 100) if total > 0 else 0
            avg_p = (stats['total_profit'] / stats['wins']) if stats['wins'] > 0 else 0
            avg_l = (stats['total_loss'] / stats['losses']) if stats['losses'] > 0 else 0
            pf = (stats['total_profit'] / stats['total_loss']) if stats['total_loss'] > 0 else float('inf')

            print(f"  {tid:<25} {stats['count']:>8} {stats['wins']:>6} {stats['losses']:>6} "
                  f"{wr:>5.1f}% {avg_p:>6.2f}% {avg_l:>6.2f}% {pf:>5.2f}")

    # Overall stats
    all_trades = results["trades"]
    if all_trades:
        wins = sum(1 for t in all_trades if t['won'])
        losses = len(all_trades) - wins
        total_pnl = sum(t['pnl_pct'] for t in all_trades)
        print(f"\n--- Overall ---")
        print(f"  Total Trades:    {len(all_trades)}")
        print(f"  Wins:            {wins} ({wins/len(all_trades)*100:.1f}%)")
        print(f"  Losses:          {losses}")
        print(f"  Total PnL:       {total_pnl:+.2f}%")
        print(f"  Avg PnL/Trade:   {total_pnl/len(all_trades):+.2f}%")

        # CP-1: Profit Factor, MaxDD, Edge, Kelly
        wins_pnl = [t['pnl_pct'] for t in all_trades if t['won']]
        loss_pnl = [abs(t['pnl_pct']) for t in all_trades if not t['won']]
        total_profit = sum(wins_pnl)
        total_loss_abs = sum(loss_pnl)
        pf = total_profit / total_loss_abs if total_loss_abs > 0 else float('inf')
        avg_win = total_profit / len(wins_pnl) if wins_pnl else 0.0
        avg_loss = total_loss_abs / len(loss_pnl) if loss_pnl else 0.0
        cum_pnl, peak_pnl, max_dd = 0.0, 0.0, 0.0
        for t in all_trades:
            cum_pnl += t['pnl_pct']
            peak_pnl = max(peak_pnl, cum_pnl)
            max_dd = max(max_dd, peak_pnl - cum_pnl)
        wr_f = wins / len(all_trades)
        rr = avg_win / avg_loss if avg_loss > 0 else 0.0
        edge = (wr_f * avg_win - (1 - wr_f) * avg_loss) / avg_loss if avg_loss > 0 else 0.0
        kelly = max(0.0, wr_f - (1 - wr_f) / rr) * 100 if rr > 0 else 0.0
        print(f"  Profit Factor:   {pf:.2f}")
        print(f"  Max Drawdown:    {max_dd:.2f}%")
        print(f"  Edge:            {edge:.2f}")
        print(f"  Kelly:           {kelly:.1f}%")

        # Weaknesses analysis
        print(f"\n--- Weakness Analysis ---")

        if results["signals_by_template"]:
            worst_template = min(
                results["signals_by_template"].items(),
                key=lambda x: (x[1]['wins'] / max(x[1]['wins'] + x[1]['losses'], 1))
            )
            print(f"  Worst template:  {worst_template[0]}")

        max_losing_streak = 0
        current_streak = 0
        for t in all_trades:
            if not t['won']:
                current_streak += 1
                max_losing_streak = max(max_losing_streak, current_streak)
            else:
                current_streak = 0
        print(f"  Max losing streak: {max_losing_streak}")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StockWise Backtest")
    parser.add_argument("--provider", default="YFINANCE", help="Data provider")
    parser.add_argument("--days", type=int, default=500, help="Days of history")
    parser.add_argument("--symbols", default="AAPL,MSFT,NVDA,GOOGL,AMZN",
                        help="Comma-separated symbols")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]

    print(f"Starting backtest: {symbols} | {args.days} days | Provider: {args.provider}")
    results = run_backtest(symbols, days_back=args.days, provider=args.provider)
    print_report(results)

    # ── CP-1: Evolution Engine Verification ──────────────────────────────────
    print("\n" + "="*70)
    print("  CP-1: SHADOW LEDGER EVALUATION (attribution + coverage + auto-disable)")
    print("="*70)

    # Step A: run evaluate_history per symbol (attribution + coverage gap data)
    dm_sl = DataSourceManager()
    fe_sl = FeatureEngine()
    hunter_sl = StockHunter(dm_sl)
    sl = ShadowLedger()
    sl._coverage_data = {}

    for sym in symbols:
        try:
            df_raw = dm_sl.get_stock_data(sym, days_back=args.days)
            if df_raw is None or len(df_raw) < 200:
                print(f"  [{sym}] Skipped — insufficient data")
                continue
            df_f = fe_sl.calculate_features(df_raw)
            state_fn = lambda df_s, _h=hunter_sl: _h.classify_stock_state(df_s)
            sl.evaluate_history(sym, df_f, stock_state_fn=state_fn)
            print(f"  [{sym}] evaluate_history complete")
        except Exception as _ex:
            print(f"  [{sym}] Error: {_ex}")

    sl._save_ledger()               # template_stats + metadata
    sl._finalize_coverage_gaps()   # coverage_gaps (merged into existing file)
    print("[Shadow Ledger] Data written to data/shadow_ledger.json")

    # Step B: evaluate auto-disable for each unique template+symbol+trend combo
    print("\n[Auto-Disable] Evaluating combos from shadow_ledger stats...")
    shadow_stats = safe_json_read('data/shadow_ledger.json', default={}).get('template_stats', {})
    matcher_sl = TemplateMatcher()
    for sym, tmpl_stats in shadow_stats.items():
        for tid, stats in tmpl_stats.items():
            # Use dominant trend from backtest trades, fall back to BULLISH
            sym_trades = [t for t in results['trades'] if t['symbol'] == sym and t['template'] == tid]
            trend_counts = {}
            for t in sym_trades:
                st = t.get('state') or {}
                tr = (st.get('trend', 'BULLISH') if isinstance(st, dict) else 'BULLISH')
                trend_counts[tr] = trend_counts.get(tr, 0) + 1
            trends_to_check = list(trend_counts.keys()) if trend_counts else ['BULLISH']
            for trend in trends_to_check:
                matcher_sl.evaluate_auto_disable(
                    template_id=tid, symbol=sym,
                    stock_state={'trend': trend},
                    shadow_stats=shadow_stats,
                )
    dc_final = safe_json_read('data/shadow_ledger.json', default={}).get('disabled_combos', [])
    print(f"[Auto-Disable] Done. Disabled combos: {len(dc_final)}")
    for c in sorted(dc_final):
        print(f"  DISABLED: {c}")
