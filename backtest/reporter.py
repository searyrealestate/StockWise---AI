"""
Generates all 5 output report files in backtest/results/.
"""

import os
import json
import logging
import math
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import pandas as pd
import numpy as np

from backtest.config import (
    MIN_WIN_RATE, STARTING_CAPITAL, TARGET_DAILY_RETURN
)

logger = logging.getLogger("backtest.reporter")

_HERE       = Path(__file__).parent
RESULTS_DIR = _HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _safe(val, default=0.0):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return default
    return float(val)


def _calc_metrics(trades: list, daily: list) -> dict:
    """Compute standard performance metrics from trades + daily returns."""
    if not trades:
        return {"win_rate": 0, "profit_factor": 0, "avg_daily_return": 0,
                "max_drawdown": 0, "sharpe": 0, "total_trades": 0,
                "total_pnl": 0}

    wins    = [t for t in trades if t['pnl_dollars'] > 0]
    losses  = [t for t in trades if t['pnl_dollars'] <= 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0

    total_profit = sum(t['pnl_dollars'] for t in wins)
    total_loss   = abs(sum(t['pnl_dollars'] for t in losses))
    profit_factor = total_profit / total_loss if total_loss > 0 else (total_profit if total_profit > 0 else 0)

    daily_returns = [d['daily_return_pct'] for d in daily]
    avg_daily = sum(daily_returns) / len(daily_returns) if daily_returns else 0

    # Max drawdown
    peak = STARTING_CAPITAL
    max_dd = 0.0
    for d in daily:
        v = d['portfolio_value']
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    # Sharpe (annualized, assume 252 trading days, risk-free ~5%)
    if len(daily_returns) > 1:
        ret_arr = np.array(daily_returns)
        rf_daily = 5.0 / 252
        excess = ret_arr - rf_daily
        sharpe = (excess.mean() / excess.std() * math.sqrt(252)) if excess.std() > 0 else 0
    else:
        sharpe = 0

    return {
        "win_rate": round(win_rate, 2),
        "profit_factor": round(profit_factor, 2),
        "avg_daily_return": round(avg_daily, 4),
        "max_drawdown": round(max_dd, 2),
        "sharpe": round(sharpe, 3),
        "total_trades": len(trades),
        "total_pnl": round(sum(t['pnl_dollars'] for t in trades), 2),
    }


def generate_reports(
    all_templates: list,
    surviving_templates: list,
    val_trades: list,
    test_trades: list,
    val_daily: list,
    test_daily: list,
    train_trades: list = None,
    train_daily: list = None,
) -> None:
    """Generate all 5 report files."""

    _save_trades_log(val_trades + test_trades)
    _save_template_performance(all_templates, surviving_templates, val_trades, test_trades)
    _save_indicator_analysis(all_templates)
    _save_daily_returns(test_daily, val_daily)
    _save_summary_report(all_templates, surviving_templates, val_trades, test_trades,
                          val_daily, test_daily, train_trades or [], train_daily or [])

    logger.info("All 5 reports generated in backtest/results/")


def _save_trades_log(trades: list):
    if not trades:
        pd.DataFrame().to_csv(RESULTS_DIR / "trades_log.csv", index=False)
        return
    rows = []
    for t in trades:
        row = {k: v for k, v in t.items()
               if not isinstance(v, (dict, list))}
        row['stock_state'] = json.dumps(t.get('stock_state', {}))
        row['kinetic_phases'] = str(t.get('kinetic_phases_visited', []))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "trades_log.csv", index=False)
    logger.info(f"trades_log.csv: {len(df)} rows")


def _save_template_performance(all_templates, surviving, val_trades, test_trades):
    rows = []
    surviving_ids = {t['id'] for t in surviving}

    for tmpl in all_templates:
        tid = tmpl['id']
        train_wr = _safe(tmpl['statistics'].get('win_rate', 0))

        val_t   = [t for t in val_trades  if t['template_id'] == tid]
        test_t  = [t for t in test_trades if t['template_id'] == tid]

        val_wins = sum(1 for t in val_t  if t['pnl_dollars'] > 0)
        tst_wins = sum(1 for t in test_t if t['pnl_dollars'] > 0)

        val_wr  = val_wins  / len(val_t)  * 100 if val_t  else 0
        test_wr = tst_wins  / len(test_t) * 100 if test_t else 0
        overfit = abs(train_wr - test_wr)

        blocks_str = "|".join(c['block'] for c in tmpl.get('conditions', []))
        avg_profit = sum(t['pnl_dollars'] for t in test_t if t['pnl_dollars'] > 0) / max(len([t for t in test_t if t['pnl_dollars'] > 0]), 1)
        avg_loss   = sum(t['pnl_dollars'] for t in test_t if t['pnl_dollars'] <= 0) / max(len([t for t in test_t if t['pnl_dollars'] <= 0]), 1)
        test_profit = sum(t['pnl_dollars'] for t in test_t if t['pnl_dollars'] > 0)
        test_loss   = abs(sum(t['pnl_dollars'] for t in test_t if t['pnl_dollars'] <= 0))
        pf = test_profit / test_loss if test_loss > 0 else (test_profit if test_profit > 0 else 0)

        rows.append({
            'template_id': tid,
            'blocks': blocks_str,
            'win_rate_train': round(train_wr, 2),
            'win_rate_val': round(val_wr, 2),
            'win_rate_test': round(test_wr, 2),
            'total_trades': len(test_t),
            'avg_profit': round(avg_profit, 2),
            'avg_loss': round(avg_loss, 2),
            'profit_factor': round(pf, 2),
            'overfit_score': round(overfit, 2),
            'survived_validation': tid in surviving_ids,
        })

    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        'template_id','blocks','win_rate_train','win_rate_val','win_rate_test',
        'total_trades','avg_profit','avg_loss','profit_factor','overfit_score','survived_validation'
    ])
    df.to_csv(RESULTS_DIR / "template_performance.csv", index=False)
    logger.info(f"template_performance.csv: {len(df)} templates")


def _save_indicator_analysis(templates: list):
    """Save indicator importance. Falls back to recomputing if file doesn't exist."""
    out = RESULTS_DIR / "indicator_analysis.csv"
    if out.exists():
        return  # Already written by template_optimizer

    from collections import defaultdict
    block_freq  = defaultdict(int)
    block_wr    = defaultdict(list)
    for t in templates:
        wr = _safe(t['statistics'].get('win_rate', 0))
        for cond in t.get('conditions', []):
            name = cond['block']
            block_freq[name] += 1
            block_wr[name].append(wr)

    rows = []
    for name, freq in sorted(block_freq.items(), key=lambda x: -x[1]):
        rows.append({
            'block_name': name,
            'frequency': freq,
            'avg_win_rate_contribution': round(sum(block_wr[name]) / len(block_wr[name]), 2) if block_wr[name] else 0,
        })
    pd.DataFrame(rows).to_csv(out, index=False)


def _save_daily_returns(test_daily: list, val_daily: list = None):
    all_daily = (val_daily or []) + (test_daily or [])
    if not all_daily:
        pd.DataFrame(columns=['date','portfolio_value','daily_return_pct',
                               'cumulative_return_pct','open_positions','cash']).to_csv(
            RESULTS_DIR / "daily_returns.csv", index=False)
        return
    df = pd.DataFrame(all_daily)
    df.to_csv(RESULTS_DIR / "daily_returns.csv", index=False)
    logger.info(f"daily_returns.csv: {len(df)} rows")


def _save_summary_report(all_templates, surviving, val_trades, test_trades,
                          val_daily, test_daily, train_trades, train_daily):
    val_m  = _calc_metrics(val_trades,  val_daily)
    test_m = _calc_metrics(test_trades, test_daily)
    train_m = _calc_metrics(train_trades, train_daily) if train_trades else {}

    # Monthly breakdown from test
    monthly = defaultdict(list)
    for d in test_daily:
        month = d['date'][:7]
        monthly[month].append(d['daily_return_pct'])

    # Template rankings
    def template_test_pf(t):
        tid = t['id']
        test_t = [tr for tr in test_trades if tr['template_id'] == tid]
        wins = sum(p['pnl_dollars'] for p in test_t if p['pnl_dollars'] > 0)
        losses = abs(sum(p['pnl_dollars'] for p in test_t if p['pnl_dollars'] <= 0))
        return wins / losses if losses > 0 else wins

    surviving_sorted = sorted(surviving, key=template_test_pf, reverse=True)
    top5     = surviving_sorted[:5]

    # Overfit suspects: high train WR, low test WR
    def overfit_score(t):
        train_wr = _safe(t['statistics'].get('win_rate', 0))
        test_t = [tr for tr in test_trades if tr['template_id'] == t['id']]
        if not test_t:
            return 0
        test_wr = sum(1 for tr in test_t if tr['pnl_dollars'] > 0) / len(test_t) * 100
        return train_wr - test_wr

    worst5 = sorted(all_templates, key=overfit_score, reverse=True)[:5]

    # Indicator importance top 20
    ia_path = RESULTS_DIR / "indicator_analysis.csv"
    indicator_lines = ""
    if ia_path.exists():
        try:
            ia_df = pd.read_csv(ia_path).head(20)
            indicator_lines = ia_df.to_string(index=False)
        except Exception:
            pass

    pass_fail = "PASS" if test_m.get('win_rate', 0) >= MIN_WIN_RATE else "FAIL"

    lines = [
        "=" * 70,
        "STOCKWISE BACKTEST & VALIDATION REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        "── PERFORMANCE SUMMARY ──────────────────────────────────────────────",
        f"{'Metric':<30} {'Train':>10} {'Val':>10} {'Test':>10}",
        f"{'Win Rate (%)':<30} {train_m.get('win_rate','N/A'):>10} {val_m['win_rate']:>10} {test_m['win_rate']:>10}",
        f"{'Total Trades':<30} {train_m.get('total_trades','N/A'):>10} {val_m['total_trades']:>10} {test_m['total_trades']:>10}",
        f"{'Profit Factor':<30} {train_m.get('profit_factor','N/A'):>10} {val_m['profit_factor']:>10} {test_m['profit_factor']:>10}",
        f"{'Avg Daily Return (%)':<30} {train_m.get('avg_daily_return','N/A'):>10} {val_m['avg_daily_return']:>10} {test_m['avg_daily_return']:>10}",
        f"{'Max Drawdown (%)':<30} {train_m.get('max_drawdown','N/A'):>10} {val_m['max_drawdown']:>10} {test_m['max_drawdown']:>10}",
        f"{'Sharpe Ratio':<30} {train_m.get('sharpe','N/A'):>10} {val_m['sharpe']:>10} {test_m['sharpe']:>10}",
        f"{'Total P&L ($)':<30} {train_m.get('total_pnl','N/A'):>10} {val_m['total_pnl']:>10} {test_m['total_pnl']:>10}",
        "",
        "── TEMPLATE DISCOVERY ───────────────────────────────────────────────",
        f"Templates discovered (train): {len(all_templates)}",
        f"Survived validation (>=60% WR): {len(surviving)}",
        f"Survival rate: {len(surviving)/max(len(all_templates),1)*100:.1f}%",
        "",
        "── TOP 5 TEMPLATES (by test profit factor) ──────────────────────────",
    ]

    for t in top5:
        blocks = ", ".join(c['block'] for c in t.get('conditions', []))
        test_t = [tr for tr in test_trades if tr['template_id'] == t['id']]
        test_wr = sum(1 for tr in test_t if tr['pnl_dollars'] > 0) / max(len(test_t), 1) * 100
        lines.append(f"  {t['id']}: {blocks[:60]}")
        lines.append(f"    Train WR: {t['statistics']['win_rate']:.1f}%  Test WR: {test_wr:.1f}%  Trades: {len(test_t)}")

    lines += [
        "",
        "── WORST 5 (OVERFIT SUSPECTS) ───────────────────────────────────────",
    ]
    for t in worst5:
        train_wr = _safe(t['statistics'].get('win_rate', 0))
        test_t = [tr for tr in test_trades if tr['template_id'] == t['id']]
        test_wr = sum(1 for tr in test_t if tr['pnl_dollars'] > 0) / max(len(test_t), 1) * 100
        lines.append(f"  {t['id']}: train_WR={train_wr:.1f}% test_WR={test_wr:.1f}% overfit={train_wr-test_wr:.1f}%")

    lines += [
        "",
        "── MONTHLY RETURNS (Test Period) ────────────────────────────────────",
    ]
    for month, rets in sorted(monthly.items()):
        monthly_total = sum(rets)
        lines.append(f"  {month}: {monthly_total:+.2f}% ({len(rets)} trading days)")

    lines += [
        "",
        "── TOP 20 INDICATOR BLOCKS ──────────────────────────────────────────",
        indicator_lines or "  (run template_optimizer to generate)",
        "",
        "=" * 70,
        f"VERDICT: {pass_fail}",
    ]

    if pass_fail == "PASS":
        lines.append(f"System meets the {MIN_WIN_RATE}% win rate target on unseen test data.")
    else:
        deficit = MIN_WIN_RATE - test_m.get('win_rate', 0)
        import backtest.config as _bc
        lines += [
            f"Test win rate {test_m.get('win_rate',0):.1f}% is {deficit:.1f}% below the {MIN_WIN_RATE}% target.",
            "",
            "RECOMMENDATIONS:",
            f"  1. Increase MIN_ACTIVATIONS (currently {_bc.MIN_ACTIVATIONS}) to reduce noise",
            "  2. Tighten MIN_PROFIT_FACTOR threshold",
            "  3. Add more diverse symbols to training data",
            "  4. Review overfit suspects above — consider removing them",
            "  5. Extend training window (adjust TRAIN_END in config.py)",
        ]

    lines.append("=" * 70)

    report_text = "\n".join(str(l) for l in lines)
    out_path = RESULTS_DIR / "summary_report.txt"
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    logger.info(f"summary_report.txt written ({len(lines)} lines)")
    print(report_text)
