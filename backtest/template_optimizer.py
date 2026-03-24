"""
Discovers profitable trading templates from TRAIN data only.
Generates block combinations, evaluates on historical data, and filters by quality metrics.
"""

import os
import json
import logging
import itertools
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import pandas as pd
import numpy as np

from backtest.config import (
    MIN_BLOCKS, MAX_BLOCKS, MAX_COMBOS, MIN_ACTIVATIONS,
    MIN_WIN_RATE, MIN_PROFIT_FACTOR, MIN_STOCKS_PROFITABLE
)

logger = logging.getLogger("backtest.template_optimizer")

_HERE       = Path(__file__).parent
RESULTS_DIR = _HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _load_blocks():
    from setup_templates import CONDITION_BLOCKS, STOP_BLOCKS, TARGET_BLOCKS
    return CONDITION_BLOCKS, STOP_BLOCKS, TARGET_BLOCKS


def _load_param_ranges():
    try:
        from system_config import PARAM_RANGES
        return PARAM_RANGES
    except Exception:
        # Fallback minimal ranges if system_config import fails
        return {
            "close_above_sma": [[50], [200]],
            "rsi_between": [[40, 70], [30, 60]],
            "macd_above_signal": [[]],
            "bullish_candle": [[]],
            "volume_surge": [[1.5]],
            "sma_above_sma": [[50, 200]],
        }


def _generate_combos(block_names: list, param_ranges: dict, max_combos: int) -> list:
    """
    Generate combinations of MIN_BLOCKS..MAX_BLOCKS condition blocks with param variations.
    Returns list of: [(block_name, params), ...]
    """
    # Only include blocks that have param ranges defined
    usable = [(name, param_ranges.get(name, [[]])) for name in block_names
              if name in param_ranges or name in block_names]

    # Build all (block_name, params) options
    block_options = []
    for name in block_names:
        ranges = param_ranges.get(name, [[]])
        if not ranges:
            ranges = [[]]
        for params in ranges:
            block_options.append((name, params))

    combos = []
    for n in range(MIN_BLOCKS, MAX_BLOCKS + 1):
        for combo in itertools.combinations(block_options, n):
            # Avoid duplicate block names in same combo
            names_in_combo = [c[0] for c in combo]
            if len(names_in_combo) != len(set(names_in_combo)):
                continue
            combos.append(list(combo))
            if len(combos) >= max_combos:
                return combos
    return combos


def _evaluate_combo(combo: list, train_data: dict, condition_blocks: dict, lookahead: int = 5) -> dict:
    """
    Test a block combo on all train symbols.
    Returns stats dict with win_rate, activations, profit_factor, per-stock breakdown.
    """
    per_stock = {}
    total_wins = 0
    total_losses = 0
    total_profit = 0.0
    total_loss_amt = 0.0

    for symbol, df in train_data.items():
        if len(df) < lookahead + 5:
            continue

        wins = 0
        losses = 0
        consecutive_losses = 0
        max_consec = 0

        rows = df.reset_index()
        for i in range(len(rows) - lookahead):
            row = rows.iloc[i].to_dict()

            # Check all conditions
            try:
                all_pass = all(
                    condition_blocks[block_name](row, params)
                    for block_name, params in combo
                    if block_name in condition_blocks
                )
            except Exception:
                continue

            if not all_pass:
                continue

            # Lookahead: did price hit +2% before -3%?
            entry = row.get('close', 0)
            if entry <= 0:
                continue

            win_target  = entry * 1.02
            loss_target = entry * 0.97

            outcome = None
            for j in range(1, lookahead + 1):
                if i + j >= len(rows):
                    break
                future = rows.iloc[i + j]
                if future.get('high', 0) >= win_target:
                    outcome = 'win'
                    break
                if future.get('low', 9999999) <= loss_target:
                    outcome = 'loss'
                    break

            if outcome is None:
                # Check final close vs entry
                final_close = rows.iloc[i + min(lookahead, len(rows) - i - 1)].get('close', entry)
                outcome = 'win' if final_close > entry else 'loss'

            if outcome == 'win':
                wins += 1
                total_wins += 1
                total_profit += 2.0
                consecutive_losses = 0
            else:
                losses += 1
                total_losses += 1
                total_loss_amt += 3.0
                consecutive_losses += 1
                max_consec = max(max_consec, consecutive_losses)

        if wins + losses > 0:
            per_stock[symbol] = {
                'wins': wins,
                'losses': losses,
                'max_consec_losses': max_consec
            }

    total_trades = total_wins + total_losses
    if total_trades == 0:
        return None

    win_rate = (total_wins / total_trades) * 100
    profit_factor = (total_profit / total_loss_amt) if total_loss_amt > 0 else (total_profit if total_profit > 0 else 0)
    profitable_stocks = sum(
        1 for s in per_stock.values()
        if s['wins'] > s['losses']
    )
    max_consec_any = max((s['max_consec_losses'] for s in per_stock.values()), default=0)

    return {
        'win_rate': win_rate,
        'activations': total_trades,
        'profit_factor': profit_factor,
        'profitable_stocks': profitable_stocks,
        'max_consec_losses': max_consec_any,
        'per_stock': per_stock,
        'total_wins': total_wins,
        'total_losses': total_losses,
    }


def discover_templates(train_data: dict) -> list:
    """
    Main entry point. Discovers templates from TRAIN data only.
    Returns list of template dicts (JSON-serializable).
    """
    CONDITION_BLOCKS, STOP_BLOCKS, TARGET_BLOCKS = _load_blocks()
    PARAM_RANGES = _load_param_ranges()

    block_names = list(CONDITION_BLOCKS.keys())
    logger.info(f"Generating combinations from {len(block_names)} blocks (cap={MAX_COMBOS})")
    combos = _generate_combos(block_names, PARAM_RANGES, MAX_COMBOS)
    logger.info(f"Testing {len(combos)} combinations on {len(train_data)} symbols...")

    winners = []
    for idx, combo in enumerate(combos):
        stats = _evaluate_combo(combo, train_data, CONDITION_BLOCKS)
        if stats is None:
            continue

        # Apply filters
        if stats['activations'] < MIN_ACTIVATIONS:
            continue
        if stats['win_rate'] < MIN_WIN_RATE:
            continue
        if stats['profit_factor'] < MIN_PROFIT_FACTOR:
            continue
        if stats['profitable_stocks'] < MIN_STOCKS_PROFITABLE:
            continue
        if stats['max_consec_losses'] > 5:
            continue

        template_id = f"DISCOVERED_{len(winners)+1:04d}"
        block_desc = "_".join(b for b, _ in combo)

        template = {
            "id": template_id,
            "name": f"Discovered: {block_desc[:50]}",
            "description": f"Auto-discovered template with {len(combo)} conditions",
            "version": 1,
            "source": "discovered",
            "enabled": True,
            "required_state": {
                "trend": ["BULLISH"],
                "structure": ["OPEN_FIELD", "NEAR_SUPPORT"],
                "volume": ["HEALTHY", "SURGING"],
                "volatility": ["NORMAL", "COMPRESSED"]
            },
            "conditions": [{"block": b, "params": p} for b, p in combo],
            "stop_loss": {"method": "atr", "atr_multiplier": 2.0, "fallback_pct": 0.02},
            "take_profit": {"method": "atr", "atr_multiplier": 3.0, "use_runner_mode": True},
            "statistics": {
                "total_activations": stats['activations'],
                "wins": stats['total_wins'],
                "losses": stats['total_losses'],
                "avg_profit_pct": 2.0,
                "avg_loss_pct": -3.0,
                "win_rate": round(stats['win_rate'], 2),
                "last_activated": None,
                "best_tickers": {},
                "worst_tickers": {},
                "best_conditions": [],
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            },
            "_backtest_stats": stats
        }
        winners.append(template)

        if idx % 1000 == 0 and idx > 0:
            logger.info(f"  {idx}/{len(combos)} combos tested, {len(winners)} winners so far")

    logger.info(f"Discovery complete: {len(winners)} templates found from {len(combos)} combos")

    # Save to results/
    out_path = RESULTS_DIR / "discovered_templates.json"
    # Remove internal stats before saving (keep it clean)
    clean = []
    for t in winners:
        tc = {k: v for k, v in t.items() if k != '_backtest_stats'}
        clean.append(tc)
    with open(out_path, 'w') as f:
        json.dump(clean, f, indent=2)
    logger.info(f"Saved {len(clean)} templates → {out_path}")

    # Indicator importance analysis
    _analyze_indicators(winners)

    return winners


def _analyze_indicators(templates: list):
    """Count which blocks appear most in winning templates. Save to indicator_analysis.csv."""
    import pandas as pd

    block_freq = defaultdict(int)
    block_wr_sum = defaultdict(float)
    block_wr_count = defaultdict(int)

    for t in templates[:20]:  # Top 20 by order of discovery (filtered already)
        wr = t['statistics']['win_rate']
        for cond in t['conditions']:
            block_name = cond['block']
            block_freq[block_name] += 1
            block_wr_sum[block_name] += wr
            block_wr_count[block_name] += 1

    rows = []
    for block_name, freq in sorted(block_freq.items(), key=lambda x: -x[1]):
        avg_wr = block_wr_sum[block_name] / block_wr_count[block_name] if block_wr_count[block_name] > 0 else 0
        rows.append({
            'block_name': block_name,
            'frequency': freq,
            'avg_win_rate_contribution': round(avg_wr, 2)
        })

    if rows:
        df = pd.DataFrame(rows)
        out_path = RESULTS_DIR / "indicator_analysis.csv"
        df.to_csv(out_path, index=False)
        logger.info(f"Indicator analysis saved → {out_path}")
