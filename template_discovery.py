# template_discovery.py

"""
StockWise Gen-13 Template Discovery Engine
==========================================
Discovers new profitable trading templates by backtesting combinations
of condition blocks against historical data.

Pipeline:
1. Fetch 2 years of history for target stocks (with API throttling)
2. Calculate all features using FeatureEngine
3. Generate smart combinations of 3-5 condition blocks
4. For each combination, scan every historical day:
   - Did all blocks return True?
   - If yes, what happened in the next 5 days? (win/loss)
5. Filter combinations that meet quality thresholds
6. Save winners as new template JSON files

Runs offline: nightly or weekend. Not during trading hours.
"""

import os
import json
import logging
import time
import itertools
import numpy as np
import pandas as pd
from datetime import datetime
import system_config as cfg
from setup_templates import (
    CONDITION_BLOCKS, STOP_BLOCKS, TARGET_BLOCKS,
    SetupTemplate, TemplateManager, _safe_get
)

logger = logging.getLogger("TemplateDiscovery")


class TemplateDiscovery:
    """
    Discovers new templates by backtesting block combinations on historical data.
    """

    def __init__(self, data_manager=None, feature_engine=None):
        """
        Args:
            data_manager: DataSourceManager instance (for fetching historical data)
            feature_engine: FeatureEngine instance (for calculating indicators)
        """
        self.dm = data_manager
        self.fe = feature_engine
        self.config = getattr(cfg, 'DISCOVERY_CONFIG', {})
        self.tm = TemplateManager()

        # Block names available for combination
        self.block_names = list(CONDITION_BLOCKS.keys())

        # Incompatible block pairs (don't combine these)
        self.incompatible_pairs = [
            ("rsi_below", "rsi_above"),           # Can't be both oversold and overbought
            ("rsi_below", "rsi_between"),          # Overlapping RSI checks
            ("rsi_above", "rsi_between"),          # Overlapping RSI checks
            ("squeeze_active", "bb_width_below"),  # Squeeze implies narrow BB
        ]

        # Results storage
        self.discovery_results = []

    def fetch_and_prepare_data(self, symbols):
        """
        Fetch historical data and calculate features for each symbol.
        Respects API rate limits with configurable throttling.

        Returns: dict of {symbol: DataFrame_with_features}
        """
        history_days = self.config.get('history_days', 500)
        throttle = self.config.get('api_throttle_seconds', 1.0)
        min_rows = self.config.get('min_history_rows', 200)

        datasets = {}

        for i, symbol in enumerate(symbols):
            try:
                logger.info(f"[{i+1}/{len(symbols)}] Fetching {symbol} ({history_days} days)...")

                # Fetch raw data
                df = self.dm.get_stock_data(symbol, days_back=history_days)

                if df is None or len(df) < min_rows:
                    logger.warning(f"[{symbol}] Insufficient data ({len(df) if df is not None else 0} rows). Skipping.")
                    continue

                # Calculate all features
                df = self.fe.calculate_features(df, strategy_config={"active_indicators": ["all"]})

                datasets[symbol] = df
                logger.info(f"[{symbol}] Prepared: {len(df)} rows, {len(df.columns)} features")

                # API throttle
                if i < len(symbols) - 1:
                    time.sleep(throttle)

            except Exception as e:
                logger.error(f"[{symbol}] Failed to prepare data: {e}")
                continue

        logger.info(f"Data preparation complete: {len(datasets)}/{len(symbols)} stocks ready")
        return datasets

    def generate_smart_combos(self):
        """
        Generate combinations of blocks, filtering out incompatible pairs.
        Returns list of tuples, each containing block names.
        """
        min_blocks = self.config.get('min_blocks_per_combo', 3)
        max_blocks = self.config.get('max_blocks_per_combo', 5)
        max_combos = self.config.get('max_combos_to_test', 5000)

        all_combos = []

        for k in range(min_blocks, max_blocks + 1):
            for combo in itertools.combinations(self.block_names, k):
                # Check for incompatible pairs
                is_compatible = True
                for b1, b2 in self.incompatible_pairs:
                    if b1 in combo and b2 in combo:
                        is_compatible = False
                        break

                if is_compatible:
                    all_combos.append(combo)

        # Cap total combos
        if len(all_combos) > max_combos:
            logger.info(f"Capping combos from {len(all_combos)} to {max_combos}")
            # Prioritize smaller combos (simpler = often better)
            all_combos.sort(key=lambda c: len(c))
            all_combos = all_combos[:max_combos]

        logger.info(f"Generated {len(all_combos)} smart combinations "
                    f"({min_blocks}-{max_blocks} blocks, {len(self.incompatible_pairs)} incompatible pairs filtered)")

        return all_combos

    def backtest_combo(self, combo, datasets):
        """
        Backtest a single block combination across all stock datasets.

        Args:
            combo: tuple of block names, e.g. ("rsi_between", "macd_above_signal", "volume_surge")
            datasets: dict of {symbol: DataFrame}

        Returns:
            dict with backtest results or None if below quality thresholds
        """
        lookahead = self.config.get('lookahead_days', 5)
        profit_target = self.config.get('profit_target_pct', 0.02)
        stop_target = self.config.get('stop_target_pct', 0.03)

        total_signals = 0
        wins = 0
        losses = 0
        total_profit = 0.0
        total_loss = 0.0
        per_stock = {}

        # Default params for blocks (use common defaults)
        default_params = {
            "rsi_between": [40, 70],
            "rsi_below": [30],
            "rsi_above": [50],
            "close_above_sma": [50],
            "sma_above_sma": [50, 200],
            "close_above_ema": [12],
            "er_slow_above": [0.55],
            "trend_alignment": [],
            "macd_above_signal": [],
            "macd_histogram_positive": [],
            "volume_surge": [1.3],
            "rvol_above": [1.2],
            "squeeze_active": [],
            "squeeze_momentum_positive": [],
            "bb_width_below": [0.15],
            "atr_percent_above": [0.01],
            "bullish_candle": [],
            "close_above_ref": ["bb_upper"],
            "close_below_ref": ["bb_lower"],
        }

        for symbol, df in datasets.items():
            if len(df) < lookahead + 10:
                continue

            stock_wins = 0
            stock_losses = 0

            # Scan each row (except last N for lookahead)
            for i in range(len(df) - lookahead):
                row = df.iloc[i]

                # Check all blocks in this combo
                all_pass = True
                for block_name in combo:
                    if block_name not in CONDITION_BLOCKS:
                        all_pass = False
                        break
                    params = default_params.get(block_name, [])
                    try:
                        if not CONDITION_BLOCKS[block_name](row, params):
                            all_pass = False
                            break
                    except Exception:
                        all_pass = False
                        break

                if not all_pass:
                    continue

                # Signal triggered! Check what happened next
                total_signals += 1
                entry_price = row.get('close', 0)

                if entry_price <= 0:
                    continue

                # Look ahead: find max gain and max loss in next N days
                future = df.iloc[i+1:i+1+lookahead]
                if future.empty:
                    continue

                max_high = future['high'].max()
                min_low = future['low'].min()

                max_gain = (max_high - entry_price) / entry_price
                max_loss = (entry_price - min_low) / entry_price

                # Determine win/loss
                hit_target = max_gain >= profit_target
                hit_stop = max_loss >= stop_target

                if hit_target and not hit_stop:
                    wins += 1
                    stock_wins += 1
                    total_profit += max_gain
                elif hit_stop:
                    losses += 1
                    stock_losses += 1
                    total_loss += max_loss
                else:
                    # Neither hit -- check closing price
                    exit_price = future.iloc[-1]['close']
                    pnl = (exit_price - entry_price) / entry_price
                    if pnl > 0:
                        wins += 1
                        stock_wins += 1
                        total_profit += pnl
                    else:
                        losses += 1
                        stock_losses += 1
                        total_loss += abs(pnl)

            if stock_wins + stock_losses > 0:
                per_stock[symbol] = {
                    "wins": stock_wins,
                    "losses": stock_losses,
                    "win_rate": round(stock_wins / (stock_wins + stock_losses) * 100, 1)
                }

        # Calculate aggregate metrics
        total_trades = wins + losses
        if total_trades == 0:
            return None

        win_rate = (wins / total_trades) * 100
        avg_profit = (total_profit / wins) * 100 if wins > 0 else 0
        avg_loss = (total_loss / losses) * 100 if losses > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        stocks_profitable = sum(1 for s in per_stock.values() if s['win_rate'] > 50)

        return {
            "combo": combo,
            "total_signals": total_signals,
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 1),
            "avg_profit_pct": round(avg_profit, 2),
            "avg_loss_pct": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "stocks_profitable": stocks_profitable,
            "per_stock": per_stock,
        }

    def meets_quality_threshold(self, result):
        """Check if a backtest result meets minimum quality requirements."""
        if result is None:
            return False

        min_activations = self.config.get('min_activations', 10)
        min_win_rate = self.config.get('min_win_rate', 55.0)
        min_avg_profit = self.config.get('min_avg_profit_pct', 1.0)
        min_pf = self.config.get('min_profit_factor', 1.5)
        min_stocks = self.config.get('min_stocks_profitable', 3)

        return (
            result['total_trades'] >= min_activations and
            result['win_rate'] >= min_win_rate and
            result['avg_profit_pct'] >= min_avg_profit and
            result['profit_factor'] >= min_pf and
            result['stocks_profitable'] >= min_stocks
        )

    def combo_to_template(self, result, default_params):
        """
        Convert a successful combo backtest result into a SetupTemplate JSON.
        """
        combo = result['combo']

        # Build conditions from blocks
        conditions = []
        for block_name in combo:
            params = default_params.get(block_name, [])
            conditions.append({"block": block_name, "params": params})

        # Generate a descriptive ID
        template_id = "DISCOVERED_" + "_".join(
            b.upper()[:8] for b in combo[:3]
        ) + f"_{int(time.time()) % 10000}"

        # Determine required_state based on blocks used
        required_state = self._infer_required_state(combo)

        template_data = {
            "id": template_id,
            "name": f"Discovered: {' + '.join(combo)}",
            "description": (f"Auto-discovered template from {result['total_trades']} historical trades. "
                            f"Win rate: {result['win_rate']}%, Profit factor: {result['profit_factor']}"),
            "version": 1,
            "source": "discovered",
            "enabled": True,
            "required_state": required_state,
            "conditions": conditions,
            "entry": {"type": "close", "confirmation_candles": 1},
            "stop_loss": {"method": "atr", "atr_multiplier": 1.5, "fallback_pct": 0.02},
            "take_profit": {"method": "atr", "atr_multiplier": 3.0, "use_runner_mode": True},
            "statistics": {
                "total_activations": result['total_trades'],
                "wins": result['wins'],
                "losses": result['losses'],
                "avg_profit_pct": result['avg_profit_pct'],
                "avg_loss_pct": result['avg_loss_pct'],
                "win_rate": result['win_rate'],
                "last_activated": None,
                "best_tickers": result.get('per_stock', {}),
                "worst_tickers": {},
                "best_conditions": [],
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
        }

        return template_data

    def _infer_required_state(self, combo):
        """
        Infer what market state a combo needs based on which blocks it uses.
        """
        state = {}

        # If combo uses trend blocks -> needs BULLISH
        trend_blocks = {"close_above_sma", "sma_above_sma", "close_above_ema",
                        "er_slow_above", "trend_alignment"}
        if any(b in trend_blocks for b in combo):
            state["trend"] = ["BULLISH"]

        # If combo uses oversold -> works in SIDEWAYS/BEARISH
        if "rsi_below" in combo:
            state["trend"] = ["SIDEWAYS", "BEARISH"]
            state["structure"] = ["NEAR_SUPPORT"]

        # If combo uses squeeze -> needs COMPRESSED
        if "squeeze_active" in combo or "bb_width_below" in combo:
            state["volatility"] = ["COMPRESSED"]

        # If combo uses volume surge -> needs good volume
        if "volume_surge" in combo or "rvol_above" in combo:
            state["volume"] = ["HEALTHY", "SURGING"]

        return state

    def run_discovery(self, symbols=None):
        """
        Main entry point: runs the full discovery pipeline.

        Args:
            symbols: list of ticker symbols. If None, uses DEFAULT_TRAINING_SYMBOLS from config.

        Returns:
            list of discovered template dicts (also saved to disk)
        """
        logger.info("=" * 60)
        logger.info("TEMPLATE DISCOVERY ENGINE -- STARTING")
        logger.info("=" * 60)

        start_time = time.time()

        # 1. Get symbols
        if symbols is None:
            symbols = getattr(cfg, 'DEFAULT_TRAINING_SYMBOLS',
                              ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN'])

        logger.info(f"Target stocks: {symbols}")

        # 2. Fetch and prepare data
        if self.dm is None or self.fe is None:
            logger.error("Discovery requires DataSourceManager and FeatureEngine. Aborting.")
            return []

        datasets = self.fetch_and_prepare_data(symbols)

        if not datasets:
            logger.error("No data available. Discovery aborted.")
            return []

        # 3. Generate combinations
        combos = self.generate_smart_combos()

        # 4. Backtest each combo
        discovered = []
        default_params = {
            "rsi_between": [40, 70],
            "rsi_below": [30],
            "rsi_above": [50],
            "close_above_sma": [50],
            "sma_above_sma": [50, 200],
            "close_above_ema": [12],
            "er_slow_above": [0.55],
            "trend_alignment": [],
            "macd_above_signal": [],
            "macd_histogram_positive": [],
            "volume_surge": [1.3],
            "rvol_above": [1.2],
            "squeeze_active": [],
            "squeeze_momentum_positive": [],
            "bb_width_below": [0.15],
            "atr_percent_above": [0.01],
            "bullish_candle": [],
            "close_above_ref": ["bb_upper"],
            "close_below_ref": ["bb_lower"],
        }

        logger.info(f"Testing {len(combos)} combinations against {len(datasets)} stocks...")

        for i, combo in enumerate(combos):
            if (i + 1) % 500 == 0:
                elapsed = time.time() - start_time
                logger.info(f"  Progress: {i+1}/{len(combos)} combos tested ({elapsed:.0f}s elapsed)")

            result = self.backtest_combo(combo, datasets)

            if self.meets_quality_threshold(result):
                discovered.append(result)
                logger.info(f"  WINNER: {combo} | "
                            f"WR: {result['win_rate']}% | "
                            f"PF: {result['profit_factor']} | "
                            f"Trades: {result['total_trades']} | "
                            f"Stocks: {result['stocks_profitable']}")

        # 5. Save discovered templates
        saved_count = 0
        for result in discovered:
            template_data = self.combo_to_template(result, default_params)
            if self.tm.add_template(template_data):
                saved_count += 1

        elapsed = time.time() - start_time

        logger.info("=" * 60)
        logger.info(f"DISCOVERY COMPLETE in {elapsed:.0f} seconds")
        logger.info(f"  Combos tested: {len(combos)}")
        logger.info(f"  Winners found: {len(discovered)}")
        logger.info(f"  Templates saved: {saved_count}")
        logger.info(f"  Total templates in library: {len(self.tm.templates)}")
        logger.info("=" * 60)

        self.discovery_results = discovered
        return discovered


if __name__ == "__main__":
    """
    Standalone execution for offline discovery.
    Usage: python template_discovery.py
    """
    from data_source_manager import DataSourceManager
    from feature_engine import FeatureEngine

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s | %(levelname)s | [%(name)s] | %(message)s')

    dm = DataSourceManager()
    fe = FeatureEngine()

    discovery = TemplateDiscovery(data_manager=dm, feature_engine=fe)
    results = discovery.run_discovery()

    if results:
        print(f"\nDiscovered {len(results)} new templates!")
        for r in results:
            print(f"  {r['combo']}: WR={r['win_rate']}%, PF={r['profit_factor']}, Trades={r['total_trades']}")
    else:
        print("\nNo new templates discovered. Try adjusting DISCOVERY_CONFIG thresholds.")
