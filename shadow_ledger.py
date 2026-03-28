# shadow_ledger.py

"""
StockWise Gen-13 Shadow Ledger
==============================
SPEC v13.4 §4: Candle-by-candle template evaluation engine.

Walks through historical data bar-by-bar, evaluates ALL templates at each candle,
records virtual signals, and tracks outcomes (target hit / stop hit).

Runs OFFLINE (weekends) per DDR Part C — does not block nightly scan.
Output: Per-template statistics used by template_matcher for confidence scoring.

Architecture:
  run_full_evaluation(symbols)
    → per symbol: fetch data → calculate_features → evaluate_history()
      → per candle: classify state → match templates → evaluate conditions
        → on SIGNAL: check cooldown → record virtual entry → resolve outcome via lookahead

Phase 2 planned: MTFA (Multi-Timeframe Analysis) will add 4H/1H/15m confluence scoring.
Current version: Daily candles only.
"""

import logging
import pandas as pd
from datetime import datetime
import system_config as cfg
from safe_json_io import safe_json_read, safe_json_write
from setup_templates import TemplateManager

logger = logging.getLogger("ShadowLedger")


class ShadowLedger:
    """
    Candle-by-candle template evaluation engine.
    Tracks 100% of signals virtually — independent of user execution.
    """

    def __init__(self, template_manager=None):
        self.config = getattr(cfg, 'SHADOW_LEDGER_CONFIG', {})
        self.ledger_path = self.config.get('ledger_path', 'data/shadow_ledger.json')
        self.tm = template_manager or TemplateManager()
        self.lookahead = self.config.get('lookahead_candles', 20)
        self.cooldown = self.config.get('min_bars_between_signals', 20)
        self.ledger = self._load_ledger()

    def _load_ledger(self):
        """Load existing ledger or initialize empty."""
        return safe_json_read(self.ledger_path, default={
            "metadata": {"last_run": None, "version": "13.4"},
            "template_stats": {},
        })

    def _save_ledger(self):
        """Persist ledger atomically."""
        self.ledger["metadata"]["last_run"] = datetime.now().isoformat()
        safe_json_write(self.ledger_path, self.ledger)

    def evaluate_history(self, symbol, df, stock_state_fn=None):
        """
        Walk through df candle-by-candle, evaluate all templates at each bar.

        Args:
            symbol: Ticker symbol
            df: Full historical DataFrame with features already calculated
            stock_state_fn: Callable(df_slice) → state dict. If None, skips state filtering.

        Returns:
            dict with per-template stats for this symbol
        """
        min_candles = self.config.get('min_candles_for_eval', 200)
        if df is None or len(df) < min_candles:
            logger.debug(
                f"[{symbol}] Insufficient data "
                f"({len(df) if df is not None else 0} < {min_candles})"
            )
            return {}

        templates = self.tm.get_enabled()
        if not templates:
            logger.warning("No enabled templates found")
            return {}

        # Per-template tracking for this symbol
        results = {}
        # Cooldown tracker: {template_id: last_signal_bar_index}
        last_signal_bar = {}

        for t in templates:
            results[t.id] = {
                "signal_count": 0, "wins": 0, "losses": 0,
                "total_pnl_pct": 0.0,
            }
            last_signal_bar[t.id] = -self.cooldown  # Allow first signal immediately

        # Walk candle-by-candle (skip first min_candles for indicator warmup)
        eval_end = len(df) - self.lookahead
        for i in range(min_candles, eval_end):
            row = df.iloc[i]

            # Optional: classify state at this candle
            state = {}
            if stock_state_fn:
                try:
                    state = stock_state_fn(df.iloc[:i + 1])
                except Exception:
                    state = {}

            # Filter templates by state if we have one; otherwise use all enabled
            matching = self.tm.get_for_state(state) if state else templates

            for template in matching:
                try:
                    # ═══ COOLDOWN CHECK ═══
                    if (i - last_signal_bar[template.id]) < self.cooldown:
                        continue

                    # evaluate_conditions returns (bool, list_of_dicts)
                    all_passed, _details = template.evaluate_conditions(row)
                    if not all_passed:
                        continue

                    # Signal detected — record virtual entry
                    entry_price = float(row.get('close', 0))
                    if entry_price <= 0:
                        continue

                    stop_loss = template.calculate_stop_loss(row)
                    take_profit = template.calculate_take_profit(row)

                    if stop_loss >= entry_price or take_profit <= entry_price:
                        continue

                    # ═══ MARK COOLDOWN ═══
                    last_signal_bar[template.id] = i

                    # Look ahead to determine outcome
                    outcome = self._resolve_outcome(
                        df, i, entry_price, stop_loss, take_profit
                    )

                    # Record result
                    results[template.id]["signal_count"] += 1
                    if outcome["hit"] == "target":
                        results[template.id]["wins"] += 1
                        results[template.id]["total_pnl_pct"] += outcome["pnl_pct"]
                    elif outcome["hit"] == "stop":
                        results[template.id]["losses"] += 1
                        results[template.id]["total_pnl_pct"] += outcome["pnl_pct"]
                    # "neither" = open after lookahead window — not counted as win or loss

                except Exception as e:
                    logger.debug(
                        f"[{symbol}] Template {template.id} eval error at bar {i}: {e}"
                    )
                    continue

        # Calculate derived stats
        for tid, stats in results.items():
            sc = stats["signal_count"]
            if sc > 0:
                stats["win_rate"] = round(stats["wins"] / sc * 100, 1)
                stats["avg_pnl_pct"] = round(stats["total_pnl_pct"] / sc, 2)
            else:
                stats["win_rate"] = 0.0
                stats["avg_pnl_pct"] = 0.0

        # Store in ledger
        if "template_stats" not in self.ledger:
            self.ledger["template_stats"] = {}
        self.ledger["template_stats"][symbol] = results

        total_signals = sum(r['signal_count'] for r in results.values())
        logger.info(
            f"[{symbol}] Shadow evaluation complete: "
            f"{total_signals} signals across {len(templates)} templates"
        )
        return results

    def _resolve_outcome(self, df, entry_idx, entry_price, stop_loss, take_profit):
        """
        Look ahead from entry candle to determine if target or stop was hit first.
        Stop is checked FIRST per conservative evaluation (worst-case intra-bar assumption).
        """
        end_idx = min(entry_idx + self.lookahead + 1, len(df))
        for j in range(entry_idx + 1, end_idx):
            candle = df.iloc[j]
            low = float(candle.get('low', entry_price))
            high = float(candle.get('high', entry_price))

            # Check stop first (conservative)
            if low <= stop_loss:
                pnl = (stop_loss - entry_price) / entry_price * 100
                return {"hit": "stop", "pnl_pct": round(pnl, 2), "bars": j - entry_idx}

            # Then check target
            if high >= take_profit:
                pnl = (take_profit - entry_price) / entry_price * 100
                return {"hit": "target", "pnl_pct": round(pnl, 2), "bars": j - entry_idx}

        return {"hit": "neither", "pnl_pct": 0.0, "bars": self.lookahead}

    def get_template_stats(self, symbol=None):
        """
        Get per-template statistics.

        Args:
            symbol: If given → returns per-stock stats for that symbol.
                    If None → returns global stats aggregated across all symbols.

        Returns:
            dict of {template_id: {signal_count, wins, losses, win_rate, avg_pnl_pct}}
        """
        all_stats = self.ledger.get("template_stats", {})

        if symbol:
            return all_stats.get(symbol, {})

        # Aggregate across all symbols
        global_stats = {}
        for sym_stats in all_stats.values():
            for tid, stats in sym_stats.items():
                if tid not in global_stats:
                    global_stats[tid] = {
                        "signal_count": 0, "wins": 0,
                        "losses": 0, "total_pnl_pct": 0.0,
                    }
                global_stats[tid]["signal_count"] += stats.get("signal_count", 0)
                global_stats[tid]["wins"] += stats.get("wins", 0)
                global_stats[tid]["losses"] += stats.get("losses", 0)
                global_stats[tid]["total_pnl_pct"] += stats.get("total_pnl_pct", 0.0)

        for tid, stats in global_stats.items():
            sc = stats["signal_count"]
            stats["win_rate"] = round(stats["wins"] / sc * 100, 1) if sc > 0 else 0.0
            stats["avg_pnl_pct"] = round(stats["total_pnl_pct"] / sc, 2) if sc > 0 else 0.0

        return global_stats

    def apply_decay(self):
        """
        SPEC v13.4 §4: Apply vectorized decay to all stored template stats.

        Recent signals are weighted more; old signals fade at a rate that depends
        on template category. VSA/institutional templates retain memory longer
        than momentum templates.

        Adds 'decayed_win_rate', 'decay_weight', and 'decay_category' to each
        template's stats entry. template_matcher reads decayed_win_rate for
        confidence scoring.

        Decay formula: decayed_win_rate = raw_wr * weight + 50.0 * (1 - weight)
        As weight → min_weight, win_rate regresses to 50% (neutral/unknown),
        NOT to 0% — an old template should be uncertain, not condemned.
        """
        decay_config = getattr(cfg, 'VECTORIZED_DECAY_CONFIG', {})
        if not decay_config.get('enabled', True):
            return

        decay_rates = decay_config.get('decay_rates', {})
        period_days = decay_config.get('decay_period_days', 7)
        min_weight = decay_config.get('min_weight', 0.05)

        last_run = self.ledger.get("metadata", {}).get("last_run")
        if not last_run:
            logger.debug("No previous run timestamp — skipping decay")
            return

        now = datetime.now()
        try:
            last_dt = datetime.fromisoformat(last_run)
            days_since = (now - last_dt).days
        except (ValueError, TypeError):
            logger.warning(f"Invalid last_run timestamp: {last_run}")
            return

        if days_since <= 0:
            return  # Same day — no decay

        periods = days_since / period_days

        # Build id → template map for category lookup
        templates_by_id = {t.id: t for t in self.tm.get_enabled()}

        for symbol, sym_stats in self.ledger.get("template_stats", {}).items():
            for tid, stats in sym_stats.items():
                template = templates_by_id.get(tid)
                if template and hasattr(template, 'get_category'):
                    category = template.get_category()
                else:
                    category = getattr(template, 'category', 'default') if template else 'default'

                rate = decay_rates.get(category, decay_rates.get('default', 0.95))
                weight = max(rate ** periods, min_weight)

                raw_wr = stats.get("win_rate", 50.0)
                stats["decayed_win_rate"] = round(raw_wr * weight + 50.0 * (1 - weight), 1)
                stats["decay_weight"] = round(weight, 4)
                stats["decay_category"] = category

        logger.info(
            f"Vectorized decay applied: {days_since}d since last run, "
            f"{periods:.1f} periods (rates by category)"
        )

    def run_full_evaluation(self, data_source_manager, symbols=None, feature_engine=None):
        """
        Batch evaluation: run candle-by-candle for all symbols.
        Intended for OFFLINE/weekend execution per DDR Part C.

        Args:
            data_source_manager: DSM instance for fetching data
            symbols: List of symbols. Defaults to DEFAULT_TRAINING_SYMBOLS from config.
            feature_engine: FeatureEngine instance for indicator calculation. Optional.
        """
        if symbols is None:
            symbols = list(getattr(cfg, 'DEFAULT_TRAINING_SYMBOLS', []))

        days_back = self.config.get('eval_days_back', 1095)
        min_candles = self.config.get('min_candles_for_eval', 200)

        logger.info(
            f"Shadow Ledger: Starting full evaluation for "
            f"{len(symbols)} symbols, {days_back} days back"
        )

        evaluated = 0
        skipped = 0

        for symbol in symbols:
            try:
                df = data_source_manager.get_stock_data(
                    symbol, days_back=days_back, interval='1d'
                )
                if df is None or len(df) < min_candles:
                    logger.debug(f"[{symbol}] Skipped — insufficient data")
                    skipped += 1
                    continue

                if feature_engine is not None:
                    df = feature_engine.calculate_features(df)

                self.evaluate_history(symbol, df)
                evaluated += 1
                # Log per-symbol summary
                sym_stats = self.ledger.get("template_stats", {}).get(symbol, {})
                sym_total = sum(s.get("signal_count", 0) for s in sym_stats.values())
                logger.info(f"[{symbol}] Evaluation complete: {sym_total} total signals")

            except Exception as e:
                logger.error(f"[{symbol}] Shadow evaluation failed: {e}")
                skipped += 1
                continue

        # Apply vectorized decay before saving (SPEC v13.4 §4)
        self.apply_decay()

        # Log per-symbol per-template detail for simulator compatibility
        for symbol in symbols:
            sym_stats = self.ledger.get("template_stats", {}).get(symbol, {})
            for tid, stats in sym_stats.items():
                sc = stats.get("signal_count", 0)
                if sc > 0:
                    wr = stats.get("win_rate", 0.0)
                    avg_pnl = stats.get("avg_pnl_pct", 0.0)
                    logger.debug(
                        f"[{symbol}] {tid}: {sc} signals, "
                        f"WR={wr:.1f}%, AvgPnL={avg_pnl:+.2f}%"
                    )

        self._save_ledger()
        logger.info(
            f"Shadow Ledger: Complete. Evaluated: {evaluated}, Skipped: {skipped}. "
            f"Saved to {self.ledger_path}"
        )


# ════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ════════════════════════════════════════════════════════════════
# Usage:
#   python shadow_ledger.py
#   python shadow_ledger.py --symbols AAPL MSFT NVDA
#   python shadow_ledger.py --symbols AAPL --days-back 365
#
# Default: runs on DEFAULT_TRAINING_SYMBOLS from system_config.py
# Intended for offline/weekend execution (DDR Part C).
# Output: data/shadow_ledger.json (used by template_matcher for
#         DDR #1 Asset-Specific win rates)
# ════════════════════════════════════════════════════════════════

def _print_summary(sl):
    """Print human-readable summary of evaluation results to stdout."""
    all_stats = sl.ledger.get("template_stats", {})
    if not all_stats:
        print("\n[ShadowLedger] No evaluation results to summarize.")
        return

    # Aggregate across all symbols
    global_stats = {}
    symbols_evaluated = list(all_stats.keys())

    for sym_stats in all_stats.values():
        for tid, stats in sym_stats.items():
            if tid not in global_stats:
                global_stats[tid] = {
                    "signal_count": 0, "wins": 0, "losses": 0,
                    "total_pnl_pct": 0.0,
                }
            global_stats[tid]["signal_count"] += stats.get("signal_count", 0)
            global_stats[tid]["wins"] += stats.get("wins", 0)
            global_stats[tid]["losses"] += stats.get("losses", 0)
            global_stats[tid]["total_pnl_pct"] += stats.get("total_pnl_pct", 0.0)

    total_signals = sum(s["signal_count"] for s in global_stats.values())

    print(f"\n{'=' * 55}")
    print(f" Shadow Ledger Evaluation Complete")
    print(f"{'=' * 55}")
    print(f" Symbols evaluated: {len(symbols_evaluated)}")
    print(f" Total signals:     {total_signals}")
    print(f"{'-' * 55}")
    print(f" {'Template':<25} {'Signals':>8} {'Wins':>6} {'WR%':>7} {'AvgPnL':>8}")
    print(f"{'-' * 55}")

    for tid in sorted(global_stats.keys()):
        s = global_stats[tid]
        sc = s["signal_count"]
        wr = round(s["wins"] / sc * 100, 1) if sc > 0 else 0.0
        avg_pnl = round(s["total_pnl_pct"] / sc, 2) if sc > 0 else 0.0
        print(f" {tid:<25} {sc:>8} {s['wins']:>6} {wr:>6.1f}% {avg_pnl:>+7.2f}%")

    print(f"{'-' * 55}")

    # Per-symbol breakdown (top 10 by signal count)
    print(f"\n Per-symbol signal counts (top contributors):")
    sym_signals = []
    for sym, sym_stats in all_stats.items():
        sym_total = sum(s.get("signal_count", 0) for s in sym_stats.values())
        sym_signals.append((sym, sym_total))
    sym_signals.sort(key=lambda x: x[1], reverse=True)

    for sym, count in sym_signals[:10]:
        print(f"   {sym:<8} {count:>5} signals")

    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    import argparse
    import sys
    import time

    # ── Parse CLI arguments ──────────────────────────────────
    parser = argparse.ArgumentParser(
        description="StockWise Shadow Ledger — offline candle-by-candle template evaluation",
        epilog="Output: data/shadow_ledger.json (feeds DDR #1 Asset-Specific win rates)"
    )
    parser.add_argument(
        "--symbols", nargs="+", default=None,
        help="Symbols to evaluate (default: DEFAULT_TRAINING_SYMBOLS from config)"
    )
    parser.add_argument(
        "--days-back", type=int, default=None,
        help="Days of history to evaluate (default: from SHADOW_LEDGER_CONFIG.eval_days_back)"
    )
    args = parser.parse_args()

    # ── Setup logging to console ─────────────────────────────
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S"
    )

    # ── Resolve symbols ──────────────────────────────────────
    symbols = args.symbols
    if symbols is None:
        symbols = list(getattr(cfg, 'DEFAULT_TRAINING_SYMBOLS', []))
    if not symbols:
        print("[ShadowLedger] ERROR: No symbols provided and DEFAULT_TRAINING_SYMBOLS is empty.")
        print("Usage: python shadow_ledger.py --symbols AAPL MSFT NVDA")
        sys.exit(1)

    # ── Override days_back if provided ───────────────────────
    sl = ShadowLedger()
    if args.days_back is not None:
        sl.config['eval_days_back'] = args.days_back

    days_back = sl.config.get('eval_days_back', 1095)

    print(f"[ShadowLedger] Starting evaluation:")
    print(f"  Symbols:   {len(symbols)} ({', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''})")
    print(f"  Days back: {days_back}")
    print(f"  Output:    {sl.ledger_path}")
    print()

    # ── Initialize dependencies ──────────────────────────────
    try:
        from data_source_manager import DataSourceManager
        from feature_engine import FeatureEngine
        dsm = DataSourceManager()
        fe = FeatureEngine()
    except Exception as e:
        print(f"[ShadowLedger] ERROR: Failed to initialize dependencies: {e}")
        sys.exit(1)

    # ── Run evaluation ───────────────────────────────────────
    start_time = time.time()

    sl.run_full_evaluation(
        data_source_manager=dsm,
        symbols=symbols,
        feature_engine=fe
    )

    elapsed = time.time() - start_time

    # ── Print summary ────────────────────────────────────────
    _print_summary(sl)

    print(f"[ShadowLedger] Duration: {elapsed:.1f}s")
    print(f"[ShadowLedger] Results saved to: {sl.ledger_path}")
    print(f"[ShadowLedger] template_matcher will now use per-stock win rates (DDR #1)")
