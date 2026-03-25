# template_matcher.py

"""
StockWise Gen-13 Template Matcher
=================================
The Bridge between Stock State and Trading Signals.

Pipeline:
1. Receive stock DataFrame + pre-classified state (from stock_hunter)
2. Filter templates that match the stock's state (trend/structure/volume/volatility)
3. For each matching template, evaluate all condition blocks against latest candle
4. If ALL conditions pass -> generate signal with entry/stop/target
5. Return list of matched signals, sorted by template win_rate

Anti-Overflow: If no templates match or all conditions fail, logs the reason
so the system can track idle periods and adjust thresholds.
"""

import logging
import time
from datetime import datetime
import system_config as cfg
from setup_templates import TemplateManager, CONDITION_BLOCKS, STOP_BLOCKS, TARGET_BLOCKS
from safe_json_io import safe_json_read

logger = logging.getLogger("TemplateMatcher")


class TemplateMatcher:
    """
    Evaluates all enabled templates against a stock's current data.
    Returns actionable signals with entry, stop-loss, and take-profit.
    """

    def __init__(self, template_manager=None):
        """
        Initialize with a TemplateManager instance.
        If none provided, creates one (loads all templates from disk).
        """
        self.tm = template_manager or TemplateManager()

        # Anti-overflow tracking
        self.idle_tracker = {}  # symbol -> {"last_signal_time": ..., "scans_without_signal": 0}

        # Statistics
        self.total_scans = 0
        self.total_signals = 0

    def scan_ticker(self, symbol, df, stock_state=None):
        """
        Main entry point: scan a single ticker against all matching templates.

        Args:
            symbol: Ticker symbol (e.g., "AAPL")
            df: DataFrame with all features calculated (from feature_engine)
            stock_state: Pre-classified state dict from stock_hunter.
                         If None, will be logged as warning but scan continues
                         (templates without required_state will still match).

        Returns:
            list of signal dicts, sorted by template win_rate (best first):
            [
                {
                    "symbol": "AAPL",
                    "template_id": "MOMENTUM_BREAKOUT",
                    "template_name": "Momentum Breakout",
                    "action": "BUY",
                    "entry_price": 150.25,
                    "stop_loss": 147.00,
                    "take_profit": 159.75,
                    "risk_reward_ratio": 2.9,
                    "template_win_rate": 72.5,
                    "template_total_trades": 40,
                    "conditions_detail": [...],
                    "stock_state": {...},
                    "confidence_score": 78.5,
                    "timestamp": "2026-03-10T15:30:00"
                }
            ]
        """
        self.total_scans += 1

        if df is None or df.empty:
            logger.warning(f"[{symbol}] Empty DataFrame, skipping template scan")
            return []

        last_row = df.iloc[-1]

        if stock_state is None:
            stock_state = {}
            logger.debug(f"[{symbol}] No stock_state provided, matching without state filter")

        # Step 1: Filter templates that match this stock's state
        matching_templates = self.tm.get_for_state(stock_state)

        if not matching_templates:
            self._track_idle(symbol, "no_templates_match_state")
            logger.debug(f"[{symbol}] No templates match state: {stock_state}")
            return []

        logger.debug(f"[{symbol}] {len(matching_templates)} templates match state. Evaluating conditions...")

        # Step 2: Evaluate each matching template's conditions
        signals = []

        for template in matching_templates:
            all_passed, details = template.evaluate_conditions(last_row)

            if all_passed:
                # Step 3: Calculate entry, stop, target
                signal = self._build_signal(symbol, template, last_row, details, stock_state)
                if signal:
                    signals.append(signal)
                    logger.info(f"[{symbol}] SIGNAL: {template.name} | "
                                f"Entry: ${signal['entry_price']:.2f} | "
                                f"Stop: ${signal['stop_loss']:.2f} | "
                                f"Target: ${signal['take_profit']:.2f} | "
                                f"R:R {signal['risk_reward_ratio']:.1f} | "
                                f"WinRate: {signal['template_win_rate']:.0f}%")
            else:
                # Log which blocks failed for debugging
                failed = [d['block'] for d in details if not d.get('passed')]
                logger.debug(f"[{symbol}] {template.id}: FAILED blocks: {failed}")

        # Step 4: Sort by confidence (win_rate + conditions strength)
        signals.sort(key=lambda s: s.get('confidence_score', 0), reverse=True)

        if signals:
            self.total_signals += len(signals)
            self._reset_idle(symbol)
        else:
            self._track_idle(symbol, "all_conditions_failed")

        return signals

    def _build_signal(self, symbol, template, row, conditions_detail, stock_state):
        """
        Builds a complete signal dict from a matched template.
        Calculates entry, stop, target, risk/reward, and confidence score.
        """
        try:
            close = row.get('close', 0)
            if close <= 0:
                return None

            # Entry price (current close for now; confirmation_candles handled by live engine)
            entry_price = round(float(close), 2)

            # Stop-loss from template's stop block
            stop_loss = template.calculate_stop_loss(row)

            # Take-profit from template's target block
            take_profit = template.calculate_take_profit(row)

            # Validate stop/target make sense
            if stop_loss >= entry_price:
                logger.debug(f"[{symbol}] {template.id}: Stop ${stop_loss} >= Entry ${entry_price}, skipping")
                return None

            if take_profit <= entry_price:
                logger.debug(f"[{symbol}] {template.id}: Target ${take_profit} <= Entry ${entry_price}, skipping")
                return None

            # Risk/Reward ratio
            risk = entry_price - stop_loss
            reward = take_profit - entry_price
            rr_ratio = round(reward / risk, 2) if risk > 0 else 0

            # Minimum R:R check (from config)
            min_rr = cfg.FRICTION_AND_ALPHA.get('min_net_rr', 1.2)
            if rr_ratio < min_rr:
                logger.debug(f"[{symbol}] {template.id}: R:R {rr_ratio} < min {min_rr}, skipping")
                return None

            # Confidence score: combination of template win rate + conditions passed + R:R quality
            # Asset-specific win rate (DDR #1): per-symbol from Shadow Ledger, cold start fallback
            asset_cfg = getattr(cfg, 'ASSET_SPECIFIC_CONFIG', {})
            if asset_cfg.get('enabled', False):
                win_rate = self.get_template_win_rate(template.id, symbol)
            else:
                win_rate = template.get_win_rate()
            total_trades = template.statistics.get('total_activations', 0)

            # New templates (< 10 trades) get a neutral confidence, not 0%
            if total_trades < 10:
                confidence = 50.0 + (rr_ratio * 5)  # Baseline + R:R bonus
            else:
                confidence = (
                    (win_rate * 0.6)
                    + (min(rr_ratio / 3.0, 1.0) * 100 * 0.2)
                    + (min(total_trades / 50, 1.0) * 100 * 0.2)
                )

            confidence = round(min(confidence, 100.0), 1)

            return {
                "symbol": symbol,
                "template_id": template.id,
                "template_name": template.name,
                "action": "BUY",
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "risk_reward_ratio": rr_ratio,
                "risk_pct": round((risk / entry_price) * 100, 2),
                "reward_pct": round((reward / entry_price) * 100, 2),
                "template_win_rate": round(win_rate, 1),
                "template_total_trades": total_trades,
                "conditions_detail": conditions_detail,
                "stock_state": stock_state,
                "confidence_score": confidence,
                "use_runner_mode": template.take_profit.get('use_runner_mode', False),
                "confirmation_candles": template.entry.get('confirmation_candles', 0),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"[{symbol}] Failed to build signal from {template.id}: {e}")
            return None

    def _track_idle(self, symbol, reason):
        """Track scans without signal for Anti-Overflow protection."""
        if symbol not in self.idle_tracker:
            self.idle_tracker[symbol] = {
                "last_signal_time": None,
                "scans_without_signal": 0,
                "last_idle_reason": reason
            }

        tracker = self.idle_tracker[symbol]
        tracker["scans_without_signal"] += 1
        tracker["last_idle_reason"] = reason

        # Log warning if idle too long
        idle_count = tracker["scans_without_signal"]
        if idle_count > 0 and idle_count % 50 == 0:
            logger.warning(f"[{symbol}] IDLE ALERT: {idle_count} scans without signal. "
                           f"Last reason: {reason}")

    def _reset_idle(self, symbol):
        """Reset idle tracker when a signal is generated."""
        self.idle_tracker[symbol] = {
            "last_signal_time": time.time(),
            "scans_without_signal": 0,
            "last_idle_reason": None
        }

    def get_idle_report(self):
        """Returns a summary of idle tickers for anti-overflow monitoring."""
        report = []
        for symbol, tracker in self.idle_tracker.items():
            if tracker["scans_without_signal"] > 10:
                report.append({
                    "symbol": symbol,
                    "idle_scans": tracker["scans_without_signal"],
                    "reason": tracker["last_idle_reason"]
                })
        return sorted(report, key=lambda x: x["idle_scans"], reverse=True)

    def get_scan_statistics(self):
        """Returns overall scan statistics."""
        return {
            "total_scans": self.total_scans,
            "total_signals": self.total_signals,
            "signal_rate": round(self.total_signals / max(self.total_scans, 1) * 100, 1),
            "templates_loaded": len(self.tm.templates),
            "templates_enabled": len(self.tm.get_enabled()),
            "idle_tickers": len([t for t in self.idle_tracker.values()
                                 if t["scans_without_signal"] > 10])
        }

    # ========================================
    # ASSET-SPECIFIC WIN RATE (DDR #1)
    # ========================================
    def get_template_win_rate(self, template_id, symbol):
        """
        DDR #1: Asset-specific template win rate with cold start fallback.

        Priority:
        1. If per-stock signals >= cold_start_min → blended (70% per-stock + 30% global)
        2. If per-stock signals < cold_start_min → global average only
        3. If no shadow ledger data at all → fall back to template.get_win_rate()

        Args:
            template_id: Template ID string
            symbol: Stock ticker symbol
        Returns:
            float: Win rate percentage (0-100)
        """
        asset_config = getattr(cfg, 'ASSET_SPECIFIC_CONFIG', {})
        if not asset_config.get('enabled', False):
            template = self._get_template_by_id(template_id)
            return template.get_win_rate() if template else 50.0

        shadow_stats = self._load_shadow_stats()
        if not shadow_stats:
            template = self._get_template_by_id(template_id)
            return template.get_win_rate() if template else 50.0

        cold_start_min = asset_config.get('cold_start_min_signals', 5)
        per_weight = asset_config.get('per_stock_weight', 0.7)
        global_weight = asset_config.get('global_weight', 0.3)

        per_stock = shadow_stats.get(symbol, {}).get(template_id, {})
        per_stock_signals = per_stock.get('signal_count', 0)

        global_stat = self._aggregate_global_stats(shadow_stats, template_id)

        if per_stock_signals < cold_start_min:
            # Cold start — not enough per-symbol history, use global only
            return global_stat.get('win_rate', 50.0)

        per_wr = per_stock.get('win_rate', 50.0)
        global_wr = global_stat.get('win_rate', 50.0)
        blended = (per_wr * per_weight) + (global_wr * global_weight)
        return round(blended, 1)

    def _load_shadow_stats(self):
        """Load shadow ledger template_stats section. Returns {} if file missing or unreadable."""
        asset_config = getattr(cfg, 'ASSET_SPECIFIC_CONFIG', {})
        path = asset_config.get('shadow_ledger_path', 'data/shadow_ledger.json')
        try:
            data = safe_json_read(path, default={})
            return data.get('template_stats', {})
        except Exception:
            return {}

    def _aggregate_global_stats(self, shadow_stats, template_id):
        """Aggregate template stats across all symbols for the global average."""
        total_signals = 0
        total_wins = 0
        total_pnl = 0.0

        for sym_stats in shadow_stats.values():
            t_stats = sym_stats.get(template_id, {})
            total_signals += t_stats.get('signal_count', 0)
            total_wins += t_stats.get('wins', 0)
            total_pnl += t_stats.get('total_pnl_pct', 0.0)

        if total_signals == 0:
            return {"win_rate": 50.0, "avg_pnl_pct": 0.0, "signal_count": 0}

        return {
            "win_rate": round(total_wins / total_signals * 100, 1),
            "avg_pnl_pct": round(total_pnl / total_signals, 2),
            "signal_count": total_signals,
        }

    def _get_template_by_id(self, template_id):
        """Find template by ID. Delegates to TemplateManager's registry."""
        return self.tm.get_template_by_id(template_id)
