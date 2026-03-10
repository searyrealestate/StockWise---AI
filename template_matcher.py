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
