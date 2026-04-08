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
import math
import time
from datetime import datetime
import system_config as cfg
from setup_templates import TemplateManager, CONDITION_BLOCKS, STOP_BLOCKS, TARGET_BLOCKS
from safe_json_io import safe_json_read, safe_json_write

logger = logging.getLogger("TemplateMatcher")

try:
    from decision_logger import DecisionLogger as _DecisionLogger
    _dl = _DecisionLogger()
except Exception:
    _dl = None


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
        self._trust_cache = None   # In-memory trust matrix (set by BacktestEngine)
        self._suit_cache  = None   # In-memory suit assignments (set by BacktestEngine)

    def scan_ticker(self, symbol, df, stock_state=None, timeframe=None):
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

        # CP-3: Build state key and get suit context (before template evaluation)
        state_key = self._build_state_key(stock_state)
        sa_cfg = getattr(cfg, 'TEMPLATE_EVOLUTION_CONFIG', {}).get("suit_assignment", {})
        suit = None
        is_exploration = False
        if sa_cfg.get("enabled", False):
            suit = self.get_suit(symbol, state_key)
            is_exploration = self._is_exploration_bar()
            # State-mismatch override: log when assigned template's state ≠ current state
            if suit and suit.get("assigned_template"):
                assigned_tmpl_obj = self._get_template_by_name(suit["assigned_template"])
                if assigned_tmpl_obj and not self._template_state_matches(
                        assigned_tmpl_obj, state_key):
                    logger.info(
                        f"[SUIT-OVERRIDE] {symbol}:{state_key} | "
                        f"assigned={suit['assigned_template']} | "
                        f"reason=state_mismatch | evaluating_all"
                    )

        # Step 1: Filter templates that match this stock's state (and timeframe if set)
        matching_templates = self.tm.get_for_state(stock_state, symbol=symbol, timeframe=timeframe)

        if not matching_templates:
            self._track_idle(symbol, "no_templates_match_state")
            logger.debug(f"[{symbol}] No templates match state: {stock_state}")
            return []

        logger.debug(f"[{symbol}] {len(matching_templates)} templates match state. Evaluating conditions...")

        # Step 2: Evaluate each matching template's conditions
        signals = []

        for template in matching_templates:
            # Auto-disable check: skip template+symbol+state combos marked as disabled
            if self._is_combo_disabled(template.id, symbol, stock_state):
                logger.info(f"[{symbol}] {template.id}: DISABLED for this symbol/state combo — skipping")
                continue

            all_passed, details = template.evaluate_conditions(last_row)

            if all_passed:
                # Step 3: Calculate entry, stop, target
                signal = self._build_signal(symbol, template, last_row, details, stock_state)
                if signal:
                    # CP-2: Contextual trust score (informational — does NOT block signals)
                    try:
                        trust = self.get_trust_score(template.id, symbol, stock_state)
                        signal["trust"] = trust
                        logger.info(
                            f"[TRUST] {template.id} | {symbol} | "
                            f"lifecycle={trust['lifecycle']} | score={trust['score']:.3f} | "
                            f"decayed_wr={trust['decayed_wr']:.1%} | "
                            f"signals={trust['total']} | level={trust['level_used']}"
                        )
                    except Exception as _trust_ex:
                        logger.debug(f"[{symbol}] Trust score failed: {_trust_ex}")
                    # CP-3: Tag whether this signal is from the assigned (suit) template
                    signal["is_assigned"] = (
                        suit is not None
                        and suit.get("assigned_template") == template.id
                    )
                    signals.append(signal)
                    if _dl:
                        try: _dl.log_signal(symbol=symbol, template_id=template.id, confidence=float(signal.get('confidence_score', 0)), regime=str(stock_state.get('trend', '') if stock_state else ''))
                        except Exception: pass
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

        # Step 4: Sort and return — suit-aware ranking (CP-3) or confidence fallback
        if sa_cfg.get("enabled", False) and signals:
            # Log each signal with its suit tag
            for sig in signals:
                assigned_tag = "ASSIGNED" if sig.get("is_assigned") else "NON-ASSIGNED"
                logger.info(
                    f"[SUIT-SIGNAL] {symbol}:{state_key} | "
                    f"template={sig['template_id']} | "
                    f"trust={sig.get('trust', {}).get('score', 0):.3f} | "
                    f"lifecycle={sig.get('trust', {}).get('lifecycle', 'N/A')} | "
                    f"tag={assigned_tag}"
                )

            if is_exploration:
                logger.info(
                    f"[SUIT-EXPLORE] {symbol}:{state_key} | "
                    f"exploration_bar=True | all_signals={len(signals)} | "
                    f"best={signals[0]['template_id']}"
                )

            # Sort by trust score (higher is better); tie-break by total signals
            signals.sort(key=lambda s: (
                -s.get('trust', {}).get('score', 0),
                -s.get('trust', {}).get('total', 0)
            ))

            # In best_single mode: return only the top-ranked signal
            mode = sa_cfg.get("mode", "best_single")
            if mode == "best_single":
                signals = [signals[0]]
        else:
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

            signal = {
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

            # Signal direction context — shows market regime in Telegram (CP-4)
            dir_cfg = getattr(cfg, 'TEMPLATE_EVOLUTION_CONFIG', {}).get("signal_direction", {})
            if dir_cfg.get("enabled", True):
                trend = stock_state.get("trend", "BULLISH") if stock_state else "BULLISH"
                direction_info = dir_cfg.get("icons", {}).get(trend, {"icon": "📈", "label": "Signal"})
                signal["direction_icon"] = direction_info.get("icon", "📈")
                signal["direction_label"] = direction_info.get("label", "Signal")
                signal["is_reversal"] = trend in ("BEARISH",)
                if signal["is_reversal"] and dir_cfg.get("show_warning_for_reversal", True):
                    signal["reversal_warning"] = dir_cfg.get(
                        "reversal_warning_text",
                        "⚠️ Mean reversion — buying against the trend"
                    )

            return signal

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

        # Prefer decayed_win_rate if available (W4-5), fall back to raw win_rate
        per_wr = per_stock.get('decayed_win_rate', per_stock.get('win_rate', 50.0))
        global_wr = global_stat.get('decayed_win_rate', global_stat.get('win_rate', 50.0))
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

        result = {
            "win_rate": round(total_wins / total_signals * 100, 1),
            "avg_pnl_pct": round(total_pnl / total_signals, 2),
            "signal_count": total_signals,
        }

        # Weighted average of decayed_win_rate across symbols (W4-5)
        total_decayed_wr = 0.0
        decayed_count = 0
        for sym_stats in shadow_stats.values():
            t_stats = sym_stats.get(template_id, {})
            sc = t_stats.get('signal_count', 0)
            if 'decayed_win_rate' in t_stats and sc > 0:
                total_decayed_wr += t_stats['decayed_win_rate'] * sc
                decayed_count += sc

        if decayed_count > 0:
            result["decayed_win_rate"] = round(total_decayed_wr / decayed_count, 1)

        return result

    def _get_template_by_id(self, template_id):
        """Find template by ID. Delegates to TemplateManager's registry."""
        return self.tm.get_template_by_id(template_id)

    # ========================================
    # TEMPLATE AUTO-DISABLE (TEMPLATE EVOLUTION)
    # ========================================

    def _disable_combo_key(self, template_id, symbol, stock_state):
        """Build a string key for a (template, symbol, trend_state) combo.
        Accepts stock_state as dict or as a 'trend:...' state key string."""
        if isinstance(stock_state, str):
            trend = stock_state.split(":")[0] if stock_state else ""
        else:
            trend = stock_state.get("trend", "") if stock_state else ""
        return f"{template_id}::{symbol}::{trend}"

    def _load_disable_list(self):
        """
        Load the set of disabled combo keys from shadow_ledger.json.
        Returns a set of string keys (template_id::symbol::trend).
        """
        evo_cfg = getattr(cfg, 'TEMPLATE_EVOLUTION_CONFIG', {})
        path = evo_cfg.get("auto_disable", {}).get("disable_list_path", "data/shadow_ledger.json")
        try:
            data = safe_json_read(path, default={})
            return set(data.get("disabled_combos", []))
        except Exception:
            return set()

    def _save_disable_list(self, disabled_set):
        """
        Persist the updated disabled combo set back to shadow_ledger.json.
        Only writes the 'disabled_combos' key — leaves all other ledger data intact.
        """
        evo_cfg = getattr(cfg, 'TEMPLATE_EVOLUTION_CONFIG', {})
        path = evo_cfg.get("auto_disable", {}).get("disable_list_path", "data/shadow_ledger.json")
        try:
            data = safe_json_read(path, default={})
            data["disabled_combos"] = sorted(disabled_set)
            safe_json_write(path, data)
        except Exception as e:
            logger.error(f"[AutoDisable] Failed to save disable list: {e}")

    def _is_combo_disabled(self, template_id, symbol, stock_state):
        """
        Returns True if this template+symbol+trend combo is on the disable list.
        Auto-disable must be enabled in TEMPLATE_EVOLUTION_CONFIG.
        """
        evo_cfg = getattr(cfg, 'TEMPLATE_EVOLUTION_CONFIG', {})
        if not evo_cfg.get("auto_disable", {}).get("enabled", False):
            return False
        key = self._disable_combo_key(template_id, symbol, stock_state)
        return key in self._load_disable_list()

    def evaluate_auto_disable(self, template_id, symbol, stock_state, shadow_stats=None,
                              notifier=None):
        """
        Evaluate whether a template+symbol+state combo should be auto-disabled.

        Disable criteria (from TEMPLATE_EVOLUTION_CONFIG):
          - signal_count >= min_signals_to_evaluate AND loss_rate > max_loss_rate
          - OR loss_streak >= min_loss_streak

        If criteria met: adds combo to disable list, sends Telegram notification.
        If combo is already disabled but global win rate has recovered above
        re_enable_win_rate threshold: re-enables and notifies.

        Args:
            template_id: Template ID string
            symbol: Stock ticker
            stock_state: Current state dict (uses 'trend' key for combo key)
            shadow_stats: template_stats section of shadow_ledger (optional, loads if None)
            notifier: NotificationManager instance for Telegram alerts (optional)
        """
        evo_cfg = getattr(cfg, 'TEMPLATE_EVOLUTION_CONFIG', {})
        ad_cfg = evo_cfg.get("auto_disable", {})
        if not ad_cfg.get("enabled", False):
            return

        min_signals = ad_cfg.get("min_signals_to_evaluate", 10)
        max_loss_rate = ad_cfg.get("max_loss_rate", 0.65)
        min_streak = ad_cfg.get("min_loss_streak", 5)
        re_enable_wr = ad_cfg.get("re_enable_win_rate", 0.50)

        # Load per-symbol stats from shadow_ledger if not provided
        if shadow_stats is None:
            shadow_stats = self._load_shadow_stats()

        per_stock = shadow_stats.get(symbol, {}).get(template_id, {})
        signal_count = per_stock.get("signal_count", 0)
        wins = per_stock.get("wins", 0)
        loss_streak = per_stock.get("loss_streak", 0)

        loss_rate = 1.0 - (wins / signal_count) if signal_count > 0 else 0.0

        key = self._disable_combo_key(template_id, symbol, stock_state)
        disabled_set = self._load_disable_list()

        if key in disabled_set:
            # Check for re-enable: global win rate recovered
            global_stat = self._aggregate_global_stats(shadow_stats, template_id)
            global_wr = global_stat.get("win_rate", 0.0) / 100.0
            if global_wr >= re_enable_wr:
                disabled_set.discard(key)
                self._save_disable_list(disabled_set)
                logger.info(
                    f"[AutoDisable] RE-ENABLED {key} | "
                    f"global_wr={global_wr:.1%} | "
                    f"threshold={re_enable_wr:.1%} | "
                    f"status=RE-ENABLED"
                )
                if notifier:
                    try:
                        notifier.send_auto_disable_notification(template_id, symbol, stock_state,
                                                                action="re_enabled")
                    except Exception:
                        pass
            return

        # Evaluate disable criteria
        should_disable = False
        reason = ""
        if signal_count >= min_signals and loss_rate > max_loss_rate:
            should_disable = True
            reason = f"loss_rate={loss_rate:.0%} > {max_loss_rate:.0%} over {signal_count} signals"
        elif loss_streak >= min_streak:
            should_disable = True
            reason = f"loss_streak={loss_streak} >= {min_streak}"

        if should_disable:
            disabled_set.add(key)
            self._save_disable_list(disabled_set)
            logger.warning(
                f"[AutoDisable] DISABLED {key} | "
                f"WR={1-loss_rate:.1%} | signals={signal_count} | "
                f"avg_pnl={per_stock.get('avg_pnl', 0):.2%} | "
                f"loss_streak={loss_streak} | "
                f"best_pnl={per_stock.get('best_pnl', 0):.2%} | "
                f"worst_pnl={per_stock.get('worst_pnl', 0):.2%} | "
                f"status=DISABLED | reason={reason}"
            )
            if notifier:
                try:
                    notifier.send_auto_disable_notification(template_id, symbol, stock_state,
                                                            action="disabled", reason=reason)
                except Exception:
                    pass
        else:
            # Watchlist: underperforming but not yet at disable threshold — log for analysis
            watchlist_lr = ad_cfg.get("watchlist_loss_rate", 0.60)
            if signal_count >= min_signals and loss_rate > watchlist_lr:
                logger.warning(
                    f"[AutoDisable-WATCHLIST] {key} | "
                    f"WR={1-loss_rate:.1%} | signals={signal_count} | "
                    f"avg_pnl={per_stock.get('avg_pnl', 0):.2%} | "
                    f"loss_streak={loss_streak} | "
                    f"best_pnl={per_stock.get('best_pnl', 0):.2%} | "
                    f"worst_pnl={per_stock.get('worst_pnl', 0):.2%} | "
                    f"status=WATCHLIST"
                )

    # ========================================
    # CONTEXTUAL TRUST SYSTEM (CP-2)
    # ========================================

    def _load_trust_matrix(self):
        """Load trust_matrix section from shadow_ledger.json."""
        if self._trust_cache is not None:
            return self._trust_cache
        evo_cfg = getattr(cfg, 'TEMPLATE_EVOLUTION_CONFIG', {})
        path = evo_cfg.get("auto_disable", {}).get(
            "disable_list_path", "data/shadow_ledger.json"
        )
        try:
            data = safe_json_read(path, default={})
            return data.get("trust_matrix", {})
        except Exception:
            return {}

    def _save_trust_matrix(self, trust_matrix):
        """Persist trust_matrix — writes only the trust_matrix key."""
        if self._trust_cache is not None:
            self._trust_cache = trust_matrix
            return
        evo_cfg = getattr(cfg, 'TEMPLATE_EVOLUTION_CONFIG', {})
        path = evo_cfg.get("auto_disable", {}).get(
            "disable_list_path", "data/shadow_ledger.json"
        )
        try:
            data = safe_json_read(path, default={})
            data["trust_matrix"] = trust_matrix
            safe_json_write(path, data)
        except Exception as e:
            logger.error(f"[Trust] Failed to save trust matrix: {e}")

    def _build_state_key(self, stock_state):
        """Build 'trend:structure:volume:volatility' key from state dict."""
        if not stock_state:
            return ":::"
        trend = stock_state.get("trend", "")
        structure = stock_state.get("structure", "")
        volume = stock_state.get("volume", "")
        volatility = stock_state.get("volatility", "")
        return f"{trend}:{structure}:{volume}:{volatility}"

    def _state_key_to_dict(self, state_key):
        """Convert 'trend:structure:volume:volatility' string to state dict.
        Accepts dict (passthrough) or None."""
        if isinstance(state_key, dict) or state_key is None:
            return state_key or {}
        parts = (state_key or "").split(":")
        keys = ["trend", "structure", "volume", "volatility"]
        return {keys[i]: parts[i] for i in range(min(len(parts), len(keys)))}

    def _get_state_group_keys(self, stock_state):
        """Return (L3, L2, L1) grouping keys for hierarchical fallback.

        L3 = full state  "trend:structure:volume:volatility"
        L2 = trend+volatility  "trend:*:*:volatility"
        L1 = trend only  "trend:*:*:*"

        Accepts stock_state as dict or string.
        """
        if isinstance(stock_state, str):
            stock_state = self._state_key_to_dict(stock_state)
        trend = stock_state.get("trend", "") if stock_state else ""
        volatility = stock_state.get("volatility", "") if stock_state else ""
        l3 = self._build_state_key(stock_state)
        l2 = f"{trend}:*:*:{volatility}"
        l1 = f"{trend}:*:*:*"
        return l3, l2, l1

    def _calculate_decayed_wr(self, signals_list):
        """Compute exponentially decayed win rate (most recent = highest weight).

        Args:
            signals_list: list of dicts with 'won' bool, ordered oldest→newest
        Returns:
            float: decayed win rate in [0, 1], or 0.5 if no signals
        """
        if not signals_list:
            return 0.5
        ct_cfg = getattr(cfg, 'TEMPLATE_EVOLUTION_CONFIG', {}).get("contextual_trust", {})
        decay = ct_cfg.get("decay_rate", 0.95)
        n = len(signals_list)
        total_weight = 0.0
        weighted_wins = 0.0
        for i, sig in enumerate(signals_list):
            # most recent signal (i=n-1) gets weight decay^0 = 1.0
            w = decay ** (n - 1 - i)
            total_weight += w
            if sig.get("won", False):
                weighted_wins += w
        return weighted_wins / total_weight if total_weight > 0 else 0.5

    def _wilson_confidence_interval(self, wins, n, z=1.96):
        """Wilson 95% confidence interval for a proportion.

        Args:
            wins: number of winning signals
            n: total signals
            z: z-score (default 1.96 for 95% CI)
        Returns:
            (lower, upper) floats in [0, 1]
        """
        if n == 0:
            return (0.0, 1.0)
        p_hat = wins / n
        z2 = z * z
        denom = 1.0 + z2 / n
        center = (p_hat + z2 / (2.0 * n)) / denom
        margin = z * math.sqrt(
            max(p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n), 0.0)
        ) / denom
        return (max(0.0, round(center - margin, 4)), min(1.0, round(center + margin, 4)))

    def _calculate_bayesian_score(self, local_wr, global_wr, n, prior=0.43):
        """Blend local WR, global WR, and prior using Bayesian-style weighting.

        Local weight scales with n: more data → trust local more.

        Args:
            local_wr: per-(symbol, state) win rate in [0, 1]
            global_wr: cross-symbol win rate for this template in [0, 1]
            n: local signal count
            prior: base-rate prior (default 0.43)
        Returns:
            float: blended trust score in [0, 1]
        """
        ct_cfg = getattr(cfg, 'TEMPLATE_EVOLUTION_CONFIG', {}).get("contextual_trust", {})
        prior_w = ct_cfg.get("bayesian_prior_weight", 0.4)
        global_w = ct_cfg.get("global_fallback_weight", 0.3)
        local_w = ct_cfg.get("local_weight", 0.7)
        burn_in = ct_cfg.get("burn_in_signals", 20)
        local_scale = min(n / max(burn_in, 1), 1.0)
        w_local = local_w * local_scale
        w_global = global_w
        w_prior = prior_w
        total = w_local + w_global + w_prior
        score = (w_local * local_wr + w_global * global_wr + w_prior * prior) / max(total, 1e-9)
        return round(score, 4)

    def _determine_lifecycle(self, wins, n, decayed_wr, config):
        """Classify a trust cell into a lifecycle state.

        Delegates to ShadowLedger.determine_lifecycle which implements
        correct hysteresis (downgrade-only, not unconditional threshold deflation).

        Args:
            wins: total wins (unused — kept for backward-compatible signature)
            n: total signals
            decayed_wr: exponentially decayed win rate
            config: contextual_trust config dict (unused — sl reads from cfg directly)
        Returns:
            str: lifecycle state name (BURN_IN/PROVEN/MONITORING/DEGRADED/DISABLED)
        """
        try:
            from shadow_ledger import ShadowLedger
            sl_tmp = ShadowLedger.__new__(ShadowLedger)
            return sl_tmp.determine_lifecycle(n, decayed_wr)
        except Exception:
            # Fallback: strict thresholds, no hysteresis (conservative)
            min_signals = config.get("lifecycle_check_min_signals", 10)
            if n < min_signals:
                return "BURN_IN"
            if decayed_wr >= config.get("proven_wr_threshold", 0.50) and n >= config.get("min_signals_for_proven", 20):
                return "PROVEN"
            elif decayed_wr >= config.get("monitoring_wr_threshold", 0.35):
                return "MONITORING"
            elif decayed_wr >= config.get("degraded_wr_threshold", 0.20):
                return "DEGRADED"
            return "DISABLED"

    def _aggregate_grouped_cells(self, trust_matrix, template_id, symbol,
                                 key_levels, min_signals=5):
        """Try each (key, level) in order; return first cell with >= min_signals.

        Falls back through L3→L2→L1→None.

        Args:
            trust_matrix: full trust_matrix dict
            template_id: template ID string
            symbol: stock ticker
            key_levels: list of (key, level_name) tuples in priority order
            min_signals: minimum signals needed to use a cell
        Returns:
            (cell_dict | None, level_str)
        """
        sym_data = trust_matrix.get(template_id, {}).get(symbol, {})
        for key, level in key_levels:
            cell = sym_data.get(key)
            if cell and cell.get("total", 0) >= min_signals:
                return cell, level
        # No cell has enough data — return first available (sparse fallback)
        for key, level in key_levels:
            cell = sym_data.get(key)
            if cell:
                return cell, level
        return None, "PRIOR"

    def _get_template_global_wr(self, trust_matrix, template_id):
        """Compute global win rate across all symbols and states for a template.

        Returns:
            float: global win rate in [0, 1], or 0.43 (base prior) if no data
        """
        tmpl_data = trust_matrix.get(template_id, {})
        total_wins = 0
        total_signals = 0
        for sym_data in tmpl_data.values():
            for cell in sym_data.values():
                total_wins += cell.get("wins", 0)
                total_signals += cell.get("total", 0)
        if total_signals == 0:
            return 0.43
        return round(total_wins / total_signals, 4)

    def get_trust_score(self, template_id, symbol, stock_state):
        """Compute contextual trust score for a template+symbol+state triple.

        Uses hierarchical state grouping (L3→L2→L1) for sparse-cell fallback.

        Args:
            template_id: template ID string
            symbol: stock ticker
            stock_state: state dict with trend/structure/volume/volatility,
                         OR a state key string "trend:structure:volume:volatility"
        Returns:
            dict: {
                "score": float [0,1],
                "lifecycle": str,
                "wins": int,
                "total": int,
                "decayed_wr": float,
                "ci_lower": float,
                "ci_upper": float,
                "level_used": str ("L3"/"L2"/"L1"/"PRIOR"),
            }
        """
        # Normalise: accept string state key or dict
        if isinstance(stock_state, str):
            stock_state = self._state_key_to_dict(stock_state)

        ct_cfg = getattr(cfg, 'TEMPLATE_EVOLUTION_CONFIG', {}).get("contextual_trust", {})
        if not ct_cfg.get("enabled", False):
            return {
                "score": 0.5, "lifecycle": "BURN_IN", "wins": 0, "total": 0,
                "decayed_wr": 0.5, "ci_lower": 0.0, "ci_upper": 1.0, "level_used": "PRIOR"
            }

        trust_matrix = self._load_trust_matrix()
        l3_key, l2_key, l1_key = self._get_state_group_keys(stock_state)
        min_signals = ct_cfg.get("min_signals_per_cell", 5)

        cell, level = self._aggregate_grouped_cells(
            trust_matrix, template_id, symbol,
            [(l3_key, "L3"), (l2_key, "L2"), (l1_key, "L1")],
            min_signals
        )

        global_wr = self._get_template_global_wr(trust_matrix, template_id)

        if not cell:
            score = self._calculate_bayesian_score(0.43, global_wr, 0)
            ci = self._wilson_confidence_interval(0, 0)
            return {
                "score": score, "lifecycle": "BURN_IN", "wins": 0, "total": 0,
                "decayed_wr": 0.5, "ci_lower": ci[0], "ci_upper": ci[1], "level_used": "PRIOR"
            }

        wins = cell.get("wins", 0)
        total = cell.get("total", 0)
        signals_list = cell.get("signals", [])

        use_decayed = ct_cfg.get("use_decayed_wr", True)
        decayed_wr = (
            self._calculate_decayed_wr(signals_list)
            if (use_decayed and signals_list)
            else wins / max(total, 1)
        )

        local_wr = wins / max(total, 1)
        score = self._calculate_bayesian_score(local_wr, global_wr, total)
        lifecycle = self._determine_lifecycle(wins, total, decayed_wr, ct_cfg)
        ci = self._wilson_confidence_interval(wins, total)

        return {
            "score": score,
            "lifecycle": lifecycle,
            "wins": wins,
            "total": total,
            "decayed_wr": round(decayed_wr, 4),
            "ci_lower": ci[0],
            "ci_upper": ci[1],
            "level_used": level,
        }

    # ========================================
    # SUIT ASSIGNMENT ENGINE (CP-3)
    # ========================================

    def _get_ledger_path(self):
        """Return path to shadow_ledger.json."""
        evo_cfg = getattr(cfg, 'TEMPLATE_EVOLUTION_CONFIG', {})
        return evo_cfg.get("auto_disable", {}).get(
            "disable_list_path", "data/shadow_ledger.json"
        )

    def _current_date_iso(self):
        """Today's date as YYYY-MM-DD string."""
        return datetime.now().strftime("%Y-%m-%d")

    def _get_template_by_name(self, template_id):
        """Find template object by ID. Returns None if not found."""
        return self._get_template_by_id(template_id)

    def _template_state_matches(self, template, state_key):
        """Return True if template's required_state covers the given state_key string."""
        parts = state_key.split(":") if isinstance(state_key, str) else []
        if len(parts) != 4:
            return True  # Can't determine — allow
        state_dict = {
            "trend": parts[0], "structure": parts[1],
            "volume": parts[2], "volatility": parts[3],
        }
        req = getattr(template, 'required_state', {})
        if not req:
            return True  # No required state — always matches
        for axis in ("trend", "structure", "volume", "volatility"):
            req_vals = req.get(axis, [])
            if req_vals and state_dict.get(axis, "") not in req_vals:
                return False
        return True

    def _state_matches_group(self, state_key, group_key):
        """Check if state_key matches a group pattern like 'BULLISH:*:*:COMPRESSED'."""
        s_parts = state_key.split(":")
        g_parts = group_key.split(":")
        if len(s_parts) != len(g_parts):
            return False
        return all(gp == "*" or gp == sp for gp, sp in zip(g_parts, s_parts))

    def _is_exploration_bar(self):
        """Return True ~exploration_pct of the time (deterministic per eval counter)."""
        import hashlib
        sa_cfg = getattr(cfg, 'TEMPLATE_EVOLUTION_CONFIG', {}).get("suit_assignment", {})
        pct = sa_cfg.get("exploration_pct", 0.15)
        self._eval_counter = getattr(self, '_eval_counter', 0) + 1
        hash_val = int(
            hashlib.md5(str(self._eval_counter).encode()).hexdigest()[:8], 16
        )
        return (hash_val % 100) < int(pct * 100)

    def _load_assignments(self):
        """Load suit assignments from shadow_ledger.json['suit_assignments']['assignments']."""
        if self._suit_cache is not None:
            return self._suit_cache
        try:
            data = safe_json_read(self._get_ledger_path(), default={})
            return data.get("suit_assignments", {}).get("assignments", {})
        except Exception:
            return {}

    def _save_assignments(self, assignments):
        """Persist suit assignments — writes only the suit_assignments key."""
        if self._suit_cache is not None:
            self._suit_cache = assignments
            return
        try:
            data = safe_json_read(self._get_ledger_path(), default={})
            if "suit_assignments" not in data:
                data["suit_assignments"] = {}
            data["suit_assignments"]["assignments"] = assignments
            data["suit_assignments"]["last_assignment"] = self._current_date_iso()
            safe_json_write(self._get_ledger_path(), data)
        except Exception as e:
            logger.error(f"[Suit] Failed to save assignments: {e}")

    def _load_assignment_history(self):
        """Load assignment history list from shadow_ledger.json."""
        try:
            data = safe_json_read(self._get_ledger_path(), default={})
            return data.get("suit_assignments", {}).get("history", [])
        except Exception:
            return []

    def _record_assignment_changes(self, changes):
        """Append assignment changes to history, capped at max_history_entries × 13."""
        sa_cfg = getattr(cfg, 'TEMPLATE_EVOLUTION_CONFIG', {}).get("suit_assignment", {})
        max_entries = sa_cfg.get("max_history_entries", 52) * 13
        try:
            data = safe_json_read(self._get_ledger_path(), default={})
            if "suit_assignments" not in data:
                data["suit_assignments"] = {}
            history = data["suit_assignments"].get("history", [])
            history.extend(changes)
            if len(history) > max_entries:
                history = history[-max_entries:]
            data["suit_assignments"]["history"] = history
            safe_json_write(self._get_ledger_path(), data)
        except Exception as e:
            logger.error(f"[Suit] Failed to record assignment changes: {e}")

    def _find_suit_clusters(self, assignments):
        """Find groups of symbols that share identical sets of assigned templates."""
        suit_fingerprints = {}
        for symbol, sym_data in assignments.items():
            templates = sorted(set(
                a.get("assigned_template", "NONE")
                for a in sym_data.get("by_state", {}).values()
                if a.get("assigned_template")
            ))
            if templates:
                fingerprint = "|".join(templates)
                suit_fingerprints.setdefault(fingerprint, []).append(symbol)
        return [
            {"templates": fp.split("|"), "symbols": syms}
            for fp, syms in suit_fingerprints.items()
            if len(syms) > 1
        ]

    def get_suit_sharing_report(self):
        """Cross-stock analysis: template usage counts and clusters of similar symbols."""
        assignments = self._load_assignments()
        template_usage = {}
        symbol_suits = {}

        for symbol, sym_data in assignments.items():
            symbol_suits[symbol] = set()
            for state_key, assignment in sym_data.get("by_state", {}).items():
                tmpl = assignment.get("assigned_template")
                if tmpl:
                    template_usage.setdefault(tmpl, set()).add(symbol)
                    symbol_suits[symbol].add(tmpl)

        clusters = self._find_suit_clusters(assignments)
        return {
            "template_usage": {t: sorted(list(s)) for t, s in template_usage.items()},
            "symbol_diversity": {s: len(t) for s, t in symbol_suits.items()},
            "clusters": clusters,
        }

    def _log_suit_summary(self, assignments, changes):
        """Log [SUIT-SUMMARY], [SUIT-ASSIGN/GAP], [SUIT-CHANGE], [SUIT-CLUSTER] lines."""
        total = len(assignments)
        assigned_count = sum(
            1 for s in assignments.values()
            if any(a.get("assigned_template") for a in s.get("by_state", {}).values())
        )
        gap_count = total - assigned_count
        logger.info(
            f"[SUIT-SUMMARY] symbols={total} | "
            f"with_suit={assigned_count} | gaps={gap_count} | changes={len(changes)}"
        )
        for symbol, sym_data in assignments.items():
            for state_key, assignment in sym_data.get("by_state", {}).items():
                tmpl = assignment.get("assigned_template")
                if tmpl:
                    logger.info(
                        f"[SUIT-ASSIGN] {symbol}:{state_key} | "
                        f"template={tmpl} | score={assignment.get('score', 0):.3f} | "
                        f"wr={assignment.get('wr', 0):.1%} | "
                        f"lifecycle={assignment.get('lifecycle', 'N/A')} | "
                        f"confidence={assignment.get('confidence', 'N/A')} | "
                        f"runner_up={assignment.get('runner_up', 'NONE')} | "
                        f"candidates={assignment.get('candidates_count', 0)}"
                    )
                else:
                    logger.warning(
                        f"[SUIT-GAP] {symbol}:{state_key} | "
                        f"assigned=NONE | reason={assignment.get('reason', 'unknown')}"
                    )
        for change in changes:
            logger.info(
                f"[SUIT-CHANGE] {change['symbol']}:{change['state']} | "
                f"from={change['from_template']} | to={change['to_template']} | "
                f"reason={change['reason']} | "
                f"old_score={change.get('old_score', 0):.3f} | "
                f"new_score={change.get('new_score', 0):.3f}"
            )
        # Cluster report
        sharing = self.get_suit_sharing_report()
        for cluster in sharing.get("clusters", []):
            logger.info(
                f"[SUIT-CLUSTER] templates={cluster['templates']} | "
                f"symbols={cluster['symbols']}"
            )

    def get_suit(self, symbol, state_key):
        """Get assigned template for symbol+state. Returns assignment dict or None.

        Lookup order: exact state → L2 group → L1 group → default.
        """
        sa_cfg = getattr(cfg, 'TEMPLATE_EVOLUTION_CONFIG', {}).get("suit_assignment", {})
        if not sa_cfg.get("enabled", False):
            return None

        assignments = self._load_assignments()
        sym_data = assignments.get(symbol, {})

        # Exact state match
        state_assignment = sym_data.get("by_state", {}).get(state_key)
        if state_assignment and state_assignment.get("assigned_template"):
            return state_assignment

        # Grouped fallback: L2 then L1
        l3, l2, l1 = self._get_state_group_keys(self._state_key_to_dict(state_key))
        for group_key in (l2, l1):
            for assigned_state, assignment in sym_data.get("by_state", {}).items():
                if (self._state_matches_group(assigned_state, group_key)
                        and assignment.get("assigned_template")):
                    return assignment

        # Default suit fallback
        default_template = sym_data.get("default")
        if default_template:
            return {
                "assigned_template": default_template,
                "score": 0.0,
                "source": "default",
                "confidence": "LOW",
            }

        return None

    def assign_suits(self):
        """Weekly: assign best template per symbol+state based on trust scores.

        Returns:
            dict: assignments by symbol
        """
        sa_cfg = getattr(cfg, 'TEMPLATE_EVOLUTION_CONFIG', {}).get("suit_assignment", {})
        if not sa_cfg.get("enabled", False):
            return {}

        matrix = self._load_trust_matrix()
        old_assignments = self._load_assignments()
        disabled_set = self._load_disable_list()

        min_score = sa_cfg.get("min_trust_score_to_assign", 0.20)
        min_signals = sa_cfg.get("min_signals_to_assign", 10)
        min_confidence_signals = sa_cfg.get("min_signals_for_high_confidence", 20)
        default_min_lifecycle = sa_cfg.get("default_min_lifecycle", "MONITORING")
        lifecycle_rank = {
            "PROVEN": 4, "MONITORING": 3, "DEGRADED": 2, "BURN_IN": 1, "DISABLED": 0
        }

        # Collect all template IDs and symbols from trust_matrix
        all_templates = set(matrix.keys())
        all_symbols = set()
        for sym_dict in matrix.values():
            all_symbols.update(sym_dict.keys())

        assignments = {}
        changes = []

        for symbol in sorted(all_symbols):
            assignments[symbol] = {"by_state": {}, "default": None}

            # Collect all state keys seen for this symbol across all templates
            all_states = set()
            for template_id in all_templates:
                all_states.update(matrix.get(template_id, {}).get(symbol, {}).keys())

            for state_key in sorted(all_states):
                candidates = []
                for template_id in all_templates:
                    # Exclude disabled combos
                    trend = state_key.split(":")[0] if state_key else ""
                    combo_key = f"{template_id}::{symbol}::{trend}"
                    if combo_key in disabled_set:
                        continue

                    trust = self.get_trust_score(template_id, symbol, state_key)
                    if trust["total"] >= min_signals and trust["score"] >= min_score:
                        candidates.append({
                            "template": template_id,
                            "score": trust["score"],
                            "wr": trust["decayed_wr"],
                            "signals": trust["total"],
                            "lifecycle": trust["lifecycle"],
                            "ci": (trust["ci_lower"], trust["ci_upper"]),
                            "source": trust["level_used"],
                        })

                if candidates:
                    candidates.sort(key=lambda c: (-c["score"], -c["signals"]))
                    best = candidates[0]
                    runner_up = candidates[1] if len(candidates) > 1 else None
                    confidence = (
                        "HIGH" if best["signals"] >= min_confidence_signals else "LOW"
                    )
                    assignment = {
                        "assigned_template": best["template"],
                        "score": best["score"],
                        "wr": best["wr"],
                        "signals": best["signals"],
                        "lifecycle": best["lifecycle"],
                        "ci": best["ci"],
                        "confidence": confidence,
                        "runner_up": runner_up["template"] if runner_up else None,
                        "runner_up_score": runner_up["score"] if runner_up else None,
                        "candidates_count": len(candidates),
                        "assigned_at": self._current_date_iso(),
                    }
                    # Track template swaps
                    old = (old_assignments.get(symbol, {})
                           .get("by_state", {}).get(state_key, {}))
                    old_tmpl = old.get("assigned_template")
                    if old_tmpl and old_tmpl != best["template"]:
                        changes.append({
                            "date": self._current_date_iso(),
                            "symbol": symbol,
                            "state": state_key,
                            "from_template": old_tmpl,
                            "to_template": best["template"],
                            "reason": "score_improvement",
                            "old_score": old.get("score", 0),
                            "new_score": best["score"],
                        })
                    assignments[symbol]["by_state"][state_key] = assignment
                else:
                    assignments[symbol]["by_state"][state_key] = {
                        "assigned_template": None,
                        "score": 0.0,
                        "reason": "NO_QUALIFIED_CANDIDATE",
                        "candidates_count": 0,
                        "assigned_at": self._current_date_iso(),
                    }

            # Default suit: best template for this symbol with lifecycle >= default_min_lifecycle
            global_candidates = []
            for template_id in all_templates:
                sym_cells = matrix.get(template_id, {}).get(symbol, {})
                if not sym_cells:
                    continue
                total_wins = sum(c.get("wins", 0) for c in sym_cells.values())
                total_sigs = sum(c.get("total", 0) for c in sym_cells.values())
                if total_sigs < min_signals:
                    continue
                global_wr = total_wins / total_sigs
                lifecycles = [
                    c.get("lifecycle", "BURN_IN")
                    for c in sym_cells.values() if c.get("total", 0) > 0
                ]
                dominant_lc = (
                    max(lifecycles, key=lambda lc: lifecycle_rank.get(lc, 0))
                    if lifecycles else "BURN_IN"
                )
                if lifecycle_rank.get(dominant_lc, 0) >= lifecycle_rank.get(default_min_lifecycle, 0):
                    global_candidates.append({
                        "template": template_id,
                        "score": global_wr,
                        "signals": total_sigs,
                        "lifecycle": dominant_lc,
                    })

            if global_candidates:
                global_candidates.sort(key=lambda c: (-c["score"], -c["signals"]))
                assignments[symbol]["default"] = global_candidates[0]["template"]

        self._save_assignments(assignments)
        if changes and sa_cfg.get("track_assignment_history", True):
            self._record_assignment_changes(changes)
        self._log_suit_summary(assignments, changes)

        return assignments
