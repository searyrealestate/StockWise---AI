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
import math
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

                    # Record block-level statistics (P1 #7A)
                    if not all_passed:
                        try:
                            template.record_block_results(
                                _details, symbol=symbol, all_passed=False,
                                outcome=None
                            )
                        except Exception:
                            pass
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

                    # Record block stats WITH outcome for passed signals
                    try:
                        template.record_block_results(
                            _details, symbol=symbol, all_passed=True,
                            outcome=outcome
                        )
                    except Exception:
                        pass

                    # Record result
                    results[template.id]["signal_count"] += 1
                    if outcome["hit"] == "target":
                        results[template.id]["wins"] += 1
                        results[template.id]["total_pnl_pct"] += outcome["pnl_pct"]
                    elif outcome["hit"] == "stop":
                        results[template.id]["losses"] += 1
                        results[template.id]["total_pnl_pct"] += outcome["pnl_pct"]
                    # "neither" = open after lookahead window — not counted as win or loss

                    # Attribution analytics (SPEC §4)
                    try:
                        self._record_signal_attribution(
                            template, symbol, df, i, outcome,
                            entry_price, stop_loss, take_profit
                        )
                    except Exception as _attr_exc:
                        logger.warning(
                            f"[{symbol}] Attribution failed for {template.id}: {_attr_exc}"
                        )

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

    # ═══════════════════════════════════════════════════════════════
    # ATTRIBUTION ANALYTICS  (SPEC §4)
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _safe_float(val, ndigits=4):
        """Return float rounded to ndigits, or None for NaN/Inf/invalid."""
        try:
            v = float(val)
            return None if (math.isnan(v) or math.isinf(v)) else round(v, ndigits)
        except (TypeError, ValueError):
            return None

    # ── A. Kill candle classification ─────────────────────────────

    def _classify_kill_type(self, prev_close, open_price, high, low, close, stop_price):
        """Return "gap_down" | "wick" | "drift" | "reversal"."""
        try:
            sf = self._safe_float
            pc = sf(prev_close); op = sf(open_price)
            lo = sf(low); cl = sf(close); sp = sf(stop_price)
            if None in (pc, op, lo, cl, sp):
                return "reversal"

            gap_pct = (op - pc) / pc * 100 if pc != 0 else 0
            if gap_pct < -0.5 and op <= sp:
                return "gap_down"

            body = cl - op
            tail = min(op, cl) - lo
            if abs(body) > 0 and tail > abs(body) * 2:
                return "wick"

            if op != 0 and abs(cl - op) / op * 100 < 0.3:
                return "drift"

            return "reversal"
        except Exception:
            return "reversal"

    def _build_kill_candle_data(self, bars, entry_idx, exit_idx, entry_price, stop_price):
        """Build kill candle analytics dict for a losing trade."""
        try:
            sf = self._safe_float
            if exit_idx is None or exit_idx >= len(bars):
                return None

            kb = bars.iloc[exit_idx]
            pb = bars.iloc[exit_idx - 1] if exit_idx > 0 else kb

            op = sf(kb.get('open')); cl = sf(kb.get('close'))
            hi = sf(kb.get('high')); lo = sf(kb.get('low'))
            pc = sf(pb.get('close'))
            vol = sf(kb.get('volume')); va = sf(kb.get('vol_avg_20'))

            kill_type = self._classify_kill_type(pc, op, hi, lo, cl, stop_price)

            body_pct = sf((cl - op) / op * 100) if op else None
            wick_pct = sf((hi - max(op or 0, cl or 0)) / op * 100) if op and hi and cl else None
            tail_pct = sf((min(op or 0, cl or 0) - lo) / op * 100) if op and lo and cl else None
            gap_pct = sf((op - pc) / pc * 100) if pc and pc != 0 and op else None
            vol_ratio = sf(vol / va) if vol and va and va != 0 else None

            max_fav_pct = None
            ep = self._safe_float(entry_price)
            if ep:
                highs = [sf(bars.iloc[j].get('high')) for j in range(entry_idx + 1, exit_idx + 1)
                         if sf(bars.iloc[j].get('high')) is not None]
                if highs:
                    max_fav_pct = sf((max(highs) - ep) / ep * 100)

            bars_in_trade = exit_idx - entry_idx if entry_idx is not None else None
            stop_dist_pct = sf((stop_price - entry_price) / entry_price * 100) if entry_price else None

            return {
                "kill_type": kill_type,
                "candle_body_pct": body_pct,
                "candle_wick_pct": wick_pct,
                "candle_tail_pct": tail_pct,
                "gap_pct": gap_pct,
                "volume_ratio": vol_ratio,
                "phase_at_death": None,
                "bars_in_trade": bars_in_trade,
                "max_favorable_pct": max_fav_pct,
                "stop_distance_pct": stop_dist_pct,
            }
        except Exception as e:
            logger.warning(f"[Attribution] _build_kill_candle_data failed: {e}")
            return None

    # ── C. Entry quality ──────────────────────────────────────────

    def _build_entry_quality(self, bars, entry_idx, entry_price):
        """Score the quality of entry timing."""
        try:
            sf = self._safe_float
            eb = bars.iloc[entry_idx]
            ep = sf(entry_price)
            bar_low = sf(eb.get('low'))
            bar_open = sf(eb.get('open'))

            entry_vs_low = sf((ep - bar_low) / bar_low * 100) if ep and bar_low else None
            entry_vs_open = sf((ep - bar_open) / bar_open * 100) if ep and bar_open else None

            # Bars to first profitable close
            bars_to_profit = None
            for j in range(entry_idx + 1, min(entry_idx + 21, len(bars))):
                cl = sf(bars.iloc[j].get('close'))
                if cl and ep and cl > ep:
                    bars_to_profit = j - entry_idx
                    break

            # Max drawdown in first 3 bars
            imm_dd = None
            lows = [sf(bars.iloc[j].get('low')) for j in range(entry_idx + 1, min(entry_idx + 4, len(bars)))
                    if sf(bars.iloc[j].get('low')) is not None]
            if lows and ep:
                imm_dd = sf((min(lows) - ep) / ep * 100)

            return {
                "entry_vs_low_pct": entry_vs_low,
                "entry_vs_open_pct": entry_vs_open,
                "bars_to_first_profit": bars_to_profit,
                "immediate_drawdown_pct": imm_dd,
            }
        except Exception as e:
            logger.warning(f"[Attribution] _build_entry_quality failed: {e}")
            return None

    # ── D. Volume profile ─────────────────────────────────────────

    def _build_volume_profile(self, bars, entry_idx, exit_idx):
        """Volume ratios at entry/exit and trend across the trade."""
        try:
            sf = self._safe_float
            clamp_exit = min(exit_idx, len(bars) - 1)
            eb = bars.iloc[entry_idx]
            xb = bars.iloc[clamp_exit]

            va_e = sf(eb.get('vol_avg_20'))
            vol_at_entry = sf(sf(eb.get('volume')) / va_e) if sf(eb.get('volume')) and va_e else None

            va_x = sf(xb.get('vol_avg_20')) or va_e
            vol_at_exit = sf(sf(xb.get('volume')) / va_x) if sf(xb.get('volume')) and va_x else None

            ratios = []
            for j in range(entry_idx, clamp_exit + 1):
                v = sf(bars.iloc[j].get('volume'))
                va = sf(bars.iloc[j].get('vol_avg_20')) or va_e
                if v and va:
                    ratios.append(v / va)

            avg_vol = sf(sum(ratios) / len(ratios)) if ratios else None

            trend = "flat"
            if len(ratios) >= 2:
                mid = len(ratios) // 2
                f_avg = sum(ratios[:mid]) / mid if mid else 0
                s_avg = sum(ratios[mid:]) / (len(ratios) - mid) if (len(ratios) - mid) else 0
                if f_avg and s_avg > f_avg * 1.2:
                    trend = "increasing"
                elif f_avg and s_avg < f_avg * 0.8:
                    trend = "decreasing"

            return {
                "volume_at_entry": vol_at_entry,
                "volume_at_exit": vol_at_exit,
                "avg_volume_during_trade": avg_vol,
                "volume_trend": trend,
            }
        except Exception as e:
            logger.warning(f"[Attribution] _build_volume_profile failed: {e}")
            return None

    # ── E. Market context (SPY) ───────────────────────────────────

    def _build_market_context(self, spy_bars, entry_idx, exit_idx):
        """SPY return on exit day and during trade. Returns None if no SPY data."""
        if spy_bars is None:
            return None
        try:
            sf = self._safe_float
            clamp = min(exit_idx, len(spy_bars) - 1)
            if entry_idx >= len(spy_bars):
                return None

            eb = spy_bars.iloc[entry_idx]
            xb = spy_bars.iloc[clamp]

            spy_open_x = sf(xb.get('open'))
            spy_close_x = sf(xb.get('close'))
            spy_day = sf((spy_close_x - spy_open_x) / spy_open_x * 100) if spy_open_x else None

            spy_entry_cl = sf(eb.get('close'))
            spy_trade = sf((spy_close_x - spy_entry_cl) / spy_entry_cl * 100) if spy_entry_cl else None

            spy_trend = "BULLISH" if (spy_trade is not None and spy_trade > 0) else "BEARISH"

            return {
                "spy_return_on_day": spy_day,
                "spy_return_during_trade": spy_trade,
                "spy_trend": spy_trend,
            }
        except Exception as e:
            logger.warning(f"[Attribution] _build_market_context failed: {e}")
            return None

    # ── F. Indicator snapshot ─────────────────────────────────────

    def _build_indicator_snapshot(self, bars, entry_idx, exit_idx):
        """RSI/ER/ATR/BB/ADX at entry, exit, and delta between them."""
        try:
            sf = self._safe_float
            COLS = ['rsi', 'er_fast', 'er_slow', 'atr', 'bb_width', 'volume_ratio', 'adx']
            clamp = min(exit_idx, len(bars) - 1)
            eb = bars.iloc[entry_idx]
            xb = bars.iloc[clamp]

            def read(row):
                return {c: sf(row.get(c)) for c in COLS}

            at_entry = read(eb)
            at_exit = read(xb)
            delta = {
                c: sf(at_entry[c] - at_exit[c]) if at_entry[c] is not None and at_exit[c] is not None else None
                for c in COLS
            }
            return {"at_entry": at_entry, "at_exit": at_exit, "delta": delta}
        except Exception as e:
            logger.warning(f"[Attribution] _build_indicator_snapshot failed: {e}")
            return None

    # ── G. Weakest block ──────────────────────────────────────────

    def _compute_block_margin(self, block_name, row, params):
        """Margin between actual indicator value and block threshold (larger = safer)."""
        sf = self._safe_float

        if block_name == 'rsi_between' and len(params) >= 2:
            rsi = sf(row.get('rsi'))
            if rsi is None: return None
            return min(rsi - params[0], params[1] - rsi)

        if block_name in ('er_slow_above', 'er_fast_above') and params:
            col = 'er_slow' if block_name == 'er_slow_above' else 'er_fast'
            v = sf(row.get(col))
            return (v - params[0]) if v is not None else None

        if block_name == 'close_above_sma' and params:
            close = sf(row.get('close'))
            sma = sf(row.get(f'sma_{int(params[0])}'))
            return (close - sma) if close is not None and sma is not None else None

        if block_name == 'volume_surge' and params:
            vol = sf(row.get('volume')); va = sf(row.get('vol_avg_20'))
            if vol is None or va is None or va == 0: return None
            return (vol / va) - params[0]

        if block_name == 'macd_above_signal':
            macd = sf(row.get('macd')); sig = sf(row.get('macd_signal'))
            return (macd - sig) if macd is not None and sig is not None else None

        if block_name == 'adx_above' and params:
            adx = sf(row.get('adx'))
            return (adx - params[0]) if adx is not None else None

        if block_name == 'sma_above_sma' and len(params) >= 2:
            fast = sf(row.get(f'sma_{int(params[0])}'))
            slow = sf(row.get(f'sma_{int(params[1])}'))
            return (fast - slow) if fast is not None and slow is not None else None

        return None

    def _build_weakest_block(self, template, bars, entry_idx):
        """Find the condition block with the smallest margin to its threshold."""
        try:
            row = bars.iloc[entry_idx]
            min_margin = float('inf')
            weakest = None

            for block_spec in template.conditions:
                block_name = block_spec.get('block', '')
                params = block_spec.get('params', [])
                try:
                    margin = self._compute_block_margin(block_name, row, params)
                    if margin is None or math.isnan(float(margin)):
                        continue
                    margin = float(margin)
                    if margin < min_margin:
                        min_margin = margin
                        threshold = params[0] if params else 0
                        weakest = {
                            "block_name": block_name,
                            "value_at_entry": self._safe_float(margin + float(threshold)) if isinstance(threshold, (int, float)) else None,
                            "threshold": self._safe_float(threshold) if isinstance(threshold, (int, float)) else None,
                            "margin": self._safe_float(margin),
                        }
                except Exception:
                    continue

            return weakest
        except Exception as e:
            logger.warning(f"[Attribution] _build_weakest_block failed: {e}")
            return None

    # ── H. Risk/Reward ────────────────────────────────────────────

    def _build_risk_reward(self, entry_price, stop_price, target_price,
                           exit_price, bars, entry_idx, exit_idx):
        """Planned vs realized R:R and max favorable excursion."""
        try:
            sf = self._safe_float
            ep = sf(entry_price); sp = sf(stop_price)
            tp = sf(target_price); xp = sf(exit_price)

            risk = (ep - sp) if ep and sp else None
            reward = (tp - ep) if tp and ep else None

            planned_rr = sf(reward / risk) if risk and risk != 0 else None
            realized_rr = sf((xp - ep) / risk) if xp and ep and risk and risk != 0 else None
            tgt_dist = sf((tp - ep) / ep * 100) if tp and ep else None
            stop_dist = sf((ep - sp) / ep * 100) if ep and sp else None

            clamp = min(exit_idx, len(bars) - 1)
            highs = [sf(bars.iloc[j].get('high')) for j in range(entry_idx + 1, clamp + 1)
                     if sf(bars.iloc[j].get('high')) is not None]
            max_fav = None; max_fav_rr = None
            if highs and ep:
                max_fav = sf((max(highs) - ep) / ep * 100)
                if risk and risk != 0:
                    max_fav_rr = sf((max(highs) - ep) / risk)

            return {
                "planned_rr": planned_rr,
                "realized_rr": realized_rr,
                "target_distance_pct": tgt_dist,
                "stop_distance_pct": stop_dist,
                "max_favorable_pct": max_fav,
                "max_favorable_rr": max_fav_rr,
            }
        except Exception as e:
            logger.warning(f"[Attribution] _build_risk_reward failed: {e}")
            return None

    # ── I. Time context ───────────────────────────────────────────

    def _build_time_context(self, bars, entry_idx, exit_idx):
        """Day-of-week, dates, and bars/calendar days in trade."""
        try:
            idx = bars.index
            entry_ts = idx[entry_idx]
            clamp = min(exit_idx, len(bars) - 1)
            exit_ts = idx[clamp]

            entry_dt = pd.Timestamp(entry_ts)
            exit_dt = pd.Timestamp(exit_ts)
            cal_days = (exit_dt - entry_dt).days

            return {
                "entry_day_of_week": entry_dt.day_name(),
                "entry_date": str(entry_dt.date()),
                "exit_date": str(exit_dt.date()),
                "bars_in_trade": clamp - entry_idx,
                "calendar_days_in_trade": cal_days,
            }
        except Exception as e:
            logger.warning(f"[Attribution] _build_time_context failed: {e}")
            return None

    # ── J. Preceding candles ──────────────────────────────────────

    def _build_preceding_candles(self, bars, entry_idx):
        """Multi-window price action analysis before entry bar."""
        try:
            sf = self._safe_float
            evo_cfg = getattr(cfg, 'TEMPLATE_EVOLUTION_CONFIG', {}).get("attribution", {})
            windows = evo_cfg.get("preceding_candle_windows", [3, 5, 10])
            result = {"windows": windows}

            for w in windows:
                start = entry_idx - w
                if start < 0:
                    result[f"window_{w}"] = None
                    continue

                slice_df = bars.iloc[start:entry_idx]
                if len(slice_df) == 0:
                    result[f"window_{w}"] = None
                    continue

                opens = [sf(r.get('open')) for _, r in slice_df.iterrows()]
                closes = [sf(r.get('close')) for _, r in slice_df.iterrows()]
                highs = [sf(r.get('high')) for _, r in slice_df.iterrows()]
                lows = [sf(r.get('low')) for _, r in slice_df.iterrows()]
                vols = [sf(r.get('volume')) for _, r in slice_df.iterrows()]
                vas = [sf(r.get('vol_avg_20')) for _, r in slice_df.iterrows()]

                # Valid pairs only
                valid = [(o, c) for o, c in zip(opens, closes) if o and c]
                green_count = sum(1 for o, c in valid if c > o)
                green_pct = green_count / len(valid) if valid else 0.5

                if green_pct > 0.7:
                    pattern = "bullish"
                elif green_pct < 0.3:
                    pattern = "bearish"
                elif all(o and c and abs(c - o) / o * 100 < 0.2 for o, c in valid if o):
                    pattern = "doji_sequence"
                else:
                    pattern = "mixed"

                first_close = next((sf(r.get('close')) for _, r in slice_df.iterrows()
                                    if sf(r.get('close'))), None)
                last_close = next((sf(r.get('close')) for _, r in reversed(list(slice_df.iterrows()))
                                   if sf(r.get('close'))), None)
                momentum_pct = sf((last_close - first_close) / first_close * 100) if first_close and last_close else None

                # Volume trend
                vol_ratios = [v / va for v, va in zip(vols, vas) if v and va and va != 0]
                v_trend = "flat"
                if len(vol_ratios) >= 2:
                    mid = len(vol_ratios) // 2
                    fa = sum(vol_ratios[:mid]) / mid if mid else 0
                    sa = sum(vol_ratios[mid:]) / (len(vol_ratios) - mid) if (len(vol_ratios) - mid) else 0
                    if fa and sa > fa * 1.2:
                        v_trend = "increasing"
                    elif fa and sa < fa * 0.8:
                        v_trend = "decreasing"

                all_highs = [h for h in highs if h]
                all_lows = [l for l in lows if l]
                ep = sf(bars.iloc[entry_idx].get('close'))
                hh_dist = sf((ep - max(all_highs)) / max(all_highs) * 100) if all_highs and ep else None
                ll_dist = sf((ep - min(all_lows)) / min(all_lows) * 100) if all_lows and ep else None

                bodies = [abs(c - o) / o * 100 for o, c in valid if o]
                avg_body = sf(sum(bodies) / len(bodies)) if bodies else None

                result[f"window_{w}"] = {
                    "pattern": pattern,
                    "momentum_pct": momentum_pct,
                    "volume_trend": v_trend,
                    "highest_high_dist_pct": hh_dist,
                    "lowest_low_dist_pct": ll_dist,
                    "green_candle_pct": sf(green_pct * 100),
                    "avg_body_pct": avg_body,
                }

            return result
        except Exception as e:
            logger.warning(f"[Attribution] _build_preceding_candles failed: {e}")
            return None

    # ── K. Key levels ─────────────────────────────────────────────

    def _build_key_levels(self, bars, entry_idx, entry_price):
        """Distance to SMA50/200 and recent swing high/low (20-bar lookback)."""
        try:
            sf = self._safe_float
            ep = sf(entry_price)
            row = bars.iloc[entry_idx]

            sma50 = sf(row.get('sma_50'))
            sma200 = sf(row.get('sma_200'))
            dist_sma50 = sf((ep - sma50) / sma50 * 100) if ep and sma50 else None
            dist_sma200 = sf((ep - sma200) / sma200 * 100) if ep and sma200 else None

            lookback = min(20, entry_idx)
            swing_slice = bars.iloc[max(0, entry_idx - lookback):entry_idx]
            resist = support = None
            if len(swing_slice):
                highs = [sf(r.get('high')) for _, r in swing_slice.iterrows() if sf(r.get('high'))]
                lows = [sf(r.get('low')) for _, r in swing_slice.iterrows() if sf(r.get('low'))]
                resist = max(highs) if highs else None
                support = min(lows) if lows else None

            dist_res = sf((ep - resist) / resist * 100) if ep and resist else None
            dist_sup = sf((ep - support) / support * 100) if ep and support else None

            return {
                "distance_to_resistance_pct": dist_res,
                "distance_to_support_pct": dist_sup,
                "distance_to_sma200_pct": dist_sma200,
                "distance_to_sma50_pct": dist_sma50,
            }
        except Exception as e:
            logger.warning(f"[Attribution] _build_key_levels failed: {e}")
            return None

    # ── L. Concurrent signals ─────────────────────────────────────

    def _build_concurrent_signals(self, template_name, symbol, signal_date, all_signals_cache):
        """Count signals fired on the same date from the cache."""
        if all_signals_cache is None:
            return None
        try:
            day_signals = all_signals_cache.get(signal_date, [])
            wins = sum(1 for s in day_signals if s.get("outcome") == "win")
            losses = sum(1 for s in day_signals if s.get("outcome") == "loss")
            same_tmpl = sum(1 for s in day_signals if s.get("template") == template_name)
            return {
                "signals_same_day": len(day_signals),
                "wins_same_day": wins,
                "losses_same_day": losses,
                "same_template_same_day": same_tmpl,
            }
        except Exception as e:
            logger.warning(f"[Attribution] _build_concurrent_signals failed: {e}")
            return None

    # ── M. Record attribution ─────────────────────────────────────

    def _record_attribution(self, template_name, symbol, attribution_data):
        """Append attribution record to shadow_ledger.json under attributions key."""
        try:
            evo_cfg = getattr(cfg, 'TEMPLATE_EVOLUTION_CONFIG', {}).get("attribution", {})
            max_records = evo_cfg.get("max_attribution_records", 500)

            data = safe_json_read(self.ledger_path, default={})
            if "attributions" not in data:
                data["attributions"] = {}
            if template_name not in data["attributions"]:
                data["attributions"][template_name] = {}
            if symbol not in data["attributions"][template_name]:
                data["attributions"][template_name][symbol] = []

            records = data["attributions"][template_name][symbol]
            records.append(attribution_data)

            # Rolling window — keep newest max_records
            if len(records) > max_records:
                data["attributions"][template_name][symbol] = records[-max_records:]

            safe_json_write(self.ledger_path, data)
        except Exception as e:
            logger.error(f"[Attribution] _record_attribution failed for {template_name}:{symbol}: {e}")

    # ── Orchestrator ──────────────────────────────────────────────

    def _record_signal_attribution(self, template, symbol, df, entry_idx,
                                   outcome, entry_price, stop_loss, take_profit):
        """Build and persist all attribution fields for one virtual signal."""
        evo_cfg = getattr(cfg, 'TEMPLATE_EVOLUTION_CONFIG', {}).get("attribution", {})
        if not evo_cfg.get("enabled", False):
            return

        exit_idx = min(entry_idx + outcome.get("bars", 1), len(df) - 1)
        target_hit = outcome["hit"] == "target"
        pnl = outcome.get("pnl_pct", 0.0)
        exit_price = stop_loss if outcome["hit"] == "stop" else (
            take_profit if target_hit else self._safe_float(df.iloc[exit_idx].get('close', entry_price))
        )

        try:
            entry_ts = df.index[entry_idx]
            current_date_iso = str(pd.Timestamp(entry_ts).date())
        except Exception:
            current_date_iso = datetime.now().date().isoformat()

        attribution = {
            "date": current_date_iso,
            "outcome": "win" if target_hit else "loss",
            "pnl_pct": round(pnl, 4),
        }

        def _try(key, fn, *args, **kwargs):
            try:
                attribution[key] = fn(*args, **kwargs)
            except Exception as exc:
                logger.warning(f"[Attribution] {key} builder failed: {exc}")
                attribution[key] = None

        if evo_cfg.get("track_kill_candle") and not target_hit:
            _try("kill_candle", self._build_kill_candle_data,
                 df, entry_idx, exit_idx, entry_price, stop_loss)

        if evo_cfg.get("track_entry_quality"):
            _try("entry_quality", self._build_entry_quality, df, entry_idx, entry_price)

        if evo_cfg.get("track_volume_profile"):
            _try("volume_profile", self._build_volume_profile, df, entry_idx, exit_idx)

        if evo_cfg.get("track_market_context"):
            _try("market_context", self._build_market_context, None, entry_idx, exit_idx)

        if evo_cfg.get("track_indicator_snapshot"):
            _try("indicator_snapshot", self._build_indicator_snapshot, df, entry_idx, exit_idx)

        if evo_cfg.get("track_weakest_block"):
            _try("weakest_block", self._build_weakest_block, template, df, entry_idx)

        if evo_cfg.get("track_risk_reward"):
            _try("risk_reward", self._build_risk_reward,
                 entry_price, stop_loss, take_profit, exit_price, df, entry_idx, exit_idx)

        if evo_cfg.get("track_time_context"):
            _try("time_context", self._build_time_context, df, entry_idx, exit_idx)

        if evo_cfg.get("track_preceding_candles"):
            _try("preceding_candles", self._build_preceding_candles, df, entry_idx)

        if evo_cfg.get("track_key_levels"):
            _try("key_levels", self._build_key_levels, df, entry_idx, entry_price)

        if evo_cfg.get("track_concurrent_signals"):
            _try("concurrent_signals", self._build_concurrent_signals,
                 template.name, symbol, current_date_iso, None)

        self._record_attribution(template.name, symbol, attribution)

        kc = attribution.get("kill_candle") or {}
        tc = attribution.get("time_context") or {}
        vp = attribution.get("volume_profile") or {}
        mc = attribution.get("market_context") or {}
        wb = attribution.get("weakest_block") or {}
        logger.info(
            f"[ATTRIBUTION] {template.name}:{symbol} | "
            f"outcome={attribution['outcome']} | pnl={pnl:.2%} | "
            f"kill_type={kc.get('kill_type', 'N/A')} | "
            f"phase={kc.get('phase_at_death', 'N/A')} | "
            f"bars={tc.get('bars_in_trade', 'N/A')} | "
            f"vol_exit={vp.get('volume_at_exit', 'N/A')} | "
            f"spy_day={mc.get('spy_return_on_day', 'N/A')} | "
            f"weakest={wb.get('block_name', 'N/A')}"
        )

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

        # Save updated block_stats back to template JSON files
        try:
            self.tm.save_all()
            logger.info("Template block_stats saved to disk")
        except Exception as e:
            logger.warning(f"Failed to save template block_stats: {e}")

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

    # Block-level statistics summary
    print(f"\n{'-' * 55}")
    print(f" Top Blockers (blocks that most often kill signals):")
    print(f"{'-' * 55}")

    all_block_stats = {}
    try:
        from setup_templates import TemplateManager
        tm = TemplateManager()
        for template in tm.get_enabled():
            bstats = template.statistics.get("block_stats", {})
            for block_name, bs in bstats.items():
                if block_name not in all_block_stats:
                    all_block_stats[block_name] = {
                        "evaluated": 0, "passed": 0, "failed": 0,
                        "was_the_blocker": 0,
                        "when_passed_trades": 0, "when_passed_wins": 0,
                    }
                agg = all_block_stats[block_name]
                agg["evaluated"] += bs.get("evaluated", 0)
                agg["passed"] += bs.get("passed", 0)
                agg["failed"] += bs.get("failed", 0)
                agg["was_the_blocker"] += bs.get("was_the_blocker", 0)
                wp = bs.get("when_passed", {})
                agg["when_passed_trades"] += wp.get("total_trades", 0)
                agg["when_passed_wins"] += wp.get("wins", 0)
    except Exception:
        pass

    if all_block_stats:
        print(f" {'Block':<28} {'Eval':>6} {'Pass%':>6} {'Blkr':>5} {'WR%':>6}")
        print(f"{'-' * 55}")
        sorted_blocks = sorted(
            all_block_stats.items(),
            key=lambda x: x[1]["was_the_blocker"],
            reverse=True
        )
        for block_name, bs in sorted_blocks:
            ev = bs["evaluated"]
            pr = round(bs["passed"] / ev * 100, 1) if ev > 0 else 0
            blkr = bs["was_the_blocker"]
            wp_trades = bs["when_passed_trades"]
            wr = round(bs["when_passed_wins"] / wp_trades * 100, 1) if wp_trades > 0 else 0
            print(f" {block_name:<28} {ev:>6} {pr:>5.1f}% {blkr:>5} {wr:>5.1f}%")
    else:
        print(" (no block stats collected yet)")

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
