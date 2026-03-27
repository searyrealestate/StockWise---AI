"""
backtest_engine.py — StockWise Gen-13 Portfolio Backtest Engine
===============================================================
Chronological portfolio simulation using ALL system components:
  - FeatureEngine for indicator calculation
  - TemplateMatcher for signal detection (real templates, real conditions)
  - PortfolioRiskManager for risk gates (correlation, drawdown, exposure)
  - Kinetic Stop phases simulated bar-by-bar (PHASE_1 → PAUSE → PHASE_4)

Includes full survivability analysis:
  - Risk of Ruin (analytical + Monte Carlo)
  - Kelly Criterion optimal sizing
  - Max consecutive losses to survive
  - Capital floor events
  - Months to ruin

Usage (standalone):
    python backtest_engine.py --capital 100000 --symbols SPY NVDA AAPL
    python backtest_engine.py --symbols AAPL --no-risk-gates

Usage (from validation_runner as Phase 6):
    from backtest_engine import BacktestEngine
    engine = BacktestEngine(data_cache=cached_feature_frames)
    results = engine.run()

Output: data/backtest_results.json
"""

import argparse
import logging
import math
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import system_config as cfg
from safe_json_io import safe_json_write
from feature_engine import FeatureEngine
from setup_templates import TemplateManager
from template_matcher import TemplateMatcher
from portfolio_risk import PortfolioRiskManager

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BacktestEngine")

BACKTEST_RESULTS_PATH = "data/backtest_results.json"

# ── Config (all values sourced from system_config where available) ────────────
_ks = getattr(cfg, "KINETIC_STOP_CONFIG", {})
_pm = getattr(cfg, "POSITION_MANAGEMENT_CONFIG", {})
_pr = getattr(cfg, "PORTFOLIO_RISK_CONFIG", {})
_sl = getattr(cfg, "SHADOW_LEDGER_CONFIG", {})

BACKTEST_CONFIG = {
    "initial_capital":        100_000,
    "position_size_pct":      0.05,     # 5% of capital per position
    "max_positions":          5,
    "commission_per_trade":   1.0,      # $ per trade leg
    "slippage_pct":           0.05,     # 0.05% slippage on entry/exit
    "min_candles_warmup":     _sl.get("min_candles_for_eval", 200),
    "ruin_threshold_pct":     50,       # 50% capital loss = ruin
    "min_trade_capital":      500,      # Cannot trade below this
    "days_back":              _sl.get("eval_days_back", 1095),
    # Kinetic stop phases — from KINETIC_STOP_CONFIG
    "phase1_atr_mult":                _ks.get("phase1_atr_mult", 2.0),
    "phase2_breakeven_trigger_pct":   _ks.get("phase2_breakeven_trigger_pct", 0.015),
    "phase3_parabolic_trigger_pct":   _ks.get("phase3_parabolic_trigger_pct", 0.03),
    "phase3_atr_mult":                _ks.get("phase3_atr_mult", 1.0),
    "runner_atr_mult":                _ks.get("runner_atr_mult", 0.5),
    "runner_min_distance_pct":        _ks.get("runner_min_distance_pct", 0.008),
    # Phase PAUSE thresholds — from POSITION_MANAGEMENT_CONFIG
    "max_healthy_pullback_pct":       _pm.get("max_healthy_pullback_pct", 0.03),
    "min_er_for_pause":               _pm.get("min_er_for_pause", 0.45),
    "min_rsi_for_pause":              _pm.get("min_rsi_for_pause", 40),
    # Phase 4 runner trigger — 2.5x phase3 parabolic trigger
    "phase4_runner_trigger_pct": _ks.get("phase3_parabolic_trigger_pct", 0.03) * 2.5,
    # Monte Carlo
    "monte_carlo_sims":       1000,
}


# ═══════════════════════════════════════════════════════════════════════════
# Position
# ═══════════════════════════════════════════════════════════════════════════

class Position:
    """An open position in the backtest portfolio."""

    __slots__ = [
        "symbol", "template_id", "template_name",
        "entry_price", "entry_date", "shares",
        "stop_loss", "take_profit",
        "phase", "highest_high", "current_stop",
        "exit_price", "exit_date", "exit_reason",
        "pnl", "pnl_pct", "bars_held",
    ]

    def __init__(self, symbol, template_id, template_name,
                 entry_price, entry_date, shares,
                 stop_loss, take_profit, initial_stop):
        self.symbol        = symbol
        self.template_id   = template_id
        self.template_name = template_name
        self.entry_price   = entry_price
        self.entry_date    = entry_date
        self.shares        = shares
        self.stop_loss     = stop_loss     # original signal stop
        self.take_profit   = take_profit   # reference (kinetic stop drives exits)
        # Kinetic stop state
        self.phase         = "PHASE_1_BREATHING"
        self.highest_high  = entry_price
        self.current_stop  = initial_stop
        # Filled on close
        self.exit_price    = None
        self.exit_date     = None
        self.exit_reason   = None
        self.pnl           = 0.0
        self.pnl_pct       = 0.0
        self.bars_held     = 0


# ═══════════════════════════════════════════════════════════════════════════
# BacktestEngine
# ═══════════════════════════════════════════════════════════════════════════

class BacktestEngine:
    """
    Chronological portfolio-level backtest.

    Day-by-day loop:
      1. Update open positions → stop-loss check + kinetic phase advance
      2. Scan for new signals → risk gates → open positions
      3. Record equity, drawdown, cash
    """

    def __init__(self, data_cache=None, symbols=None,
                 initial_capital=None, config_overrides=None,
                 use_risk_gates=True):
        self.symbols = symbols or list(getattr(cfg, "DEFAULT_TRAINING_SYMBOLS", []))
        self.config = dict(BACKTEST_CONFIG)
        if config_overrides:
            self.config.update(config_overrides)
        if initial_capital is not None:
            self.config["initial_capital"] = initial_capital

        self.use_risk_gates = use_risk_gates
        self.data_cache = data_cache or {}   # symbol → DataFrame (with features)

        # Components
        self.fe      = FeatureEngine()
        self.matcher = TemplateMatcher()
        try:
            from stock_hunter import StockHunter

            class _MockDM:  # minimal stub — classify_stock_state doesn't use data_manager
                stock_client = None
            self.hunter = StockHunter(_MockDM())
        except Exception as exc:
            logger.warning(f"StockHunter unavailable: {exc}")
            self.hunter = None
        try:
            self.risk_mgr = PortfolioRiskManager() if use_risk_gates else None
        except Exception as exc:
            logger.warning(f"PortfolioRiskManager unavailable: {exc}")
            self.risk_mgr = None

        # Runtime state
        self.capital        = self.config["initial_capital"]
        self.open_positions = []   # list[Position]
        self.closed_trades  = []   # list[dict]
        self.equity_curve   = []   # list[{date, equity, cash, open_positions}]

    # ───────────────────────────────────────────────────────────────────────
    # Public entry point
    # ───────────────────────────────────────────────────────────────────────

    def run(self) -> dict:
        t0 = time.perf_counter()
        logger.info("=" * 60)
        logger.info(f"Backtest start | capital=${self.config['initial_capital']:,.0f} "
                    f"| {len(self.symbols)} symbols | risk_gates={self.use_risk_gates}")
        logger.info("=" * 60)

        self._ensure_data()
        if not self.data_cache:
            return {"error": "No data available", "trades": [], "equity_curve": []}

        timeline = self._build_timeline()
        if not timeline:
            return {"error": "Empty timeline after warmup exclusion",
                    "trades": [], "equity_curve": []}

        logger.info(f"Timeline: {timeline[0].date()} to {timeline[-1].date()} "
                    f"({len(timeline)} trading days)")

        for trading_day in timeline:
            self._process_day(trading_day)

        self._close_remaining(timeline[-1])

        elapsed = round(time.perf_counter() - t0, 1)
        results = self._compute_results()
        results["metadata"] = {
            "run_timestamp":    datetime.now(timezone.utc).isoformat(),
            "elapsed_s":        elapsed,
            "symbols":          self.symbols,
            "config":           {k: v for k, v in self.config.items()
                                 if not k.startswith("_")},
            "timeline_days":    len(timeline),
            "date_range":       f"{timeline[0].date()} to {timeline[-1].date()}",
            "risk_gates_enabled": self.use_risk_gates,
        }

        try:
            safe_json_write(BACKTEST_RESULTS_PATH, results)
            logger.info(f"Results saved to {BACKTEST_RESULTS_PATH}")
        except Exception as exc:
            logger.warning(f"Could not save results: {exc}")

        return results

    # ───────────────────────────────────────────────────────────────────────
    # Data preparation
    # ───────────────────────────────────────────────────────────────────────

    def _ensure_data(self):
        if self.data_cache:
            logger.info(f"Using {len(self.data_cache)} pre-cached DataFrames")
            return
        logger.info("Fetching + computing features (no cache provided)…")
        try:
            from data_source_manager import DataSourceManager
            dsm = DataSourceManager(use_ibkr=False, allow_fallback=True)
        except Exception as exc:
            logger.error(f"Cannot init DataSourceManager: {exc}")
            return
        warmup = self.config["min_candles_warmup"]
        for sym in self.symbols:
            try:
                df = dsm.get_stock_data(sym, days_back=self.config["days_back"],
                                        interval="1d")
                if df is None or len(df) < warmup:
                    logger.warning(f"  {sym}: insufficient data")
                    continue
                df_feat = self.fe.calculate_features(df.copy())
                if df_feat is not None and len(df_feat) >= warmup:
                    self.data_cache[sym] = df_feat
                    logger.info(f"  {sym}: {len(df_feat)} rows x {len(df_feat.columns)} features")
            except Exception as exc:
                logger.warning(f"  {sym}: {exc}")

    def _build_timeline(self) -> list:
        warmup = self.config["min_candles_warmup"]
        all_dates: set = set()
        for df in self.data_cache.values():
            if len(df) > warmup:
                all_dates.update(df.index[warmup:])
        return sorted(all_dates)

    # ───────────────────────────────────────────────────────────────────────
    # Day-by-day loop
    # ───────────────────────────────────────────────────────────────────────

    def _process_day(self, trading_day):
        self._update_positions(trading_day)
        if len(self.open_positions) < self.config["max_positions"]:
            self._scan_for_signals(trading_day)
        equity = self._calc_equity(trading_day)
        self.equity_curve.append({
            "date":            str(trading_day.date()),
            "equity":          round(equity, 2),
            "cash":            round(self.capital, 2),
            "open_positions":  len(self.open_positions),
        })

    def _update_positions(self, trading_day):
        to_close = []
        for pos in self.open_positions:
            df = self.data_cache.get(pos.symbol)
            if df is None or trading_day not in df.index:
                pos.bars_held += 1
                continue

            row   = df.loc[trading_day]
            low   = float(row.get("low",   pos.entry_price))
            high  = float(row.get("high",  pos.entry_price))
            close = float(row.get("close", pos.entry_price))

            pos.bars_held += 1
            if high > pos.highest_high:
                pos.highest_high = high

            # Advance kinetic stop (stop can only increase)
            self._advance_kinetic_phase(pos, row, close)

            # Check stop hit (low touches or crosses stop)
            if low <= pos.current_stop:
                self._close_position(pos, pos.current_stop, trading_day,
                                     f"STOP_HIT({pos.phase})")
                to_close.append(pos)

        for pos in to_close:
            self.open_positions.remove(pos)

    def _advance_kinetic_phase(self, pos, row, price):
        """Simulate kinetic stop phase transitions using KINETIC_STOP_CONFIG keys."""
        c = self.config
        atr = float(row.get("atr", 0))
        if atr <= 0:
            atr = abs(price * 0.02)   # 2% ATR fallback

        gain_pct = (price - pos.entry_price) / pos.entry_price

        if pos.phase == "PHASE_1_BREATHING":
            p1_stop = pos.entry_price - atr * c["phase1_atr_mult"]
            pos.current_stop = max(pos.current_stop, p1_stop)
            if gain_pct >= c["phase2_breakeven_trigger_pct"]:
                pos.phase = "PHASE_2_BREAKEVEN"

        elif pos.phase == "PHASE_2_BREAKEVEN":
            # Snap to near breakeven
            p2_stop = pos.entry_price - atr * 0.2
            pos.current_stop = max(pos.current_stop, p2_stop)
            if gain_pct >= c["phase3_parabolic_trigger_pct"]:
                pos.phase = "PHASE_3_PARABOLIC"

        elif pos.phase == "PHASE_3_PARABOLIC":
            p3_stop = pos.highest_high - atr * c["phase3_atr_mult"]
            pos.current_stop = max(pos.current_stop, p3_stop)

            # Check PHASE_PAUSE: healthy pullback with trend intact
            rsi = float(row.get("rsi", 100))
            er  = float(row.get("er_slow", row.get("er_fast", 1.0)))
            pullback = (pos.highest_high - price) / pos.highest_high \
                if pos.highest_high > 0 else 0.0
            if (c["phase2_breakeven_trigger_pct"] <= pullback <= c["max_healthy_pullback_pct"]
                    and rsi >= c["min_rsi_for_pause"]
                    and er  >= c["min_er_for_pause"]):
                pos.phase = "PHASE_PAUSE"
                return   # stop frozen during pause

            if gain_pct >= c["phase4_runner_trigger_pct"]:
                pos.phase = "PHASE_4_RUNNER"

        elif pos.phase == "PHASE_PAUSE":
            # Stop FROZEN — resume when price recovers to ~99% of highest high
            if price >= pos.highest_high * 0.99:
                pos.phase = "PHASE_3_PARABOLIC"
            return   # no stop movement during pause

        elif pos.phase == "PHASE_4_RUNNER":
            runner_floor = pos.highest_high * (1.0 - c["runner_min_distance_pct"])
            runner_atr   = pos.highest_high - atr * c["runner_atr_mult"]
            p4_stop      = max(runner_floor, runner_atr)
            pos.current_stop = max(pos.current_stop, p4_stop)

    def _scan_for_signals(self, trading_day):
        open_syms = {p.symbol for p in self.open_positions}
        for sym in self.symbols:
            if len(self.open_positions) >= self.config["max_positions"]:
                break
            if sym in open_syms:
                continue
            df = self.data_cache.get(sym)
            if df is None or trading_day not in df.index:
                continue

            # Slice up to today (no lookahead bias)
            try:
                loc = df.index.get_loc(trading_day)
                if isinstance(loc, slice):
                    loc = loc.stop - 1
                elif isinstance(loc, np.ndarray):
                    loc = int(np.flatnonzero(loc)[0])
            except Exception:
                continue

            df_slice = df.iloc[:loc + 1]
            if len(df_slice) < self.config["min_candles_warmup"]:
                continue

            # Classify state so templates can match by regime
            stock_state = {}
            if self.hunter:
                try:
                    stock_state = self.hunter.classify_stock_state(df_slice)
                except Exception:
                    pass

            # Template matching — returns list of signal dicts
            try:
                signals = self.matcher.scan_ticker(sym, df_slice, stock_state)
            except Exception:
                continue
            if not signals:
                continue

            # Best signal (list sorted by template win_rate descending)
            sig = signals[0]
            entry_price   = float(sig["entry_price"])
            stop_loss     = float(sig["stop_loss"])
            take_profit   = float(sig["take_profit"])
            template_id   = sig.get("template_id",   "UNKNOWN")
            template_name = sig.get("template_name", template_id)

            if entry_price <= 0 or stop_loss >= entry_price:
                continue

            # Portfolio risk gates
            if self.risk_mgr and self.use_risk_gates:
                if not self._check_risk_gates(sym, df_slice):
                    continue

            # Position sizing
            cap_to_risk = self.capital * self.config["position_size_pct"]
            if cap_to_risk < self.config["min_trade_capital"] or cap_to_risk < entry_price:
                continue

            slippage   = entry_price * self.config["slippage_pct"] / 100
            actual_px  = entry_price + slippage
            commission = self.config["commission_per_trade"]

            shares = int(cap_to_risk / actual_px)
            if shares <= 0:
                continue
            cost = actual_px * shares + commission
            if cost > self.capital:
                shares = int((self.capital - commission) / actual_px)
                if shares <= 0:
                    continue
                cost = actual_px * shares + commission

            self.capital -= cost
            pos = Position(
                symbol=sym, template_id=template_id, template_name=template_name,
                entry_price=actual_px, entry_date=str(trading_day.date()),
                shares=shares, stop_loss=stop_loss, take_profit=take_profit,
                initial_stop=stop_loss,
            )
            self.open_positions.append(pos)
            logger.debug(f"  OPEN {sym} @{actual_px:.2f} x{shares} stop={stop_loss:.2f} "
                         f"tmpl={template_id}")

    def _check_risk_gates(self, symbol, df) -> bool:
        """Run check_all_gates; return True if approved."""
        try:
            open_pos_dict = {
                p.symbol: {"entry_price": p.entry_price, "qty": p.shares}
                for p in self.open_positions
            }
            pv = self._calc_equity_fast()
            approved, _ = self.risk_mgr.check_all_gates(
                symbol, df, open_pos_dict,
                market_data=None, portfolio_value=pv,
            )
            return approved
        except Exception:
            return True   # fail-open in backtest context

    def _close_position(self, pos, exit_px, exit_day, reason):
        slippage  = exit_px * self.config["slippage_pct"] / 100
        net_exit  = (exit_px - slippage) * pos.shares - self.config["commission_per_trade"]
        self.capital += net_exit

        pos.exit_price  = round(exit_px, 2)
        pos.exit_date   = str(exit_day.date())
        pos.exit_reason = reason
        pos.pnl         = round((exit_px - pos.entry_price) * pos.shares, 2)
        pos.pnl_pct     = round((exit_px - pos.entry_price) / pos.entry_price * 100, 2)

        self.closed_trades.append({
            "symbol":        pos.symbol,
            "template_id":   pos.template_id,
            "template_name": pos.template_name,
            "entry_price":   round(pos.entry_price, 2),
            "exit_price":    pos.exit_price,
            "entry_date":    pos.entry_date,
            "exit_date":     pos.exit_date,
            "shares":        pos.shares,
            "pnl":           pos.pnl,
            "pnl_pct":       pos.pnl_pct,
            "exit_reason":   pos.exit_reason,
            "bars_held":     pos.bars_held,
            "final_phase":   pos.phase,
        })

    def _close_remaining(self, last_day):
        for pos in list(self.open_positions):
            df = self.data_cache.get(pos.symbol)
            last_close = float(df["close"].iloc[-1]) \
                if df is not None and not df.empty else pos.entry_price
            self._close_position(pos, last_close, last_day, "BACKTEST_END")
        self.open_positions.clear()

    # ───────────────────────────────────────────────────────────────────────
    # Equity helpers
    # ───────────────────────────────────────────────────────────────────────

    def _calc_equity(self, trading_day) -> float:
        equity = self.capital
        for pos in self.open_positions:
            df = self.data_cache.get(pos.symbol)
            if df is not None and trading_day in df.index:
                price = float(df.loc[trading_day, "close"])
            else:
                price = pos.entry_price
            equity += price * pos.shares
        return equity

    def _calc_equity_fast(self) -> float:
        return self.capital + sum(p.entry_price * p.shares for p in self.open_positions)

    # ───────────────────────────────────────────────────────────────────────
    # Results computation
    # ───────────────────────────────────────────────────────────────────────

    def _compute_results(self) -> dict:
        trades   = self.closed_trades
        initial  = self.config["initial_capital"]
        equities = [e["equity"] for e in self.equity_curve]

        if not trades:
            logger.warning("No trades executed — check template conditions + data quality")
            return {
                "summary": {
                    "initial_capital": initial,
                    "final_equity": round(self.capital, 2),
                    "total_return_pct": round((self.capital - initial) / initial * 100, 2),
                    "total_trades": 0,
                },
                "survivability": {"survival_verdict": "NO_TRADES"},
                "monthly_returns": [], "per_template": {}, "per_symbol": {},
                "phase_distribution": {}, "equity_curve": self.equity_curve, "trades": [],
            }

        wins   = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]

        total_pnl    = sum(t["pnl"] for t in trades)
        final_equity = initial + total_pnl
        total_ret    = round((final_equity - initial) / initial * 100, 2)
        win_rate     = round(len(wins) / len(trades) * 100, 1)
        avg_win      = round(float(np.mean([t["pnl_pct"] for t in wins])),   2) if wins   else 0.0
        avg_loss     = round(float(np.mean([t["pnl_pct"] for t in losses])), 2) if losses else 0.0
        gross_wins   = sum(t["pnl"] for t in wins)
        gross_loss   = abs(sum(t["pnl"] for t in losses))
        pf           = round(gross_wins / gross_loss, 2) if gross_loss > 0 else float("inf")

        # Drawdown
        if equities:
            eq_arr  = np.array(equities, dtype=float)
            peak    = np.maximum.accumulate(eq_arr)
            dd_arr  = (eq_arr - peak) / peak * 100
            max_dd  = round(float(abs(dd_arr.min())), 2)
        else:
            dd_arr = np.array([])
            max_dd = 0.0

        # Sharpe / Sortino / Calmar
        if len(equities) > 1:
            daily_ret = np.diff(equities) / np.array(equities[:-1]) * 100
            mu        = float(np.mean(daily_ret))
            sigma     = float(np.std(daily_ret, ddof=1)) or 1e-9
            sharpe    = round(mu / sigma * math.sqrt(252), 2)
            downside  = daily_ret[daily_ret < 0]
            ds_std    = float(np.std(downside, ddof=1)) if len(downside) > 1 else 1e-9
            sortino   = round(mu / ds_std * math.sqrt(252), 2)
            calmar    = round(total_ret / max_dd, 2) if max_dd > 0 else 0.0
        else:
            sharpe = sortino = calmar = 0.0

        max_consec = cur_consec = 0
        for t in trades:
            if t["pnl"] <= 0:
                cur_consec += 1
                max_consec = max(max_consec, cur_consec)
            else:
                cur_consec = 0

        avg_bars = round(float(np.mean([t["bars_held"] for t in trades])), 1)

        phase_dist = defaultdict(int)
        for t in trades:
            phase_dist[t["final_phase"]] += 1

        monthly      = self._compute_monthly()
        survivability = self._compute_survivability(
            trades, wins, losses, avg_win, avg_loss, monthly, dd_arr, equities
        )

        return {
            "summary": {
                "initial_capital":        initial,
                "final_equity":           round(final_equity, 2),
                "total_return_pct":       total_ret,
                "total_pnl":              round(total_pnl, 2),
                "total_trades":           len(trades),
                "wins":                   len(wins),
                "losses":                 len(losses),
                "win_rate":               win_rate,
                "avg_win_pct":            avg_win,
                "avg_loss_pct":           avg_loss,
                "profit_factor":          pf,
                "win_loss_ratio":  round(abs(avg_win / avg_loss), 2) if avg_loss != 0 else 0.0,
                "max_consecutive_losses": max_consec,
                "max_drawdown_pct":       max_dd,
                "sharpe_ratio":           sharpe,
                "sortino_ratio":          sortino,
                "calmar_ratio":           calmar,
                "avg_bars_held":          avg_bars,
            },
            "survivability":     survivability,
            "monthly_returns":   monthly,
            "per_template":      self._group_stats(trades, "template_id"),
            "per_symbol":        self._group_stats(trades, "symbol"),
            "phase_distribution": dict(phase_dist),
            "equity_curve":      self.equity_curve,
            "trades":            trades,
        }

    @staticmethod
    def _group_stats(trades, key):
        groups = {}
        for t in trades:
            k = t.get(key, "UNKNOWN")
            groups.setdefault(k, []).append(t)
        result = {}
        for k, ts in groups.items():
            w = [t for t in ts if t["pnl"] > 0]
            result[k] = {
                "trades":      len(ts),
                "wins":        len(w),
                "win_rate":    round(len(w) / len(ts) * 100, 1),
                "total_pnl":   round(sum(t["pnl"] for t in ts), 2),
                "avg_pnl_pct": round(float(np.mean([t["pnl_pct"] for t in ts])), 2),
            }
        return result

    def _compute_monthly(self) -> list:
        if not self.equity_curve:
            return []
        monthly = {}
        for entry in self.equity_curve:
            mk = entry["date"][:7]
            monthly[mk] = entry["equity"]
        result = []
        prev = self.config["initial_capital"]
        for mk in sorted(monthly):
            eq  = monthly[mk]
            ret = round((eq - prev) / prev * 100, 2) if prev > 0 else 0.0
            result.append({"month": mk, "equity": round(eq, 2), "return_pct": ret})
            prev = eq
        return result

    # ───────────────────────────────────────────────────────────────────────
    # Survivability Analysis
    # ───────────────────────────────────────────────────────────────────────

    def _compute_survivability(self, trades, wins, losses,
                               avg_win: float, avg_loss: float,
                               monthly: list, dd_arr, equities: list) -> dict:
        """
        How long can I stay in the game?

        1. Risk of Ruin (analytical — Ralph Vince formula)
        2. Risk of Ruin (Monte Carlo, 1000 shuffled sequences)
        3. Max consecutive losses before capital < min_trade_capital
        4. Kelly Criterion optimal position sizing
        5. Capital floor events (equity below 90/80/70/50% of initial)
        6. Recovery days from max drawdown
        7. Months to ruin (if negative expectancy)
        8. Worst-case scenarios (5/10/15/20 consecutive max losses)
        9. Survival verdict: SAFE / WARNING / DANGER / CRITICAL
        """
        c        = self.config
        initial  = c["initial_capital"]
        pos_size = c["position_size_pct"]
        min_cap  = c["min_trade_capital"]

        if not trades:
            return {"survival_verdict": "NO_TRADES"}

        win_rate  = len(wins) / len(trades)
        loss_rate = 1.0 - win_rate
        avg_win_d  = abs(avg_win)  / 100
        avg_loss_d = abs(avg_loss) / 100

        # ── 1. Risk of Ruin (analytical) ─────────────────────────────────────
        ror_analytical = 1.0
        edge = 0.0
        if avg_loss_d > 0 and len(trades) >= 5:
            edge = (win_rate * avg_win_d - loss_rate * avg_loss_d) / avg_loss_d
            capital_units = 1.0 / max(pos_size * avg_loss_d, 1e-9)
            if edge > 0:
                base = (1.0 - edge) / (1.0 + edge)
                ror_analytical = round(min(base ** capital_units, 1.0), 6)
            # else: negative edge → certain ruin = 1.0

        # ── 2. Monte Carlo Risk of Ruin ───────────────────────────────────────
        pnl_list   = [t["pnl"] for t in trades]
        ruin_level = initial * (1.0 - c["ruin_threshold_pct"] / 100)
        ruin_count = 0
        sims       = c["monte_carlo_sims"]
        random.seed(42)
        for _ in range(sims):
            shuffled = random.sample(pnl_list, len(pnl_list))
            cap_sim  = initial
            for pnl in shuffled:
                cap_sim += pnl
                if cap_sim <= ruin_level or cap_sim <= min_cap:
                    ruin_count += 1
                    break
        mc_ruin_pct = round(ruin_count / sims * 100, 1)

        # ── 3. Max consecutive losses to survive ─────────────────────────────
        cap_sim     = initial
        consec_surv = 0
        if avg_loss_d > 0:
            while cap_sim > min_cap and consec_surv < 100_000:
                cap_sim    -= cap_sim * pos_size * avg_loss_d   # compounding
                consec_surv += 1
        else:
            consec_surv = 99999

        # ── 4. Kelly Criterion ────────────────────────────────────────────────
        b         = avg_win_d / avg_loss_d if avg_loss_d > 0 else 1.0
        kelly     = max(0.0, (b * win_rate - loss_rate) / b) if b > 0 else 0.0
        kelly_h   = kelly / 2.0
        kelly_pct = round(kelly * 100, 1)
        kelly_h_pct = round(kelly_h * 100, 1)
        kelly_rec = "OK" if pos_size <= kelly_h else \
                    ("REDUCE" if pos_size <= kelly else "CRITICAL_REDUCE")

        # ── 5. Capital floor events ───────────────────────────────────────────
        floors = {}
        for pct in [90, 80, 70, 50]:
            threshold = initial * pct / 100
            floors[f"below_{pct}pct"] = int(sum(1 for e in equities if e < threshold))

        # ── 6. Recovery days from max drawdown ────────────────────────────────
        recovery_days = None
        if len(dd_arr) > 0 and len(equities) > 1:
            min_idx      = int(np.argmin(dd_arr))
            peak_before  = float(max(equities[:min_idx + 1])) if min_idx >= 0 else equities[0]
            for idx in range(min_idx + 1, len(equities)):
                if equities[idx] >= peak_before:
                    recovery_days = idx - min_idx
                    break

        # ── 7. Months to ruin ─────────────────────────────────────────────────
        monthly_rets = [m["return_pct"] for m in monthly] if monthly else []
        avg_monthly  = float(np.mean(monthly_rets)) if monthly_rets else 0.0
        if avg_monthly < 0 and abs(avg_monthly) > 0:
            months_to_ruin = round(
                (initial - min_cap) / abs(avg_monthly / 100 * initial), 1
            )
        else:
            months_to_ruin = None   # positive expectancy → no ruin

        # ── 8. Worst-case consecutive loss scenarios ──────────────────────────
        worst_case = {}
        for n in [5, 10, 15, 20]:
            cap_wc = float(initial)
            for _ in range(n):
                cap_wc -= cap_wc * pos_size * avg_loss_d
            worst_case[f"{n}_consec_losses_capital"] = round(cap_wc, 2)
            worst_case[f"{n}_consec_losses_pct_remaining"] = round(cap_wc / initial * 100, 1)

        # ── 9. Verdict ────────────────────────────────────────────────────────
        if ror_analytical < 0.01 and mc_ruin_pct < 5:
            verdict = "SAFE"
        elif ror_analytical < 0.05 and mc_ruin_pct < 15:
            verdict = "WARNING"
        elif ror_analytical < 0.20 and mc_ruin_pct < 30:
            verdict = "DANGER"
        else:
            verdict = "CRITICAL"

        return {
            "risk_of_ruin_analytical":           ror_analytical,
            "risk_of_ruin_monte_carlo_pct":      mc_ruin_pct,
            "edge":                              round(edge, 4),
            "max_consecutive_losses_to_survive": consec_surv,
            "kelly_optimal_pct":                 kelly_pct,
            "kelly_half_pct":                    kelly_h_pct,
            "current_position_size_pct":         round(pos_size * 100, 1),
            "kelly_recommendation":              kelly_rec,
            "capital_floor_events":              floors,
            "recovery_days_from_max_dd":         recovery_days,
            "months_to_ruin":                    months_to_ruin,
            "avg_monthly_return_pct":            round(avg_monthly, 2),
            "ruin_threshold_pct":                c["ruin_threshold_pct"],
            "worst_case_scenarios":              worst_case,
            "survival_verdict":                  verdict,
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="StockWise Gen-13 Portfolio Backtest")
    parser.add_argument("--capital",       type=float, default=100_000)
    parser.add_argument("--symbols",       nargs="+")
    parser.add_argument("--position-size", type=float, default=0.05,
                        help="Position size as fraction of capital (0.05 = 5%%)")
    parser.add_argument("--max-positions", type=int,   default=5)
    parser.add_argument("--no-risk-gates", action="store_true",
                        help="Disable portfolio risk gates (more trades, faster)")
    parser.add_argument("--output",        default=BACKTEST_RESULTS_PATH)
    args = parser.parse_args()

    engine = BacktestEngine(
        symbols=args.symbols,
        initial_capital=args.capital,
        use_risk_gates=not args.no_risk_gates,
        config_overrides={
            "position_size_pct": args.position_size,
            "max_positions":     args.max_positions,
        },
    )
    results = engine.run()
    s  = results.get("summary", {})
    sv = results.get("survivability", {})

    print("\n" + "=" * 55)
    print("BACKTEST SUMMARY")
    print("=" * 55)
    meta = results.get("metadata", {})
    print(f"Period        : {meta.get('date_range', 'N/A')}")
    print(f"Initial Cap   : ${s.get('initial_capital', 0):>12,.2f}")
    print(f"Final Equity  : ${s.get('final_equity',   0):>12,.2f}")
    print(f"Total Return  : {s.get('total_return_pct', 0):>+.2f}%")
    print(f"Total Trades  : {s.get('total_trades', 0)}")
    print(f"Win Rate      : {s.get('win_rate', 0):.1f}%")
    print(f"Avg Win       : {s.get('avg_win_pct', 0):>+.2f}%")
    print(f"Avg Loss      : {s.get('avg_loss_pct', 0):>+.2f}%")
    print(f"Profit Factor : {s.get('profit_factor', 0):.2f}")
    print(f"Sharpe Ratio  : {s.get('sharpe_ratio', 0):.2f}")
    print(f"Sortino Ratio : {s.get('sortino_ratio', 0):.2f}")
    print(f"Calmar Ratio  : {s.get('calmar_ratio', 0):.2f}")
    print(f"Max Drawdown  : {s.get('max_drawdown_pct', 0):.2f}%")
    print(f"Avg Bars Held : {s.get('avg_bars_held', 0):.1f}")

    print("\n" + "=" * 55)
    print("SURVIVABILITY ANALYSIS")
    print("=" * 55)
    if sv.get("survival_verdict") == "NO_TRADES":
        print("  (no trades executed — survivability unavailable)")
    else:
        ror_a = sv.get('risk_of_ruin_analytical', 0)
        ror_m = sv.get('risk_of_ruin_monte_carlo_pct', 0)
        print(f"Risk of Ruin (analytical)  : {ror_a if isinstance(ror_a, str) else f'{ror_a:.6f}'}")
        print(f"Risk of Ruin (Monte Carlo) : {ror_m if isinstance(ror_m, str) else f'{ror_m:.1f}%'}")
        print(f"Edge                       : {sv.get('edge', 0):.4f}")
        print(f"Max Consec Losses (survive): {sv.get('max_consecutive_losses_to_survive', 'N/A')}")
        print(f"Kelly Optimal Size         : {sv.get('kelly_optimal_pct', 0):.1f}%")
        print(f"Kelly Half-Size            : {sv.get('kelly_half_pct', 0):.1f}%")
        print(f"Current Position Size      : {sv.get('current_position_size_pct', 0):.1f}%")
        print(f"Kelly Recommendation       : {sv.get('kelly_recommendation', 'N/A')}")
        print(f"Recovery Days (max DD)     : {sv.get('recovery_days_from_max_dd', 'N/A')}")
        mtr = sv.get('months_to_ruin')
        print(f"Months to Ruin             : {mtr if mtr is not None else 'NONE (positive expect.)'}")
        print(f"Avg Monthly Return         : {sv.get('avg_monthly_return_pct', 0):+.2f}%")

    wc = sv.get("worst_case_scenarios", {})
    if wc:
        print("\nWorst-Case Consecutive Loss Scenarios:")
        for n in [5, 10, 15, 20]:
            cap_k = f"{n}_consec_losses_capital"
            pct_k = f"{n}_consec_losses_pct_remaining"
            if cap_k in wc:
                print(f"  {n:>2} losses -> ${wc[cap_k]:>12,.2f}  "
                      f"({wc[pct_k]:.1f}% remaining)")

    print(f"\nSURVIVAL VERDICT: {sv.get('survival_verdict', 'N/A')}")
    print(f"\nResults -> {args.output}")
    print("=" * 55)

    sys.exit(0 if sv.get("survival_verdict") not in ("DANGER", "CRITICAL") else 1)


if __name__ == "__main__":
    main()
