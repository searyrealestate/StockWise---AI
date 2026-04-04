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
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import system_config as cfg
from safe_json_io import safe_json_read, safe_json_write
try:
    from versioned_save import save_versioned_copy
except ImportError:
    save_versioned_copy = None
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
        "indicator_snapshot",
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
        self.indicator_snapshot = {}


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
        self.feed_shadow_ledger = True
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
        self.block_eval_stats = {}   # Populated by _collect_block_evaluations()

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

        # Collect block-level evaluation stats (second pass, read-only)
        try:
            self._collect_block_evaluations()
        except Exception as exc:
            logger.warning(f"Block evaluation collection failed (non-fatal): {exc}")
            self.block_eval_stats = {}

        results["analytics"] = self._compute_analytics(
            self.closed_trades, results.get("summary", {})
        )

        # Save analytics report to reports_dir
        try:
            reports_dir = getattr(cfg, "REPORTS_DIR",
                                  os.path.join(os.path.dirname(BACKTEST_RESULTS_PATH), "reports"))
            os.makedirs(reports_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(reports_dir, f"analytics_{ts}.json")
            safe_json_write(report_path, results["analytics"])
            logger.info(f"Analytics report saved to {report_path}")
        except Exception as exc:
            logger.warning(f"Could not save analytics report: {exc}")

        try:
            safe_json_write(BACKTEST_RESULTS_PATH, results)
            logger.info(f"Results saved to {BACKTEST_RESULTS_PATH}")
            # Save versioned copy for comparison across runs
            if save_versioned_copy:
                save_versioned_copy(BACKTEST_RESULTS_PATH, "backtest_history")
        except Exception as exc:
            logger.warning(f"Could not save results: {exc}")

        # Feed results into Shadow Ledger for DDR #1 Asset-Specific
        if self.feed_shadow_ledger:
            backtest_trades = []
            for trade in self.closed_trades:
                pnl = trade.get("pnl_pct", 0.0)
                backtest_trades.append({
                    "symbol":      trade.get("symbol", ""),
                    "template_id": trade.get("template_id", ""),
                    "pnl_pct":     pnl,
                    "won":         pnl > 0,
                })
            self._feed_shadow_ledger(backtest_trades)

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
            # Capture indicator snapshot at entry for profiler analysis
            try:
                last_row = df_slice.iloc[-1]
                snapshot = {}
                for col in df_slice.columns:
                    val = last_row.get(col, None)
                    if val is None:
                        continue
                    if isinstance(val, (bool, np.bool_)):
                        snapshot[col] = bool(val)
                    elif isinstance(val, (int, float, np.integer, np.floating)):
                        if not (isinstance(val, float) and math.isnan(val)):
                            snapshot[col] = round(float(val), 6)
                pos.indicator_snapshot = snapshot
            except Exception:
                pos.indicator_snapshot = {}

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
            "bars_held":          pos.bars_held,
            "final_phase":        pos.phase,
            "indicators_at_entry": getattr(pos, "indicator_snapshot", {}),
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

    def _feed_shadow_ledger(self, trades):
        """
        Feed backtest results into shadow_ledger.json for DDR #1 Asset-Specific.

        Performs ADDITIVE merge: signal_count, wins, losses, total_pnl_pct
        are accumulated (not overwritten). Derived fields (win_rate, avg_pnl_pct)
        are recalculated after merge.

        Args:
            trades: list of trade dicts from backtest, each containing:
                    symbol, template_id, pnl_pct, and won (bool)
        """
        if not trades:
            logger.info("Shadow ledger feed: no trades to feed")
            return

        # Resolve shadow ledger path from config
        sl_config = getattr(cfg, 'SHADOW_LEDGER_CONFIG', {})
        ledger_path = sl_config.get('ledger_path', 'data/shadow_ledger.json')

        # Load existing ledger (or create empty)
        ledger = safe_json_read(ledger_path, default={
            "metadata": {"last_run": None, "version": "13.4"},
            "template_stats": {},
        })

        if "template_stats" not in ledger:
            ledger["template_stats"] = {}

        stats = ledger["template_stats"]

        # Build per-symbol per-template deltas from backtest trades
        deltas = {}  # {symbol: {template_id: {signal_count, wins, losses, total_pnl_pct}}}
        for trade in trades:
            sym = trade.get("symbol", "")
            tid = trade.get("template_id", "")
            pnl = trade.get("pnl_pct", 0.0)
            won = trade.get("won", False)

            if not sym or not tid:
                continue

            if sym not in deltas:
                deltas[sym] = {}
            if tid not in deltas[sym]:
                deltas[sym][tid] = {
                    "signal_count": 0, "wins": 0, "losses": 0,
                    "total_pnl_pct": 0.0,
                }

            deltas[sym][tid]["signal_count"] += 1
            if won:
                deltas[sym][tid]["wins"] += 1
            else:
                deltas[sym][tid]["losses"] += 1
            deltas[sym][tid]["total_pnl_pct"] += pnl

        # Additive merge into existing ledger
        symbols_fed = 0
        total_signals_fed = 0

        for sym, sym_deltas in deltas.items():
            if sym not in stats:
                stats[sym] = {}
            symbols_fed += 1

            for tid, delta in sym_deltas.items():
                if tid not in stats[sym]:
                    stats[sym][tid] = {
                        "signal_count": 0, "wins": 0, "losses": 0,
                        "total_pnl_pct": 0.0, "win_rate": 0.0, "avg_pnl_pct": 0.0,
                    }

                existing = stats[sym][tid]
                existing["signal_count"] += delta["signal_count"]
                existing["wins"]         += delta["wins"]
                existing["losses"]       += delta["losses"]
                existing["total_pnl_pct"] += delta["total_pnl_pct"]

                # Recalculate derived fields
                sc = existing["signal_count"]
                if sc > 0:
                    existing["win_rate"]    = round(existing["wins"] / sc * 100, 1)
                    existing["avg_pnl_pct"] = round(existing["total_pnl_pct"] / sc, 2)
                else:
                    existing["win_rate"]    = 0.0
                    existing["avg_pnl_pct"] = 0.0

                total_signals_fed += delta["signal_count"]

                logger.debug(
                    f"[{sym}] {tid}: +{delta['signal_count']} signals, "
                    f"+{delta['wins']} wins -> "
                    f"total {existing['signal_count']}, "
                    f"WR={existing['win_rate']:.1f}%"
                )

        # Update metadata
        ledger["metadata"]["last_run"] = datetime.now(timezone.utc).isoformat()
        ledger["metadata"]["last_feed_source"] = "backtest"

        # Save
        safe_json_write(ledger_path, ledger)

        logger.info(
            f"Shadow ledger fed: {symbols_fed} symbols, "
            f"{total_signals_fed} signals merged into {ledger_path}"
        )

    # ───────────────────────────────────────────────────────────────────────
    # Analytics (SPEC v13.4 §5)
    # ───────────────────────────────────────────────────────────────────────

    def _compute_analytics(self, trades: list, results_summary: dict) -> dict:
        """
        7-section analytics over closed trades.
        Returns a dict added to results["analytics"].
        """
        analytics: dict = {}

        if not trades:
            return analytics

        # ── Section 1: template_anatomy ──────────────────────────────────────
        template_anatomy: dict = {}
        templates_dir = getattr(cfg, "TEMPLATES_DIR", "data/templates")
        try:
            if os.path.isdir(templates_dir):
                for fname in os.listdir(templates_dir):
                    if not fname.endswith(".json"):
                        continue
                    fpath = os.path.join(templates_dir, fname)
                    try:
                        tdata = safe_json_read(fpath, default={})
                        tid = tdata.get("id", fname.replace(".json", ""))
                        conditions = tdata.get("conditions", [])
                        template_anatomy[tid] = {
                            "condition_count": len(conditions),
                            "block_names": [c.get("block", "?") for c in conditions],
                            "version":     tdata.get("version", 1),
                            "source":      tdata.get("source", ""),
                        }
                    except Exception:
                        pass
        except Exception:
            pass
        analytics["template_anatomy"] = template_anatomy

        # ── Section 2: trade_breakdown ────────────────────────────────────────
        per_tmpl: dict = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl_pct_sum": 0.0,
                                               "pnl_sum": 0.0, "bars_sum": 0.0})
        for t in trades:
            tid   = t.get("template_id", "UNKNOWN")
            won   = t.get("pnl", 0) > 0
            entry = per_tmpl[tid]
            entry["trades"]      += 1
            entry["wins"]        += int(won)
            entry["pnl_pct_sum"] += t.get("pnl_pct", 0.0)
            entry["pnl_sum"]     += t.get("pnl", 0.0)
            entry["bars_sum"]    += t.get("bars_held", 0)

        trade_breakdown: dict = {}
        for tid, d in per_tmpl.items():
            n = d["trades"]
            wr = round(d["wins"] / n * 100, 1) if n else 0.0
            trade_breakdown[tid] = {
                "total_trades":   n,
                "wins":           d["wins"],
                "losses":         n - d["wins"],
                "win_rate":       wr,
                "avg_pnl_pct":    round(d["pnl_pct_sum"] / n, 2) if n else 0.0,
                "total_pnl":      round(d["pnl_sum"], 2),
                "avg_bars_held":  round(d["bars_sum"] / n, 1) if n else 0.0,
            }
        analytics["trade_breakdown"] = trade_breakdown

        # ── Section 3: temporal ───────────────────────────────────────────────
        if getattr(cfg.ANALYTICS_CONFIG, "__class__", None) or isinstance(
                getattr(cfg, "ANALYTICS_CONFIG", None), dict):
            include_temporal = cfg.ANALYTICS_CONFIG.get("include_temporal", True)
        else:
            include_temporal = True

        temporal: dict = {"by_year": {}, "by_quarter": {}, "by_month": {}}
        if include_temporal:
            for t in trades:
                entry_date = t.get("entry_date", "")[:10]
                if not entry_date:
                    continue
                try:
                    yr  = entry_date[:4]
                    mo  = entry_date[:7]
                    q   = f"{yr}Q{(int(entry_date[5:7]) - 1) // 3 + 1}"
                    won = t.get("pnl", 0) > 0
                    pnl = t.get("pnl_pct", 0.0)
                    for bucket, key in [(temporal["by_year"], yr),
                                        (temporal["by_quarter"], q),
                                        (temporal["by_month"], mo)]:
                        if key not in bucket:
                            bucket[key] = {"trades": 0, "wins": 0, "pnl_pct_sum": 0.0}
                        bucket[key]["trades"]      += 1
                        bucket[key]["wins"]        += int(won)
                        bucket[key]["pnl_pct_sum"] += pnl
                except (ValueError, IndexError):
                    pass

            for bucket in temporal.values():
                for key, d in bucket.items():
                    n = d["trades"]
                    d["win_rate"]    = round(d["wins"] / n * 100, 1) if n else 0.0
                    d["avg_pnl_pct"] = round(d["pnl_pct_sum"] / n, 2) if n else 0.0
                    del d["pnl_pct_sum"]

        analytics["temporal"] = temporal

        # ── Section 4: phase_analysis ─────────────────────────────────────────
        phase_analysis: dict = {}
        for t in trades:
            phase = t.get("final_phase", "UNKNOWN")
            if phase not in phase_analysis:
                phase_analysis[phase] = {
                    "trades": 0, "wins": 0, "pnl_pct_sum": 0.0,
                    "template_counts": Counter(),
                }
            entry = phase_analysis[phase]
            entry["trades"]      += 1
            entry["wins"]        += int(t.get("pnl", 0) > 0)
            entry["pnl_pct_sum"] += t.get("pnl_pct", 0.0)
            tid = t.get("template_id", "UNKNOWN")
            entry["template_counts"][tid] += 1

        for ph, d in phase_analysis.items():
            n = d["trades"]
            d["win_rate"]    = round(d["wins"] / n * 100, 1) if n else 0.0
            d["avg_pnl_pct"] = round(d["pnl_pct_sum"] / n, 2) if n else 0.0
            d["template_counts"] = dict(d["template_counts"])
            del d["pnl_pct_sum"]
        analytics["phase_analysis"] = phase_analysis

        # ── Section 5: block_stats ────────────────────────────────────────────
        block_stats: dict = {}
        include_bs = True
        ac = getattr(cfg, "ANALYTICS_CONFIG", {})
        if isinstance(ac, dict):
            include_bs = ac.get("include_block_stats", True)

        if include_bs and os.path.isdir(templates_dir):
            for fname in os.listdir(templates_dir):
                if not fname.endswith(".json"):
                    continue
                fpath = os.path.join(templates_dir, fname)
                try:
                    tdata = safe_json_read(fpath, default={})
                    tid   = tdata.get("id", fname.replace(".json", ""))
                    stats = tdata.get("statistics", {}).get("block_stats", {})
                    if stats:
                        block_stats[tid] = stats
                except Exception:
                    pass
        analytics["block_stats"] = block_stats

        # ── Section 6: shadow_ledger_matrix ───────────────────────────────────
        shadow_matrix: dict = {}
        include_sm = True
        if isinstance(ac, dict):
            include_sm = ac.get("include_shadow_matrix", True)

        if include_sm:
            sl_config = getattr(cfg, "SHADOW_LEDGER_CONFIG", {})
            ledger_path = sl_config.get("ledger_path", "data/shadow_ledger.json")
            try:
                ledger = safe_json_read(ledger_path, default={})
                shadow_matrix = ledger.get("template_stats", {})
            except Exception:
                pass
        analytics["shadow_ledger_matrix"] = shadow_matrix

        # ── Section 7: winner_loser_profile ───────────────────────────────────
        wins_list   = [t for t in trades if t.get("pnl", 0) > 0]
        losses_list = [t for t in trades if t.get("pnl", 0) <= 0]

        bars_buckets = [2, 5, 10, 20]
        if isinstance(ac, dict) and "bars_buckets" in ac:
            bars_buckets = ac["bars_buckets"]

        def _bars_dist(tlist):
            dist = Counter()
            for t in tlist:
                bh = t.get("bars_held", 0)
                label = None
                prev = 0
                for b in bars_buckets:
                    if bh <= b:
                        label = f"{prev+1}-{b}"
                        break
                    prev = b
                if label is None:
                    label = f"{bars_buckets[-1]+1}+"
                dist[label] += 1
            return dict(dist)

        top_n = sorted(wins_list,   key=lambda t: t.get("pnl_pct", 0), reverse=True)[:5]
        bot_n = sorted(losses_list, key=lambda t: t.get("pnl_pct", 0))[:5]

        def _trade_summary(tlist):
            return [
                {
                    "symbol":      t.get("symbol", ""),
                    "template_id": t.get("template_id", ""),
                    "entry_date":  t.get("entry_date", ""),
                    "pnl_pct":     t.get("pnl_pct", 0.0),
                    "bars_held":   t.get("bars_held", 0),
                }
                for t in tlist
            ]

        analytics["winner_loser_profile"] = {
            "bars_distribution_wins":   _bars_dist(wins_list),
            "bars_distribution_losses": _bars_dist(losses_list),
            "top_5_trades":             _trade_summary(top_n),
            "worst_5_trades":           _trade_summary(bot_n),
        }

        # ── Section 8: block_evaluations (from second pass) ──────────────────
        _bes = getattr(self, "block_eval_stats", {})
        analytics["block_evaluations"] = _bes if _bes else {}

        # ── Per-Symbol Summary ────────────────────────────────────────────────
        per_symbol_summary: dict = {}
        sym_groups: dict = defaultdict(list)
        for t in trades:
            sym_groups[t["symbol"]].append(t)
        for sym, slist in sym_groups.items():
            sw = sum(1 for t in slist if t["pnl"] > 0)
            per_symbol_summary[sym] = {
                "trades":       len(slist),
                "wins":         sw,
                "wr":           round(sw / len(slist) * 100, 1) if slist else 0.0,
                "total_pnl":    round(sum(t["pnl"] for t in slist), 2),
                "avg_pnl_pct":  round(sum(t["pnl_pct"] for t in slist) / len(slist), 2)
                                if slist else 0.0,
            }
        analytics["per_symbol_summary"] = per_symbol_summary

        # ── Section 10: Indicator Profiler (WIN vs LOSS) ──────────────────────
        indicator_profiler: dict = {}
        win_trades  = [t for t in trades
                       if t.get("pnl", 0) > 0 and t.get("indicators_at_entry")]
        loss_trades = [t for t in trades
                       if t.get("pnl", 0) <= 0 and t.get("indicators_at_entry")]

        if win_trades and loss_trades:
            skip_cols = {"open", "high", "low", "close", "volume", "date", "Date",
                         "dynamic_stop_loss"}

            all_indicators: set = set()
            for t in win_trades + loss_trades:
                snap = t.get("indicators_at_entry", {})
                for k, v in snap.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        all_indicators.add(k)

            discriminators: list = []
            for ind in sorted(all_indicators - skip_cols):
                win_vals  = [t["indicators_at_entry"][ind] for t in win_trades
                             if ind in t["indicators_at_entry"]
                             and t["indicators_at_entry"][ind] is not None]
                loss_vals = [t["indicators_at_entry"][ind] for t in loss_trades
                             if ind in t["indicators_at_entry"]
                             and t["indicators_at_entry"][ind] is not None]

                if len(win_vals) < 3 or len(loss_vals) < 3:
                    continue

                win_avg  = sum(win_vals)  / len(win_vals)
                loss_avg = sum(loss_vals) / len(loss_vals)
                delta    = win_avg - loss_avg
                all_vals = win_vals + loss_vals
                val_range = max(all_vals) - min(all_vals)
                normalized_delta = abs(delta) / val_range if val_range > 0 else 0.0

                discriminators.append({
                    "indicator":        ind,
                    "win_avg":          round(win_avg,  4),
                    "loss_avg":         round(loss_avg, 4),
                    "delta":            round(delta,    4),
                    "normalized_score": round(normalized_delta, 4),
                    "win_samples":      len(win_vals),
                    "loss_samples":     len(loss_vals),
                })

            discriminators.sort(key=lambda x: -x["normalized_score"])

            # Per-symbol profile (top 10 per symbol)
            per_symbol_profile: dict = {}
            for sym in set(t["symbol"] for t in win_trades + loss_trades):
                sym_wins   = [t for t in win_trades   if t["symbol"] == sym]
                sym_losses = [t for t in loss_trades  if t["symbol"] == sym]
                if len(sym_wins) < 2 or len(sym_losses) < 2:
                    continue
                sym_disc: list = []
                for ind in all_indicators - skip_cols:
                    sw = [t["indicators_at_entry"][ind] for t in sym_wins
                          if ind in t["indicators_at_entry"]
                          and t["indicators_at_entry"][ind] is not None]
                    sl = [t["indicators_at_entry"][ind] for t in sym_losses
                          if ind in t["indicators_at_entry"]
                          and t["indicators_at_entry"][ind] is not None]
                    if len(sw) < 2 or len(sl) < 2:
                        continue
                    sw_avg = sum(sw) / len(sw)
                    sl_avg = sum(sl) / len(sl)
                    d      = sw_avg - sl_avg
                    ar     = max(sw + sl) - min(sw + sl)
                    ns     = abs(d) / ar if ar > 0 else 0.0
                    sym_disc.append({
                        "indicator":        ind,
                        "win_avg":          round(sw_avg, 4),
                        "loss_avg":         round(sl_avg, 4),
                        "delta":            round(d,      4),
                        "normalized_score": round(ns,     4),
                    })
                sym_disc.sort(key=lambda x: -x["normalized_score"])
                per_symbol_profile[sym] = sym_disc[:10]

            indicator_profiler = {
                "total_wins_with_snapshot":   len(win_trades),
                "total_losses_with_snapshot": len(loss_trades),
                "top_discriminators":         discriminators[:30],
                "per_symbol_profile":         per_symbol_profile,
            }

        analytics["indicator_profiler"] = indicator_profiler

        return analytics

    def _print_analytics(self, analytics: dict) -> None:
        """Print analytics to console in formatted sections."""
        if not analytics:
            return

        W = 55

        # ── template_anatomy ─────────────────────────────────────────────────
        anatomy = analytics.get("template_anatomy", {})
        if anatomy:
            print("\n" + "=" * W)
            print("TEMPLATE ANATOMY")
            print("=" * W)
            for tid, a in anatomy.items():
                print(f"  {tid}  v{a.get('version', 1)}  ({a.get('condition_count', 0)} conditions)")
                blocks = ", ".join(a.get("block_names", []))
                if blocks:
                    print(f"    Blocks: {blocks}")

        # ── trade_breakdown ──────────────────────────────────────────────────
        breakdown = analytics.get("trade_breakdown", {})
        if breakdown:
            print("\n" + "=" * W)
            print("TRADE BREAKDOWN BY TEMPLATE")
            print("=" * W)
            hdr = f"  {'Template':<28} {'Trades':>6} {'WR%':>6} {'AvgPnL%':>8} {'TotalPnL':>10}"
            print(hdr)
            print("  " + "-" * (W - 2))
            for tid, d in sorted(breakdown.items(), key=lambda x: -x[1]["total_trades"]):
                print(f"  {tid:<28} {d['total_trades']:>6} {d['win_rate']:>6.1f} "
                      f"{d['avg_pnl_pct']:>+8.2f} {d['total_pnl']:>+10.2f}")

        # ── per_symbol_summary ───────────────────────────────────────────────
        pss = analytics.get("per_symbol_summary", {})
        if pss:
            print("\n" + "─" * 22 + " PER-SYMBOL SUMMARY " + "─" * 22)
            print(f"  {'Symbol':<8} {'Trades':>7} {'Wins':>6} {'WR%':>7} "
                  f"{'AvgPnL%':>9} {'TotalPnL':>12}")
            print(f"  {'─' * 55}")
            for sym, d in sorted(pss.items(), key=lambda x: -x[1]["total_pnl"]):
                print(f"  {sym:<8} {d['trades']:>7} {d['wins']:>6} {d['wr']:>6.1f}% "
                      f"{d['avg_pnl_pct']:>+8.2f}% ${d['total_pnl']:>+10,.0f}")

        # ── temporal ─────────────────────────────────────────────────────────
        temporal = analytics.get("temporal", {})
        by_year = temporal.get("by_year", {})
        if by_year:
            print("\n" + "=" * W)
            print("TEMPORAL BREAKDOWN (by year)")
            print("=" * W)
            hdr = f"  {'Year':<8} {'Trades':>6} {'WR%':>6} {'AvgPnL%':>8}"
            print(hdr)
            print("  " + "-" * 30)
            for yr in sorted(by_year):
                d = by_year[yr]
                print(f"  {yr:<8} {d['trades']:>6} {d['win_rate']:>6.1f} {d['avg_pnl_pct']:>+8.2f}")

        # ── phase_analysis ────────────────────────────────────────────────────
        phase = analytics.get("phase_analysis", {})
        if phase:
            print("\n" + "=" * W)
            print("PHASE ANALYSIS")
            print("=" * W)
            hdr = f"  {'Phase':<20} {'Trades':>6} {'WR%':>6} {'AvgPnL%':>8}"
            print(hdr)
            print("  " + "-" * 42)
            for ph, d in sorted(phase.items(), key=lambda x: -x[1]["trades"]):
                print(f"  {ph:<20} {d['trades']:>6} {d['win_rate']:>6.1f} {d['avg_pnl_pct']:>+8.2f}")

        # ── block_stats ───────────────────────────────────────────────────────
        block_stats = analytics.get("block_stats", {})
        if block_stats:
            print("\n" + "=" * W)
            print("BLOCK STATISTICS")
            print("=" * W)
            for tid, bstats in block_stats.items():
                print(f"  [{tid}]")
                hdr = f"    {'Block':<25} {'Eval':>5} {'Pass%':>6} {'Blocker':>7}"
                print(hdr)
                print("    " + "-" * 42)
                for bname, bd in sorted(bstats.items()):
                    ev  = bd.get("evaluated", 0)
                    pr  = bd.get("pass_rate",  0.0)
                    blk = bd.get("was_the_blocker", 0)
                    print(f"    {bname:<25} {ev:>5} {pr:>6.1f} {blk:>7}")

        # ── shadow_ledger_matrix ──────────────────────────────────────────────
        shadow = analytics.get("shadow_ledger_matrix", {})
        if shadow:
            print("\n" + "=" * W)
            print("SHADOW LEDGER MATRIX")
            print("=" * W)
            for sym, templates in sorted(shadow.items()):
                if isinstance(templates, dict):
                    for tid, sd in templates.items():
                        sc  = sd.get("signal_count", 0)
                        wr  = sd.get("win_rate", 0.0)
                        apnl = sd.get("avg_pnl_pct", 0.0)
                        print(f"  {sym:<6} {tid:<28} sig={sc:>4}  WR={wr:>5.1f}%  "
                              f"avgPnL={apnl:>+6.2f}%")

        # ── winner_loser_profile ──────────────────────────────────────────────
        wlp = analytics.get("winner_loser_profile", {})
        if wlp:
            print("\n" + "=" * W)
            print("WINNER / LOSER PROFILE")
            print("=" * W)
            win_dist  = wlp.get("bars_distribution_wins", {})
            loss_dist = wlp.get("bars_distribution_losses", {})
            if win_dist or loss_dist:
                all_keys = sorted(set(list(win_dist) + list(loss_dist)))
                print(f"  Bars-held distribution (wins / losses):")
                for k in all_keys:
                    w_cnt = win_dist.get(k, 0)
                    l_cnt = loss_dist.get(k, 0)
                    print(f"    {k:>8} bars:  wins={w_cnt:>4}  losses={l_cnt:>4}")

            top5 = wlp.get("top_5_trades", [])
            if top5:
                print(f"\n  Top 5 trades:")
                for t in top5:
                    print(f"    {t['symbol']:<6} {t['template_id']:<25} "
                          f"{t['entry_date'][:10]}  {t['pnl_pct']:>+.2f}%  "
                          f"{t['bars_held']} bars")
            worst5 = wlp.get("worst_5_trades", [])
            if worst5:
                print(f"\n  Worst 5 trades:")
                for t in worst5:
                    print(f"    {t['symbol']:<6} {t['template_id']:<25} "
                          f"{t['entry_date'][:10]}  {t['pnl_pct']:>+.2f}%  "
                          f"{t['bars_held']} bars")

        # ── Section 8: block_evaluations ─────────────────────────────────────
        be = analytics.get("block_evaluations", {})
        if be:
            print("\n" + "─" * 43 + " BLOCK / INDICATOR STATISTICS " + "─" * 37)
            for tid, ts in sorted(be.items()):
                sf     = ts.get("state_filter", {})
                total  = sf.get("total_scans",    0)
                matched  = sf.get("state_matched",  0)
                rejected = sf.get("state_rejected", 0)
                match_pct = round(matched / total * 100, 1) if total > 0 else 0.0
                print(f"\n  [{tid}] {total} scans -> {matched} state-matched ({match_pct}%)")

                rr = sf.get("rejection_reasons", {})
                if rr:
                    parts = []
                    for axis, reasons in rr.items():
                        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
                            parts.append(f"{axis}.{reason}={count}")
                    if parts:
                        print(f"    Rejections: {', '.join(parts[:6])}")

                blocks = ts.get("blocks", {})
                if blocks:
                    print(f"    {'Block':<28} {'Eval':>6} {'Pass':>6} {'Fail':>6} "
                          f"{'Pass%':>6} {'SolBlk':>7} {'SB%':>5} {'WP->WR':>7} {'WP->PnL':>8}")
                    print(f"    {'─' * 86}")
                    for bn, bd in sorted(blocks.items(),
                                         key=lambda x: -x[1].get("was_sole_blocker", 0)):
                        ev  = bd.get("evaluated",       0)
                        pa  = bd.get("passed",          0)
                        fa  = bd.get("failed",          0)
                        pr  = bd.get("pass_rate",       0.0)
                        sb  = bd.get("was_sole_blocker", 0)
                        sbr = bd.get("sole_blocker_rate", 0.0)
                        wp  = bd.get("when_passed",     {})
                        wp_wr  = (f"{wp.get('wr', 0):.1f}%"
                                  if wp.get("total_trades", 0) > 0 else "—")
                        wp_pnl = (f"{wp.get('avg_pnl', 0):+.2f}%"
                                  if wp.get("total_trades", 0) > 0 else "—")
                        print(f"    {bn:<28} {ev:>6} {pa:>6} {fa:>6} "
                              f"{pr:>5.1f}% {sb:>7} {sbr:>4.1f}% {wp_wr:>7} {wp_pnl:>8}")

                    # Per-symbol lowest pass rates (investigation targets, min 10 evals)
                    all_sym_rates = []
                    for bn, bd in blocks.items():
                        for sym, sd in bd.get("per_symbol", {}).items():
                            if sd.get("evaluated", 0) >= 10:
                                all_sym_rates.append((sym, bn, sd.get("pass_rate", 0.0)))
                    if all_sym_rates:
                        all_sym_rates.sort(key=lambda x: x[2])
                        print(f"    Lowest per-symbol pass rates:")
                        for sym, bn, pr in all_sym_rates[:5]:
                            print(f"      {sym} -> {bn}: {pr:.1f}%")

        # ── Section 10: Indicator Profiler ────────────────────────────────────
        ip = analytics.get("indicator_profiler", {})
        if ip and ip.get("top_discriminators"):
            print("\n" + "─" * 17 + " SECTION 10: INDICATOR PROFILER (WIN vs LOSS) " + "─" * 17)
            print(f"  Wins with snapshot: {ip.get('total_wins_with_snapshot', 0)}, "
                  f"Losses with snapshot: {ip.get('total_losses_with_snapshot', 0)}")
            discs = ip["top_discriminators"]
            print(f"\n  Top Discriminators (highest WIN-LOSS separation):")
            print(f"  {'Indicator':<24} {'WIN avg':>10} {'LOSS avg':>10} "
                  f"{'Delta':>10} {'Score':>7}")
            print(f"  {'─' * 65}")
            for d in discs[:20]:
                print(f"  {d['indicator']:<24} {d['win_avg']:>10.4f} "
                      f"{d['loss_avg']:>10.4f} {d['delta']:>+10.4f} "
                      f"{d['normalized_score']:>6.3f}")
            psp = ip.get("per_symbol_profile", {})
            if psp:
                print(f"\n  Per-Symbol Top 5 Discriminators:")
                for sym in sorted(psp.keys()):
                    disc_list = psp[sym]
                    if not disc_list:
                        continue
                    top5 = ", ".join(
                        f"{d['indicator']}({d['delta']:+.3f})" for d in disc_list[:5]
                    )
                    print(f"    {sym}: {top5}")

    def _collect_block_evaluations(self):
        """
        Second-pass analysis: evaluate every template's conditions on every trading day
        to collect per-block pass/fail statistics. READ-ONLY — does not affect trades.

        Runs AFTER the backtest loop. Re-scans the timeline to record what each block
        did on each day, then links outcomes to actual closed trades.

        Output: self.block_eval_stats dict, consumed by _compute_analytics Section 8.
        """
        analytics_cfg = getattr(cfg, "ANALYTICS_CONFIG", {})
        if not analytics_cfg.get("include_block_evaluations", True):
            logger.info("Block evaluations disabled in ANALYTICS_CONFIG")
            return

        if not self.data_cache:
            logger.info("Block evaluations: no data cache, skipping")
            return

        logger.info("Collecting block-level evaluation statistics (second pass)...")
        t0 = time.perf_counter()

        warmup = self.config["min_candles_warmup"]

        # Build trade lookup: (symbol, entry_date) -> trade dict
        trade_lookup = {}
        for trade in self.closed_trades:
            key = (trade["symbol"], trade["entry_date"])
            trade_lookup[key] = trade

        # Reuse the existing TemplateManager from the matcher
        tm = self.matcher.tm
        all_templates = tm.get_enabled()

        # Initialize stats structure per template
        stats = {}
        for template in all_templates:
            block_names = [c.get("block", "?") for c in template.conditions]
            stats[template.id] = {
                "state_filter": {
                    "total_scans":    0,
                    "state_matched":  0,
                    "state_rejected": 0,
                    "rejection_reasons": defaultdict(lambda: defaultdict(int)),
                },
                "blocks": {},
            }
            for bn in block_names:
                stats[template.id]["blocks"][bn] = {
                    "evaluated":       0,
                    "passed":          0,
                    "failed":          0,
                    "was_sole_blocker": 0,
                    "when_passed": {
                        "total_trades": 0, "wins": 0, "losses": 0, "pnl_sum": 0.0,
                    },
                    "per_symbol": defaultdict(lambda: {"evaluated": 0, "passed": 0, "failed": 0}),
                }

        timeline = self._build_timeline()

        for trading_day in timeline:
            for sym in self.symbols:
                df = self.data_cache.get(sym)
                if df is None or trading_day not in df.index:
                    continue

                try:
                    loc = df.index.get_loc(trading_day)
                    if isinstance(loc, slice):
                        loc = loc.stop - 1
                    elif isinstance(loc, np.ndarray):
                        loc = int(np.flatnonzero(loc)[0])
                except Exception:
                    continue

                df_slice = df.iloc[:loc + 1]
                if len(df_slice) < warmup:
                    continue

                last_row = df_slice.iloc[-1]

                # Classify state
                stock_state = {}
                if self.hunter:
                    try:
                        stock_state = self.hunter.classify_stock_state(df_slice)
                    except Exception:
                        pass

                for template in all_templates:
                    tid = template.id
                    ts  = stats[tid]
                    ts["state_filter"]["total_scans"] += 1

                    # State filter check
                    if not tm._state_matches(template.required_state, stock_state):
                        ts["state_filter"]["state_rejected"] += 1
                        for axis, acceptable in template.required_state.items():
                            actual = stock_state.get(axis, "")
                            if actual not in acceptable:
                                ts["state_filter"]["rejection_reasons"][axis][actual] += 1
                        continue

                    ts["state_filter"]["state_matched"] += 1

                    # Evaluate all conditions
                    try:
                        all_passed, details = template.evaluate_conditions(last_row)
                    except Exception:
                        continue

                    # Record per-block stats
                    failed_blocks = []
                    for detail in details:
                        bn = detail.get("block", "?")
                        if bn not in ts["blocks"]:
                            continue
                        bs = ts["blocks"][bn]
                        bs["evaluated"] += 1
                        if sym not in bs["per_symbol"]:
                            bs["per_symbol"][sym] = {"evaluated": 0, "passed": 0, "failed": 0}
                        bs["per_symbol"][sym]["evaluated"] += 1
                        if detail.get("passed"):
                            bs["passed"] += 1
                            bs["per_symbol"][sym]["passed"] += 1
                        else:
                            bs["failed"] += 1
                            bs["per_symbol"][sym]["failed"] += 1
                            failed_blocks.append(bn)

                    # Sole blocker detection
                    if len(failed_blocks) == 1:
                        sole = failed_blocks[0]
                        if sole in ts["blocks"]:
                            ts["blocks"][sole]["was_sole_blocker"] += 1

                    # Link to trade outcome if signal was generated
                    if all_passed:
                        day_str = str(trading_day.date())
                        trade = trade_lookup.get((sym, day_str))
                        if trade and trade.get("template_id") == tid:
                            won     = trade["pnl"] > 0
                            pnl_pct = trade.get("pnl_pct", 0.0)
                            for detail in details:
                                bn = detail.get("block", "?")
                                if bn in ts["blocks"] and detail.get("passed"):
                                    wp = ts["blocks"][bn]["when_passed"]
                                    wp["total_trades"] += 1
                                    if won:
                                        wp["wins"] += 1
                                    else:
                                        wp["losses"] += 1
                                    wp["pnl_sum"] += pnl_pct

        # Compute derived metrics and convert defaultdicts for JSON serialisation
        for tid, ts in stats.items():
            sf = ts["state_filter"]
            sf["rejection_reasons"] = {
                k: dict(v) for k, v in sf["rejection_reasons"].items()
            }
            for bn, bs in ts["blocks"].items():
                ev = bs["evaluated"]
                bs["pass_rate"] = round(bs["passed"] / ev * 100, 1) if ev > 0 else 0.0
                bs["sole_blocker_rate"] = (
                    round(bs["was_sole_blocker"] / bs["failed"] * 100, 1)
                    if bs["failed"] > 0 else 0.0
                )
                wp = bs["when_passed"]
                if wp["total_trades"] > 0:
                    wp["wr"]      = round(wp["wins"] / wp["total_trades"] * 100, 1)
                    wp["avg_pnl"] = round(wp["pnl_sum"] / wp["total_trades"], 2)
                else:
                    wp["wr"]      = 0.0
                    wp["avg_pnl"] = 0.0

                bs["per_symbol"] = {
                    s: {
                        **d,
                        "pass_rate": round(d["passed"] / d["evaluated"] * 100, 1)
                                     if d["evaluated"] > 0 else 0.0,
                    }
                    for s, d in bs["per_symbol"].items()
                }

        self.block_eval_stats = stats
        elapsed = round(time.perf_counter() - t0, 1)
        logger.info(
            f"Block evaluation collection complete: {len(stats)} templates, {elapsed}s"
        )

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
# Walk-Forward Validator
# ═══════════════════════════════════════════════════════════════════════════

class WalkForwardValidator:
    """
    Walk-Forward Validation: 70/30 train/test split with no look-ahead.

    Runs BacktestEngine separately on train and test periods,
    then compares per-template performance to detect overfitting
    and validate generated templates meet CP-2 criteria.

    Usage:
        validator = WalkForwardValidator(symbols=["AAPL", "NVDA"])
        report = validator.validate()
        print(report["verdict"])  # "PASS" or "FAIL"
    """

    def __init__(self, symbols=None, initial_capital=None,
                 use_risk_gates=True, data_cache=None):
        wf_cfg = getattr(cfg, 'WALK_FORWARD_CONFIG', {})
        self.symbols = symbols or list(getattr(cfg, "DEFAULT_TRAINING_SYMBOLS", []))
        self.initial_capital = initial_capital or BACKTEST_CONFIG["initial_capital"]
        self.use_risk_gates = use_risk_gates
        self.train_pct = wf_cfg.get("train_pct", 0.70)
        self.data_cache = data_cache  # optional pre-loaded data
        self.config = wf_cfg

    def validate(self) -> dict:
        """
        Main entry point. Returns full walk-forward report with CP-2 verdict.

        Returns:
            dict with keys: split_date, train_days, test_days, train_summary,
            test_summary, per_template, cp2, overfit_warnings, verdict
        """
        logger.info("=" * 60)
        logger.info("WALK-FORWARD VALIDATION START")
        logger.info(f"Train/Test split: {self.train_pct:.0%} / {1-self.train_pct:.0%}")
        logger.info("=" * 60)

        # Step 1: Load data via BacktestEngine's loader
        loader = BacktestEngine(
            symbols=self.symbols,
            initial_capital=self.initial_capital,
            use_risk_gates=self.use_risk_gates,
            data_cache=self.data_cache,
        )
        loader._ensure_data()
        full_data = loader.data_cache

        if not full_data:
            logger.error("[WF] No data available")
            return {"error": "No data available", "verdict": "FAIL"}

        # Step 2: Split data by date
        train_data, test_data, split_info = self._split_data(full_data)

        if not train_data or not test_data:
            logger.error("[WF] Insufficient data for split")
            return {"error": "Insufficient data for train/test split", "verdict": "FAIL"}

        logger.info(
            f"[WF] Split at {split_info['split_date']} | "
            f"Train: {split_info['train_days']} days | "
            f"Test: {split_info['test_days']} days"
        )

        # Step 3: Run BacktestEngine on TRAIN period
        logger.info("[WF] === TRAIN PERIOD ===")
        train_engine = BacktestEngine(
            symbols=self.symbols,
            initial_capital=self.initial_capital,
            use_risk_gates=self.use_risk_gates,
            data_cache=train_data,
        )
        train_engine.feed_shadow_ledger = False  # Don't pollute ledger with partial data
        train_results = train_engine.run()

        # Step 4: Run BacktestEngine on TEST period
        logger.info("[WF] === TEST PERIOD ===")
        test_engine = BacktestEngine(
            symbols=self.symbols,
            initial_capital=self.initial_capital,
            use_risk_gates=self.use_risk_gates,
            data_cache=test_data,
        )
        test_engine.feed_shadow_ledger = False
        test_results = test_engine.run()

        # Step 5: Compare per-template
        per_template = self._compare_per_template(
            train_results.get("trades", train_engine.closed_trades),
            test_results.get("trades", test_engine.closed_trades),
        )

        # Step 6: Evaluate CP-2
        cp2 = self._evaluate_cp2(per_template)

        # Step 7: Detect overfit warnings
        overfit_warnings = self._detect_overfitting(per_template)

        verdict = cp2["verdict"]
        if overfit_warnings:
            for w in overfit_warnings:
                logger.warning(f"[WF-OVERFIT] {w}")

        report = {
            "split_date":      split_info["split_date"],
            "train_days":      split_info["train_days"],
            "test_days":       split_info["test_days"],
            "train_summary":   train_results.get("summary", {}),
            "test_summary":    test_results.get("summary", {}),
            "per_template":    per_template,
            "cp2":             cp2,
            "overfit_warnings": overfit_warnings,
            "verdict":         verdict,
        }

        self._log_report(report)
        return report

    def _split_data(self, full_data):
        """Split each symbol's DataFrame into train/test by global split date.

        Finds the global date range across all symbols, calculates
        split_date at train_pct of the range, then slices each DataFrame.
        Test data includes warmup bars before split_date so indicators are valid.

        Returns:
            (train_data, test_data, split_info)
        """
        warmup = BACKTEST_CONFIG.get("min_candles_warmup", 200)

        # Collect all tradeable dates (after warmup strip)
        all_dates = set()
        for df in full_data.values():
            if len(df) > warmup:
                all_dates.update(df.index[warmup:])

        if not all_dates:
            return None, None, {}

        sorted_dates = sorted(all_dates)
        split_idx = int(len(sorted_dates) * self.train_pct)

        if split_idx < 1 or split_idx >= len(sorted_dates) - 1:
            return None, None, {}

        split_date = sorted_dates[split_idx]

        train_data = {}
        test_data = {}

        for symbol, df in full_data.items():
            # Train: all rows up to and including split_date
            train_mask = df.index <= split_date
            train_df = df.loc[train_mask]

            # Test: include warmup bars before split_date so indicators are valid
            test_start_pos = max(
                0,
                df.index.get_indexer([split_date], method='ffill')[0] - warmup
            )
            test_df = df.iloc[test_start_pos:]

            if len(train_df) >= warmup:
                train_data[symbol] = train_df
            if len(test_df) >= warmup:
                test_data[symbol] = test_df

        split_info = {
            "split_date": (
                str(split_date.date()) if hasattr(split_date, 'date') else str(split_date)
            ),
            "train_days": split_idx,
            "test_days":  len(sorted_dates) - split_idx,
        }

        return train_data, test_data, split_info

    def _compare_per_template(self, train_trades, test_trades):
        """Compare per-template performance between train and test periods.

        Args:
            train_trades: list of trade dicts from train BacktestEngine
            test_trades:  list of trade dicts from test BacktestEngine
        Returns:
            dict: template_id → performance comparison dict
        """
        def _calc_stats(trades):
            if not trades:
                return {"trades": 0, "wr": 0.0, "pf": 0.0, "total_pnl": 0.0}
            wins = sum(1 for t in trades if t.get("pnl_pct", 0) > 0)
            gross_profit = sum(t["pnl_pct"] for t in trades if t.get("pnl_pct", 0) > 0)
            gross_loss   = abs(sum(t["pnl_pct"] for t in trades if t.get("pnl_pct", 0) <= 0))
            wr = wins / len(trades)
            pf = (gross_profit / gross_loss) if gross_loss > 0 else (
                float('inf') if gross_profit > 0 else 0.0
            )
            return {
                "trades":    len(trades),
                "wr":        round(wr, 4),
                "pf":        round(pf, 2),
                "total_pnl": round(sum(t.get("pnl_pct", 0) for t in trades), 2),
            }

        train_by_tmpl = {}
        for t in train_trades:
            tid = t.get("template_id", "UNKNOWN")
            train_by_tmpl.setdefault(tid, []).append(t)

        test_by_tmpl = {}
        for t in test_trades:
            tid = t.get("template_id", "UNKNOWN")
            test_by_tmpl.setdefault(tid, []).append(t)

        all_templates = set(train_by_tmpl.keys()) | set(test_by_tmpl.keys())
        overfit_threshold = self.config.get("flag_overfit_threshold", 0.20)

        result = {}
        for tid in sorted(all_templates):
            tr = _calc_stats(train_by_tmpl.get(tid, []))
            te = _calc_stats(test_by_tmpl.get(tid, []))
            wr_delta = tr["wr"] - te["wr"]
            is_gen   = tid.startswith("GEN_")

            result[tid] = {
                "train_trades": tr["trades"],
                "train_wr":     tr["wr"],
                "train_pf":     tr["pf"],
                "train_pnl":    tr["total_pnl"],
                "test_trades":  te["trades"],
                "test_wr":      te["wr"],
                "test_pf":      te["pf"],
                "test_pnl":     te["total_pnl"],
                "wr_delta":     round(wr_delta, 4),
                "is_generated": is_gen,
                "overfit_flag": wr_delta > overfit_threshold and tr["trades"] >= 5,
            }

        return result

    def _evaluate_cp2(self, per_template):
        """Evaluate CP-2 pass/fail for generated templates vs seed baseline.

        CP-2 criteria:
        - Generated templates test WR > cp2_min_wr_generated (default 25%)
        - Generated templates test PF >= cp2_min_pf (default 1.50)
        - If zero generated templates traded → pass vacuously
        - If < min_test_trades → soft fail with reason
        """
        min_test_trades = self.config.get("min_test_trades", 5)
        cp2_min_wr = self.config.get("cp2_min_wr_generated", 0.25)
        cp2_min_pf = self.config.get("cp2_min_pf", 1.50)

        gen_trades = gen_wins = 0
        gen_profit = gen_loss = 0.0
        seed_trades = seed_wins = 0
        seed_profit = seed_loss = 0.0

        for tid, stats in per_template.items():
            tt = stats["test_trades"]
            tw = stats["test_wr"]
            tp = stats["test_pnl"]
            wins_est = int(round(tw * tt)) if tt > 0 else 0

            if stats["is_generated"]:
                gen_trades += tt
                gen_wins   += wins_est
                if tp > 0: gen_profit += tp
                else:      gen_loss   += abs(tp)
            else:
                seed_trades += tt
                seed_wins   += wins_est
                if tp > 0: seed_profit += tp
                else:      seed_loss   += abs(tp)

        gen_wr  = gen_wins  / gen_trades  if gen_trades  > 0 else 0.0
        gen_pf  = (gen_profit  / gen_loss)  if gen_loss  > 0 else (
            float('inf') if gen_profit  > 0 else 0.0)
        seed_wr = seed_wins / seed_trades if seed_trades > 0 else 0.0
        seed_pf = (seed_profit / seed_loss) if seed_loss > 0 else (
            float('inf') if seed_profit > 0 else 0.0)

        reasons  = []
        pass_wr  = True
        pass_pf  = True

        if gen_trades == 0:
            # No generated templates active — pass vacuously
            pass
        elif gen_trades < min_test_trades:
            reasons.append(
                f"Generated templates had only {gen_trades} test trades (min={min_test_trades})"
            )
            pass_wr = False
            pass_pf = False
        else:
            if gen_wr < cp2_min_wr:
                pass_wr = False
                reasons.append(f"Generated WR={gen_wr:.1%} < {cp2_min_wr:.0%}")
            if gen_pf != float('inf') and gen_pf < cp2_min_pf:
                pass_pf = False
                reasons.append(f"Generated PF={gen_pf:.2f} < {cp2_min_pf:.2f}")

        verdict = "PASS" if (pass_wr and pass_pf) else "FAIL"

        return {
            "generated_test_trades": gen_trades,
            "generated_test_wr":     round(gen_wr, 4),
            "generated_test_pf":     round(gen_pf, 2) if gen_pf != float('inf') else "inf",
            "seed_test_trades":      seed_trades,
            "seed_test_wr":          round(seed_wr, 4),
            "seed_test_pf":          round(seed_pf, 2) if seed_pf != float('inf') else "inf",
            "pass_wr":               pass_wr,
            "pass_pf":               pass_pf,
            "verdict":               verdict,
            "reasons":               reasons,
        }

    def _detect_overfitting(self, per_template):
        """Return warning strings for templates where train WR >> test WR."""
        threshold = self.config.get("flag_overfit_threshold", 0.20)
        warnings = []
        for tid, stats in per_template.items():
            if stats["overfit_flag"]:
                warnings.append(
                    f"{tid}: train_wr={stats['train_wr']:.1%} → test_wr={stats['test_wr']:.1%} "
                    f"(Δ={stats['wr_delta']:.1%} > {threshold:.0%}) [OVERFIT]"
                )
        return warnings

    def _log_report(self, report):
        """Log walk-forward results summary."""
        logger.info("=" * 60)
        logger.info("WALK-FORWARD RESULTS")
        logger.info("=" * 60)
        logger.info(
            f"Split: {report.get('split_date')} | "
            f"Train: {report.get('train_days')}d | "
            f"Test: {report.get('test_days')}d"
        )

        train_s = report.get("train_summary", {})
        test_s  = report.get("test_summary",  {})
        logger.info(
            f"[TRAIN] Trades={train_s.get('total_trades', 0)} | "
            f"WR={train_s.get('win_rate', 0):.1f}% | "
            f"PF={train_s.get('profit_factor', 0):.2f} | "
            f"Return={train_s.get('total_return_pct', 0):.2f}%"
        )
        logger.info(
            f"[TEST]  Trades={test_s.get('total_trades', 0)} | "
            f"WR={test_s.get('win_rate', 0):.1f}% | "
            f"PF={test_s.get('profit_factor', 0):.2f} | "
            f"Return={test_s.get('total_return_pct', 0):.2f}%"
        )

        for tid, stats in report.get("per_template", {}).items():
            flag    = " [OVERFIT]" if stats.get("overfit_flag") else ""
            gen_tag = " [GEN]" if stats.get("is_generated") else ""
            logger.info(
                f"[WF-TMPL] {tid}{gen_tag} | "
                f"train={stats['train_trades']}t/{stats['train_wr']:.0%}wr/{stats['train_pf']:.1f}pf | "
                f"test={stats['test_trades']}t/{stats['test_wr']:.0%}wr/{stats['test_pf']:.1f}pf | "
                f"Δwr={stats['wr_delta']:+.0%}{flag}"
            )

        cp2 = report.get("cp2", {})
        logger.info(
            f"[CP-2] Generated: WR={cp2.get('generated_test_wr', 0):.1%} | "
            f"PF={cp2.get('generated_test_pf', 0)}"
        )
        logger.info(
            f"[CP-2] Seed:      WR={cp2.get('seed_test_wr', 0):.1%} | "
            f"PF={cp2.get('seed_test_pf', 0)}"
        )
        logger.info(f"[CP-2] VERDICT: {cp2.get('verdict', 'N/A')}")
        for r in cp2.get("reasons", []):
            logger.info(f"[CP-2] Reason: {r}")
        for w in report.get("overfit_warnings", []):
            logger.warning(f"[WF-OVERFIT] {w}")

    # ── Quality Gate ──────────────────────────────────────────────────────────

    def validate_single_template(self, template_id, all_data_dict, config=None):
        """WF quality gate for a single newly-generated template.

        Splits all_data_dict 70/30, runs the full pipeline on each half,
        filters trades to template_id only, then evaluates PF threshold.

        Args:
            template_id:    ID of the template to validate (e.g. "GEN_BEARISH_SQUEEZE_BREAK")
            all_data_dict:  dict of {symbol: DataFrame} with full historical feature data
            config:         optional config override (defaults to self.config)

        Returns:
            dict: {passed, test_pf, test_trades, train_pf, train_trades, reason}
        """
        cfg_used = config if config is not None else self.config
        min_trades = cfg_used.get("quality_gate_min_trades", 3)
        min_pf     = cfg_used.get("quality_gate_min_pf", 1.0)

        if not all_data_dict:
            return {"passed": True, "test_pf": 0.0, "test_trades": 0,
                    "train_pf": 0.0, "train_trades": 0,
                    "reason": "no data — BURN_IN"}

        train_data, test_data, _ = self._split_data(all_data_dict)
        if not train_data or not test_data:
            return {"passed": True, "test_pf": 0.0, "test_trades": 0,
                    "train_pf": 0.0, "train_trades": 0,
                    "reason": "insufficient data for split — BURN_IN"}

        train_trades = self._run_engine_and_filter(train_data, template_id)
        test_trades  = self._run_engine_and_filter(test_data,  template_id)

        return self._evaluate_quality_gate(
            template_id, train_trades, test_trades, min_trades, min_pf
        )

    def _run_engine_and_filter(self, data_cache, template_id):
        """Run BacktestEngine on data_cache and return only trades for template_id."""
        engine = BacktestEngine(
            symbols=list(data_cache.keys()),
            initial_capital=self.initial_capital,
            use_risk_gates=self.use_risk_gates,
            data_cache=data_cache,
        )
        engine.feed_shadow_ledger = False
        results = engine.run()
        trades = results.get("trades", getattr(engine, "closed_trades", []))
        return [t for t in trades if t.get("template_id") == template_id]

    def _evaluate_quality_gate(self, template_id, train_trades, test_trades,
                               min_trades, min_pf):
        """Pure evaluation logic — compute PF from trade lists and apply thresholds.

        Separated from engine calls so it can be unit-tested without network/disk I/O.

        Returns:
            dict: {passed, test_pf, test_trades, train_pf, train_trades, reason}
        """
        def _pf(trades):
            profit = sum(t["pnl_pct"] for t in trades if t.get("pnl_pct", 0) > 0)
            loss   = abs(sum(t["pnl_pct"] for t in trades if t.get("pnl_pct", 0) <= 0))
            if loss > 0:
                return round(profit / loss, 2)
            return float("inf") if profit > 0 else 0.0

        test_pf  = _pf(test_trades)
        train_pf = _pf(train_trades)
        n_test   = len(test_trades)
        n_train  = len(train_trades)

        if n_test < min_trades:
            return {
                "passed":       True,
                "test_pf":      test_pf,
                "test_trades":  n_test,
                "train_pf":     train_pf,
                "train_trades": n_train,
                "reason":       (
                    f"insufficient test trades ({n_test} < {min_trades}) — BURN_IN"
                ),
            }

        if test_pf >= min_pf:
            return {
                "passed":       True,
                "test_pf":      test_pf,
                "test_trades":  n_test,
                "train_pf":     train_pf,
                "train_trades": n_train,
                "reason":       f"test PF {test_pf:.2f} >= threshold {min_pf}",
            }

        return {
            "passed":       False,
            "test_pf":      test_pf,
            "test_trades":  n_test,
            "train_pf":     train_pf,
            "train_trades": n_train,
            "reason":       f"test PF {test_pf:.2f} < threshold {min_pf}",
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
    parser.add_argument("--no-feed-shadow-ledger", action="store_true", default=False,
                        help="Skip feeding results into shadow_ledger.json")
    parser.add_argument("--output",        default=BACKTEST_RESULTS_PATH)
    parser.add_argument("--walk-forward",  action="store_true",
                        help="Run Walk-Forward Validation (70/30 split, CP-2 checkpoint)")
    args = parser.parse_args()

    if args.walk_forward:
        wf = WalkForwardValidator(
            symbols=args.symbols,
            initial_capital=args.capital,
            use_risk_gates=not args.no_risk_gates,
        )
        report = wf.validate()

        cp2 = report.get("cp2", {})
        print(f"\n{'='*55}")
        print(f"WALK-FORWARD CP-2 VERDICT: {cp2.get('verdict', 'N/A')}")
        print(f"{'='*55}")
        print(f"Generated test WR : {cp2.get('generated_test_wr', 0):.1%}")
        print(f"Generated test PF : {cp2.get('generated_test_pf', 0)}")
        print(f"Seed test WR      : {cp2.get('seed_test_wr', 0):.1%}")
        print(f"Seed test PF      : {cp2.get('seed_test_pf', 0)}")
        if cp2.get("reasons"):
            for r in cp2["reasons"]:
                print(f"  → {r}")
        for w in report.get("overfit_warnings", []):
            print(f"  ⚠️  {w}")

        wf_path = args.output.replace(".json", "_walkforward.json")
        safe_json_write(wf_path, report)
        print(f"\nReport saved: {wf_path}")
        print("=" * 55)

        sys.exit(0 if cp2.get("verdict") == "PASS" else 1)

    engine = BacktestEngine(
        symbols=args.symbols,
        initial_capital=args.capital,
        use_risk_gates=not args.no_risk_gates,
        config_overrides={
            "position_size_pct": args.position_size,
            "max_positions":     args.max_positions,
        },
    )
    if args.no_feed_shadow_ledger:
        engine.feed_shadow_ledger = False
        logger.info("Shadow ledger feed disabled via --no-feed-shadow-ledger")
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

    analytics = results.get("analytics", {})
    if analytics:
        engine._print_analytics(analytics)

    print(f"\nResults -> {args.output}")
    print("=" * 55)

    sys.exit(0 if sv.get("survival_verdict") not in ("DANGER", "CRITICAL") else 1)


if __name__ == "__main__":
    main()
