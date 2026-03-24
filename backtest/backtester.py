"""
Portfolio backtester. Simulates trading day-by-day using discovered templates.
Used for both validation and test phases.
"""

import logging
import math
from datetime import timedelta
from collections import defaultdict

import pandas as pd
import numpy as np

from backtest.config import (
    STARTING_CAPITAL, MAX_POSITION_PCT, COMMISSION, SLIPPAGE,
    MIN_WIN_RATE
)

logger = logging.getLogger("backtest.backtester")

# ── Risk constants (from system_config.PORTFOLIO_RISK_CONFIG) ──────────────
MAX_SECTOR_POSITIONS = 2
MAX_TOTAL_EXPOSURE   = 0.60
MAX_CORRELATION      = 0.80
RISK_PER_TRADE_PCT   = 0.0075   # 0.75% of capital per trade


def _safe(val, default=0.0):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return default
    return val


def _check_conditions(row_dict: dict, conditions: list, condition_blocks: dict) -> bool:
    """Return True if all template conditions pass on this row."""
    try:
        for cond in conditions:
            block_name = cond.get('block', '')
            params     = cond.get('params', [])
            fn = condition_blocks.get(block_name)
            if fn is None:
                continue
            if not fn(row_dict, params):
                return False
        return True
    except Exception:
        return False


def _calc_stop(row_dict: dict, stop_cfg: dict, stop_blocks: dict) -> float:
    method = stop_cfg.get('method', 'atr')
    fn = stop_blocks.get(method)
    if fn is None:
        close = _safe(row_dict.get('close', 0))
        return close * 0.98
    atr_mult = stop_cfg.get('atr_multiplier', 2.0)
    fallback_pct = stop_cfg.get('fallback_pct', 0.02)
    close = _safe(row_dict.get('close', 0))
    try:
        stop = fn(row_dict, [atr_mult])
        if stop <= 0 or stop >= close:
            stop = close * (1 - fallback_pct)
        return stop
    except Exception:
        return close * (1 - fallback_pct)


def _calc_target(row_dict: dict, take_cfg: dict, target_blocks: dict) -> float:
    method = take_cfg.get('method', 'atr')
    fn = target_blocks.get(method)
    if fn is None:
        close = _safe(row_dict.get('close', 0))
        return close * 1.06
    atr_mult = take_cfg.get('atr_multiplier', 3.0)
    close = _safe(row_dict.get('close', 0))
    try:
        target = fn(row_dict, [atr_mult])
        if target <= close:
            target = close * 1.06
        return target
    except Exception:
        return close * 1.06


def _classify_state(df: pd.DataFrame) -> dict:
    """Simplified stock state classification (no external deps needed)."""
    if len(df) < 20:
        return {"trend": "SIDEWAYS", "structure": "OPEN_FIELD",
                "volume": "HEALTHY", "volatility": "NORMAL"}
    try:
        last = df.iloc[-1]
        close = _safe(last.get('close', 0))

        # Trend
        sma50  = _safe(last.get('sma_50', 0))
        sma200 = _safe(last.get('sma_200', 0))
        if sma50 > sma200 and close > sma50:
            trend = "BULLISH"
        elif sma50 < sma200 and close < sma50:
            trend = "BEARISH"
        else:
            trend = "SIDEWAYS"

        # Structure
        recent_high = df['high'].tail(20).max()
        recent_low  = df['low'].tail(20).min()
        if close >= recent_high * 0.98:
            structure = "NEAR_RESISTANCE"
        elif close <= recent_low * 1.02:
            structure = "NEAR_SUPPORT"
        else:
            structure = "OPEN_FIELD"

        # Volume
        vol_avg = _safe(last.get('vol_avg_20', 1))
        volume  = _safe(last.get('volume', 0))
        if vol_avg < 500_000:
            volume_state = "ILLIQUID"
        elif volume > vol_avg * 2:
            volume_state = "SURGING"
        elif volume < vol_avg * 0.5:
            volume_state = "DRYING_UP"
        else:
            volume_state = "HEALTHY"

        # Volatility
        bb_width = _safe(last.get('bb_width', 0.15))
        if bb_width < 0.10:
            volatility = "COMPRESSED"
        elif bb_width > 0.30:
            volatility = "VOLATILE"
        else:
            volatility = "NORMAL"

        return {"trend": trend, "structure": structure,
                "volume": volume_state, "volatility": volatility}
    except Exception:
        return {"trend": "SIDEWAYS", "structure": "OPEN_FIELD",
                "volume": "HEALTHY", "volatility": "NORMAL"}


def run_backtest(data: dict, templates: list, config: dict = None) -> tuple:
    """
    Simulate trading day-by-day across all symbols.

    Args:
        data:      {symbol: DataFrame}  — already split to correct period
        templates: list of template dicts from template_optimizer
        config:    optional override dict

    Returns:
        (trades_list, daily_returns)
        trades_list: list of trade dicts
        daily_returns: list of daily portfolio state dicts
    """
    from setup_templates import CONDITION_BLOCKS, STOP_BLOCKS, TARGET_BLOCKS

    if not data or not templates:
        logger.warning("run_backtest called with empty data or templates")
        return [], []

    # Build sorted list of all trading dates
    all_dates = sorted(set(
        date for df in data.values() for date in df.index
    ))

    capital   = STARTING_CAPITAL
    cash      = capital
    positions = {}   # symbol -> position dict
    trades    = []
    daily_log = []
    trade_id  = 0

    for today in all_dates:
        daily_open_positions = len(positions)
        portfolio_value = cash + sum(
            pos['qty'] * data[sym].loc[today, 'close']
            if today in data[sym].index else pos['qty'] * pos['entry_price']
            for sym, pos in positions.items()
            if sym in data
        )

        # ── Manage open positions ──────────────────────────────────────────
        exits_today = []
        for sym, pos in list(positions.items()):
            if sym not in data or today not in data[sym].index:
                continue
            row = data[sym].loc[today]
            high  = _safe(row.get('high', pos['entry_price']))
            low   = _safe(row.get('low',  pos['entry_price']))
            close = _safe(row.get('close', pos['entry_price']))
            atr   = _safe(row.get('atr', pos['entry_price'] * 0.02))

            stop  = pos['stop_loss']
            target = pos['take_profit']
            phase = pos.get('kinetic_phase', 1)
            phase_high = pos.get('phase_high', pos['entry_price'])
            phases_visited = pos.get('phases_visited', [1])

            # Update phase high watermark
            if high > phase_high:
                pos['phase_high'] = high
                phase_high = high

            exit_price  = None
            exit_reason = None

            # Kinetic stop progression
            gain_pct = (close - pos['entry_price']) / pos['entry_price'] * 100

            if phase == 1:
                # Trail at 2.0 ATR below phase_high
                new_stop = phase_high - 2.0 * atr
                if new_stop > stop:
                    stop = new_stop
                    pos['stop_loss'] = stop
                if gain_pct >= 1.5 and phase == 1:
                    pos['kinetic_phase'] = 2
                    phases_visited.append(2)
                    pos['phases_visited'] = phases_visited

            elif phase == 2:
                # Breakeven + buffer
                be_stop = pos['entry_price'] * 1.005
                if be_stop > stop:
                    stop = be_stop
                    pos['stop_loss'] = stop
                if gain_pct >= 3.0:
                    pos['kinetic_phase'] = 3
                    phases_visited.append(3)
                    pos['phases_visited'] = phases_visited

            elif phase == 3:
                # Choke: trail at 1.0 ATR from high
                new_stop = phase_high - 1.0 * atr
                if new_stop > stop:
                    stop = new_stop
                    pos['stop_loss'] = stop
                if gain_pct >= 5.0:
                    pos['kinetic_phase'] = 4
                    phases_visited.append(4)
                    pos['phases_visited'] = phases_visited

            elif phase == 4:
                # Runner: trail at 0.5 ATR from high
                new_stop = phase_high - 0.5 * atr
                if new_stop > stop:
                    stop = new_stop
                    pos['stop_loss'] = stop

            # Check stop hit
            if low <= stop:
                exit_price  = stop
                exit_reason = "STOP_HIT"
            # Check target hit
            elif high >= target:
                if pos.get('runner_mode', False):
                    # Already in runner — will be caught by trailing stop above
                    pass
                else:
                    pos['runner_mode'] = True
                    # Activate phase 4 immediately on target
                    if phase < 4:
                        pos['kinetic_phase'] = 4
                        if 4 not in phases_visited:
                            phases_visited.append(4)
                            pos['phases_visited'] = phases_visited

            if exit_reason:
                exit_price = max(exit_price, stop)  # never exit below stop
                qty = pos['qty']
                gross_pnl = (exit_price - pos['entry_price']) * qty
                cost = (pos['entry_price'] + exit_price) * qty * COMMISSION
                slippage_cost = exit_price * qty * SLIPPAGE
                net_pnl = gross_pnl - cost - slippage_cost
                tax = max(0, net_pnl * TAX_RATE) if net_pnl > 0 else 0
                net_pnl -= tax

                cash += exit_price * qty - cost - slippage_cost - tax
                hold_days = (today - pd.Timestamp(pos['entry_date'])).days

                # Drawdown during hold
                max_dd = 0.0
                try:
                    hold_slice = data[sym].loc[pos['entry_date']:today]['low']
                    if not hold_slice.empty:
                        lowest = hold_slice.min()
                        max_dd = (pos['entry_price'] - lowest) / pos['entry_price'] * 100
                except Exception:
                    pass

                trade_id += 1
                trade = {
                    "trade_id": trade_id,
                    "symbol": sym,
                    "entry_date": str(pos['entry_date'])[:10],
                    "exit_date": str(today)[:10],
                    "entry_price": round(pos['entry_price'], 4),
                    "exit_price": round(exit_price, 4),
                    "stop_loss": round(pos['stop_loss'], 4),
                    "take_profit": round(pos['take_profit'], 4),
                    "qty": qty,
                    "pnl_dollars": round(net_pnl, 2),
                    "pnl_pct": round((exit_price - pos['entry_price']) / pos['entry_price'] * 100, 3),
                    "exit_reason": exit_reason,
                    "template_id": pos['template_id'],
                    "confidence_score": pos['confidence_score'],
                    "hold_duration_days": hold_days,
                    "max_drawdown_pct": round(max_dd, 3),
                    "stock_state": pos['stock_state'],
                    "entry_indicators": pos['entry_indicators'],
                    "exit_indicators": _row_to_indicators(data[sym].loc[today] if today in data[sym].index else None),
                    "kinetic_phases_visited": phases_visited,
                    "portfolio_value_at_entry": round(pos['portfolio_value_at_entry'], 2),
                    "portfolio_value_at_exit": round(portfolio_value, 2),
                }
                trades.append(trade)
                exits_today.append(sym)

        for sym in exits_today:
            del positions[sym]

        # ── Scan for new entries ───────────────────────────────────────────
        total_exposure_pct = sum(
            pos['qty'] * pos['entry_price'] / portfolio_value
            for pos in positions.values()
        ) if portfolio_value > 0 else 0.0

        sector_counts = defaultdict(int)
        for pos in positions.values():
            sector_counts[pos.get('sector', 'UNKNOWN')] += 1

        for symbol, df in data.items():
            if symbol in positions:
                continue
            if today not in df.index:
                continue
            if total_exposure_pct >= MAX_TOTAL_EXPOSURE:
                break
            if cash < STARTING_CAPITAL * 0.05:
                break

            row = df.loc[today]
            row_dict = row.to_dict()
            close = _safe(row_dict.get('close', 0))
            if close <= 0:
                continue

            # Classify state
            df_up_to_today = df[df.index <= today]
            stock_state = _classify_state(df_up_to_today)

            # Match templates
            signal = None
            for tmpl in templates:
                # Check state compatibility
                rs = tmpl.get('required_state', {})
                if rs:
                    if stock_state['trend'] not in rs.get('trend', [stock_state['trend']]):
                        continue
                    if stock_state['volume'] not in rs.get('volume', [stock_state['volume']]):
                        continue
                    if stock_state['volatility'] not in rs.get('volatility', [stock_state['volatility']]):
                        continue

                if _check_conditions(row_dict, tmpl['conditions'], CONDITION_BLOCKS):
                    stop_cfg   = tmpl.get('stop_loss', {})
                    target_cfg = tmpl.get('take_profit', {})
                    stop   = _calc_stop(row_dict, stop_cfg, STOP_BLOCKS)
                    target = _calc_target(row_dict, target_cfg, TARGET_BLOCKS)
                    risk   = close - stop
                    if risk <= 0 or risk / close > 0.10:
                        continue
                    signal = {
                        'template': tmpl,
                        'stop': stop,
                        'target': target,
                        'risk': risk,
                        'confidence': _safe(tmpl['statistics'].get('win_rate', 70)),
                    }
                    break

            if signal is None:
                continue

            # Position sizing
            risk_dollars  = portfolio_value * RISK_PER_TRADE_PCT
            max_dollars   = portfolio_value * MAX_POSITION_PCT
            qty = int(min(
                risk_dollars / signal['risk'],
                max_dollars / close
            ))
            if qty < 1:
                continue

            cost_basis = close * qty * (1 + COMMISSION + SLIPPAGE)
            if cost_basis > cash:
                qty = int(cash / (close * (1 + COMMISSION + SLIPPAGE)))
                if qty < 1:
                    continue
                cost_basis = close * qty * (1 + COMMISSION + SLIPPAGE)

            cash -= cost_basis
            total_exposure_pct += (close * qty) / portfolio_value

            positions[symbol] = {
                'entry_date': today,
                'entry_price': close,
                'stop_loss': signal['stop'],
                'take_profit': signal['target'],
                'qty': qty,
                'template_id': signal['template']['id'],
                'confidence_score': signal['confidence'],
                'stock_state': stock_state,
                'entry_indicators': _row_to_indicators(row),
                'kinetic_phase': 1,
                'phase_high': close,
                'phases_visited': [1],
                'runner_mode': False,
                'sector': 'UNKNOWN',
                'portfolio_value_at_entry': portfolio_value,
            }

        # ── Daily snapshot ─────────────────────────────────────────────────
        end_value = cash + sum(
            pos['qty'] * (data[sym].loc[today, 'close'] if sym in data and today in data[sym].index else pos['entry_price'])
            for sym, pos in positions.items()
        )
        prev_value = daily_log[-1]['portfolio_value'] if daily_log else STARTING_CAPITAL
        daily_ret  = (end_value - prev_value) / prev_value * 100 if prev_value > 0 else 0.0
        cum_ret    = (end_value - STARTING_CAPITAL) / STARTING_CAPITAL * 100

        daily_log.append({
            'date': str(today)[:10],
            'portfolio_value': round(end_value, 2),
            'daily_return_pct': round(daily_ret, 4),
            'cumulative_return_pct': round(cum_ret, 4),
            'open_positions': len(positions),
            'cash': round(cash, 2),
        })

    # Close any remaining positions at last available price (END_OF_PERIOD)
    for sym, pos in positions.items():
        if sym in data and not data[sym].empty:
            last_row = data[sym].iloc[-1]
            exit_price = _safe(last_row.get('close', pos['entry_price']))
            qty = pos['qty']
            gross_pnl = (exit_price - pos['entry_price']) * qty
            net_pnl = gross_pnl - exit_price * qty * (COMMISSION + SLIPPAGE)
            tax = max(0, net_pnl * TAX_RATE) if net_pnl > 0 else 0
            net_pnl -= tax
            trade_id += 1
            trades.append({
                "trade_id": trade_id,
                "symbol": sym,
                "entry_date": str(pos['entry_date'])[:10],
                "exit_date": str(all_dates[-1])[:10] if all_dates else "N/A",
                "entry_price": round(pos['entry_price'], 4),
                "exit_price": round(exit_price, 4),
                "stop_loss": round(pos['stop_loss'], 4),
                "take_profit": round(pos['take_profit'], 4),
                "qty": qty,
                "pnl_dollars": round(net_pnl, 2),
                "pnl_pct": round((exit_price - pos['entry_price']) / pos['entry_price'] * 100, 3),
                "exit_reason": "END_OF_PERIOD",
                "template_id": pos['template_id'],
                "confidence_score": pos['confidence_score'],
                "hold_duration_days": (all_dates[-1] - pd.Timestamp(pos['entry_date'])).days if all_dates else 0,
                "max_drawdown_pct": 0.0,
                "stock_state": pos['stock_state'],
                "entry_indicators": pos['entry_indicators'],
                "exit_indicators": _row_to_indicators(last_row),
                "kinetic_phases_visited": pos.get('phases_visited', [1]),
                "portfolio_value_at_entry": round(pos['portfolio_value_at_entry'], 2),
                "portfolio_value_at_exit": round(exit_price * qty, 2),
            })

    logger.info(f"Backtest complete: {len(trades)} trades, {len(daily_log)} trading days")
    return trades, daily_log


# Reference TAX_RATE from config
from backtest.config import TAX_RATE


def _row_to_indicators(row) -> dict:
    """Convert a DataFrame row to a serializable indicators dict."""
    if row is None:
        return {}
    try:
        if hasattr(row, 'to_dict'):
            d = row.to_dict()
        else:
            d = dict(row)
        # Convert numpy types to Python native
        return {k: float(v) if hasattr(v, 'item') else v
                for k, v in d.items()
                if not (isinstance(v, float) and math.isnan(v))}
    except Exception:
        return {}


def calc_win_rate(trades: list) -> float:
    """Return win rate as a percentage (0-100)."""
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t['pnl_dollars'] > 0)
    return round(wins / len(trades) * 100, 2)


def template_val_wr(template: dict, trades: list) -> float:
    """Return win rate for a specific template from a trades list."""
    template_trades = [t for t in trades if t['template_id'] == template['id']]
    return calc_win_rate(template_trades)
