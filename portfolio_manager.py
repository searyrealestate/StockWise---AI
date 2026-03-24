# portfolio_manager.py

"""
StockWise Gen-12 Portfolio Manager (The Accountant)
===================================================
The Heart of Treasury Management.
Responsible for:
1. Maintaining the 'Shadow Ledger' (Paper/Live record keeping).
2. Calculating PnL, Taxes, and Commissions.
3. Managing Risk (Position Sizing & Stop Loss Tracking).
4. Ensuring we never trade more than available cash.
"""

import json
import os
import logging
from datetime import datetime
import system_config as cfg
import numpy as np
import pandas as pd
from safe_json_io import safe_json_read, safe_json_write

logger = logging.getLogger("PortfolioManager")

class PortfolioManager:
    """
    Electronic Ledger for tracking the Portfolio State.
    Handles Cash, Equity, and active Trade History.
    """
    def __init__(self):
        self.file_path = os.path.join(cfg.PROJECT_ROOT, "shadow_portfolio.json")
        self.portfolio = self._load_portfolio()
        
    def _load_portfolio(self):
        """Loads the JSON ledger or initializes a new account."""
        default = {
            "cash": cfg.RISK_CONFIG["starting_capital"],
            "equity": cfg.RISK_CONFIG["starting_capital"],
            "trades": []  # List of active trade dictionary objects
        }
        return safe_json_read(self.file_path, default=default)

    def _save_portfolio(self):
        """Persists the ledger to disk (atomic write via safe_json_io)."""
        safe_json_write(self.file_path, self.portfolio)

    def calculate_commission(self, qty):
        """
        Calculates brokerage fees based on config.
        Default: higher of $1.00 or $0.005/share.
        """
        costs = cfg.COSTS_CONFIG
        raw_comm = qty * costs["commission_per_share"]
        return max(costs["min_commission"], raw_comm)

    def apply_slippage(self, price, is_buy=True):
        """
        Simulates Real-World Execution.
        Paper Mode Only: Adds a penalty (slippage) to the execution price.
        Buys are executed higher (+0.1%), Sells are executed lower (-0.1%).
        """
        if cfg.MODE != "PAPER": return price
        
        slip_pct = cfg.COSTS_CONFIG["slippage_pct"]
        if is_buy:
            return price * (1 + slip_pct)
        else:
            return price * (1 - slip_pct)

    def record_trade(self, symbol, action, price, qty, strategy, stop_loss, target):
        """
        Executes an ORDER and logs it as an OPEN TRADE.
        Deducts cash including estimated costs.
        """
        # 1. Simulate Slippage (Worse execution price)
        exec_price = self.apply_slippage(price, is_buy=True)
        
        # 2. Calculate Commission
        comm = self.calculate_commission(qty)
        
        total_cost = (exec_price * qty) + comm
        
        # 3. Solvency Check
        if self.portfolio["cash"] < total_cost:
            logger.warning(f"Insufficient Funds for {symbol}")
            return False
            
        # 4. Update Ledger
        self.portfolio["cash"] -= total_cost
        
        trade = {
            "symbol": symbol,
            "entry_price": exec_price,
            "qty": qty,
            "strategy": strategy,
            "stop_loss": stop_loss,
            "target": target,
            "entry_time": datetime.now().isoformat(),
            "commission_paid": comm,
            "status": "OPEN"
        }
        
        self.portfolio["trades"].append(trade)
        self._save_portfolio()
        logger.info(f"TRADE OPEN: {qty} {symbol} @ {exec_price:.2f} (Comm: ${comm:.2f})")
        return True

    def close_trade(self, symbol, price, reason):
        """
        Closes an active position and realizes Profit/Loss.
        Handles Tax calculation on gains.
        """
        for trade in self.portfolio["trades"]:
            if trade["symbol"] == symbol and trade["status"] == "OPEN":
                # 1. Apply Slippage (Sell lower)
                exit_price = self.apply_slippage(price, is_buy=False)
                
                # 2. Calculate Exit Commission
                comm = self.calculate_commission(trade["qty"])
                
                # 3. Gross PnL (Proceeds - Cost Basis)
                gross_proceeds = (exit_price * trade["qty"])
                cost_basis = (trade["entry_price"] * trade["qty"])
                gross_pnl = gross_proceeds - cost_basis
                
                # 4. Tax Estimation (Capital Gains)
                # Tax is only applied on PROFITS.
                tax = 0.0
                if gross_pnl > 0:
                    tax = gross_pnl * cfg.COSTS_CONFIG["tax_rate"]
                
                # 5. Net PnL Calculation
                # Net = Gross - EntryFees - ExitFees - Tax
                net_pnl = gross_pnl - trade["commission_paid"] - comm - tax
                
                # 6. Update Portfolio (Return cash to pool)
                self.portfolio["cash"] += (gross_proceeds - comm - tax) 
                
                # 7. Update Trade Record
                trade["status"] = "CLOSED"
                trade["exit_price"] = exit_price
                trade["exit_time"] = datetime.now().isoformat()
                trade["reason"] = reason
                trade["net_pnl"] = net_pnl
                trade["tax_paid"] = tax
                trade["exit_commission"] = comm
                
                self._save_portfolio()
                logger.info(f"TRADE CLOSED: {symbol} | Net PnL: ${net_pnl:.2f} (Tax: ${tax:.2f})")
                return True
        
        return False

    def get_active_position(self, symbol):
        """Helper to find an active trade for a symbol."""
        for trade in self.portfolio["trades"]:
            if trade["symbol"] == symbol and trade["status"] == "OPEN":
                return trade
        return None

class RiskManager:
    """
    Sub-system dealing with Risk Math.
    Calculates Position Sizing (Shares to buy) and validates Stop Losses.
    """
    def __init__(self, portfolio_value, global_risk_pct=1.0):
        """
        :param portfolio_value: Total value of the trading account (Cash + Equity)
        :param global_risk_pct: Max % of total portfolio to loose on a single trade (Risk Unit).
        """
        self.portfolio_value = portfolio_value
        self.global_risk_pct = global_risk_pct
        self.max_risk_dollars_per_trade = self.portfolio_value * (self.global_risk_pct / 100.0)
        logger.info(f"RiskManager initialized. Max risk per trade: ${self.max_risk_dollars_per_trade:.2f}")

    def update_portfolio_value(self, new_value):
        """Re-calibrates risk unit as portfolio grows/shrinks (Auto-Scaling)."""
        self.portfolio_value = new_value
        self.max_risk_dollars_per_trade = self.portfolio_value * (self.global_risk_pct / 100.0)
        logger.info(
            f"Portfolio value updated to ${new_value:.2f}. New max risk: ${self.max_risk_dollars_per_trade:.2f}")

    def calculate_position_size(self, entry_price, stop_loss_price):
        """
        Fixed Fractional Risk Sizing.
        Determines quantity based on the distance to the Stop Loss.
        Formula: Qty = (Risk Amount) / (Entry - StopLoss)
        """
        if entry_price <= 0 or stop_loss_price >= entry_price:
            logger.warning(f"Invalid position size calculation: Entry ${entry_price}, SL ${stop_loss_price}")
            return 0  # Invalid parameters

        # 1. Expectancy (Risk per share)
        risk_per_share = entry_price - stop_loss_price

        # 2. Risk Calculation
        num_shares = self.max_risk_dollars_per_trade / risk_per_share

        # 3. Capital Ceiling Check
        # Ensure we don't try to buy more than we have (or leverage max)
        investment_amount = num_shares * entry_price
        if investment_amount > self.portfolio_value:
            num_shares = self.portfolio_value / entry_price
            logger.warning("Position size capped by total portfolio value.")

        logger.info(f"Position size calculated: {np.floor(num_shares)} shares.")
        return np.floor(num_shares)  # Always round down to whole shares

    def manage_open_position(self, current_day_data: pd.Series, position_data: dict):
        """
        Dynamic Exit Management.
        Called on every bar to check:
        1. Structural Failures (Price < SMA 150)
        2. Static Stop Loss
        3. Trailing Stops (Volatility Based)
        """
        try:
            current_low = current_day_data['low']
            current_close = current_day_data['close']

            # --- 1. Structural Stop-Loss (Trend Change) ---
            # If price closes below the 150-day Long Term moving average, exit immediately.
            if 'sma_150' in current_day_data:
                current_sma_150 = current_day_data['sma_150']
                if current_close < current_sma_150:
                    logger.info(
                        f"EXIT_SIGNAL: Structural stop hit. Close ({current_close:.2f}) < 150-day SMA ({current_sma_150:.2f}).")
                    return "EXIT_SIGNAL", position_data

            # --- 2. Stop-Loss Checks ---

            # 2a. Static Stop Check (Initial SL)
            # If not using trailing stop, just check the fixed level
            if not position_data.get('use_trailing_stop', False):
                if current_low <= position_data['current_stop_loss']:
                    logger.info(f"EXIT_SIGNAL: Static stop-loss hit at ${position_data['current_stop_loss']}.")
                    return "EXIT_SIGNAL", position_data
                return "HOLD", position_data

            # 2b. Trailing Stop Logic (ATR Based)
            # Moves stop loss UP as price rises, locking in profits.
            current_high = current_day_data['high']
            atr_value = current_day_data.get('atr_14', 0)
            
            if atr_value == 0:
                logger.warning("ATR value is 0, trailing stop will not work correctly.")
                return "HOLD", position_data 

            atr_mult = position_data.get('atr_multiplier', 2.5)

            # Calculate where the stop SHOULD be based on today's high (Standard Chandelier Exit logic)
            new_potential_stop = current_high - (atr_value * atr_mult)
            
            # Ratchet Logic: Stop can ONLY move UP, never down.
            new_stop_loss = max(position_data['current_stop_loss'], new_potential_stop)

            # Check for Breach: Did price hit our (potentially updated) stop *today*?
            # Note: We check against current_low.
            if current_low <= new_stop_loss:
                logger.info(f"EXIT_SIGNAL: Trailing stop-loss hit at ${new_stop_loss:.2f}.")
                position_data['current_stop_loss'] = new_stop_loss  # Log final stop
                return "EXIT_SIGNAL", position_data

            # Update the stored stop loss if it moved up
            if new_stop_loss > position_data['current_stop_loss']:
                logger.debug(f"Trailing stop raised to ${new_stop_loss:.2f}")
                position_data['current_stop_loss'] = new_stop_loss

            return "HOLD", position_data

        except KeyError as e:
            logger.error(f"Missing expected data in current_day_data: {e}. Holding position as failsafe.",
                         exc_info=True)
            return "HOLD", position_data
        except Exception as e:
            logger.error(f"Error in manage_open_position: {e}. Holding position as failsafe.", exc_info=True)
            return "HOLD", position_data
