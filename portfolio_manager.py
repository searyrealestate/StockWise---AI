
"""
StockWise Portfolio Manager
===========================
Manages Real vs. Shadow portfolios.
Tracks system performance (Shadow) separately from user performance (Real).
"""

import json
import os
import logging
from datetime import datetime
import system_config as cfg

logger = logging.getLogger("PortfolioManager")

class PortfolioManager:
    def __init__(self):
        self.shadow_file = "shadow_portfolio.json"
        self.user_file = "user_portfolio.json"
        
        # Load Ledgers
        self.shadow_portfolio = self._load_json(self.shadow_file)
        self.user_portfolio = self._load_json(self.user_file)
        
        # Ensure 'trades' list exists
        if "trades" not in self.shadow_portfolio: self.shadow_portfolio["trades"] = []
        if "trades" not in self.user_portfolio: self.user_portfolio["trades"] = []

    def _load_json(self, filepath):
        """Load JSON file or return empty dict if not found."""
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load {filepath}: {e}")
                return {}
        return {}

    def _save_json(self, filepath, data):
        """Save data to JSON file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save {filepath}: {e}")

    def add_shadow_trade(self, ticker, entry_price, stop_loss, target_price, timestamp=None):
        """
        Record a virtual trade for system tracking.
        Called automatically by the engine on signal.
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
            
        trade = {
            "id": f"SHADOW_{len(self.shadow_portfolio['trades']) + 1}",
            "ticker": ticker,
            "entry_price": float(entry_price),
            "stop_loss": float(stop_loss),
            "target_price": float(target_price),
            "timestamp": str(timestamp),
            "status": "OPEN",
            "pnl": 0.0,
            "exit_price": 0.0,
            "exit_reason": None,
            "allocation": 1000.0 # Virtual $1000 per trade
        }
        
        self.shadow_portfolio["trades"].append(trade)
        self._save_json(self.shadow_file, self.shadow_portfolio)
        logger.info(f"[Shadow] Trade Recorded: {ticker} @ {entry_price}")

    def add_user_trade(self, ticker, entry_price, qty, timestamp=None):
        """
        Record a real trade manually entered by user.
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
            
        trade = {
            "id": f"USER_{len(self.user_portfolio['trades']) + 1}",
            "ticker": ticker,
            "entry_price": float(entry_price),
            "qty": float(qty),
            "timestamp": str(timestamp),
            "status": "OPEN",
            "pnl": 0.0,
            "exit_price": 0.0,
            "exit_reason": None
        }
        
        self.user_portfolio["trades"].append(trade)
        self._save_json(self.user_file, self.user_portfolio)
        logger.info(f"[User] Trade Recorded: {ticker} @ {entry_price} x {qty}")
        return trade["id"]

    def close_user_trade(self, ticker, exit_price, qty, timestamp=None):
        """
        Close a real trade manually entered by user.
        Strategies: FIFO (First In First Out) if multiple open.
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
            
        exit_price = float(exit_price)
        qty_to_close = float(qty)
        closed_ids = []
        
        # FIFO : Find oldest OPEN trade for this ticker
        # We iterate a copy to safely modify list potentially (appending new records)
        # Actually we iterate direct list but appended items go to end so it's fine for FIFO if we stop when satisfied
        
        # We need to index them to update in place
        for i, trade in enumerate(self.user_portfolio["trades"]):
            if trade["status"] == "OPEN" and trade["ticker"] == ticker:
                if qty_to_close <= 0:
                    break
                    
                available_qty = trade["qty"]
                
                # --- SCENARIO A: PARTIAL CLOSE ---
                if qty_to_close < available_qty:
                    # 1. Update Open Position (Reduce Qty)
                    trade["qty"] = available_qty - qty_to_close
                    
                    # 2. Create New "Closed" Record for the portion sold
                    closed_part = trade.copy()
                    closed_part["id"] = f"{trade['id']}_CL_{int(datetime.now().timestamp())}"
                    closed_part["qty"] = qty_to_close
                    closed_part["status"] = "CLOSED"
                    closed_part["exit_price"] = exit_price
                    closed_part["exit_reason"] = "USER_SELL"
                    closed_part["exit_timestamp"] = timestamp
                    closed_part["pnl"] = (exit_price - trade["entry_price"]) * qty_to_close
                    
                    # Add to ledger
                    self.user_portfolio["trades"].append(closed_part)
                    
                    closed_ids.append(closed_part["id"])
                    qty_to_close = 0 # Done
                    
                # --- SCENARIO B: FULL CLOSE ---
                else:         
                    trade["status"] = "CLOSED"
                    trade["exit_price"] = exit_price
                    trade["exit_reason"] = "USER_SELL"
                    trade["exit_timestamp"] = timestamp
                    
                    # PnL
                    pnl = (exit_price - trade["entry_price"]) * trade["qty"]
                    trade["pnl"] = pnl
                    
                    closed_ids.append(trade["id"])
                    qty_to_close -= available_qty
                
        self._save_json(self.user_file, self.user_portfolio)
        if closed_ids:
            logger.info(f"[User] Trade Closed: {ticker} @ {exit_price}. IDs: {closed_ids}")
            return closed_ids
        else:
            return None

    def get_active_positions(self, type='BOTH'):
        """Return list of open trades."""
        active = []
        
        if type in ['BOTH', 'SHADOW']:
            for t in self.shadow_portfolio.get("trades", []):
                if t["status"] == "OPEN":
                    t["type"] = "SHADOW"
                    active.append(t)
                    
        if type in ['BOTH', 'USER']:
            for t in self.user_portfolio.get("trades", []):
                if t["status"] == "OPEN":
                    t["type"] = "USER"
                    active.append(t)
                    
        return active

    def get_user_position_summary(self):
        """Return a string summary of all open user positions."""
        active = []
        for t in self.user_portfolio.get("trades", []):
            if t["status"] == "OPEN":
                active.append(t)
        
        if not active:
            return "No Active Positions."
            
        lines = ["📋 **Active Positions**"]
        for t in active:
            lines.append(f"• {t['ticker']}: {t['entry_price']} x {t['qty']} (ID: {t['id']})")
            
        return "\n".join(lines)

    def calculate_user_pnl(self, timeframe='ALL'):
        """
        Calculate Realized PnL for USER portfolio.
        timeframe: 'TODAY', 'MONTH', 'ALL'
        """
        total_pnl = 0.0
        count = 0
        now = datetime.now()
        
        for t in self.user_portfolio.get("trades", []):
            if t["status"] == "CLOSED" and t.get("pnl") is not None:
                # Timestamps check
                ts_str = t.get("exit_timestamp")
                if not ts_str: continue
                
                # Careful parsing: Isoformat might vary slightly
                try:
                    exit_dt = datetime.fromisoformat(ts_str)
                except:
                    continue
                
                include = False
                if timeframe == 'ALL':
                    include = True
                elif timeframe == 'TODAY':
                    if exit_dt.date() == now.date():
                        include = True
                elif timeframe == 'MONTH':
                    if exit_dt.year == now.year and exit_dt.month == now.month:
                        include = True
                        
                if include:
                    total_pnl += t["pnl"]
                    count += 1
                    
        return total_pnl, count

    def update_position_status(self, ticker, current_price):
        """
        Check shadow trades for Exit triggers.
        Returns list of closed trades details.
        """
        closed_trades = []
        
        for trade in self.shadow_portfolio.get("trades", []):
            if trade["status"] == "OPEN" and trade["ticker"] == ticker:
                
                # Check Target
                # Assuming LONG only for now
                if current_price >= trade["target_price"]:
                    trade["status"] = "CLOSED"
                    trade["exit_price"] = current_price
                    trade["exit_reason"] = "TARGET"
                    # Simple PnL Calc: (Exit - Entry) / Entry * Allocation
                    pct_change = (current_price - trade["entry_price"]) / trade["entry_price"]
                    trade["pnl"] = pct_change * trade["allocation"]
                    closed_trades.append(trade)
                    
                # Check Stop
                elif current_price <= trade["stop_loss"]:
                    trade["status"] = "CLOSED"
                    trade["exit_price"] = current_price
                    trade["exit_reason"] = "STOP"
                    pct_change = (current_price - trade["entry_price"]) / trade["entry_price"]
                    trade["pnl"] = pct_change * trade["allocation"]
                    closed_trades.append(trade)
                    
        if closed_trades:
            self._save_json(self.shadow_file, self.shadow_portfolio)
            
        return closed_trades
