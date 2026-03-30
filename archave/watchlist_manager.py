#watchlist_manager.py

import json
import os
import logging
import system_config as cfg
from portfolio_manager import PortfolioManager

logger = logging.getLogger("WatchlistManager")

class WatchlistManager_old:
    def __init__(self):
        self.filepath = os.path.join(cfg.DB_DIR, "dynamic_watchlist.json")
        self.pm = PortfolioManager()
        # The Seed List: Top 10 S&P 500 by weight (Start here)
        self.seed_list = [
            "NVDA", "MSFT", "AAPL", "AMZN", "META", 
            "GOOGL", "TSLA", "BRK.B", "LLY", "AVGO"
        ]

    def get_active_watchlist(self):
        """Returns the current list of 'learned' stocks."""
        if not os.path.exists(self.filepath):
            self._initialize_db()
        
        try:
            with open(self.filepath, 'r') as f:
                data = json.load(f)
                return data.get("tickers", self.seed_list)
        except Exception as e:
            logger.error(f"Failed to load watchlist: {e}")
            return self.seed_list

    def add_new_candidates(self, new_tickers):
        """Adds new stocks to the database, ensuring no duplicates."""
        current = set(self.get_active_watchlist())
        added = []
        
        for t in new_tickers:
            if t not in current:
                current.add(t)
                added.append(t)
        
        if added:
            self._save(list(current))
            logger.info(f"✅ Watchlist expanded! Added {len(added)} new stocks: {added}")
        else:
            logger.info("No new stocks added (all candidates already exist).")
            
        return added

    def prune_watchlist(self):
        """
        Optional: Remove stocks to keep the list size manageable (e.g., max 100).
        CRITICAL: Never remove a stock that is currently held in the Portfolio.
        """
        current_list = self.get_active_watchlist()
        if len(current_list) <= 100: return # Only prune if we grow too big

        # 1. Get Protected Stocks (Active Positions)
        # We need to peek at the PortfolioManager's data
        try:
            with open(self.pm.file_path, 'r') as f:
                port_data = json.load(f)
                held_positions = {t['symbol'] for t in port_data.get('trades', []) if t['status'] == 'OPEN'}
        except:
            held_positions = set()

        # 2. Prune logic (FIFO - First In First Out, but protect holdings)
        # This is a simple implementation. You could add 'performance based' pruning later.
        keep_list = []
        removed_count = 0
        
        # Keep the most recent 100, plus any older ones that are held
        # Assuming the list is somewhat ordered by insertion (if we append new ones)
        # We'll reverse it to keep new ones, then check holdings
        
        # Simpler: Just slice the last 100, but add back holdings if they got cut
        new_core = current_list[-100:]
        for old_stock in current_list[:-100]:
            if old_stock in held_positions:
                new_core.insert(0, old_stock) # Keep it
            else:
                removed_count += 1
        
        if removed_count > 0:
            self._save(new_core)
            logger.info(f"✂️ Pruned {removed_count} old stocks from watchlist.")

    def _initialize_db(self):
        """Creates the initial JSON file."""
        self._save(self.seed_list)
        logger.info("Initialized Dynamic Watchlist with Seed List.")

    def _save(self, tickers):
        with open(self.filepath, 'w') as f:
            json.dump({"tickers": tickers, "last_updated": str(os.path.getmtime(self.filepath) if os.path.exists(self.filepath) else 0)}, f, indent=4)
