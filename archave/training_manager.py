# training_manager.py

"""
Training Manager
================
Orchestrates the "Just-in-Time" training workflow for the Dynamic Watchlist.
When the Scanner finds a new candidate, this manager:
1. Adds it to the Watchlist (via WatchlistManager).
2. Cleanses the 'data/gold' directory (removes non-watchlist files).
3. Fetches & Processes the new stock using DataEngineer.
4. Triggers ModelTrainer to retrain the AI on the updated universe.
"""

import os
import glob
import logging
from concurrent.futures import ThreadPoolExecutor

import system_config as cfg
from watchlist_manager import WatchlistManager
from data_engineer import DataEngineer
# Note: ModelTrainer is in train_model.py, need to import carefully or use subprocess
# We can import the class directly
from train_model import ModelTrainer 

logger = logging.getLogger("TrainingManager")

class TrainingManager:
    def __init__(self):
        self.wm = WatchlistManager()
        self.engineer = DataEngineer()
        self.trainer = ModelTrainer()
        self.gold_dir = os.path.join(cfg.PROJECT_ROOT, "data", "gold")

    def sync_gold_data(self, active_watchlist):
        """
        Ensures the data/gold directory perfectly matches the active watchlist.
        1. Removes parquet files for stocks NO LONGER in the watchlist.
        2. Returns list of stocks that are MISSING from data/gold.
        """
        # 1. Cleanup Old Data
        existing_files = glob.glob(os.path.join(self.gold_dir, "*.parquet"))
        active_files = {f"{symbol}.parquet" for symbol in active_watchlist}
        
        for f_path in existing_files:
            f_name = os.path.basename(f_path)
            if f_name not in active_files:
                try:
                    os.remove(f_path)
                    logger.info(f"🧹 Removed stale training data: {f_name}")
                except Exception as e:
                    logger.error(f"Failed to remove {f_name}: {e}")

        # 2. Identify Missing Data
        missing = []
        for symbol in active_watchlist:
            expected_path = os.path.join(self.gold_dir, f"{symbol}.parquet")
            if not os.path.exists(expected_path):
                missing.append(symbol)
                
        return missing

    def recruit_new_stock(self, symbol):
        """
        The "Talent Scout" Handler.
        Called when Scanner finds a new high-score candidate.
        """
        logger.info(f"🎓 RECRUITING NEW TALENT: {symbol}")
        
        # 1. Add to Watchlist (Persistent DB)
        # This returns the newly added list (or empty if existed)
        added = self.wm.add_new_candidates([symbol])
        if not added:
            logger.info(f"{symbol} is already in the class.")
            return

        # 2. Get Fresh Watchlist
        current_watchlist = self.wm.get_active_watchlist()
        
        # 3. Sync Data Directory (Remove old junk)
        missing_stocks = self.sync_gold_data(current_watchlist)
        
        # 4. Engineer Data for the New Recruit (and any other missing ones)
        if missing_stocks:
            logger.info(f"🛠️ Engineering Golden Data for: {missing_stocks}")
            # Use DataEngineer to build parquet
            with ThreadPoolExecutor(max_workers=5) as executor:
                executor.map(self.engineer.process_ticker, missing_stocks)
        
        # 5. Just-In-Time Training
        # Now that we have the new data, we must retrain the brain.
        logger.info("🧠 Triggering Just-In-Time AI Retraining...")
        self.trainer.train()
        logger.info(f"✅ {symbol} successfully integrated into the AI Model.")

if __name__ == "__main__":
    # Test Run
    tm = TrainingManager()
    # Mock recruitment
    # tm.recruit_new_stock("NVDA") 
