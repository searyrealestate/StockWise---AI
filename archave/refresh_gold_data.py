# refresh_gold_data.py

import os
import glob
import logging
from training_manager import TrainingManager
from watchlist_manager import WatchlistManager
import system_config as cfg

# Configure basic logging to see output in console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RefreshGold")

def refresh():
    print("--- Starting Gold Data Refresh ---")
    tm = TrainingManager()
    wm = WatchlistManager()
    watchlist = wm.get_active_watchlist()
    
    print(f"Active Watchlist ({len(watchlist)}): {watchlist}")
    
    # 1. Delete all existing gold data to force regeneration with new features
    # Be careful only to delete .parquet files in data/gold
    gold_pattern = os.path.join(cfg.PROJECT_ROOT, "data", "gold", "*.parquet")
    files = glob.glob(gold_pattern)
    print(f"Found {len(files)} existing parquet files. Deleting...")
    
    for f in files:
        try:
            os.remove(f)
        except Exception as e:
            print(f"Error deleting {f}: {e}")
            
    # 2. Sync (will find all missing)
    missing = tm.sync_gold_data(watchlist)
    print(f"Stocks to process: {missing}")
    
    if not missing:
        print("No stocks to process? Something is wrong.")
        return

    # 3. Process
    # We can use the thread pool logic from TrainingManager manually, or just loop
    from concurrent.futures import ThreadPoolExecutor
    
    print("--- Engineering Features (including SuperTrend) ---")
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(tm.engineer.process_ticker, missing))
        
    print("--- Data Engineering Complete ---")
    
    # 4. Train
    print("--- Starting Model Training ---")
    try:
        tm.trainer.train()
        print("SUCCESS: Model Training Complete.")
    except Exception as e:
        print(f"FAILURE: Model Training Failed: {e}")

if __name__ == "__main__":
    refresh()
