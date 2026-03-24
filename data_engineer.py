# data_engineer.py

"""
🛠️ Data Engineer - StockWise Gen-12
==================================
The Architect of the Golden Dataset.

Responsibilities:
1.  Fetch Deep History (2 Years) for all Watchlist stocks.
2.  Apply Feature Engineering (Indicators).
3.  Apply Master Patterns (63 Functions).
4.  Label Data (Ground Truth).
5.  Save as optimized Parquet files for the AI Trainer.

Usage:
    python data_engineer.py
"""

import pandas as pd
import numpy as np
import os
import logging
import time
from datetime import datetime, timedelta

# Import StockWise Modules
import system_config as cfg
from data_source_manager import DataSourceManager
from feature_engine import RobustFeatureCalculator, MasterPatternLib, calculate_ground_truth, TechnicalAnalyzer

# Configure Logging
logger = logging.getLogger("DataEngineer")
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

def run_data_pipeline():
    """
    Main Execution Pipeline.
    """
    logger.info("Starting Data Engineering Pipeline...")
    
    # 1. Initialize Data Source Manager
    # We allow fallbacks to get the best possible data
    dsm = DataSourceManager(use_ibkr=True, allow_fallback=True)
    
    # 2. Load Watchlist
    watchlist = cfg.WATCHLIST
    logger.info(f"Watchlist Loaded: {len(watchlist)} symbols")
    
    success_count = 0
    
    # 3. Iterate and Process
    for px, symbol in enumerate(watchlist):
        try:
            logger.info(f"\n[{px+1}/{len(watchlist)}] Processing {symbol}...")
            
            # A. Fetch Data (2 Years = ~730 Days)
            # We want DAILY data for the core model
            df = dsm.get_stock_data(symbol, days_back=730, interval='1d', source='AUTO')
            
            if df.empty or len(df) < 200:
                logger.warning(f"Insufficient data for {symbol} ({len(df)} rows). Skipping.")
                continue
                
            # B. Feature Engineering (Technical Indicators)
            logger.info(f"Calculating Indicators...")
            calc = RobustFeatureCalculator()
            df = calc.calculate_features(df)
            
            # C. Master Patterns (Geometric & Candlesticks)
            logger.info(f"Identifying Patterns...")
            
            # We need to run patterns row-by-row or using the library's vectorized methods where available
            # The MasterPatternLib was designed for single-slice analysis in live trading, 
            # but we can apply it to the whole dataframe efficiently.
            
            # 1. Add Smart Candlestick Patterns (Vectorized)
            df = calc.add_candlestick_patterns(df)
            df = calc.add_advanced_patterns(df)
            
            # 2. Add Master Scores (Looping required for complex patterns if not vectorized)
            # For efficiency in training generation, we might rely on the vectorized signals above.
            # But let's add a placeholder for the aggregate score if needed.
            df['master_score'] = 0 # Placeholder for now, or implement rolling calculation
            
            # D. Labeling (Ground Truth)
            # This is the "God Mode" step that looks into the future to create labels
            logger.info(f"Generating Labels (Ground Truth)...")
            df = calculate_ground_truth(df, lookahead=10) # 10 Days lookahead for trend
            
            # E. Clean and Save
            # Drop NaNs created by indicators/lookahead
            df.dropna(inplace=True)
            
            if len(df) > 100:
                # Ensure directory exists
                gold_dir = os.path.join(cfg.DB_DIR, "gold")
                os.makedirs(gold_dir, exist_ok=True)
                
                save_path = os.path.join(gold_dir, f"{symbol}.parquet")
                df.to_parquet(save_path)
                logger.info(f"Saved Golden Dataset: {save_path} ({len(df)} rows)")
                success_count += 1
            else:
                logger.warning(f"Data grew too small after cleaning ({len(df)} rows).")

        except Exception as e:
            logger.error(f"Failed to process {symbol}: {e}")
            import traceback
            # logger.error(traceback.format_exc())
            
    logger.info(f"\nPipeline Complete. Successfully processed {success_count}/{len(watchlist)} stocks.")

if __name__ == "__main__":
    run_data_pipeline()
