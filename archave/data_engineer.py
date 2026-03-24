# data_engineer.py

"""
Data Engineer - StockWise Gen-7
===============================
Responsible for creating the "Golden Dataset".
Fetches raw data, applies Feature Engine + Ground Truth, and saves to optimized Parquet files.
This enables high-speed LSTM training without redundant API calls.
"""

import os
import logging
import pandas as pd
import glob
from concurrent.futures import ThreadPoolExecutor

import system_config as cfg
from data_source_manager import DataSourceManager
from feature_engine import RobustFeatureCalculator, calculate_ground_truth

# Setup Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataEngineer")

GOLD_DIR = os.path.join(cfg.PROJECT_ROOT, "data", "gold")
os.makedirs(GOLD_DIR, exist_ok=True)

class DataEngineer:
    def __init__(self):
        self.dm = DataSourceManager()
        self.calc = RobustFeatureCalculator()
        
    def process_ticker(self, symbol):
        """
        Full Pipeline for a single ticker.
        Fetch -> Feature Eng -> Ground Truth -> Parquet
        """
        try:
            logger.info(f"Processing {symbol}...")
            
            # 1. Fetch 2 Years of Hourly Data
            # Gen-7 requires deep history for LSTM context
            days_back = 730 
            df = self.dm.get_stock_data(symbol, days_back=days_back, interval="1h", source="AUTO")
            
            if df.empty or len(df) < 500:
                logger.warning(f"Skipping {symbol}: Insufficient data ({len(df)} bars)")
                return
            
            # 2. Apply Features (Gen-7 Signal Stack)
            # Includes WaveTrend, Keltner, VSA, etc.
            df = self.calc.calculate_features(df)
            
            # 3. Apply Ground Truth (The Oracle)
            # Look ahead 15 days (approx 100 hourly bars?) 
            # 15 days * 7 trading hours = 105 bars. Let's say 100.
            calculate_ground_truth(df, lookahead=100) 
            
            # 4. Save to Parquet
            # Parquet is columnar, compressed, and much faster for ML loading
            filename = os.path.join(GOLD_DIR, f"{symbol}.parquet")
            df.to_parquet(filename)
            
            logger.info(f"✅ Saved {symbol} to {filename} ({len(df)} rows)")
            
        except Exception as e:
            logger.error(f"Failed to process {symbol}: {e}")

    def build_golden_dataset(self):
        """
        Runs the pipeline for the entire watchlist.
        """
        logger.info(f"Starting Golden Dataset Build for {len(cfg.WATCHLIST)} symbols...")
        logger.info(f"Target Directory: {GOLD_DIR}")
        
        # Use ThreadPool for speed (I/O bound)
        with ThreadPoolExecutor(max_workers=5) as executor:
            executor.map(self.process_ticker, cfg.WATCHLIST)
            
        logger.info("Golden Dataset Build Complete.")

if __name__ == "__main__":
    engineer = DataEngineer()
    engineer.build_golden_dataset()
