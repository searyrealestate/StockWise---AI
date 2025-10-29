"""
Intraday Data Merger with Daily Context
=========================================

This script is a critical post-processing step in the data generation pipeline.
Its purpose is to take the pre-calculated daily feature files (named `_daily_context.parquet`)
and enrich high-frequency intraday data with this daily-level context.

By merging these two datasets, each 15-minute data point is annotated with features
that describe the broader daily trend, volatility, and market conditions. This provides
the AI models with essential context that isn't available from looking at intraday
data alone.

How it Works:
-------------
1.  **Scans for Work**: The script scans all strategy subdirectories (e.g.,
    `dynamic_profit`, `2per_profit`) in both the training and testing sets to find
    all temporary `_daily_context.parquet` files.
2.  **Parallel Processing**: It uses a `ThreadPoolExecutor` to process multiple
    stocks in parallel, significantly speeding up the entire operation.
3.  **Downloads Intraday Data**: For each stock, it downloads up to 730 days
    of 15-minute intraday data from Yahoo Finance.
4.  **Merges Datasets**: It merges the daily context features into the 15-minute
    DataFrame, matching them by date.
5.  **Forward Fills Context**: The daily features are forward-filled (`ffill`) to
    ensure that every 15-minute bar throughout a given day has the same daily
    context information.
6.  **Saves Final Output**: The final, merged DataFrame is saved as the definitive
    `_features.parquet` file, which is the direct input for the model trainer.
7.  **Cleans Up**: After successfully creating the final features file, it
    deletes the temporary `_daily_context.parquet` file.

Usage:
------
The script is designed to be run after the main `Create_parquet_file_NASDAQ.py`
pipeline has completed.

    python merge_intraday_with_context.py
"""


# merge_intraday_with_context.py (Optimized & Corrected Version)

import pandas as pd
import yfinance as yf
import os
import glob
from tqdm import tqdm
import logging
import concurrent.futures

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("IntradayMerger")

# --- Configuration ---
BASE_FEATURES_DIR = "models/NASDAQ-training set/features"
BASE_TEST_FEATURES_DIR = "models/NASDAQ-testing set/features"
MAX_WORKERS = 10 # Number of parallel download threads

def process_single_merge(context_file_path: str):
    """
    Processes a single context file: downloads intraday, merges, and saves.
    This function is designed to be run in a separate thread.
    """
    try:
        symbol = os.path.basename(context_file_path).replace("_daily_context.parquet", "")

        # 1. Load the daily context data
        daily_df = pd.read_parquet(context_file_path)
        daily_df['date_only'] = daily_df.index.date

        # 2. Download up to 730 days of 15-minute data from yfinance
        intraday_df = yf.download(
            symbol,
            period="730d", # Increased from 60d
            interval="15m",
            auto_adjust=True,
            progress=False
        )
        if intraday_df.empty:
            logger.warning(f"No 15-min data found for {symbol}. Skipping.")
            os.remove(context_file_path) # Clean up context file
            return

        # Flatten multi-level columns if present (from yfinance)
        if isinstance(intraday_df.columns, pd.MultiIndex):
            intraday_df.columns = intraday_df.columns.droplevel(1)

        intraday_df.columns = [col.lower() for col in intraday_df.columns]
        intraday_df['date_only'] = intraday_df.index.date

        # 3. Merge intraday with daily context features
        merged_df = pd.merge(
            intraday_df.reset_index(),
            daily_df,
            on='date_only',
            how='left',
            suffixes=('', '_daily')
        )

        # 4. Clean up and set index
        merged_df.set_index('Datetime', inplace=True)
        cols_to_drop = [col for col in merged_df.columns if '_daily' in col or col == 'date_only']
        merged_df.drop(columns=cols_to_drop, inplace=True)

        # Forward-fill missing data from weekends/holidays and drop any remaining NaNs
        merged_df.ffill(inplace=True)
        merged_df.dropna(inplace=True)

        # 5. Save the final, merged feature file with the correct name
        output_path = context_file_path.replace("_daily_context.parquet", "_features.parquet")
        merged_df.to_parquet(output_path)

        # 6. Clean up by deleting the temporary context file
        os.remove(context_file_path)

    except Exception as e:
        logger.error(f"Failed to merge data for {context_file_path}: {e}")

def merge_data_for_strategy(strategy_dir: str):
    """
    Finds all daily_context files and processes them in parallel.
    """
    context_files = glob.glob(os.path.join(strategy_dir, "*_daily_context.parquet"))
    if not context_files:
        logger.warning(f"No context files found in {strategy_dir}. Skipping.")
        return

    logger.info(f"Processing {len(context_files)} symbols in parallel for: {os.path.basename(strategy_dir)}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        list(tqdm(executor.map(process_single_merge, context_files), total=len(context_files), desc=f"Merging {os.path.basename(strategy_dir)}"))


if __name__ == "__main__":
    all_strategy_dirs = [d for d in glob.glob(os.path.join(BASE_FEATURES_DIR, '*')) if os.path.isdir(d)]
    all_test_strategy_dirs = [d for d in glob.glob(os.path.join(BASE_TEST_FEATURES_DIR, '*')) if os.path.isdir(d)]
    all_dirs = all_strategy_dirs + all_test_strategy_dirs

    for directory in all_dirs:
        merge_data_for_strategy(directory)

    logger.info("🎉 All intraday data merging complete!")