"""
Missing Stock Finder Utility
============================

This script is a diagnostic and recovery tool for the main data generation pipeline.
Its purpose is to identify which stocks were expected to be processed but do not have
a corresponding output Parquet file in the features directory.

This is useful for finding stocks that may have failed during a parallel processing
run due to transient network errors, API limits, or other issues.

How it Works:
-------------
1.  **Reads Expected Tickers**: It reads the master lists of training and testing
    tickers from their respective `.txt` files in the `logs/` directory to build a
    complete set of all stocks that *should* have been processed.
2.  **Scans for Processed Tickers**: It recursively scans all subdirectories
    within the `models/NASDAQ-training set/features/` directory, looking for any
    `.parquet` files. It extracts the stock symbol from each filename to build a
    set of stocks that were *actually* processed.
3.  **Identifies Missing Tickers**: It calculates the difference between the
    "expected" set and the "processed" set to find the missing stocks.
4.  **Generates Rerun File**: It saves the list of missing tickers as a single,
    comma-separated line into a new file named `manual_rerun.txt`.

Usage:
------
Simply run the script from the command line after a data generation run:

    python find_missing_stocks.py

If any stocks are found to be missing,

import os
import glob

# --- Configuration ---
TRAIN_LIST_FILE = "logs/nasdaq_train_comprehensive.txt"
TEST_LIST_FILE = "logs/nasdaq_test_comprehensive.txt"
FEATURES_BASE_DIR = "models/NASDAQ-training set/features"
OUTPUT_FILE = "manual_rerun.txt"

def find_missing_stocks():
    # 1. Get the full list of expected tickers
    all_expected_tickers = set()
    with open(TRAIN_LIST_FILE, 'r') as f:
        all_expected_tickers.update(line.strip() for line in f if line.strip())
    with open(TEST_LIST_FILE, 'r') as f:
        all_expected_tickers.update(line.strip() for line in f if line.strip())
    print(f"Found {len(all_expected_tickers)} total expected tickers.")

    # 2. Get the set of successfully processed tickers
    processed_tickers = set()
    # Note: Using glob with recursive=True to find all parquet files
    search_pattern = os.path.join(FEATURES_BASE_DIR, '**', '*.parquet')
    for file_path in glob.glob(search_pattern, recursive=True):
        # Extract symbol from filename like 'AAPL_daily_context.parquet'
        symbol = os.path.basename(file_path).split('_')[0]
        processed_tickers.add(symbol)
    print(f"Found {len(processed_tickers)} successfully processed tickers.")

    # 3. Find the difference
    missing_tickers = all_expected_tickers - processed_tickers
    print(f"Identified {len(missing_tickers)} missing tickers.")

    # 4. Save the missing tickers to the output file
    if missing_tickers:
        with open(OUTPUT_FILE, 'w') as f:
            f.write(','.join(sorted(list(missing_tickers))))
        print(f"✅ Success! Saved {len(missing_tickers)} missing tickers to '{OUTPUT_FILE}'.")
        print(f"You can now run the main script with: python Create_parquet_file_NASDAQ.py --ticker-file {OUTPUT_FILE}")
    else:
        print("✅ No missing tickers found. All files appear to be processed.")


if __name__ == "__main__":
    find_missing_stocks()