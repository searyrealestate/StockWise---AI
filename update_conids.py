"""
IBKR Contract ID (ConID) Updater
=================================

This script automates the process of finding and adding Interactive Brokers
Contract IDs (conIDs) to a CSV file containing a list of stock symbols.

It connects to a running instance of TWS or IB Gateway, reads a specified CSV
file, and iterates through each row. If a row is missing a conID, the script
uses the `DataSourceManager` to look up the conID for that stock symbol and
updates the DataFrame.

To be respectful of the TWS API's rate limits, the script pauses for one second
between each lookup request.

Finally, it saves the updated DataFrame back to the original CSV file.

Pre-conditions for running:
---------------------------
- A live, running instance of Interactive Brokers Trader Workstation (TWS) or
  Gateway must be available and configured for API connections.
- The input CSV file must contain a column named 'Symbol' (case-insensitive).

Usage:
------
The script can be run from the command line. You can optionally specify the
path to your CSV file using the --file argument.

# To run with the default file 'nasdaq_stocks_with_conids.csv'
python update_conids.py

# To specify a different file
python update_conids.py --file path/to/your/stocks.csv
"""


# update_conids.py
# how to run: python update_conids.py
#  python update_conids.py --file nasdaq_stocks.csv

import pandas as pd
import argparse
import time
import numpy as np
from data_source_manager import DataSourceManager


def update_csv_with_conids(file_path: str):
    """
    Reads a CSV of stock symbols, finds their IBKR ConID, and saves the updated CSV.
    """
    print(f"--- Starting ConID Updater for '{file_path}' ---")

    # Step 1: Connect to the DataSourceManager
    data_manager = DataSourceManager(use_ibkr=True)
    if not data_manager.connect_to_ibkr():
        print("❌ FATAL: Could not connect to IBKR. Please ensure TWS or Gateway is running and configured.")
        return

    # Step 2: Read the CSV file
    try:
        df = pd.read_csv(file_path)
        # Standardize columns to handle 'symbol' or 'Symbol'
        df.columns = [col.strip().title() for col in df.columns]
    except FileNotFoundError:
        print(f"❌ FATAL: File not found at '{file_path}'")
        return
    except Exception as e:
        print(f"❌ FATAL: Could not read CSV file. Error: {e}")
        return

    if 'Symbol' not in df.columns:
        print("❌ FATAL: CSV file must contain a 'Symbol' column.")
        return

    # Step 3: Add 'Conid' column if it doesn't exist
    if 'Conid' not in df.columns:
        df['Conid'] = np.nan

    # Step 4: Loop through the DataFrame and find missing Conids
    updated_count = 0
    for index, row in df.iterrows():
        # Check if Conid is missing (NaN or 0)
        if pd.isna(row['Conid']) or row['Conid'] == 0:
            symbol = row['Symbol']
            print(f"\nProcessing symbol: {symbol}")

            conid = data_manager.find_conid(symbol)

            if conid:
                df.at[index, 'Conid'] = conid
                updated_count += 1

            # Be respectful of the API rate limit - wait 1 second between requests
            time.sleep(1)

    # Step 5: Save the updated file
    if updated_count > 0:
        df.to_csv(file_path, index=False)
        print(f"\n--- 🎉 Finished ---")
        print(f"✅ Successfully found and updated {updated_count} Conids.")
        print(f"✅ The file '{file_path}' has been updated.")
    else:
        print(f"\n--- Finished ---")
        print("✅ No missing Conids found. Your file is already up to date.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automate fetching of IBKR ConIDs for a stock list.")
    parser.add_argument(
        '--file',
        type=str,
        default="nasdaq_stocks_with_conids.csv",
        help="Path to the CSV file containing stock symbols."
    )
    args = parser.parse_args()
    update_csv_with_conids(args.file)