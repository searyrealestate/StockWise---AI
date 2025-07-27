# update_stocks.py
import pandas as pd
from datetime import datetime
import os

def load_nasdaq_symbols_from_ftp():
    """
    🔄 Loads NASDAQ-listed symbols and names from NASDAQ's official FTP.
    Returns a DataFrame with columns: ['symbol', 'name']
    """
    url = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt"
    try:
        df = pd.read_csv(url, sep="|")
        df = df[["Symbol", "Security Name"]]
        df.columns = ["symbol", "name"]
        df = df[df["symbol"] != "File Creation Time"]  # remove footer row
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to fetch NASDAQ data: {e}")


def save_nasdaq_stock_list(output_path="nasdaq_stocks.csv"):
    """
    📥 Downloads and saves NASDAQ stock list to a CSV file.
    """
    try:
        df = load_nasdaq_symbols_from_ftp()
        df.to_csv(output_path, index=False)
        print(f"✅ Saved {len(df)} stocks to {output_path}")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    save_nasdaq_stock_list()
