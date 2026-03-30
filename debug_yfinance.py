
import yfinance as yf
import pandas as pd

print("--- DEBUG YFINANCE OUTPUT ---")
tickers = ["BRK-B", "IBM"]
for t in tickers:
    print(f"\nDownloading {t}...")
    df = yf.download(t, period="1mo", interval="1d", auto_adjust=True, progress=False)
    print("Columns:", df.columns)
    print("Index:", df.index)
    print("Head:\n", df.head())
    
    if isinstance(df.columns, pd.MultiIndex):
        print("Multilevel Columns detected.")
        flat_cols = [
            "_".join([str(part) for part in col if part is not None and part != ""]).lower()
            for col in df.columns
        ]
        print("Flattened:", flat_cols)
    else:
        print("Single Level columns.")
