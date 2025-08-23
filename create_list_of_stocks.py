import pandas as pd
import yfinance as yf
from tqdm import tqdm  # Add this for progress bar

# Load NASDAQ tickers
nasdaq_tickers = pd.read_csv("https://datahub.io/core/nasdaq-listings/r/nasdaq-listed.csv")["Symbol"].tolist()

results = []

# Wrap the loop with tqdm for progress indication
for symbol in tqdm(nasdaq_tickers[:1200], desc="Processing NASDAQ tickers"):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        market_cap = info.get("marketCap", 0)

        if market_cap and market_cap > 150_000_000:
            hist = stock.history(period="1y")
            if not hist.empty:
                start_price = hist["Close"].iloc[0]
                end_price = hist["Close"].iloc[-1]
                performance = ((end_price - start_price) / start_price) * 100

                results.append({
                    "Symbol": symbol,
                    "Market Cap": market_cap,
                    "1Y Performance (%)": round(performance, 2)
                })

    except Exception:
        continue  # Skip problematic tickers

# Create DataFrame and sort
df = pd.DataFrame(results)
df_sorted = df.sort_values(by="1Y Performance (%)", ascending=False)

# Export to text file
df_sorted.to_csv("nasdaq_top_1000.txt", sep="\t", index=False)
print("✅ Saved to nasdaq_top_1000.txt")
