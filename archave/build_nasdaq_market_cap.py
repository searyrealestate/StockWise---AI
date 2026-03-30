import pandas as pd
import os
import yfinance as yf
import argparse
from time import sleep


def apply_trending_filters(df, min_cap, max_cap, min_volume):
    df["marketcap"] = pd.to_numeric(df["marketcap"], errors="coerce")
    df["avg_volume"] = pd.to_numeric(df["avg_volume"], errors="coerce")

    trending = df[
        (df["marketcap"] > min_cap) &
        (df["avg_volume"] > min_volume)
    ]

    breakout = df[
        (df["marketcap"].between(100_000_000, 300_000_000))
    ]

    mid_caps = df[
        (df["marketcap"].between(500_000_000, 1_000_000_000))
    ]

    print(f"📈 Trending stocks: {len(trending)}")
    print(f"🚀 Breakout candidates: {len(breakout)}")
    print(f"⚖️ Mid-caps: {len(mid_caps)}")

    return trending, breakout, mid_caps


def enrich_with_yfinance(symbols, delay=0.2):
    data = []
    for i, symbol in enumerate(symbols, 1):
        try:
            info = yf.Ticker(symbol).info
            cap = info.get("marketCap")
            vol = info.get("averageVolume")
            if cap and vol:
                data.append({
                    "symbol": symbol,
                    "marketcap": cap,
                    "avg_volume": vol,
                    "shortName": info.get("shortName", ""),
                    "sector": info.get("sector", ""),
                    "industry": info.get("industry", "")
                })
            print(f"{i}/{len(symbols)} ✅ {symbol}")
        except Exception as e:
            print(f"{i}/{len(symbols)} ❌ {symbol}: {e}")
        sleep(delay)
    return pd.DataFrame(data)

def main():
    parser = argparse.ArgumentParser(description="NASDAQ Screener")
    parser.add_argument("--min_cap", type=int, default=1_000_000_000, help="Minimum market cap")
    parser.add_argument("--min_volume", type=int, default=1_000_000, help="Minimum average volume")
    parser.add_argument("--include_etfs", action="store_true", help="Include ETFs")
    parser.add_argument("--delay", type=float, default=0.2, help="Delay between API calls (seconds)")
    args = parser.parse_args()

    url = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt"
    df_raw = pd.read_csv(url, sep="|")

    df_clean = df_raw[
        (df_raw["Symbol"].notna()) &
        (df_raw["Test Issue"] != "Y") &
        (df_raw["Financial Status"] != "Delinquent")
    ]
    if not args.include_etfs:
        df_clean = df_clean[df_clean["ETF"] != "Y"]

    symbols = df_clean["Symbol"].unique().tolist()
    print(f"✅ Pulled {len(symbols)} NASDAQ tickers")

    df_enriched = enrich_with_yfinance(symbols, delay=args.delay)
    df_enriched.to_csv("nasdaq_enriched_data.csv", index=False)

    trending, breakout, mid = apply_trending_filters(
        df_enriched,
        min_cap=args.min_cap,
        max_cap=5_000_000_000_000,
        min_volume=args.min_volume
    )

    trending.to_csv("nasdaq_trending.csv", index=False)
    breakout.to_csv("nasdaq_breakout.csv", index=False)
    mid.to_csv("nasdaq_mid_caps.csv", index=False)

    print("✅ Results saved: trending.csv, breakout.csv, mid_caps.csv")


if __name__ == "__main__":
    main()
