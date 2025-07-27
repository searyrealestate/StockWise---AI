import os, glob, gc, joblib
import pandas as pd
import yfinance as yf
from xgboost import XGBClassifier
from datetime import datetime
import requests
import ta
from io import StringIO
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from build_nasdaq_market_cap import enrich_with_yfinance
import numpy as np

"""
📊 StockWise NASDAQ Pipeline
────────────────────────────

This script builds a full machine learning pipeline for analyzing NASDAQ-listed stocks
based on volume and price action. It includes scraping, data preparation, feature engineering,
model training, and file export.

🔧 Key Components:
──────────────────
1. generate_nasdaq_ticker_lists()
   - Scrapes NASDAQ stock symbols from stockanalysis.com
   - Saves training and testing ticker lists to .txt files

2. load_ticker_list(path)
   - Loads tickers from a saved text file

3. download_stock_data(symbol)
   - Downloads historical stock data using yfinance
   - Renames columns with ticker suffixes

4. add_volume_features_and_labels(df, symbol)
   - Adds volume-based features (e.g., relative volume, turnover)
   - Labels rows as BUY signals if future return exceeds threshold

5. train_model(df, symbol)
   - Trains an XGBoost classifier on the engineered features
   - Returns the trained model and cleaned DataFrame

6. process_ticker_list(tickers, output_dir, train=True)
   - Runs the full pipeline for each ticker:
     → Downloads data
     → Adds features
     → Trains model (optional)
     → Saves model and features
     → Logs skipped tickers

7. main(ticker_file, train=True)
   - Entry point to run the pipeline on a list of tickers

📁 Output:
──────────
- Trained models (.pkl)
- Feature datasets (.parquet)
- Skipped tickers log (.txt)

Use this script to generate training/testing datasets and models for use in
backtesting, prediction, or integration with a Streamlit dashboard.
"""

debug = False

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

# 👇 Update these with your actual local paths
TRAIN_DIR = r"C:\Users\user\PycharmProjects\StockWise\models\NASDAQ-training set"
TEST_DIR = r"C:\Users\user\PycharmProjects\StockWise\models\NASDAQ-testing set"

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)


def log_run(description, log_file="additional needed files/run_log.txt"):
    """
    📝 Appends a timestamped description of the current run to a log file.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(f"[{timestamp}] {description}\n")


def tag_economic_regime(df, regime_file="additional needed files/stock_event_tags.csv"):
    """
    🏷️ Tags each row in the stock DataFrame with macroeconomic event info.
    Adds columns: Event, ImpactType, AffectedSector
    """
    regimes = pd.read_csv(regime_file, parse_dates=["start_date", "end_date"])
    df = df.copy()
    df["Date"] = df.index if df.index.name == "Date" else pd.to_datetime(df.index)

    df["Event"] = "None"
    df["ImpactType"] = "None"
    df["AffectedSector"] = "None"

    for _, row in regimes.iterrows():
        mask = (df["Date"] >= row["start_date"]) & (df["Date"] <= row["end_date"])
        df.loc[mask, "Event"] = row["event_name"]
        df.loc[mask, "ImpactType"] = row["impact_type"]
        df.loc[mask, "AffectedSector"] = row["affected_sector"]

    return df


def generate_nasdaq_ticker_lists(train_file="nasdaq_train_400.txt", test_file="nasdaq_test_400.txt"):
    """
    📥 Scrapes all NASDAQ stock symbols from StockAnalysis.com.
    ✍️ Dynamically splits available tickers into training and testing sets.

    """

    url = "https://finviz.com/screener.ashx?v=111&f=exch_nasd"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/122.0.0.0 Safari/537.36"
    }

    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))

    response = session.get(url, headers=headers, timeout=10)

    response.raise_for_status()  # Raise an error if the request failed

    tables = pd.read_html(StringIO(response.text))
    if debug:
        print(f"🧪 Found {len(tables)} tables")
        for i, table in enumerate(tables):
            print(f"📄 Table {i} columns: {table.columns.tolist()}")

    # df = None
    # for i, t in enumerate(tables):
    #     print(f"\n🔍 Table {i} Columns: {list(t.columns)}")

        # if "Ticker" in t.columns:
        #     ticker_col = t["Ticker"]
        #     non_na_count = ticker_col.notna().sum()
        #     print(f"✅ Found 'Ticker' in Table {i}, Non-NaN entries: {non_na_count}")
        #     print(f"🧪 Sample tickers: {ticker_col.dropna().astype(str).head(10).tolist()}")
        #
        #     if df is None:
        #         raise ValueError("No valid ticker table found. Please check the data source format.")
        #
        #     if non_na_count >= 100 and ticker_col.map(lambda x: isinstance(x, str)).all():
        #         df = t
        #         print(f"🎯 Selected Table {i} as valid ticker source")
        #         break
        #     else:
        #         print(f"⚠️ Rejected Table {i}: Not enough valid tickers or type mismatch")
        #
        # else:
        #     print(f"🚫 Table {i} skipped — missing 'Ticker' column")
    df = None
    for i, t in enumerate(tables):
        print(f"\n🔍 Table {i} Columns: {list(t.columns)}")

        if "Ticker" in t.columns:
            tickers_raw = t["Ticker"].dropna().astype(str)

            # Basic filtering for legit symbols (max 5 letters, no numbers/symbols)
            clean_tickers = [tk for tk in tickers_raw if tk.isalpha() and len(tk) <= 5]

            print(f"🧪 Raw tickers: {tickers_raw.tolist()[:5]}")
            print(f"✅ Clean tickers ({len(clean_tickers)}): {clean_tickers[:5]}")

            # Decide if this table is good
            if len(clean_tickers) >= 20:  # 👈 Accept table if it has at least 20 good tickers
                df = t
                print(f"🎯 Selected table {i} for NASDAQ tickers.")
                break
            else:
                print(f"⚠️ Rejected table {i} — too few valid tickers.")
        else:
            print(f"🚫 Skipped table {i} — no 'Ticker' column.")

    if df is None:
        raise ValueError("❌ No valid ticker table found. Please check if the website layout changed.")

    tickers = df["Ticker"].dropna().astype(str).tolist()
    print(f"🔢 Total tickers scraped: {len(tickers)}")

    tickers = [t for t in tickers if t.isalpha() and len(t) <= 5]
    print(f"✅ Cleaned to {len(tickers)} valid tickers")

    tickers = [t for t in tickers if t.isalpha() and len(t) <= 5]
    print(f"✅ Filtered to {len(tickers)} valid tickers.")

    # 💾 Save full list of tickers for reproducibility
    with open("nasdaq_all_tickers_train.txt", "w") as f:
        f.writelines([t + "\n" for t in tickers])

    # 🎯 Force 400 training tickers if available
    train_size = min(400, len(tickers))
    train_tickers = tickers[:train_size]
    test_tickers = tickers[train_size:]

    with open(train_file, "w") as f:
        f.writelines([t + "\n" for t in train_tickers])

    with open(test_file, "w") as f:
        f.writelines([t + "\n" for t in test_tickers])

    print(f"✅ Saved {len(train_tickers)} training tickers to {train_file}")
    print(f"✅ Saved {len(test_tickers)} testing tickers to {test_file}")


def load_ticker_list(path):
    """
    📂 Loads a list of tickers from a text file.
    🔄 Strips whitespace and filters empty lines.
    """

    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def download_stock_data(symbol, start_date="2000-01-01"):
    """
        📈 Downloads historical stock data via yfinance.
        🧹 Renames price/volume columns with ticker suffix.
        """

    df = yf.download(symbol, start=start_date, progress=False)
    if df.empty:
        raise ValueError(f"Empty data for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    # Keep standard column names: Open, High, Low, Close, Volume
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

    df["Ticker"] = symbol
    return df


def add_volume_features_and_labels(df, symbol, window=20, threshold=0.05, forward_days=5, debug=False):
    """
    📊 Adds volume features, technical indicators, and future return label.
    ✅ Skips stocks with too few rows for indicator computation.
    """

    # 🚫 Skip short series
    if len(df) < 30:
        print(f"[{symbol}] Skipped — only {len(df)} rows (too short for indicators)")
        return pd.DataFrame()

    import ta
    close_col = "Close"
    volume_col = "Volume"
    df = df.copy()
    df[volume_col] = pd.to_numeric(df[volume_col], errors="coerce").fillna(0)

    # Volume-based features
    df["Volume_MA"] = df[volume_col].rolling(window=window).mean()
    df["Volume_Relative"] = df[volume_col] / df["Volume_MA"]
    df["Volume_Delta"] = df[volume_col].diff()
    df["Turnover"] = df[close_col] * df[volume_col]
    df["Volume_Spike"] = (df["Volume_Relative"] > 1.5).astype(int)

    # Technical indicators (multi-window)
    try:
        for w in [5, 7, 10, 14]:
            df[f"rsi_{w}"] = ta.momentum.RSIIndicator(close=df[close_col], window=w).rsi()

        for w in [5, 10, 20]:
            df[f"ema_{w}"] = ta.trend.EMAIndicator(close=df[close_col], window=w).ema_indicator()
            df[f"sma_{w}"] = ta.trend.SMAIndicator(close=df[close_col], window=w).sma_indicator()

        for fast, slow in [(5, 12), (7, 14), (10, 20)]:
            macd = ta.trend.MACD(close=df[close_col], window_fast=fast, window_slow=slow)
            df[f"macd_diff_{fast}_{slow}"] = macd.macd_diff()

        for w in [5, 10, 14]:
            df[f"adx_{w}"] = ta.trend.ADXIndicator(high=df["High"], low=df["Low"], close=df[close_col], window=w).adx()

        df["obv"] = ta.volume.OnBalanceVolumeIndicator(close=df[close_col], volume=df[volume_col]).on_balance_volume()

    except Exception as e:
        print(f"[{symbol}] Failed to compute indicators: {e}")
        return pd.DataFrame()

    # Label: 5-day forward return
    future_return = df[close_col].shift(-forward_days) / df[close_col] - 1
    df["Target"] = (future_return > threshold).astype(int)

    # 🏷️ Tag macroeconomic regimes and encode them
    df = tag_economic_regime(df)

    # One-hot encode event-related features
    df = pd.get_dummies(df, columns=["Event", "ImpactType", "AffectedSector"], prefix=["Event", "Impact", "Sector"])

    # Clean up
    df.dropna(inplace=True)
    return df


def train_model(df, symbol):
    """
    🧠 Trains XGBoost classifier using volume-based features.
    ✅ Returns fitted model and cleaned training DataFrame.
    """

    feature_cols = ["Volume_Relative", "Volume_Delta", "Turnover", "Volume_Spike", "Target"]
    df_clean = df.dropna(subset=feature_cols).copy()
    print(f"🧪 {symbol}: {df_clean.shape[0]} rows after dropna on required features")

    df_clean["logMarketCap"] = np.log1p(df_clean["marketCap"])
    df_clean["logAvgVolume"] = np.log1p(df_clean["avgVolume"])
    feature_cols.extend(["logMarketCap", "logAvgVolume"])

    if df_clean.empty:
        print(f"⚠️ Skipped {symbol} — all rows dropped after cleaning.")
        return None, pd.DataFrame()
    if df_clean["Target"].nunique() < 2:
        print(f"⚠️ Skipped {symbol} — Target column has only one class.")
        return None, pd.DataFrame()

    X = df_clean[feature_cols[:-1]]
    y = df_clean["Target"]

    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        use_label_encoder=False,
        eval_metric="logloss"
    )
    model.fit(X, y)
    return model, df_clean


def process_ticker_list(tickers, output_dir, train=True, meta_df=None):
    all_dfs = []
    failed_verbose = []

    for i, symbol in enumerate(tickers, 1):
        try:
            df = download_stock_data(symbol)
            df = add_volume_features_and_labels(df, symbol=symbol, debug=False)
            if meta_df is not None and symbol in meta_df.index:
                df["marketCap"] = meta_df.loc[symbol]["marketcap"]
                df["avgVolume"] = meta_df.loc[symbol]["avg_volume"]
            else:
                df["marketCap"] = None
                df["avgVolume"] = None

            expected_cols = ["Open", "High", "Low", "Close", "Volume", "Volume_MA", "Volume_Relative", "Volume_Delta",
                             "Turnover", "Volume_Spike", "Target"]
            missing = [col for col in expected_cols if col not in df.columns]
            if missing:
                raise ValueError(f"{symbol} is missing columns: {missing}")

            df["Ticker"] = symbol
            all_dfs.append(df)

            if train:
                model, df_trained = train_model(df, symbol)
                if df_trained.empty:
                    continue
                joblib.dump(model, os.path.join(output_dir, f"{symbol}_model_{run_id}.pkl"))

            df.dropna().to_parquet(
                os.path.join(output_dir, f"{symbol}_features_{run_id}.parquet"), compression="snappy"
            )
            print(f"✅ Processed {symbol} ({i}/{len(tickers)})")

        except Exception as e:
            print(f"❌ Skipped {symbol}: {e}")
            failed_verbose.append((symbol, str(e)))

    if failed_verbose:
        with open(os.path.join(output_dir, f"skipped_verbose_{run_id}.txt"), "w") as f:
            for symbol, err in failed_verbose:
                f.write(f"{symbol}\t{err}\n")
        print(f"📝 Verbose skipped tickers saved: {len(failed_verbose)}")


def main(ticker_file, train=True):
    tickers = load_ticker_list(ticker_file)
    print(f"📦 Loaded {len(tickers)} tickers from {ticker_file}")
    enriched_df = enrich_with_yfinance(tickers)
    if "symbol" not in enriched_df.columns:
        print("⚠️ No 'symbol' column found in enriched_df. Dumping columns for inspection:")
        print(enriched_df.columns.tolist())
        raise ValueError("Enrichment failed — missing 'symbol' column in metadata.")
    print("🔍 Enrichment DataFrame preview:")
    print(enriched_df.head())

    enriched_df.set_index("symbol", inplace=True)

    enriched_df.reset_index().to_csv(f"metadata_marketcap_volume_{run_id}.csv", index=False)
    target_dir = TRAIN_DIR if train else TEST_DIR
    process_ticker_list(tickers, output_dir=target_dir, train=train,meta_df=enriched_df)


if __name__ == "__main__":
    """
    🚀 Runs full pipeline on a list of tickers.
    📁 Chooses output dir based on training/testing mode.
    """
    log_run("Regime tagging added + retry logic for ticker scraping")

    # Create training and testing stock list files
    generate_nasdaq_ticker_lists(train_file="nasdaq_train_400.txt", test_file="nasdaq_test_400.txt")
    # Example usage:
    main("nasdaq_train_400.txt", train=True)
    main("nasdaq_test_400.txt", train=False)

    pass

