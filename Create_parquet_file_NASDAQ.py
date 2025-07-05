import os, glob, gc, joblib
import pandas as pd
import yfinance as yf
from xgboost import XGBClassifier
from datetime import datetime
import requests

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


run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

# 👇 Update these with your actual local paths
TRAIN_DIR = r"C:\Users\user\PycharmProjects\StockWise\models\NASDAQ-training set"
TEST_DIR = r"C:\Users\user\PycharmProjects\StockWise\models\NASDAQ-testing set"

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)


def generate_nasdaq_ticker_lists(train_file="nasdaq_train_100.txt", test_file="nasdaq_test_400.txt"):
    """
    📥 Scrapes all NASDAQ stock symbols from StockAnalysis.com.
    ✍️ Saves first 100 to training list, next 400 to testing list.
    """

    url = "https://stockanalysis.com/list/nasdaq-stocks/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/122.0.0.0 Safari/537.36"
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Raise an error if the request failed

    tables = pd.read_html(response.text)
    df = tables[0]

    if "Symbol" not in df.columns:
        raise ValueError("Could not find 'Symbol' column in scraped table.")

    tickers = df["Symbol"].dropna().astype(str).tolist()

    if len(tickers) < 500:
        raise ValueError(f"Only found {len(tickers)} tickers — expected at least 500.")

    train_tickers = tickers[:100]
    test_tickers = tickers[101:500]

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

    df["Symbol"] = symbol
    return df


def add_volume_features_and_labels(df, symbol, window=20, threshold=0.05, forward_days=5, debug=False):
    """
    📊 Adds rolling volume features and future return label.
    ✅ Label is 1 if 5-day return exceeds threshold.
    """

    close_col = "Close"
    volume_col = "Volume"
    df = df.copy()
    df[volume_col] = pd.to_numeric(df[volume_col], errors="coerce").fillna(0)

    df["Volume_MA"] = df[volume_col].rolling(window=window).mean()
    df["Volume_Relative"] = df[volume_col] / df["Volume_MA"]
    df["Volume_Delta"] = df[volume_col].diff()
    df["Turnover"] = df[close_col] * df[volume_col]
    df["Volume_Spike"] = (df["Volume_Relative"] > 1.5).astype(int)

    future_return = df[close_col].shift(-forward_days) / df[close_col] - 1
    df["Target"] = (future_return > threshold).astype(int)
    return df


def train_model(df, symbol):
    """
    🧠 Trains XGBoost classifier using volume-based features.
    ✅ Returns fitted model and cleaned training DataFrame.
    """

    feature_cols = ["Volume_Relative", "Volume_Delta", "Turnover", "Volume_Spike", "Target"]
    df_clean = df.dropna(subset=feature_cols).copy()
    print(f"🧪 {symbol}: {df_clean.shape[0]} rows after dropna on required features")

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


def process_ticker_list(tickers, output_dir, train=True):
    """
    🔄 Main pipeline: downloads data, adds features, trains model (optional), saves outputs.
    📦 Stores features in Parquet, models in joblib, and logs skipped symbols.
    """
    all_dfs, failed = [], []

    for i, symbol in enumerate(tickers, 1):
        try:
            df = download_stock_data(symbol)
            df = add_volume_features_and_labels(df, symbol=symbol, debug=False)
            expected_cols = ["Open", "High", "Low", "Close", "Volume", "Volume_MA", "Volume_Relative", "Volume_Delta",
                             "Turnover", "Volume_Spike", "Target"]
            missing = [col for col in expected_cols if col not in df.columns]
            if missing:
                raise ValueError(f"{symbol} is missing columns: {missing}")

            df["Symbol"] = symbol
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
            failed.append(symbol)

    if failed:
        with open(os.path.join(output_dir, f"skipped_{run_id}.txt"), "w") as f:
            f.writelines([t + "\n" for t in failed])
        print(f"📝 Skipped tickers saved: {len(failed)}")


def main(ticker_file, train=True):
    tickers = load_ticker_list(ticker_file)
    print(f"📦 Loaded {len(tickers)} tickers from {ticker_file}")
    target_dir = TRAIN_DIR if train else TEST_DIR
    process_ticker_list(tickers, output_dir=target_dir, train=train)


if __name__ == "__main__":
    """
    🚀 Runs full pipeline on a list of tickers.
    📁 Chooses output dir based on training/testing mode.
    """
    # Create training and testing stock list files
    # generate_nasdaq_ticker_lists(train_file="nasdaq_train_100.txt", test_file="nasdaq_test_400.txt")
    # Example usage:
    # main("nasdaq_train_100.txt", train=True)
    main("nasdaq_test_400.txt", train=False)

    pass

