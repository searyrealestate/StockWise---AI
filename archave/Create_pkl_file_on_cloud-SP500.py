import os, glob, gc, joblib
import pandas as pd
import yfinance as yf
from xgboost import XGBClassifier
from google.colab import drive
from datetime import datetime
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

"""
📊 StockWise S&P 500 Model Builder (Google Colab Edition)
──────────────────────────────────────────────────────────

This script builds a machine learning pipeline for analyzing S&P 500 stocks using
volume-based features. It is designed to run in Google Colab and save models and
feature datasets to Google Drive.

🔧 Key Components:
──────────────────
1. get_sp500_tickers()
   - Scrapes the current list of S&P 500 companies from Wikipedia

2. download_stock_data(symbol)
   - Downloads historical stock data using yfinance
   - Renames columns with ticker suffixes

3. add_volume_features_and_labels(df, symbol)
   - Adds volume-based features (e.g., relative volume, turnover)
   - Labels rows as BUY signals if future return exceeds a threshold

4. train_model(df, symbol)
   - Trains an XGBoost classifier on the engineered features
   - Returns the trained model and cleaned DataFrame

5. train_all_models(df_all)
   - Trains models for all tickers in a combined DataFrame
   - Saves models and feature sets in batches

6. save_models_and_features(models, dataframes, path)
   - Saves trained models (.pkl) and feature DataFrames (.pkl) to disk

7. validate_generated_pkls(path)
   - Validates saved feature files to check for corruption or missing data

8. main()
   - Orchestrates the full pipeline:
     → Loads tickers
     → Downloads and processes data
     → Trains models
     → Saves combined dataset and logs skipped tickers

📁 Output:
──────────
- Trained models (.pkl)
- Feature datasets (.pkl)
- Combined dataset (.pkl)
- Skipped tickers log (.txt)

Designed for use in Google Colab with Google Drive integration.
"""


# Mount Drive
drive.mount('/content/drive')

# Set working directory
BASE_DIR = "/content/drive/MyDrive/StockWise/models"
os.makedirs(BASE_DIR, exist_ok=True)


def get_sp500_tickers():
    """
    📥 Scrapes the current list of S&P 500 stock symbols from Wikipedia.
    📥 Input: None
    📤 return: List[str] — list of ticker symbols (e.g., ["AAPL", "MSFT", "GOOGL"])
    """

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    tickers = tables[0]["Symbol"].tolist()
    return tickers


def download_stock_data(symbol, start_date="2000-01-01"):
    """

    📈 Downloads historical stock data for a given symbol using yfinance.
     Renames columns to include the symbol (e.g., Close_AAPL).

    :param symbol: stock ticker
    :param start_date: start date for historical data (default: 2000-01-01)
    :return: stock data with renamed columns (e.g., Close_AAPL, Volume_AAPL)
    """
    df = yf.download(symbol, start=start_date, progress=False)
    if df.empty:
        raise ValueError(f"Empty data for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df = df.rename(columns={k: f"{k}_{symbol}" for k in ["Open", "High", "Low", "Close", "Volume"]})
    df["Symbol"] = symbol
    return df


def add_volume_features_and_labels(
    df, symbol,
    window=20, threshold=0.05, forward_days=5,
    debug=False
):
    """
    📊 Adds engineered features based on volume (e.g., relative volume, turnover) and creates a binary target label based on future return.

    :param df: stock data
    :param symbol: stock ticker
    :param window: moving average window for volume
    :param threshold: return threshold for labeling
    :param forward_days: number of days to look ahead
    :param debug: True to print debugging information
    :return: enriched with features and Target column
    """
    close_col = f"Close_{symbol}"
    volume_col = f"Volume_{symbol}"
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

    🤖 Trains an XGBoost classifier using the volume-based features.
    Returns the trained model and the cleaned DataFrame used for training. Skips training if data is insufficient.

    Parameters:
        df : pd.DataFrame
            The stock's processed data including engineered features and 'Target'.
        symbol : str
            Stock symbol (used in logs or filenames).

    Returns:
        model : trained XGBClassifier or None
        df_clean : the cleaned feature DataFrame used for training
    """
    feature_cols = [
        "Volume_Relative",
        "Volume_Delta",
        "Turnover",
        "Volume_Spike",
        "Target"
    ]

    # Drop rows with missing values only in required columns
    df_clean = df.dropna(subset=feature_cols).copy()

    # Log post-cleaning shape
    print(f"🧪 {symbol}: {df_clean.shape[0]} rows after dropna on required features")

    # Guard: no rows left
    if df_clean.empty:
        print(f"⚠️ Skipped {symbol} — all rows dropped after cleaning.")
        return None, pd.DataFrame()

    # Guard: binary classification requires at least two classes
    if df_clean["Target"].nunique() < 2:
        print(f"⚠️ Skipped {symbol} — Target column has only one class.")
        return None, pd.DataFrame()

    # Separate features and labels
    X = df_clean[feature_cols[:-1]]  # all except Target
    y = df_clean["Target"]

    # Train model
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        use_label_encoder=False,
        eval_metric="logloss"
    )
    model.fit(X, y)

    return model, df_clean


def save_models_and_features(models, dataframes, path):
    """
    💾 Saves trained models and their corresponding feature DataFrames to disk as .pkl files.

    :param models: trained models
    :param dataframes: symbol-feature pairs
    :param path: output directory
    :return: Saves .pkl files to disk (no return)
    """
    for s, m in models.items():
        joblib.dump(m, os.path.join(path, f"{s}_model.pkl"))
    for s, df in dataframes:
        df.to_pickle(os.path.join(path, f"{s}_features.pkl"))
    print(f"💾 Saved {len(models)} models and feature sets.")


def train_all_models(df_all, save_every=10):
    """
    🔁 Loops through all symbols in the combined dataset,
    trains models, and saves them in batches (e.g., every 10 models).

    :param df_all: pd.DataFrame — merged dataset with all tickers
    :param save_every: int — how often to save models (default: every 10)
    :return: Saves models and features to disk (no return)
    """
    symbols = df_all["Symbol"].unique()
    trained_models = {}
    trained_dataframes = []

    for i, symbol in enumerate(symbols, start=1):
        df_symbol = df_all[df_all["Symbol"] == symbol].copy()
        try:
            model, df_trained = train_model(df_symbol, symbol)
            if df_trained.empty:
                print(f"⚠️ Skipped {symbol} (empty after cleaning)")
                continue
            trained_models[symbol] = model
            trained_dataframes.append((symbol, df_trained))
            print(f"✅ Trained {symbol} ({i}/{len(symbols)})")
        except Exception as e:
            print(f"❌ Skipped {symbol}: {e}")
            continue

        if len(trained_models) >= save_every:
            save_models_and_features(trained_models, trained_dataframes, BASE_DIR)
            trained_models.clear()
            trained_dataframes.clear()
            gc.collect()

    # Final flush
    if trained_models:
        save_models_and_features(trained_models, trained_dataframes, BASE_DIR)


def validate_generated_pkls(path=BASE_DIR):
    """
    ✅ Scans saved feature files to check for corruption, empty data, or invalid datetime indices.
    Logs valid and broken files.
    :param path: str — directory containing saved .pkl files
    :return: Prints summary of valid and broken files (no return)
    """
    features = sorted(glob.glob(os.path.join(path, "*_features_*.pkl")))
    if not features:
        print("❌ No feature files found.")
        return

    valid, broken = [], []
    for file in features:
        symbol = os.path.basename(file).replace("_features.pkl", "")
        try:
            df = pd.read_pickle(file)
            if df.empty:
                broken.append((symbol, "Empty DataFrame"))
            elif not pd.to_datetime(df.index, errors="coerce").notna().all():
                broken.append((symbol, "Invalid DateTime Index"))
            else:
                valid.append((symbol, df.shape[0]))
        except Exception as e:
            broken.append((symbol, f"Error: {e}"))

    print(f"\n✅ Valid: {len(valid)}")
    print(f"❌ Broken: {len(broken)}")
    print(f"📦 Total scanned: {len(features)}")


def main():
    """
    🚀 Orchestrates the full pipeline: loads tickers, downloads and processes data, trains models, saves outputs, and logs skipped tickers.

    📥 Input: None (uses hardcoded top 100 S&P 500 tickers)
    :return:  Saves models, features, combined dataset, and skipped tickers log
    """
    tickers = get_sp500_tickers()[:100]  # Or full 500
    all_dfs, failed = [], []

    for i, symbol in enumerate(tickers, 1):
        try:
            df = download_stock_data(symbol)
            df = add_volume_features_and_labels(df, symbol=symbol, debug=False)
            df["Symbol"] = symbol
            all_dfs.append(df)
            print(f"✅ Processed {symbol} ({i}/{len(tickers)})")
        except Exception as e:
            print(f"❌ Skipped {symbol}: {e}")
            failed.append(symbol)

    if not all_dfs:
        print("❌ No data to train on.")
        return

    df_all = pd.concat(all_dfs)
    df_all.to_pickle(os.path.join(BASE_DIR, f"sp500_all_{run_id}.pkl"))
    print("💾 Saved merged dataset.")
    train_all_models(df_all)

    # Save combined features
    try:
        paths = glob.glob(os.path.join(BASE_DIR, "*_features.pkl"))
        combined = pd.concat([pd.read_pickle(p) for p in paths])
        combined.to_pickle(os.path.join(BASE_DIR, f"sp500_features_all{run_id}.pkl"))
        print("📊 Saved combined dataset.")
    except Exception as e:
        print(f"⚠️ Could not combine dataset: {e}")

    if failed:
        with open(os.path.join(BASE_DIR, f"skipped_tickers{run_id}.txt"), "w") as f:
            f.writelines([s + "\n" for s in failed])


if __name__ == "__main__":
    """
    🧭 Entry point: runs main() and then validates the saved feature files.
    """
    main()
    validate_generated_pkls(BASE_DIR)

