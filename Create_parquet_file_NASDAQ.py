"""
📊 StockWise NASDAQ Pipeline - COMPREHENSIVE VERSION
───────────────────────────────────────────────────

This version gets ALL NASDAQ stocks (1000+) from multiple sources:
- Official NASDAQ FTP (most reliable)
- Finviz with pagination (comprehensive)
- Yahoo Finance (backup)

Usage:
    python Create_parquet_file_NASDAQ.py                    # Process all stocks
    python Create_parquet_file_NASDAQ.py --debug            # Debug mode
    python Create_parquet_file_NASDAQ.py --debug --stock AAPL  # Single stock debug
    python Create_parquet_file_NASDAQ.py --max-stocks 500   # Limit number of stocks
"""

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
import argparse
import time

# Parse command line arguments
parser = argparse.ArgumentParser(description='StockWise NASDAQ Data Processor - Comprehensive')
parser.add_argument('--debug', action='store_true', help='Enable debug mode with detailed output')
parser.add_argument('--stock', type=str, help='Process only specific stock (for debugging)')
parser.add_argument('--max-stocks', type=int, default=1000, help='Maximum number of stocks to process')
parser.add_argument('--quick', action='store_true', help='Quick mode - use fallback stocks only')
args = parser.parse_args()

debug = args.debug
debug_stock = args.stock
max_stocks = args.max_stocks
quick_mode = args.quick

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

# Your existing directories
TRAIN_DIR = r"C:\Users\user\PycharmProjects\StockWise\models\NASDAQ-training set"
TEST_DIR = r"C:\Users\user\PycharmProjects\StockWise\models\NASDAQ-testing set"

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)


def debug_print(message, level="INFO"):
    """🐛 Debug printing with levels"""
    if debug:
        timestamp = datetime.now().strftime("%H:%M:%S")
        levels = {
            "INFO": "ℹ️", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌",
            "FEATURE": "🔧", "DATA": "📊"
        }
        icon = levels.get(level, "🔸")
        print(f"[{timestamp}] {icon} DEBUG: {message}")


def log_run(description, log_file="additional needed files/run_log.txt"):
    """📝 Logging function"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {description}"
    if debug:
        log_entry += " [DEBUG MODE]"
    with open(log_file, "a") as f:
        f.write(log_entry + "\n")
    debug_print(f"Logged: {description}")


def get_nasdaq_from_official_ftp():
    """📥 Get ALL NASDAQ stocks from official FTP"""
    debug_print("Fetching from official NASDAQ FTP...")

    try:
        url = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt"
        df = pd.read_csv(url, sep="|")

        # Clean up the data
        df = df[df['Symbol'].notna()]
        df = df[df['Test Issue'] != 'Y']
        df = df[df['Financial Status'] != 'Delinquent']

        # Get regular stocks (not ETFs)
        stocks_df = df[df['ETF'] != 'Y']

        # Clean symbols
        symbols = stocks_df['Symbol'].tolist()
        clean_symbols = [s.strip() for s in symbols if s and isinstance(s, str) and s.isalpha() and len(s) <= 5]

        debug_print(f"Official NASDAQ FTP: Found {len(clean_symbols)} stocks", "SUCCESS")
        print(f"✅ Official NASDAQ FTP: Found {len(clean_symbols)} stocks")

        return clean_symbols

    except Exception as e:
        debug_print(f"Official FTP failed: {e}", "ERROR")
        print(f"❌ Official NASDAQ FTP failed: {e}")
        return []


def get_nasdaq_from_finviz_comprehensive():
    """🔍 Get NASDAQ stocks from Finviz with pagination"""
    debug_print("Fetching from Finviz (all pages)...")
    print("🔄 Fetching from Finviz (comprehensive scraping)...")

    all_symbols = []
    page = 1
    max_pages = 100  # Increased limit

    try:
        while page <= max_pages:
            offset = (page - 1) * 20
            url = f"https://finviz.com/screener.ashx?v=111&f=exch_nasd&r={offset + 1}"

            debug_print(f"Fetching page {page} (offset {offset})...")

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()

            tables = pd.read_html(StringIO(response.text))

            found_tickers = False
            page_symbols = []

            for table in tables:
                if "Ticker" in table.columns:
                    tickers = table["Ticker"].dropna().astype(str).tolist()
                    clean_tickers = [t for t in tickers if t.isalpha() and len(t) <= 5]
                    page_symbols.extend(clean_tickers)
                    found_tickers = True

            if not found_tickers or len(page_symbols) == 0:
                debug_print(f"Page {page}: No more data found", "INFO")
                break

            all_symbols.extend(page_symbols)
            debug_print(f"Page {page}: Found {len(page_symbols)} symbols", "SUCCESS")

            if page % 10 == 0:
                print(f"   📄 Processed {page} pages, found {len(all_symbols)} symbols so far...")

            time.sleep(0.3)
            page += 1

        unique_symbols = list(set(all_symbols))
        debug_print(f"Finviz comprehensive: Found {len(unique_symbols)} unique stocks", "SUCCESS")
        print(f"✅ Finviz comprehensive: Found {len(unique_symbols)} unique stocks")

        return unique_symbols

    except Exception as e:
        debug_print(f"Finviz comprehensive failed: {e}", "ERROR")
        print(f"❌ Finviz comprehensive failed: {e}")
        return []


def get_fallback_nasdaq_tickers():
    """🔄 Quality fallback list"""
    tickers = [
        # Large cap tech
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'ADBE',
        # Other large caps
        'INTC', 'CMCSA', 'CSCO', 'AVGO', 'TXN', 'QCOM', 'COST', 'SBUX', 'GILD', 'MDLZ',
        'ADP', 'VRTX', 'REGN', 'ISRG', 'BKNG', 'AMD', 'MRNA', 'PYPL', 'ZM', 'DXCM',
        # Mid/Small caps
        'ILMN', 'BIIB', 'SGEN', 'ALGN', 'CTSH', 'EA', 'EBAY', 'FAST', 'FISV', 'IDXX',
        'INTU', 'KLAC', 'LRCX', 'MCHP', 'MELI', 'MNST', 'NTES', 'OKTA', 'PAYX', 'ROKU',
        'TEAM', 'TTWO', 'VRSK', 'WDAY', 'ZS', 'CRWD', 'SNOW', 'DDOG', 'NET', 'SHOP',
        # Additional quality stocks
        'ABNB', 'DOCU', 'ZOOM', 'PTON', 'UBER', 'LYFT', 'PINS', 'SNAP', 'TWTR', 'RBLX',
        'COIN', 'HOOD', 'SOFI', 'PLTR', 'WISH', 'CLOV', 'AMC', 'GME', 'BB', 'NOK',
        # Biotech
        'GILEAD', 'MODERNA', 'BNTX', 'NVAX', 'VRTX', 'REGN', 'GILD', 'AMGN', 'CELG', 'BMRN',
        # More tech
        'CRM', 'NOW', 'WDAY', 'VEEV', 'SPLK', 'PANW', 'FTNT', 'CYBR', 'OKTA', 'MDB'
    ]
    debug_print(f"Using fallback ticker list with {len(tickers)} quality stocks", "INFO")
    print(f"✅ Using fallback list: {len(tickers)} quality stocks")
    return tickers


def generate_comprehensive_nasdaq_ticker_lists(train_file="nasdaq_train_comprehensive.txt",
                                               test_file="nasdaq_test_comprehensive.txt"):
    """📥 Generate comprehensive NASDAQ ticker lists"""

    if quick_mode:
        print("🚀 QUICK MODE: Using fallback ticker list")
        tickers = get_fallback_nasdaq_tickers()
    else:
        print("🚀 COMPREHENSIVE MODE: Fetching ALL NASDAQ stocks")
        print("=" * 70)

        all_symbols = []

        # Method 1: Official NASDAQ FTP (most reliable)
        ftp_symbols = get_nasdaq_from_official_ftp()
        if ftp_symbols:
            all_symbols.extend(ftp_symbols)

        # Method 2: Finviz comprehensive (if FTP didn't work or we want more)
        if len(all_symbols) < 500:  # If we don't have enough from FTP
            finviz_symbols = get_nasdaq_from_finviz_comprehensive()
            if finviz_symbols:
                new_symbols = [s for s in finviz_symbols if s not in all_symbols]
                all_symbols.extend(new_symbols)
                print(f"   📈 Added {len(new_symbols)} additional symbols from Finviz")

        # Method 3: Fallback if both failed
        if len(all_symbols) < 100:
            print("⚠️ Primary methods failed, using fallback list")
            fallback_symbols = get_fallback_nasdaq_tickers()
            new_symbols = [s for s in fallback_symbols if s not in all_symbols]
            all_symbols.extend(new_symbols)

        tickers = sorted(list(set(all_symbols)))

        print("=" * 70)
        print(f"🎯 TOTAL NASDAQ SYMBOLS FOUND: {len(tickers)}")
        print("=" * 70)

    # Limit if requested
    if max_stocks and len(tickers) > max_stocks:
        print(f"📊 Limiting to {max_stocks} stocks (from {len(tickers)} available)")
        tickers = tickers[:max_stocks]

    # Save comprehensive list
    with open(f"nasdaq_all_comprehensive_{run_id}.txt", "w") as f:
        for ticker in tickers:
            f.write(f"{ticker}\n")

    # Split into train/test (70/30)
    split_point = int(len(tickers) * 0.7)
    train_tickers = tickers[:split_point]
    test_tickers = tickers[split_point:]

    # Ensure minimum test size
    if len(test_tickers) < 10:
        test_size = max(10, len(tickers) // 4)
        train_tickers = tickers[:-test_size]
        test_tickers = tickers[-test_size:]

    with open(train_file, "w") as f:
        for ticker in train_tickers:
            f.write(f"{ticker}\n")

    with open(test_file, "w") as f:
        for ticker in test_tickers:
            f.write(f"{ticker}\n")

    print(f"✅ Saved {len(train_tickers)} training tickers to {train_file}")
    print(f"✅ Saved {len(test_tickers)} testing tickers to {test_file}")
    print(f"💾 Complete list saved: nasdaq_all_comprehensive_{run_id}.txt")

    debug_print(f"Training tickers: {train_tickers[:10]}{'...' if len(train_tickers) > 10 else ''}", "DATA")
    debug_print(f"Testing tickers: {test_tickers[:10]}{'...' if len(test_tickers) > 10 else ''}", "DATA")

    return train_tickers, test_tickers


# Keep all your existing functions but replace the ticker generation
def load_ticker_list(path):
    """📂 Load tickers"""
    with open(path, "r") as f:
        tickers = [line.strip() for line in f if line.strip()]
    debug_print(f"Loaded {len(tickers)} tickers from {path}", "DATA")
    return tickers


def download_stock_data(symbol, start_date="2000-01-01"):
    """📈 Download stock data"""
    debug_print(f"Downloading data for {symbol} from {start_date}", "INFO")

    df = yf.download(symbol, start=start_date, progress=False, auto_adjust=True)
    if df.empty:
        debug_print(f"No data received for {symbol}", "ERROR")
        raise ValueError(f"Empty data for {symbol}")

    debug_print(f"Downloaded {len(df)} rows for {symbol}", "SUCCESS")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df["Ticker"] = symbol

    debug_print(f"Date range: {df.index.min()} to {df.index.max()}", "DATA")

    return df


def tag_economic_regime(df, regime_file="additional needed files/stock_event_tags.csv"):
    """🏷️ Economic regime tagging"""
    debug_print(f"Tagging economic regimes from {regime_file}")

    try:
        if not os.path.exists(regime_file):
            debug_print(f"Regime file not found: {regime_file}", "WARNING")
            return df

        regimes = pd.read_csv(regime_file, parse_dates=["start_date", "end_date"])
        debug_print(f"Loaded {len(regimes)} economic regime periods")

        df = df.copy()
        df["Date"] = df.index if df.index.name == "Date" else pd.to_datetime(df.index)

        df["Event"] = "None"
        df["ImpactType"] = "None"
        df["AffectedSector"] = "None"

        tagged_rows = 0
        for _, row in regimes.iterrows():
            mask = (df["Date"] >= row["start_date"]) & (df["Date"] <= row["end_date"])
            rows_affected = mask.sum()
            if rows_affected > 0:
                df.loc[mask, "Event"] = row["event_name"]
                df.loc[mask, "ImpactType"] = row["impact_type"]
                df.loc[mask, "AffectedSector"] = row["affected_sector"]
                tagged_rows += rows_affected

        debug_print(f"Economic regime tagging complete: {tagged_rows} total rows tagged", "SUCCESS")
        return df

    except Exception as e:
        debug_print(f"Regime tagging failed: {e}", "ERROR")
        return df


def add_volume_features_and_labels(df, symbol, window=20, threshold=0.05, forward_days=5, debug_mode=None):
    """📊 Enhanced feature engineering with debug"""

    if debug_mode is None:
        debug_mode = debug

    debug_print(f"=== FEATURE ENGINEERING START for {symbol} ===", "FEATURE")
    debug_print(f"Input shape: {df.shape}", "DATA")

    # Skip short series
    if len(df) < 30:
        debug_print(f"Skipping {symbol} - only {len(df)} rows (need 30+ for indicators)", "WARNING")
        print(f"[{symbol}] Skipped — only {len(df)} rows (too short for indicators)")
        return pd.DataFrame()

    close_col = "Close"
    volume_col = "Volume"
    df = df.copy()
    df[volume_col] = pd.to_numeric(df[volume_col], errors="coerce").fillna(0)

    debug_print(f"Volume stats: mean={df[volume_col].mean():.0f}, max={df[volume_col].max():.0f}", "DATA")

    # === ORIGINAL VOLUME FEATURES ===
    debug_print("Adding original volume features...", "FEATURE")

    df["Volume_MA"] = df[volume_col].rolling(window=window).mean()
    df["Volume_Relative"] = df[volume_col] / df["Volume_MA"]
    df["Volume_Delta"] = df[volume_col].diff()
    df["Turnover"] = df[close_col] * df[volume_col]
    df["Volume_Spike"] = (df["Volume_Relative"] > 1.5).astype(int)

    spike_count = df["Volume_Spike"].sum()
    debug_print(f"Original volume features added. Volume spikes: {spike_count}", "SUCCESS")

    # === ENHANCED FEATURES ===
    enhanced_count = 0

    try:
        # Enhanced volume features
        for period in [5, 10]:
            df[f"Volume_MA_{period}"] = df[volume_col].rolling(period).mean()
            df[f"Volume_Ratio_{period}"] = df[volume_col] / df[f"Volume_MA_{period}"]
            enhanced_count += 2

        df["Volume_Momentum"] = df[volume_col].rolling(5).mean() / df[volume_col].rolling(20).mean()
        df["Volume_Breakout"] = (df["Volume_Relative"] > 2.0).astype(int)
        df["Volume_Dry_Up"] = (df["Volume_Relative"] < 0.5).astype(int)
        enhanced_count += 3

        # Technical indicators
        for w in [7, 14, 21]:
            df[f"rsi_{w}"] = ta.momentum.RSIIndicator(close=df[close_col], window=w).rsi()
            enhanced_count += 1

        for w in [5, 10, 20]:
            df[f"ema_{w}"] = ta.trend.EMAIndicator(close=df[close_col], window=w).ema_indicator()
            df[f"sma_{w}"] = ta.trend.SMAIndicator(close=df[close_col], window=w).sma_indicator()
            enhanced_count += 2

        # MACD
        for fast, slow in [(12, 26), (5, 35)]:
            macd = ta.trend.MACD(close=df[close_col], window_fast=fast, window_slow=slow)
            df[f"macd_{fast}_{slow}"] = macd.macd()
            df[f"macd_signal_{fast}_{slow}"] = macd.macd_signal()
            df[f"macd_diff_{fast}_{slow}"] = macd.macd_diff()
            enhanced_count += 3

        # ADX
        for w in [10, 14]:
            df[f"adx_{w}"] = ta.trend.ADXIndicator(high=df["High"], low=df["Low"], close=df[close_col], window=w).adx()
            enhanced_count += 1

        # OBV
        df["obv"] = ta.volume.OnBalanceVolumeIndicator(close=df[close_col], volume=df[volume_col]).on_balance_volume()
        df["obv_ma"] = df["obv"].rolling(10).mean()
        enhanced_count += 2

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close=df[close_col])
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_middle"] = bb.bollinger_mavg()
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        df["bb_position"] = (df[close_col] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
        enhanced_count += 5

        # Stochastic
        stoch = ta.momentum.StochasticOscillator(high=df["High"], low=df["Low"], close=df[close_col])
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()
        enhanced_count += 2

        # MFI
        df["mfi"] = ta.volume.MFIIndicator(high=df["High"], low=df["Low"], close=df[close_col],
                                           volume=df[volume_col]).money_flow_index()
        enhanced_count += 1

        # Market structure
        df["higher_high"] = (df["High"] > df["High"].shift(1)).astype(int)
        df["lower_low"] = (df["Low"] < df["Low"].shift(1)).astype(int)
        df["gap_up"] = (df["Open"] > df[close_col].shift(1)).astype(int)
        df["gap_down"] = (df["Open"] < df[close_col].shift(1)).astype(int)
        df["gap_size"] = abs(df["Open"] - df[close_col].shift(1)) / df[close_col].shift(1)
        enhanced_count += 5

        # Support/Resistance
        for period in [10, 20]:
            df[f"high_{period}d"] = df["High"].rolling(period).max()
            df[f"low_{period}d"] = df["Low"].rolling(period).min()
            df[f"distance_high_{period}d"] = (df[close_col] - df[f"high_{period}d"]) / df[close_col]
            df[f"distance_low_{period}d"] = (df[close_col] - df[f"low_{period}d"]) / df[close_col]
            enhanced_count += 4

        debug_print(f"Enhanced features added: {enhanced_count} new features", "SUCCESS")

    except Exception as e:
        debug_print(f"Enhanced features warning: {e}", "ERROR")

    # === TARGET LABELS ===
    # Original target
    future_return = df[close_col].shift(-forward_days) / df[close_col] - 1
    df["Target"] = (future_return > threshold).astype(int)

    original_targets = df["Target"].sum()
    debug_print(f"Original target (5d, 5%): {original_targets} positive signals", "DATA")

    # Multi-timeframe targets
    for days in [10, 15, 20]:
        future_ret = df[close_col].shift(-days) / df[close_col] - 1
        df[f"Target_{days}d"] = (future_ret > threshold).astype(int)
        df[f"Target_{days}d_3pct"] = (future_ret > 0.03).astype(int)

    # Economic regime tagging
    df = tag_economic_regime(df)

    try:
        df = pd.get_dummies(df, columns=["Event", "ImpactType", "AffectedSector"],
                            prefix=["Event", "Impact", "Sector"])
    except Exception as e:
        debug_print(f"One-hot encoding warning: {e}", "ERROR")

    # Final cleanup
    initial_rows = len(df)
    df = df.dropna()
    final_rows = len(df)
    dropped_rows = initial_rows - final_rows

    debug_print(f"Data cleanup: {initial_rows} → {final_rows} rows ({dropped_rows} dropped)", "DATA")
    debug_print(f"=== FEATURE ENGINEERING COMPLETE for {symbol} ===", "FEATURE")
    debug_print(f"Final dataset shape: {df.shape}", "SUCCESS")

    return df


def train_model(df, symbol):
    """🧠 Model training"""
    debug_print(f"=== MODEL TRAINING START for {symbol} ===", "INFO")

    feature_cols = ["Volume_Relative", "Volume_Delta", "Turnover", "Volume_Spike", "Target"]
    df_clean = df.dropna(subset=feature_cols).copy()
    debug_print(f"Training data shape: {df_clean.shape}", "DATA")

    print(f"🧪 {symbol}: {df_clean.shape[0]} rows after dropna on required features")

    # Handle market cap data
    if "marketCap" in df_clean.columns and "avgVolume" in df_clean.columns:
        df_clean["marketCap"] = pd.to_numeric(df_clean["marketCap"], errors='coerce').fillna(0)
        df_clean["avgVolume"] = pd.to_numeric(df_clean["avgVolume"], errors='coerce').fillna(0)

        if df_clean["marketCap"].sum() > 0:
            df_clean["logMarketCap"] = np.log1p(df_clean["marketCap"])
            feature_cols.append("logMarketCap")

        if df_clean["avgVolume"].sum() > 0:
            df_clean["logAvgVolume"] = np.log1p(df_clean["avgVolume"])
            feature_cols.append("logAvgVolume")

    if df_clean.empty:
        debug_print("Training data is empty", "ERROR")
        print(f"⚠️ Skipped {symbol} — all rows dropped after cleaning.")
        return None, pd.DataFrame()

    if df_clean["Target"].nunique() < 2:
        debug_print(f"Target has only one class", "ERROR")
        print(f"⚠️ Skipped {symbol} — Target column has only one class.")
        return None, pd.DataFrame()

    X = df_clean[feature_cols[:-1]]
    y = df_clean["Target"]

    debug_print(f"Features: {feature_cols[:-1]}", "DATA")
    debug_print(f"Target distribution: {y.value_counts().to_dict()}", "DATA")

    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        eval_metric="logloss"
    )
    model.fit(X, y)

    debug_print(f"=== MODEL TRAINING COMPLETE for {symbol} ===", "SUCCESS")
    return model, df_clean


def process_ticker_list(tickers, output_dir, train=True, meta_df=None):
    """🔄 Process ticker list"""

    if debug_stock and debug_stock in tickers:
        process_tickers = [debug_stock]
        debug_print(f"DEBUG MODE: Processing only {debug_stock}", "WARNING")
    else:
        process_tickers = tickers
        debug_print(f"Processing all {len(tickers)} tickers", "INFO")

    if not process_tickers:
        print("⚠️ No tickers to process")
        return

    debug_print(f"Output directory: {output_dir}", "INFO")
    debug_print(f"Training mode: {train}", "INFO")

    all_dfs = []
    failed_verbose = []
    successful_count = 0

    for i, symbol in enumerate(process_tickers, 1):
        debug_print(f"=== PROCESSING {symbol} ({i}/{len(process_tickers)}) ===", "INFO")

        try:
            df = download_stock_data(symbol)
            df = add_volume_features_and_labels(df, symbol=symbol, debug_mode=debug)

            if df.empty:
                debug_print(f"{symbol} resulted in empty dataframe", "ERROR")
                failed_verbose.append((symbol, "Empty after feature engineering"))
                continue

            # Handle metadata
            if meta_df is not None and len(meta_df) > 0 and symbol in meta_df.index:
                df["marketCap"] = meta_df.loc[symbol].get("marketcap", None)
                df["avgVolume"] = meta_df.loc[symbol].get("avg_volume", None)
                debug_print(f"Added metadata for {symbol}", "DATA")
            else:
                df["marketCap"] = None
                df["avgVolume"] = None

            # Validate required columns
            expected_cols = ["Open", "High", "Low", "Close", "Volume", "Volume_MA", "Volume_Relative",
                             "Volume_Delta", "Turnover", "Volume_Spike", "Target"]
            missing = [col for col in expected_cols if col not in df.columns]
            if missing:
                debug_print(f"Missing columns: {missing}", "ERROR")
                failed_verbose.append((symbol, f"Missing columns: {missing}"))
                continue

            df["Ticker"] = symbol

            # Train model
            if train:
                model, df_trained = train_model(df, symbol)
                if model is not None and not df_trained.empty:
                    model_path = os.path.join(output_dir, f"{symbol}_model_{run_id}.pkl")
                    joblib.dump(model, model_path)
                    debug_print(f"Model saved: {model_path}", "SUCCESS")

            # Save features
            feature_path = os.path.join(output_dir, f"{symbol}_features_{run_id}.parquet")
            df.dropna().to_parquet(feature_path, compression="snappy")
            debug_print(f"Features saved: {feature_path}", "SUCCESS")

            successful_count += 1
            print(f"✅ Processed {symbol} ({i}/{len(process_tickers)})")

        except Exception as e:
            debug_print(f"Processing failed for {symbol}: {str(e)}", "ERROR")
            print(f"❌ Skipped {symbol}: {e}")
            failed_verbose.append((symbol, str(e)))

    # Summary
    debug_print(f"=== PROCESSING SUMMARY ===", "SUCCESS")
    debug_print(f"Successful: {successful_count}/{len(process_tickers)}", "SUCCESS")

    if failed_verbose:
        verbose_log_path = os.path.join(output_dir, f"skipped_verbose_{run_id}.txt")
        with open(verbose_log_path, "w") as f:
            for symbol, err in failed_verbose:
                f.write(f"{symbol}\t{err}\n")
        print(f"📝 Verbose skipped tickers saved: {len(failed_verbose)}")


def main(ticker_file, train=True):
    """🚀 Main execution"""
    debug_print(f"=== MAIN EXECUTION START ===", "INFO")

    tickers = load_ticker_list(ticker_file)
    print(f"📦 Loaded {len(tickers)} tickers from {ticker_file}")

    if not tickers:
        debug_print("No tickers found", "ERROR")
        return

    # Get metadata
    try:
        enriched_df = enrich_with_yfinance(tickers[:100])  # Limit for speed
        if enriched_df.empty or "symbol" not in enriched_df.columns:
            debug_print("Enrichment failed", "WARNING")
            enriched_df = pd.DataFrame()
        else:
            debug_print(f"Enrichment successful: {len(enriched_df)} stocks", "SUCCESS")
            enriched_df.set_index("symbol", inplace=True)

    except Exception as e:
        debug_print(f"Enrichment failed: {e}", "ERROR")
        enriched_df = pd.DataFrame()

    target_dir = TRAIN_DIR if train else TEST_DIR
    process_ticker_list(tickers, output_dir=target_dir, train=train,
                        meta_df=enriched_df if not enriched_df.empty else None)


if __name__ == "__main__":
    """🚀 Main execution"""

    if debug:
        print("🐛 DEBUG MODE ENABLED")
        if debug_stock:
            print(f"🎯 Debug stock filter: {debug_stock}")
        print("=" * 60)

    if quick_mode:
        print("⚡ QUICK MODE: Using quality fallback stocks only")

    log_run(
        f"COMPREHENSIVE: Fetching ALL NASDAQ stocks. Debug={'ON' if debug else 'OFF'}, Quick={'ON' if quick_mode else 'OFF'}")

    # Generate comprehensive ticker lists
    print("🚀 GENERATING COMPREHENSIVE NASDAQ TICKER LISTS")
    train_tickers, test_tickers = generate_comprehensive_nasdaq_ticker_lists(
        train_file="nasdaq_train_comprehensive.txt",
        test_file="nasdaq_test_comprehensive.txt"
    )

    # Process training data
    if train_tickers and (not debug_stock or debug_stock in train_tickers):
        print(f"\n🚀 Processing {len(train_tickers)} training tickers...")
        main("nasdaq_train_comprehensive.txt", train=True)

    # Process testing data
    if test_tickers and (not debug_stock or debug_stock in test_tickers):
        print(f"\n🚀 Processing {len(test_tickers)} testing tickers...")
        main("nasdaq_test_comprehensive.txt", train=False)

    print("✅ COMPREHENSIVE PROCESSING COMPLETE!")
    print(f"📊 Total stocks processed: {len(train_tickers) + len(test_tickers)}")