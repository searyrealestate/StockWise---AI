import os
import pandas as pd
import re
import ta
import numpy as np
from typing import Optional
from tqdm import tqdm

REQUIRED_COLUMNS = ["Datetime", "Target"]


class DataManager:
    """
    Handles loading, validating, and combining stock feature files for training and evaluation.
    """

    def __init__(self, feature_dir: str, label: str = "DataManager"):
        self.feature_dir = feature_dir
        self.label = label

    def get_available_symbols(self) -> list:
        files = os.listdir(self.feature_dir)
        symbols = []
        for f in files:
            if f.endswith(".parquet"):
                # Adjust this based on your actual file naming convention
                match = re.match(r"([A-Z]+)_features.*\.parquet", f)

                if match:
                    symbols.append(match.group(1))
        return sorted(set(symbols))

    def load_feature_file(self, symbol: str) -> Optional[pd.DataFrame]:
        candidates = [
            f for f in os.listdir(self.feature_dir)
            if f.startswith(f"{symbol}_features") and f.endswith(".parquet")
        ]
        if not candidates:
            print(f"[{self.label}] No file found for symbol: {symbol}")
            return None

        file_path = os.path.join(self.feature_dir, sorted(candidates)[-1])
        try:
            df = pd.read_parquet(file_path)
            # Ensure 'Datetime' is a datetime object
            if "Datetime" in df.columns:
                df["Datetime"] = pd.to_datetime(df["Datetime"])
            else:
                print(f"[{self.label}] Warning: 'Datetime' column not found in {file_path}")
                return None  # Or handle appropriately if Datetime is crucial

            # Sort by Datetime to ensure correct order for indicator calculation
            df = df.sort_values(by="Datetime").reset_index(drop=True)

            # --- Add Technical Indicators ---
            df = self._add_technical_indicators(df)

            # Ensure 'Target' column exists and is numeric, handle potential NaNs
            if "Target" not in df.columns:
                print(f"[{self.label}] Warning: 'Target' column not found for {symbol}. Skipping.")
                return None

            # Convert Target to numeric, coercing errors to NaN, then drop NaNs
            df["Target"] = pd.to_numeric(df["Target"], errors='coerce')
            df.dropna(subset=[self.label], inplace=True)  # Drop rows where target is NaN

            # Ensure expected feature columns are numeric, fill NaNs if any after indicator calc
            # This is a safe guard. It's better if indicators don't produce NaNs in final features.
            for col in ['Volume_MA_20', 'RSI_14', 'Momentum_5', 'MACD', 'MACD_Signal',
                        'MACD_Histogram', 'BB_Upper', 'BB_Lower', 'BB_Middle',
                        'BB_Position', 'Daily_Return', 'Volatility_20D']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col].fillna(df[col].mean(), inplace=True)  # Fill with mean, or 0, or drop row
                else:
                    # If a core feature is missing after calculation, log and consider returning None
                    print(
                        f"[{self.label}] Critical: Feature '{col}' is missing for {symbol} after indicator calculation.")

            return df
        except Exception as e:
            print(f"[{self.label}] Error loading or processing {file_path}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def combine_feature_files(self, symbols_to_load: list) -> pd.DataFrame:
        all_dfs = []
        print(f"[{self.label}] Attempting to combine feature files for {len(symbols_to_load)} symbols...")
        # Use tqdm directly here
        for symbol in tqdm(symbols_to_load, desc=f"Loading {self.label} data"):  # <--- tqdm used here
            df = self.load_feature_file(symbol)
            if df is not None and not df.empty:
                all_dfs.append(df)
            # else: # Optional: log symbols that failed to load
            #     print(f"[{self.label}] Skipping {symbol} due to load/process error or empty data.")

        if not all_dfs:
            print(f"[{self.label}] No valid dataframes to combine.")
            return pd.DataFrame()  # Return empty DataFrame if nothing was loaded

        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"[{self.label}] Successfully combined data from {len(all_dfs)} symbols.")
        return combined_df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Ensure essential columns exist for TA calculations
        required_ta_cols = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in df.columns for col in required_ta_cols):
            print(
                f"[{self.label}] Missing one or more required columns for TA calculation: {required_ta_cols}. Skipping indicators.")
            return df  # Return original DataFrame if essential columns are missing

        try:
            # Re-calculating all 12 expected features here to ensure consistency
            df["Volume_MA_20"] = df["Volume"].rolling(window=20).mean()  # Example: 20-day Volume MA
            df["RSI_14"] = ta.momentum.RSIIndicator(close=df["Close"], window=14).rsi()
            df["Momentum_5"] = ta.momentum.ROCIndicator(close=df["Close"], window=5).roc()  # Rate of Change (Momentum)

            macd = ta.trend.MACD(close=df["Close"])
            df["MACD"] = macd.macd()
            df["MACD_Signal"] = macd.macd_signal()
            df["MACD_Histogram"] = macd.macd_diff()

            bb = ta.volatility.BollingerBands(close=df["Close"])
            df["BB_Upper"] = bb.bollinger_hband()
            df["BB_Lower"] = bb.bollinger_lband()
            df["BB_Middle"] = bb.bollinger_mband()
            df["BB_Position"] = bb.bollinger_percent_b()  # %B (position within bands)

            df["Daily_Return"] = df["Close"].pct_change()  # Daily percentage return
            df["Volatility_20D"] = df["Daily_Return"].rolling(window=20).std()  # 20-day Volatility

            df.dropna(inplace=True)  # Clean up NaNs from indicators
        except Exception as e:
            print(f"[{self.label}] Failed to compute indicators: {e}")
            import traceback
            traceback.print_exc()  # Print full traceback for debugging

        return df


if __name__ == "__main__":
    TRAIN_FEATURE_DIR = "models/NASDAQ-training set"
    TEST_FEATURE_DIR = "models/NASDAQ-testing set"  # Corrected this from 'training set' based on context

    train_data_manager = DataManager(TRAIN_FEATURE_DIR, label="Train")
    test_data_manager = DataManager(TEST_FEATURE_DIR, label="Test")

    symbols = train_data_manager.get_available_symbols()
    print("Available symbols (first 5):", symbols[:5])

    # Test loading a few files and ensure indicators are added
    # For testing, you might want to pick a symbol you know has enough data
    test_symbol = "AAPL"  # Example symbol
    print(f"\n--- Testing data loading and indicators for {test_symbol} ---")
    df_test = train_data_manager.load_feature_file(test_symbol)
    if df_test is not None and not df_test.empty:
        print(f"DataFrame for {test_symbol} shape: {df_test.shape}")
        print(f"Columns after indicators: {df_test.columns.tolist()}")
        # Check if the critical columns are present
        expected_cols_after_indicators = [
            'Volume_MA_20', 'RSI_14', 'Momentum_5', 'MACD', 'MACD_Signal',
            'MACD_Histogram', 'BB_Upper', 'BB_Lower', 'BB_Middle',
            'BB_Position', 'Daily_Return', 'Volatility_20D', 'Target', 'Datetime'
        ]
        all_present = True
        for col in expected_cols_after_indicators:
            if col not in df_test.columns:
                print(f"⚠️ Missing expected column '{col}' after indicator calculation.")
                all_present = False
        if all_present:
            print("✅ All expected indicator columns are present.")
        else:
            print("❌ Some expected indicator columns are missing.")
    else:
        print(f"Failed to load or process data for {test_symbol}.")

    # Additional test for combining files
    print("\n--- Testing combine_feature_files ---")
    some_symbols = symbols[:5]  # Take a few symbols
    combined_df = train_data_manager.combine_feature_files(some_symbols)
    if not combined_df.empty:
        print(f"Combined DataFrame shape: {combined_df.shape}")
        print(f"Combined DataFrame columns (first 5): {combined_df.columns.tolist()[:5]}...")
        print(f"Combined DataFrame head:\n{combined_df.head()}")
    else:
        print("Combined DataFrame is empty.")
