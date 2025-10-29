import os
import pandas as pd
import re
from typing import Optional
from tqdm import tqdm


class DataManager:
    """
    Handles loading, validating, and combining pre-processed stock feature files.
    Its only job is to load the Parquet files as they are.
    """

    def __init__(self, feature_dir: str, label: str = "DataManager"):
        self.feature_dir = feature_dir
        self.label = label

    def get_available_symbols(self) -> list:
        """Scans the feature directory and returns a sorted list of unique stock symbols."""
        if not os.path.exists(self.feature_dir):
            print(f"[{self.label}] Directory not found: {self.feature_dir}")
            return []
        files = os.listdir(self.feature_dir)
        symbols = []
        for f in files:
            if f.endswith(".parquet"):
                match = re.match(r"([A-Z]+)_daily_context.*\.parquet", f)
                if match:
                    symbols.append(match.group(1))
        return sorted(list(set(symbols)))

    def load_feature_file(self, symbol: str) -> Optional[pd.DataFrame]:
        """Loads the most recent Parquet feature file for a single symbol without altering it."""
        candidates = [
            f for f in os.listdir(self.feature_dir)
            if f.startswith(f"{symbol}_daily_context") and f.endswith(".parquet")
        ]
        if not candidates:
            return None

        file_path = os.path.join(self.feature_dir, sorted(candidates)[-1])
        try:
            df = pd.read_parquet(file_path)
            return df
        except Exception as e:
            print(f"[{self.label}] Error loading {file_path}: {e}")
            return None

    def combine_feature_files(self, symbols_to_load: list) -> pd.DataFrame:
        """Loads and combines feature files for a list of symbols into a single DataFrame."""
        all_dfs = []
        print(f"[{self.label}] Attempting to load and combine feature files for {len(symbols_to_load)} symbols...")

        # This progress bar has a descriptive title
        for symbol in tqdm(symbols_to_load, desc=f"Loading {self.label} data"):
            df = self.load_feature_file(symbol)
            if df is not None and not df.empty:
                all_dfs.append(df)

        if not all_dfs:
            print(f"[{self.label}] No valid DataFrames were loaded to combine.")
            return pd.DataFrame()

        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"[{self.label}] Successfully combined data from {len(all_dfs)} symbols.")
        return combined_df


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
