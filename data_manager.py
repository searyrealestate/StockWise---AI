import os
import pandas as pd
import re
import ta  # ✅ Add this
from typing import Optional

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

            # 🛠️ Safely handle index and 'Date' column
            if df.index.name and df.index.name.lower() == "date":
                if "Date" not in df.columns:
                    df = df.reset_index()
                else:
                    df.index.name = None  # Avoid conflict
            elif "Date" in df.columns:
                pass  # Already good

            df.rename(columns={"Date": "Datetime"}, inplace=True)
            df["symbol"] = symbol

            # ✅ Add technical indicators
            df = self._add_technical_indicators(df)

            return df
        except Exception as e:
            print(f"[{self.label}] Failed to load {symbol}: {e}")
            return None

    def combine_feature_files(self, symbols: list[str]) -> pd.DataFrame:
        combined = []
        for symbol in symbols:
            df = self.load_feature_file(symbol)
            if df is not None and self._validate_columns(df):
                combined.append(df)
        if not combined:
            raise ValueError("No valid feature files found.")
        return pd.concat(combined, ignore_index=True)

    def _validate_columns(self, df: pd.DataFrame) -> bool:
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            print(f"[{self.label}] Missing columns: {missing}")
            return False
        return True

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        try:
            df["rsi"] = ta.momentum.RSIIndicator(close=df["Close"]).rsi()
            df["macd"] = ta.trend.MACD(close=df["Close"]).macd_diff()
            bb = ta.volatility.BollingerBands(close=df["Close"])
            df["bb_high"] = bb.bollinger_hband()
            df["bb_low"] = bb.bollinger_lband()
            df["adx"] = ta.trend.ADXIndicator(high=df["High"], low=df["Low"], close=df["Close"]).adx()
            df["obv"] = ta.volume.OnBalanceVolumeIndicator(close=df["Close"], volume=df["Volume"]).on_balance_volume()
            df["sma_10"] = ta.trend.SMAIndicator(close=df["Close"], window=10).sma_indicator()
            df["ema_20"] = ta.trend.EMAIndicator(close=df["Close"], window=20).ema_indicator()

            df.dropna(inplace=True)  # ✅ Clean up NaNs from indicators
        except Exception as e:
            print(f"[{self.label}] Failed to compute indicators: {e}")

        return df


if __name__ == "__main__":
    TRAIN_FEATURE_DIR = "models/NASDAQ-testing set"
    TEST_FEATURE_DIR = "models/NASDAQ-training set"

    train_data_manager = DataManager(TRAIN_FEATURE_DIR, label="Train")
    test_data_manager = DataManager(TEST_FEATURE_DIR, label="Test")

    symbols = train_data_manager.get_available_symbols()
    print("Available symbols:", symbols[:5])

    df = train_data_manager.combine_feature_files(symbols[:10])
    print("Combined shape:", df.shape)

