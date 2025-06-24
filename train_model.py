# train_model.py

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import yfinance as yf
from xgboost import XGBClassifier
import matplotlib.pyplot as plt


def compute_rsi(close, window=14):
    # Manual RSI Calculation
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(close, fast=12, slow=26, signal=9):
    # Manual MACD Calculation
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist


def load_and_prepare_data(symbol, start_date="2022-01-01", end_date="2024-12-31"):
    df = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True)

    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["MA50"] = df["Close"].rolling(window=50).mean()
    df["Volatility"] = df["Close"].rolling(window=10).std()
    df["RSI"] = compute_rsi(df["Close"])


    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = compute_macd(df["Close"])

    df["Future_Return"] = df["Close"].shift(-15) / df["Close"] - 1
    df["Target"] = (df["Future_Return"] > 0.05).astype(int)

    # Lag features: yesterday's signals
    df["RSI_lag1"] = df["RSI"].shift(1)
    df["MACD_lag1"] = df["MACD"].shift(1)
    df["MACD_Signal_lag1"] = df["MACD_Signal"].shift(1)
    df["Close_lag1"] = df["Close"].shift(1)
    df["Return_lag1"] = df["Close"].pct_change().shift(1)

    return df.dropna()


def train_model(df):
    import matplotlib.pyplot as plt

    # Define features
    features = [
        "MA20", "MA50", "Volatility", "RSI", "MACD", "MACD_Signal", "MACD_Hist",
        "RSI_lag1", "MACD_lag1", "MACD_Signal_lag1", "Close_lag1", "Return_lag1"
    ]

    # Split features and labels
    X = df[features]
    y = df["Target"]

    # Balance dataset with SMOTE
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, shuffle=True, random_state=42
    )

    # XGBoost Classifier
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluation
    print("=== Performance Report ===")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # Feature importance
    importance = model.feature_importances_
    feature_names = [str(col) if not isinstance(col, str) else col for col in X.columns]
    for name, score in zip(feature_names, importance):
        print(f"{name}: {score:.4f}")

    # Plot importance
    plt.figure(figsize=(8, 6))
    plt.barh(feature_names, importance)
    plt.title("Feature Importance (XGBoost)")
    plt.tight_layout()
    plt.show()

    return model



if __name__ == "__main__":
    df = load_and_prepare_data("AAPL")
    model = train_model(df)

