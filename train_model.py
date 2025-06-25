# train_model.py

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import yfinance as yf
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier, LogisticRegression

original_features = [
        "MA20", "MA50", "Volatility", "RSI", "MACD", "MACD_Signal", "MACD_Hist",
        "RSI_lag1", "MACD_lag1", "MACD_Signal_lag1", "Close_lag1", "Return_lag1"
    ]

enriched_features = original_features + [
    "Bullish_Engulfing", "Hammer", "Pct_from_20d_high", "Pct_from_20d_low",
    "OBV", "OBV_10_MA", "Correl_with_SPY_10"
]


def version():
    v = "SW version: 0.0.5"
    print(v)
    return v


def train_multiple_models(df):
    features = [
        "MA20", "MA50", "Volatility", "RSI", "MACD", "MACD_Signal", "MACD_Hist",
        "RSI_lag1", "MACD_lag1", "MACD_Signal_lag1", "Close_lag1", "Return_lag1"
    ]

    X = df[features]
    y = df["Target"]

    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, shuffle=True, random_state=42
    )

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", use_label_encoder=False,
            random_state=42
        ),
        "SGD Classifier": SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
    }

    for name, model in models.items():
        print(f"\n=== {name} ===")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))
        print("Accuracy:", accuracy_score(y_test, y_pred))

    results = []
    # Inside the loop:
    results.append({
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred)
    })


def predict_live(symbol, model, window=50):
    """
    Predict the probability of a stock increasing more than 5% in 15 days using a trained model.

    :param symbol: (str) Stock ticker symbol to fetch live data for (e.g. "AAPL")
    :param model: (sklearn/BaseEstimator) Trained classifier with a .predict() and .predict_proba() method
    :param window: (int) Number of historical days to include for calculating rolling indicators (default is 50)
    :return: None – prints prediction and confidence score to console
    """
    import datetime

    # Get recent data (last 90 days to be safe)
    end_date = datetime.datetime.today()
    start_date = end_date - datetime.timedelta(days=90)
    df = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True)
    df = df.reset_index().set_index("Date")

    # Ensure there's enough data to compute rolling features
    if len(df) < window:
        print(f"Not enough data to predict for {symbol}")
        return

    # Recompute features like in training
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["MA50"] = df["Close"].rolling(window=50).mean()
    df["Volatility"] = df["Close"].rolling(window=10).std()
    df["RSI"] = compute_rsi(df["Close"])
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = compute_macd(df["Close"])
    df["RSI_lag1"] = df["RSI"].shift(1)
    df["MACD_lag1"] = df["MACD"].shift(1)
    df["MACD_Signal_lag1"] = df["MACD_Signal"].shift(1)
    df["Close_lag1"] = df["Close"].shift(1)
    df["Return_lag1"] = df["Close"].pct_change().shift(1)

    df = df.dropna()

    # Extract latest row of features
    latest = df.iloc[-1]

    input_data = pd.DataFrame([{
        "MA20": latest["MA20"],
        "MA50": latest["MA50"],
        "Volatility": latest["Volatility"],
        "RSI": latest["RSI"],
        "MACD": latest["MACD"],
        "MACD_Signal": latest["MACD_Signal"],
        "MACD_Hist": latest["MACD_Hist"],
        "RSI_lag1": latest["RSI_lag1"],
        "MACD_lag1": latest["MACD_lag1"],
        "MACD_Signal_lag1": latest["MACD_Signal_lag1"],
        "Close_lag1": latest["Close_lag1"],
        "Return_lag1": latest["Return_lag1"]
    }])

    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][prediction]

    label = "Increase > 5%" if prediction == 1 else "No significant increase"
    print(f"\n📈 Prediction for {symbol}: {label}")
    print(f"Confidence: {probability:.2%}")


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
    # Download main symbol data and flatten index
    df = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True)
    df = df.reset_index().set_index("Date")

    # Core technical indicators
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["MA50"] = df["Close"].rolling(window=50).mean()
    df["Volatility"] = df["Close"].rolling(window=10).std()
    df["RSI"] = compute_rsi(df["Close"])
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = compute_macd(df["Close"])

    # === Enriched Features ===

    # Candlestick patterns
    df["Bullish_Engulfing"] = (
        (df["Close"] > df["Open"]) &
        (df["Open"].shift(1) > df["Close"].shift(1))
    ).astype(int)

    df["Hammer"] = (
        ((df["High"] - df["Low"]) > 3 * abs(df["Open"] - df["Close"])) &
        (df["Close"] > df["Open"])
    ).astype(int)

    # Price position relative to recent range
    df["Pct_from_20d_high"] = (
        (df["Close"] - df["Close"].rolling(20).max()) /
        df["Close"].rolling(20).max()
    )

    df["Pct_from_20d_low"] = (
        (df["Close"] - df["Close"].rolling(20).min()) /
        df["Close"].rolling(20).min()
    )

    # On-Balance Volume
    df["OBV"] = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()
    df["OBV_10_MA"] = df["OBV"].rolling(window=10).mean()

    # Correlation with SPY (market context)
    spy = yf.download("^GSPC", start=start_date, end=end_date, auto_adjust=True)
    spy = spy.reset_index().set_index("Date")
    spy["SPY_Return"] = spy["Close"].pct_change()
    df["SPY_Return"] = spy["SPY_Return"]
    df["Correl_with_SPY_10"] = df["Close"].pct_change().rolling(10).corr(df["SPY_Return"])

    # Future return target
    df["Future_Return"] = df["Close"].shift(-15) / df["Close"] - 1
    df["Target"] = (df["Future_Return"] > 0.05).astype(int)

    # Lag features
    df["RSI_lag1"] = df["RSI"].shift(1)
    df["MACD_lag1"] = df["MACD"].shift(1)
    df["MACD_Signal_lag1"] = df["MACD_Signal"].shift(1)
    df["Close_lag1"] = df["Close"].shift(1)
    df["Return_lag1"] = df["Close"].pct_change().shift(1)

    return df.dropna()


def compare_models(df):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    from xgboost import XGBClassifier
    from imblearn.over_sampling import SMOTE

    def run_model(feature_set, name):
        X = df[feature_set]
        y = df["Target"]

        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

        model = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", use_label_encoder=False,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))

    # Run both comparisons
    run_model(original_features, "Baseline Feature Set")
    run_model(enriched_features, "Enriched Feature Set")


def train_model(df):
    # Define features
    features = [
        "MA20", "MA50", "Volatility", "RSI", "MACD", "MACD_Signal", "MACD_Hist",
        "RSI_lag1", "MACD_lag1", "MACD_Signal_lag1", "Close_lag1", "Return_lag1",
        "Bullish_Engulfing", "Hammer", "Pct_from_20d_high", "Pct_from_20d_low",
        "OBV", "OBV_10_MA", "Correl_with_SPY_10"
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


def feature_drop_test(df, base_features, new_features):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from xgboost import XGBClassifier
    from imblearn.over_sampling import SMOTE

    print("=== Feature Drop Impact Test ===")

    # Baseline with all enriched features
    full_set = base_features + new_features

    def train(features):
        X = df[features]
        y = df["Target"]
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, y)
        X_train, X_test, y_train, y_test = train_test_split(
            X_res, y_res, test_size=0.2, random_state=42
        )
        model = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", use_label_encoder=False,
            random_state=42
        )
        model.fit(X_train, y_train)
        return accuracy_score(y_test, model.predict(X_test))

    base_acc = train(full_set)
    print(f"\nFull enriched feature set accuracy: {base_acc:.4f}")

    # Drop each feature one-by-one
    drops = {}
    for f in new_features:
        reduced = [feat for feat in full_set if feat != f]
        acc = train(reduced)
        delta = base_acc - acc
        drops[f] = delta
        print(f"Dropped {f:25s} → accuracy: {acc:.4f} | Δ: {delta:+.4f}")

    # Optional: print sorted impact
    print("\n=== Sorted Impact (Descending) ===")
    for feat, delta in sorted(drops.items(), key=lambda x: -x[1]):
        print(f"{feat:25s} → Δ accuracy: {delta:+.4f}")


def rolling_window_backtest(df, features, window_months=6, test_months=1):
    import matplotlib.pyplot as plt
    from dateutil.relativedelta import relativedelta
    from sklearn.metrics import accuracy_score
    from xgboost import XGBClassifier
    from imblearn.over_sampling import SMOTE

    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    start = df.index.min() + relativedelta(months=window_months + test_months)
    end = df.index.max() - relativedelta(months=test_months)

    current = start
    results = []

    while current <= end:
        train_end = current
        test_start = current
        test_end = test_start + relativedelta(months=test_months)

        train_data = df[df.index < train_end]
        test_data = df[(df.index >= test_start) & (df.index < test_end)]

        if len(train_data) < 150 or len(test_data) < 15:
            current += relativedelta(months=1)
            print(f"Skipping → Train: {len(train_data)} | Test: {len(test_data)} | Date: {test_end.strftime('%Y-%m')}")
            continue

        X_train = train_data[features]
        y_train = train_data["Target"]
        X_test = test_data[features]
        y_test = test_data["Target"]

        sm = SMOTE(random_state=42)
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

        model = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", use_label_encoder=False,
            random_state=42
        )
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        results.append((test_end.strftime("%Y-%m"), acc))
        print(f"{test_end.strftime('%Y-%m')} → Accuracy: {acc:.4f}")

        current += relativedelta(months=1)

    if not results:
        print("No valid rolling windows found. Try decreasing window size or using more data.")
        return

    # Plot results
    months, accs = zip(*results)
    plt.figure(figsize=(10, 4))
    plt.plot(months, accs, marker="o")
    plt.title("Rolling Window Accuracy Over Time")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = load_and_prepare_data("AAPL")
    rolling_window_backtest(df, features=enriched_features)

    # feature_drop_test(
    #     df,
    #     base_features=original_features,
    #     new_features=[
    #         "Bullish_Engulfing", "Hammer", "Pct_from_20d_high", "Pct_from_20d_low",
    #         "OBV", "OBV_10_MA", "Correl_with_SPY_10"
    #     ]
    # )
    # compare_models(df)
    # model = train_model(df)
    # # train_multiple_models(df)
    # predict_live("AAPL", model)

