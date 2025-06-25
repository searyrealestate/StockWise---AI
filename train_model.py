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
import datetime
import streamlit as st

debug = False

plt.close('all')  # clears old plots

original_features = [
        "MA20", "MA50", "Volatility", "RSI", "MACD", "MACD_Signal", "MACD_Hist",
        "RSI_lag1", "MACD_lag1", "MACD_Signal_lag1", "Close_lag1", "Return_lag1"
    ]

enriched_features = original_features + [
    "Bullish_Engulfing", "Hammer", "Pct_from_20d_high", "Pct_from_20d_low",
    "OBV", "OBV_10_MA", "Correl_with_SPY_10"
]


def version():
    v = "SW version: 0.0.6"
    st.write(v)
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
        st.write(f"\n=== {name} ===")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.write(classification_report(y_test, y_pred))
        st.write("Accuracy:", accuracy_score(y_test, y_pred))

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

    # Get recent data (last 90 days to be safe)
    end_date = datetime.datetime.today()
    start_date = end_date - datetime.timedelta(days=90)
    df = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True)
    df = df.reset_index().set_index("Date")

    # Ensure there's enough data to compute rolling features
    if len(df) < window:
        st.write(f"Not enough data to predict for {symbol}")
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
    st.write(f"\n📈 Prediction for {symbol}: {label}")
    st.write(f"Confidence: {probability:.2%}")


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
        probs = model.predict_proba(X_test)[:, 1]

        # Save to df for plotting later
        df.loc[X_test.index, f"{name}_Predicted"] = y_pred
        df.loc[X_test.index, f"{name}_Prob"] = probs
        acc = accuracy_score(y_test, y_pred)

        st.write(f"\n=== {name} ===")
        st.write(f"Accuracy: {acc:.4f}")
        st.write(classification_report(y_test, y_pred))
        # Evaluate
        st.write("=== Performance Report ===")
        st.write(classification_report(y_test, y_pred))

    # Run both comparisons
    run_model(original_features, "Baseline Feature Set")
    run_model(enriched_features, "Enriched Feature Set")


def train_model(df, plot_type=0):

    # Define features
    features = [
        "MA20", "MA50", "Volatility", "RSI", "MACD", "MACD_Signal", "MACD_Hist",
        "RSI_lag1", "MACD_lag1", "MACD_Signal_lag1", "Close_lag1", "Return_lag1",
        "Bullish_Engulfing", "Hammer", "Pct_from_20d_high", "Pct_from_20d_low",
        "OBV", "OBV_10_MA", "Correl_with_SPY_10"
    ]

    # Original (unbalanced) data for time-aligned predictions
    X_orig = df[features]
    y_orig = df["Target"]
    X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
        X_orig, y_orig, test_size=0.2, random_state=42, shuffle=True
    )

    # Resample and train using balanced data
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_orig, y_orig)
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, shuffle=True, random_state=42
    )

    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9,
        eval_metric="logloss", use_label_encoder=False,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Evaluate on balanced test set
    y_pred = model.predict(X_test)
    with st.expander("XGBClassifier", True):
        st.write("=== Performance Report ===")
        st.write(classification_report(y_test, y_pred))
        st.write("Accuracy:", accuracy_score(y_test, y_pred))

    # === Feature Importance Plot ===
    importance = model.feature_importances_
    feature_names = [str(col) if not isinstance(col, str) else col for col in X_orig.columns]

    if plot_type in ["All", "Feature Importance"]:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(feature_names, importance)
        ax.set_title("Feature Importance (XGBoost)")
        fig.tight_layout()
        st.pyplot(fig)

    # === Predict on unbalanced X_test for timeline-aligned plotting ===
    y_pred_orig = model.predict(X_test_orig)
    y_prob_orig = model.predict_proba(X_test_orig)[:, 1]

    df.loc[X_test_orig.index, "Predicted"] = y_pred_orig
    df.loc[X_test_orig.index, "Prob"] = y_prob_orig

    return model


def feature_drop_test(df, base_features, new_features):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from xgboost import XGBClassifier
    from imblearn.over_sampling import SMOTE

    st.write("=== Feature Drop Impact Test ===")

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
    st.write(f"\nFull enriched feature set accuracy: {base_acc:.4f}")

    # Drop each feature one-by-one
    drops = {}
    for f in new_features:
        reduced = [feat for feat in full_set if feat != f]
        acc = train(reduced)
        delta = base_acc - acc
        drops[f] = delta
        st.write(f"Dropped {f:25s} → accuracy: {acc:.4f} | Δ: {delta:+.4f}")

    # Optional: print sorted impact
    st.write("\n=== Sorted Impact (Descending) ===")
    for feat, delta in sorted(drops.items(), key=lambda x: -x[1]):
        st.write(f"{feat:25s} → Δ accuracy: {delta:+.4f}")


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
            st.write(f"Skipping → Train: {len(train_data)} | Test: {len(test_data)} | Date: {test_end.strftime('%Y-%m')}")
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
        st.write(f"{test_end.strftime('%Y-%m')} → Accuracy: {acc:.4f}")

        current += relativedelta(months=1)

    if not results:
        st.write("No valid rolling windows found. Try decreasing window size or using more data.")
        return

    # Plot results
    months, accs = zip(*results)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(months, accs, marker="o")
    ax.set_title("Rolling Window Accuracy Over Time")
    ax.set_ylabel("Accuracy")
    ax.set_xticklabels(months, rotation=45)
    ax.grid(True)
    fig.tight_layout()

    st.pyplot(fig)


def plot_price_with_signals(df, symbol="AAPL"):
    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot the price line
    ax.plot(df.index, df["Close"], label="Close Price", color="blue")

    # Correct & incorrect buy predictions
    correct = df[(df["Predicted"] == 1) & (df["Target"] == 1)]
    incorrect = df[(df["Predicted"] == 1) & (df["Target"] == 0)]

    ax.scatter(correct.index, correct["Close"], marker="^", color="green", label="Correct Buy Signal", alpha=0.7)
    ax.scatter(incorrect.index, incorrect["Close"], marker="v", color="red", label="Wrong Buy Signal", alpha=0.7)

    ax.set_title(f"{symbol} Price with Buy Predictions")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    st.pyplot(fig)


def plot_prediction_confidence(df, debug=False):
    df = df.copy()

    if "Prob" not in df.columns:
        st.warning("Missing 'Prob' column in DataFrame. Make sure model predictions were stored.")
        return

    df["Confidence"] = df["Prob"]  # assume this came from predict_proba

    if debug:
        st.write("✅ Prob in df:", "Prob" in df.columns)
        st.write("🔍 Sample Prob values:")
        st.dataframe(df["Prob"].dropna().head())

    fig, ax = plt.subplots(figsize=(12, 5))
    scatter = ax.scatter(
        df.index, df["Confidence"],
        c=df["Target"], cmap="coolwarm",
        s=40, alpha=0.8
    )
    cbar = fig.colorbar(scatter, ax=ax, label="Actual Target (0 = No Gain, 1 = Gain)")
    ax.set_title("Prediction Confidence Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Predicted Probability of Gain")
    ax.grid(True)
    fig.tight_layout()

    st.pyplot(fig)


def plot_signal_comparison(df, symbol="AAPL"):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(13, 6))
    plt.plot(df.index, df["Close"], label="Close Price", color="black")

    # Enriched model signals
    enriched_correct = df[(df["Enriched Feature Set_Predicted"] == 1) & (df["Target"] == 1)]
    enriched_wrong = df[(df["Enriched Feature Set_Predicted"] == 1) & (df["Target"] == 0)]

    # Baseline model signals
    base_correct = df[(df["Baseline Feature Set_Predicted"] == 1) & (df["Target"] == 1)]
    base_wrong = df[(df["Baseline Feature Set_Predicted"] == 1) & (df["Target"] == 0)]

    plt.scatter(base_correct.index, base_correct["Close"], marker="o", color="gray", label="Baseline Correct", alpha=0.5)
    plt.scatter(base_wrong.index, base_wrong["Close"], marker="o", edgecolor="gray", facecolor="none", label="Baseline Wrong", alpha=0.5)

    plt.scatter(enriched_correct.index, enriched_correct["Close"], marker="^", color="green", label="Enriched Correct", alpha=0.7)
    plt.scatter(enriched_wrong.index, enriched_wrong["Close"], marker="v", color="red", label="Enriched Wrong", alpha=0.7)

    plt.title(f"{symbol} Price with Baseline vs Enriched Model Signals")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=True)

def page_config(page_title, page_icon=":desert_island:"):
    st.title(page_title)

    # Streamlit page config
    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )


if __name__ == "__main__":
    page_config("Stock Model Dashboard")

    # 🧭 User controls which plot(s) to show
    # plot_type = 3  # Change this to 1, 2, or 3 as needed
    st.sidebar.write("\n📊 Available Plot Modes (set plot_type):")
    st.sidebar.write("All = All plots (Feature Importance, Prediction Confidence, Signal Accuracy)")
    st.sidebar.write("Feature Importance = Only Feature Importance")
    st.sidebar.write("Confidence Heatmap = Only Prediction Confidence (heatmap)")
    st.sidebar.write("Signal Accuracy = Only Accuracy (price + signal overlay)")
    plot_option = st.sidebar.radio("Choose a plot", ["All", "Feature Importance", "Confidence Heatmap", "Signal Accuracy"])
    column1, column2 = st.sidebar.columns(2)
    symbol = column1.text_input("Enter stock symbol", value="AAPL")
    run_button = st.sidebar.button("Run")
    if not run_button:
        st.stop()
    else:
        df = load_and_prepare_data(symbol)

        model = train_model(df, plot_type=plot_option)  # Pass plot_type into train_model

        if plot_option == "All" or plot_option == "Confidence Heatmap":
            plot_prediction_confidence(df)
        if plot_option == "All" or plot_option == "Signal Accuracy":
            plot_price_with_signals(df, symbol=symbol)

        # rolling_window_backtest(df, features=enriched_features)

