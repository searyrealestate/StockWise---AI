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
import plotly.express as px
import plotly.graph_objects as go
from dateutil.relativedelta import relativedelta
import uuid
from plotly.subplots import make_subplots

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
    v = "SW version: 0.0.10"
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

    # if isinstance(df.columns, pd.MultiIndex):
    #     df.columns = ['_'.join(filter(None, map(str, col))).strip() if isinstance(col, tuple) else col for col in
    #                   df.columns]
    # 🔧 Final flatten to ensure all columns are usable downstream
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(filter(None, map(str, col))) if isinstance(col, tuple) else col for col in df.columns]

    if debug:
        st.write("✅ Column names after flattening in load_and_prepare_data():", df.columns.tolist())

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

def compute_bollinger_bandwidth(df, symbol, window=20, num_std=2):
    close = df[f"Close_{symbol}"]
    ma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    df["Bollinger_Width"] = (upper - lower) / ma
    return df


def compute_atr(df, symbol, window=14):
    high = df[f"High_{symbol}"]
    low = df[f"Low_{symbol}"]
    close = df[f"Close_{symbol}"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(window=window).mean()
    return df


def train_model(df, symbol, plot_type=0):
    # === Add technical features directly ===
    df = compute_bollinger_bandwidth(df, symbol)
    df = compute_atr(df, symbol)

    expected_features = [
        "MA20", "MA50", "Volatility", "RSI", "MACD", "MACD_Signal", "MACD_Hist",
        "RSI_lag1", "MACD_lag1", "MACD_Signal_lag1", "Close_lag1", "Return_lag1",
        "Bullish_Engulfing", "Hammer", "Pct_from_20d_high", "Pct_from_20d_low",
        "OBV", "OBV_10_MA", "Correl_with_SPY_10",
        "Volume_Relative", "Volume_Spike", "Turnover",
        "ATR", "Bollinger_Width"
    ]
    features = [f for f in expected_features if f in df.columns]

    if debug:
        st.write("📌 Using features:", features)

    X = df[features]
    y = df["Target"]

    X = X.dropna()
    y = y.loc[X.index]

    X_eval, _, y_eval, _ = train_test_split(X, y, test_size=0.8, random_state=42)

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred_eval = model.predict(X_eval)

    if debug:
        with st.expander("XGBClassifier", expanded=True):
            st.write("=== Performance Report ===")
            st.write(classification_report(y_eval, y_pred_eval))
            st.write("Accuracy:", accuracy_score(y_eval, y_pred_eval))

            if plot_type in ["All", "Feature Importance"]:
                importance = model.feature_importances_
                importance_df = pd.DataFrame({
                    "Feature": features,
                    "Importance": importance
                }).sort_values(by="Importance", ascending=True)

                fig = px.bar(
                    importance_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="🔍 Feature Importance (XGBoost)",
                    height=600
                )
                fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig, use_container_width=True, key=f"plot_{uuid.uuid4()}")

    df["Predicted"] = pd.Series(model.predict(X), index=X.index).astype(int)
    df["Prob"] = pd.Series(model.predict_proba(X)[:, 1], index=X.index).astype(float)

    df = add_smart_entry_signal(df, prob_threshold=0.7, max_bollinger_width=0.08, volume_confirm=True, debug=debug)

    if debug:
        st.write("🎯 Sample predictions:", df[["Predicted", "Prob", "Entry_Signal"]].head())

    return model, df



def add_smart_entry_signal(df, prob_threshold=0.7, max_bollinger_width=0.08, volume_confirm=True, debug=False):
    """
    Adds Entry_Signal column to df based on multi-condition filters:
    - Model confidence (Prob)
    - Volume confirmation (Volume_Spike)
    - Volatility compression (Bollinger Width)

    Parameters:
        df: pd.DataFrame - your prediction-enhanced dataframe
        prob_threshold: float - minimum model confidence
        max_bollinger_width: float - threshold for Bollinger squeeze
        volume_confirm: bool - require Volume_Spike == 1
        debug: bool - print out activation stats

    Returns:
        df: pd.DataFrame with added Entry_Signal column
    """
    condition = (
        (df["Prob"] > prob_threshold) &
        (df["Bollinger_Width"] < max_bollinger_width)
    )
    if volume_confirm and "Volume_Spike" in df.columns:
        condition &= (df["Volume_Spike"] == 1)

    df["Entry_Signal"] = condition.astype(int)

    if debug:
        total_signals = df["Entry_Signal"].sum()
        st.write(f"🎯 Smart Entry Signals: {total_signals} triggered")
        st.write(df[["Prob", "Bollinger_Width", "Volume_Spike", "Entry_Signal"]].head(10))

    return df


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
    from dateutil.relativedelta import relativedelta

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
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index, y=df[f"Close_{symbol}"],
        mode="lines", name="Close Price", line=dict(color="blue")
    ))

    correct = df[(df["Predicted"] == 1) & (df["Target"] == 1)]
    incorrect = df[(df["Predicted"] == 1) & (df["Target"] == 0)]

    fig.add_trace(go.Scatter(
        x=correct.index, y=correct[f"Close_{symbol}"],
        mode="markers", name="Correct Buy",
        marker=dict(color="green", symbol="triangle-up", size=10)
    ))

    fig.add_trace(go.Scatter(
        x=incorrect.index, y=incorrect[f"Close_{symbol}"],
        mode="markers", name="Wrong Buy",
        marker=dict(color="red", symbol="triangle-down", size=10)
    ))

    fig.update_layout(
        title=f"{symbol} Price with Buy Predictions",
        xaxis_title="Date", yaxis_title="Price",
        legend=dict(x=0, y=1),
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True, key=f"plot_{uuid.uuid4()}")


def plot_prediction_confidence(df):
    df = df.copy()
    # 🔧 Flatten MultiIndex columns if necessary
    # if isinstance(df.columns, pd.MultiIndex):
    #     df.columns = ['_'.join(c).strip() if isinstance(c, tuple) else c for c in df.columns]
    # else:
    #     df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    if "Prob" not in df.columns:
        st.warning("Missing 'Prob' column in DataFrame. Make sure model predictions were stored.")
        return

    df["Confidence"] = df["Prob"]

    if debug:
        st.write("✅ 'Prob' present:", "Prob" in df.columns)
        st.dataframe(df[["Confidence", "Target"]].head())

    df["Target_Label"] = df["Target"].map({0: "No Gain", 1: "Gain"})

    fig = px.scatter(
        df,
        x=df.index,
        y="Confidence",
        color="Target_Label",
        title="Prediction Confidence Over Time",
        labels={"Confidence": "Model Confidence", "color": "Actual Target"},
        color_discrete_map={"No Gain": "red", "Gain": "green"},
        opacity=0.7
    )

    fig.update_traces(marker=dict(size=8))
    fig.update_layout(hovermode="x unified")

    st.plotly_chart(fig, use_container_width=True, key=f"plot_{uuid.uuid4()}")


def plot_signal_comparison(df, symbol="AAPL"):
    fig = go.Figure()

    close_col = f"Close_{symbol}"

    fig.add_trace(go.Scatter(
        x=df.index, y=df[close_col],
        mode="lines", name="Close Price", line=dict(color="black")
    ))

    # Enriched model signals
    enriched_correct = df[(df["Enriched Feature Set_Predicted"] == 1) & (df["Target"] == 1)]
    enriched_wrong = df[(df["Enriched Feature Set_Predicted"] == 1) & (df["Target"] == 0)]

    fig.add_trace(go.Scatter(
        x=enriched_correct.index, y=enriched_correct[close_col],
        mode="markers", name="Enriched Correct",
        marker=dict(color="green", symbol="triangle-up", size=10)
    ))

    fig.add_trace(go.Scatter(
        x=enriched_wrong.index, y=enriched_wrong[close_col],
        mode="markers", name="Enriched Wrong",
        marker=dict(color="red", symbol="triangle-down", size=10)
    ))

    # Baseline model signals
    base_correct = df[(df["Baseline Feature Set_Predicted"] == 1) & (df["Target"] == 1)]
    base_wrong = df[(df["Baseline Feature Set_Predicted"] == 1) & (df["Target"] == 0)]

    fig.add_trace(go.Scatter(
        x=base_correct.index, y=base_correct[close_col],
        mode="markers", name="Baseline Correct",
        marker=dict(color="gray", symbol="circle", size=8)
    ))

    fig.add_trace(go.Scatter(
        x=base_wrong.index, y=base_wrong[close_col],
        mode="markers", name="Baseline Wrong",
        marker=dict(color="gray", symbol="x", size=8)
    ))

    fig.update_layout(
        title=f"{symbol} Price with Baseline vs Enriched Model Signals",
        xaxis_title="Date",
        yaxis_title="Price",
        legend=dict(x=0, y=1),
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True, key=f"plot_{uuid.uuid4()}")


def page_config(page_title, page_icon=":desert_island:"):
    st.title(page_title)

    # Streamlit page config
    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )


def evaluate_exit(entry_price, future_prices, method="tp_sl", **kwargs):
    """
    Exit strategy logic: decides when to sell a trade.
    """
    if method == "tp_sl":
        tp_pct = kwargs.get("take_profit_pct", 0.07)
        sl_pct = kwargs.get("stop_loss_pct", 0.03)
        for i, price in enumerate(future_prices):
            change_pct = (price - entry_price) / entry_price
            if change_pct >= tp_pct:
                return i, price, "TP"
            elif change_pct <= -sl_pct:
                return i, price, "SL"

    elif method == "trailing_stop":
        trail_pct = kwargs.get("trail_pct", 0.03)
        peak = entry_price
        for i, price in enumerate(future_prices):
            peak = max(peak, price)
            drop_pct = (price - peak) / peak
            if drop_pct <= -trail_pct:
                return i, price, "Trailing Stop"

    return len(future_prices) - 1, future_prices[-1], "Timed"


def simulate_trades(df, symbol, take_profit_pct=0.07, stop_loss_pct=0.03,
                    max_hold_days=15, min_confidence=0.5, show_plot=True):
    trades = []
    df = df.copy()

    # Filter only smart entries
    entries = df[df["Entry_Signal"] == 1]

    for entry_time, row in entries.iterrows():
        if row["Prob"] < min_confidence:
            continue

        entry_price = row[f"Close_{symbol}"]
        hold_days = 0
        exit_reason = "MaxHold"
        exit_time = None
        exit_price = entry_price

        # Simulate forward for max_hold_days
        forward_df = df.loc[entry_time:].iloc[1:max_hold_days+1]
        for future_time, f_row in forward_df.iterrows():
            hold_days += 1
            price = f_row[f"Close_{symbol}"]

            if price >= entry_price * (1 + take_profit_pct):
                exit_price = price
                exit_reason = "TakeProfit"
                exit_time = future_time
                break
            elif price <= entry_price * (1 - stop_loss_pct):
                exit_price = price
                exit_reason = "StopLoss"
                exit_time = future_time
                break
        else:
            # Exited by time
            exit_price = forward_df[f"Close_{symbol}"].iloc[-1]
            exit_time = forward_df.index[-1]

        net_return = (exit_price / entry_price - 1) * 100
        trades.append({
            "Entry Time": entry_time,
            "Entry Price": entry_price,
            "Exit Time": exit_time,
            "Exit Price": exit_price,
            "Hold Days": hold_days,
            "Exit Reason": exit_reason,
            "Net Return %": net_return,
            "Confidence": row["Prob"]
        })

    trades_df = pd.DataFrame(trades).set_index("Entry Time")

    if show_plot and not trades_df.empty:
        st.line_chart(df[f"Close_{symbol}"])
        st.write("✅ Trades simulated:", len(trades_df))
        st.dataframe(trades_df.tail())

    return trades_df


# def test_volume_chart(df, volume_col):
#     import plotly.graph_objects as go
#     import streamlit as st
#
#     df = df.sort_index()
#     df = df[~df.index.duplicated()]
#     df[volume_col] = pd.to_numeric(df[volume_col], errors="coerce").fillna(0)
#
#     # st.write("🔍 Volume stats:", df[volume_col].describe())
#
#     fig = go.Figure()
#     fig.add_trace(go.Bar(
#         x=df.index,
#         y=df[volume_col],
#         name="Volume",
#         marker_color="gray",
#         width=0.8
#     ))
#
#     fig.update_layout(
#         title="Volume Only Test",
#         xaxis_title="Date",
#         yaxis_title="Volume",
#         bargap=0.01,
#         height=400
#     )
#
#     st.plotly_chart(fig, use_container_width=True, key="volume_only_test")
def apply_realistic_fees(trades_df, tax_rate=0.25, per_share_fee=0.01, min_fee=2.5):
    """
    Adds real-world brokerage fees and capital gains tax to simulated trades.

    Parameters:
        trades_df : pd.DataFrame — trades with at least 'Net Return %' and 'Shares'
        tax_rate : float — capital gains tax on profits (default: 25%)
        per_share_fee : float — per-share execution fee (default: $0.01)
        min_fee : float — minimum fee per side (buy/sell) per trade

    Returns:
        trades_df : pd.DataFrame — enriched with fee and tax columns
    """

    if "Shares" not in trades_df.columns:
        st.warning("Missing 'Shares' column — assuming 100 shares per trade.")
        trades_df["Shares"] = 100  # Fallback assumption

    # Buy & Sell Execution Fees
    trades_df["Buy Fee"] = trades_df["Shares"].apply(lambda x: max(per_share_fee * x, min_fee))
    trades_df["Sell Fee"] = trades_df["Shares"].apply(lambda x: max(per_share_fee * x, min_fee))

    # Capital Gains Tax: 25% of Net Profit if > 0
    trades_df["Tax Paid"] = trades_df["Net Return %"].apply(
        lambda x: x * tax_rate if x > 0 else 0
    )

    # Adjusted Net Return After Costs
    trades_df["Net Return After Fees"] = (
        trades_df["Net Return %"] - trades_df["Tax Paid"]
        - trades_df["Buy Fee"] - trades_df["Sell Fee"]
    )

    return trades_df


def display_strategy_summary(trades_df):
    if trades_df.empty:
        st.info("No trades to summarize.")
        return

    total_trades = len(trades_df)
    gross_total_return = trades_df["Net Return %"].sum()
    net_total_return = trades_df["Net Return After Fees"].sum()

    buy_fees = trades_df["Buy Fee"].sum() if "Buy Fee" in trades_df else 0.0
    sell_fees = trades_df["Sell Fee"].sum() if "Sell Fee" in trades_df else 0.0
    total_fees = buy_fees + sell_fees

    total_tax = trades_df["Tax Paid"].sum() if "Tax Paid" in trades_df else 0.0
    win_rate = (trades_df["Net Return %"] > 0).mean() * 100
    avg_net_return = trades_df["Net Return After Fees"].mean()

    st.markdown("### 📈 Strategy Summary")

    col1, col2, col3 = st.columns(3)
    col1.metric("📌 Total Trades", total_trades)
    col2.metric("📈 Gross Return (%)", f"{gross_total_return:.2f}")
    col3.metric("💰 Net Return After Fees (%)", f"{net_total_return:.2f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("🏆 Win Rate (%)", f"{win_rate:.2f}")
    col5.metric("📊 Avg Net After Fees", f"{avg_net_return:.2f}%")
    col6.metric("🧾 Tax Paid", f"{total_tax:.2f}")

    st.caption(f"💸 Total Fees Paid: {total_fees:.2f}")


def plot_price_volume_domain(df, trades_df, symbol, zoom_to_return=False):
    close_col = f"Close_{symbol}"
    open_col = f"Open_{symbol}"
    high_col = f"High_{symbol}"
    low_col = f"Low_{symbol}"
    volume_col = f"Volume_{symbol}"

    df = df.sort_index()
    df = df[~df.index.duplicated()]
    df[volume_col] = pd.to_numeric(df[volume_col], errors="coerce").fillna(0)

    volume_colors = [
        "rgba(0,200,0,0.4)" if c >= o else "rgba(200,0,0,0.4)"
        for c, o in zip(df[close_col], df[open_col])
    ]

    fig = go.Figure()

    # 🕯️ Price candles
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df[open_col],
        high=df[high_col],
        low=df[low_col],
        close=df[close_col],
        name="Price",
        increasing_line_color="green",
        decreasing_line_color="red",
        xaxis="x",
        yaxis="y"
    ))

    # 📊 Volume bars
    fig.add_trace(go.Bar(
        x=df.index,
        y=df[volume_col],
        marker_color=volume_colors,
        name="Volume",
        xaxis="x2",
        yaxis="y2"
    ))

    if not trades_df.empty:
        # ✅ Ensure required columns exist
        if "Entry Date" not in trades_df.columns and "Entry Time" in trades_df.columns:
            trades_df["Entry Date"] = pd.to_datetime(trades_df["Entry Time"])
        elif "Entry Date" not in trades_df.columns:
            trades_df["Entry Date"] = trades_df.index

        if "Exit Date" not in trades_df.columns and "Exit Time" in trades_df.columns:
            trades_df["Exit Date"] = pd.to_datetime(trades_df["Exit Time"]).dt.normalize()

        trades_df = trades_df[~trades_df.duplicated(subset=[
            "Entry Date", "Exit Date", "Entry Price", "Exit Price", "Net Return %"
        ])].copy()

        # 🔺 Entries
        fig.add_trace(go.Scatter(
            x=trades_df["Entry Date"],
            y=trades_df["Entry Price"],
            mode="markers",
            name="Entry",
            marker=dict(symbol="triangle-up", size=10, color="#00FFFF"),
            xaxis="x",
            yaxis="y",
            hovertemplate="🟢 Entry: %{x|%b %d, %Y}<br>Price: %{y:.2f}<extra></extra>"
        ))

        # 🔻 Exits
        fig.add_trace(go.Scatter(
            x=trades_df["Exit Date"],
            y=trades_df["Exit Price"],
            mode="markers",
            name="Exit",
            marker=dict(symbol="triangle-down", size=10, color="#FFD700"),
            xaxis="x",
            yaxis="y",
            hovertemplate="🔴 Exit: %{x|%b %d, %Y}<br>Price: %{y:.2f}<extra></extra>"
        ))

        # 📈 Cumulative Return
        trades_df["Return Multiplier"] = 1 + trades_df["Net Return %"].fillna(0) / 100
        daily_returns = trades_df.groupby("Exit Date")["Return Multiplier"].prod().sort_index()
        equity_curve = daily_returns.cumprod()

        full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
        equity_curve = equity_curve.reindex(full_range).ffill().fillna(1.0)
        equity_curve.iloc[0] = 1.0

        if zoom_to_return and (equity_curve > 1.0).any():
            first_move = equity_curve[equity_curve > 1.0].index.min()
            fig.update_xaxes(range=[first_move, df.index.max()])

        fig.add_trace(go.Scatter(
            x=equity_curve.index,
            y=equity_curve,
            mode="lines+markers",
            name="Net Return ×",
            xaxis="x",
            yaxis="y3",
            line=dict(color="dodgerblue", dash="dot", width=2),
            marker=dict(size=4),
            hovertemplate="Return: %{y:.2f}×<br>%{x|%b %d, %Y}<extra></extra>"
        ))

    # 🎛 Layout
    fig.update_layout(
        title=f"{symbol} Candlestick + Volume + Trades",
        height=750,
        margin=dict(l=50, r=50, t=60, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", x=0, y=1.05),
        xaxis=dict(domain=[0, 1], anchor="y", rangeslider=dict(visible=True), type="date"),
        yaxis=dict(domain=[0.4, 1], title="Price"),
        xaxis2=dict(domain=[0, 1], anchor="y2", matches="x"),
        yaxis2=dict(domain=[0, 0.25], title="Volume", showgrid=True),
        yaxis3=dict(overlaying="y", side="right", showgrid=False, title="Net Return ×")
    )

    st.plotly_chart(fig, use_container_width=True, key="domain_price_volume_trades")


def plot_backtest_price(df, trades_df, symbol):
    close_col = f"Close_{symbol}"
    open_col = f"Open_{symbol}"
    high_col = f"High_{symbol}"
    low_col = f"Low_{symbol}"
    volume_col = f"Volume_{symbol}"

    # 📦 Clean volume column and index
    df = df.sort_index()
    df = df[~df.index.duplicated()]
    df[volume_col] = pd.to_numeric(df[volume_col], errors="coerce").fillna(0)

    # 🎨 Volume bar color logic
    volume_colors = [
        "rgba(0,200,0,0.4)" if c >= o else "rgba(200,0,0,0.4)"
        for c, o in zip(df[close_col], df[open_col])
    ]

    # 📊 Set up subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.75, 0.25],
        specs=[[{"secondary_y": True}], [{}]]
    )

    # 🕯️ Candlestick chart (row 1)
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df[open_col],
        high=df[high_col],
        low=df[low_col],
        close=df[close_col],
        name="Price",
        increasing_line_color="green",
        decreasing_line_color="red"
    ), row=1, col=1)

    # 📊 Volume bars (row 2)
    fig.add_trace(go.Bar(
        x=df.index,
        y=df[volume_col],
        name="Volume",
        marker_color=volume_colors
    ), row=2, col=1)

    # 🔧 Force bar visuals
    fig.update_traces(width=0.8, marker_line_width=0, selector=dict(type="bar"))

    # ✅ Explicit y-axis settings for volume panel
    fig.update_yaxes(
        title_text="Volume",
        row=2, col=1,
        range=[0, df[volume_col].max() * 1.1],
        showgrid=True
    )

    if not trades_df.empty:
        for col in ["Confidence", "Net Return %", "Hold Days", "Outcome"]:
            if col not in trades_df.columns:
                trades_df[col] = None

        # 🔺 Entry markers
        fig.add_trace(go.Scatter(
            x=trades_df["Entry Date"],
            y=trades_df["Entry Price"],
            mode="markers",
            name="Entry",
            marker=dict(symbol="triangle-up", size=10, color="#00FFFF"),
            hovertemplate="🟢 Entry: %{x|%b %d, %Y}<br>Price: %{y:.2f}<br>Confidence: %{customdata[0]:.2f}<extra></extra>",
            customdata=trades_df[["Confidence"]].fillna(0).values
        ), row=1, col=1)

        # 🔻 Exit markers
        fig.add_trace(go.Scatter(
            x=trades_df["Exit Date"],
            y=trades_df["Exit Price"],
            mode="markers",
            name="Exit",
            marker=dict(symbol="triangle-down", size=10, color="#FFD700"),
            hovertemplate="🔴 Exit: %{x|%b %d, %Y}<br>Price: %{y:.2f}<br>Return: %{customdata[0]:.2f}%<br>Days Held: %{customdata[1]}<br>Outcome: %{customdata[2]}<extra></extra>",
            customdata=trades_df[["Net Return %", "Hold Days", "Outcome"]].fillna(0).values
        ), row=1, col=1)

        # 📈 Cumulative Net Return overlay
        trades_sorted = trades_df.sort_values("Exit Date").copy()
        trades_sorted["Cumulative"] = (1 + trades_sorted["Net Return %"].fillna(0) / 100).cumprod()
        equity_curve = (
            trades_sorted[["Exit Date", "Cumulative"]]
            .drop_duplicates(subset="Exit Date")
            .set_index("Exit Date")
        )
        full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
        equity_curve = equity_curve.reindex(full_range).ffill().fillna(1.0)
        equity_curve.iloc[0] = 1.0

        fig.add_trace(go.Scatter(
            x=equity_curve.index,
            y=equity_curve["Cumulative"],
            mode="lines+markers",
            name="Net Return ×",
            line=dict(color="#1f77b4", dash="dot", width=2),
            marker=dict(size=4),
            hovertemplate="Return: %{y:.2f}×<br>%{x|%b %d, %Y}<extra></extra>"
        ), row=1, col=1, secondary_y=True)

    # 🧭 Layout settings
    fig.update_layout(
        title=f"{symbol} Candlestick with Trades, Return & Volume",
        xaxis=dict(
            title="Date",
            rangeselector=dict(
                buttons=[
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(step="all")
                ]
            ),
            rangeslider=dict(visible=True),
            type="date"
        ),
        yaxis=dict(title="Price"),
        yaxis2=dict(title="Net Return ×", overlaying="y", side="right", showgrid=False),
        hovermode="x unified",
        height=750,
        bargap=0.01,
        margin=dict(l=50, r=50, t=60, b=40)
    )

    st.plotly_chart(fig, use_container_width=True, key="stacked_trade_chart")

def add_volume_features_and_labels(
    df,
    symbol,
    window=20,
    entry_return_threshold=0.05,
    forward_days=5
):
    """
    Adds volume-based features and entry target labels to a stock DataFrame.

    Parameters:
        df : pd.DataFrame
            Stock data indexed by date.
        symbol : str
            The asset symbol (used for column names like 'Close_SYMBOL').
        window : int
            Rolling window length for average volume.
        entry_return_threshold : float
            Threshold for labeling a 'good entry' (e.g., 0.05 = +5%).
        forward_days : int
            How many days to look ahead when labeling.
        debug : bool
            Print debug information if True.
    Returns:
        pd.DataFrame
            Enhanced DataFrame with new features and entry labels.
    """
    close_col = f"Close_{symbol}"
    volume_col = f"Volume_{symbol}"

    df = df.copy()
    df[volume_col] = pd.to_numeric(df[volume_col], errors="coerce").fillna(0)

    # === Volume-based features ===
    df["Volume_MA"] = df[volume_col].rolling(window=window).mean()
    df["Volume_Relative"] = df[volume_col] / df["Volume_MA"]
    df["Volume_Delta"] = df[volume_col].diff()
    df["Turnover"] = df[close_col] * df[volume_col]
    df["Volume_Spike"] = (df["Volume_Relative"] > 1.5).astype(int)

    # === Entry target label ===
    future_return = df[close_col].shift(-forward_days) / df[close_col] - 1
    # df["Target_Entry"] = (future_return > entry_return_threshold).astype(int)
    df["Target"] = (future_return > entry_return_threshold).astype(int)

    if debug:
        st.subheader("🔎 Volume Feature Debug")
        st.write("📊 Volume stats:", df[volume_col].describe())
        st.write("📈 Volume_MA stats:", df["Volume_MA"].dropna().describe())
        st.write("⚡ Relative Volume sample:", df["Volume_Relative"].dropna().head(10))
        st.write("🔥 Volume Spike counts:", df["Volume_Spike"].value_counts())
        st.write("🎯 Entry Label Breakdown:", df["Target_Entry"].value_counts(normalize=True))

    return df


def compute_atr(df, symbol, window=14):
    high = df[f"High_{symbol}"]
    low = df[f"Low_{symbol}"]
    close = df[f"Close_{symbol}"]

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = true_range.rolling(window=window).mean()
    return df


def compute_bollinger_bandwidth(df, symbol, window=20, num_std=2):
    close = df[f"Close_{symbol}"]
    ma = close.rolling(window).mean()
    std = close.rolling(window).std()

    upper = ma + num_std * std
    lower = ma - num_std * std
    bandwidth = (upper - lower) / ma

    df["Bollinger_Width"] = bandwidth
    return df

# def compare_volume_feature_impact(symbol, take_profit, stop_loss, max_hold, min_conf, debug=False):
#     st.header("📊 Model Comparison: With vs. Without Volume Features")
#
#     # === Base data load
#     df_base = load_and_prepare_data(symbol)
#
#     # === Model A: Without Volume Features
#     if debug:
#         st.subheader("🤖 Training Model WITHOUT Volume Features")
#     model_plain, df_plain = train_model(df_base.copy(), plot_type=None)
#
#     trades_plain = simulate_trades(
#         df_plain, symbol,
#         take_profit_pct=take_profit,
#         stop_loss_pct=stop_loss,
#         max_hold_days=max_hold,
#         min_confidence=min_conf,
#         show_plot=False
#     )
#
#     # === Model B: With Volume Features
#     if debug:
#         st.subheader("🤖 Training Model WITH Volume Features")
#     df_with_volume = add_volume_features_and_labels(df_base.copy(), symbol=symbol, debug=debug)
#     model_volume, df_volume = train_model(df_with_volume.copy(), plot_type=None)
#
#     trades_volume = simulate_trades(
#         df_volume, symbol,
#         take_profit_pct=take_profit,
#         stop_loss_pct=stop_loss,
#         max_hold_days=max_hold,
#         min_confidence=min_conf,
#         show_plot=False
#     )
#
#     # === Comparison Output
#     def summarize(trades, label):
#         if trades.empty:
#             st.warning(f"🚫 No trades found for {label}")
#             return
#
#         win_rate = (trades["Net Return %"] > 0).mean() * 100
#         total_return = trades["Net Return %"].sum()
#         avg_return = trades["Net Return %"].mean()
#         num_trades = len(trades)
#
#         st.markdown(f"### ⚖️ {label}")
#         st.write(f"""
#         - 📈 **Win Rate**: {win_rate:.2f}%
#         - 💰 **Total Net Return**: {total_return:.2f}%
#         - 📊 **Avg Return / Trade**: {avg_return:.2f}%
#         - 📎 **Number of Trades**: {num_trades}
#         """)
#
#     st.subheader("🔍 Performance Comparison")
#     summarize(trades_plain, label="Baseline Model (No Volume)")
#     summarize(trades_volume, label="Enhanced Model (With Volume Features)")


if "model" not in st.session_state:
    st.session_state.model = None
if "df" not in st.session_state:
    st.session_state.df = None

if __name__ == "__main__":
    page_config("Stock Model Dashboard")

    admin_mode = st.sidebar.checkbox("🛠️ Admin Mode")
    if admin_mode:
        st.sidebar.markdown("#### Historical Backtest Window")
        backtest_start = st.sidebar.date_input("From", value=datetime.date(2023, 1, 1))
        backtest_end = st.sidebar.date_input("To", value=datetime.date(2023, 1, 30))

    if debug:
        plot_option = st.sidebar.radio("Choose a Plot", ["All", "Feature Importance", "Confidence Heatmap", "Signal Accuracy"])
    else:
        plot_option = None

    st.sidebar.subheader("🧠 Trade Simulation Settings")
    take_profit = st.sidebar.slider("Take Profit (%)", 1, 20, value=7) / 100
    stop_loss = st.sidebar.slider("Stop Loss (%)", 1, 20, value=3) / 100
    max_hold = st.sidebar.slider("Max Hold Days", 5, 30, value=15)
    min_conf = st.sidebar.slider("Minimum Confidence", 0.0, 1.0, value=0.5, step=0.05)

    col1, col2 = st.sidebar.columns(2)
    symbol = col1.text_input("Enter stock symbol", value="AAPL")
    run_simulate = st.sidebar.button("Run & Simulate")

    if run_simulate:
        # Load and enrich dataset
        df = load_and_prepare_data(symbol)
        df = add_volume_features_and_labels(df, symbol=symbol)

        # Train model
        model, df_updated = train_model(df.copy(), symbol, plot_type=plot_option)

        # Limit to backtest window if enabled
        if admin_mode:
            df_slice = df_updated[
                (df_updated.index >= pd.to_datetime(backtest_start)) &
                (df_updated.index <= pd.to_datetime(backtest_end))
            ]
            st.markdown(
                f"🕰️ **Backtest Mode Active**: Simulating between **{backtest_start}** and **{backtest_end}**"
            )
        else:
            df_slice = df_updated

        # Simulate trades
        trades_df = simulate_trades(
            df_slice,
            symbol=symbol,
            take_profit_pct=take_profit,
            stop_loss_pct=stop_loss,
            max_hold_days=max_hold,
            min_confidence=min_conf,
            show_plot=True
        )

        # Add Entry/Exit Dates
        trades_df["Entry Date"] = trades_df.index
        trades_df["Exit Date"] = pd.to_datetime(trades_df["Exit Time"]).dt.normalize()

        # ✅ Apply realistic brokerage fees & tax
        trades_df = apply_realistic_fees(trades_df)
        # ✅ Show performance summary
        display_strategy_summary(trades_df)

        # # ✅ Add missing 'Entry Date' column from index
        # if not trades_df.empty and "Entry Date" not in trades_df.columns:
        #     trades_df["Entry Date"] = trades_df.index

        if debug:
            st.write("✅ Trades found:", len(trades_df))
            st.write("🧬 trades_df columns:", trades_df.columns.tolist())

        zoom_to_return = st.sidebar.checkbox("📌 Zoom to first trade", value=True)

        # Plot
        plot_price_volume_domain(df, trades_df, symbol, zoom_to_return)

        # Persist session state
        st.session_state.df = df_updated
        st.session_state.model = model

    elif st.session_state.df is not None and st.session_state.model is not None:
        df = st.session_state.df

        if plot_option in ["All", "Confidence Heatmap"]:
            with st.expander("Prediction Confidence", expanded=False):
                plot_prediction_confidence(df)

        if plot_option in ["All", "Signal Accuracy"]:
            with st.expander("Price With Signals", expanded=False):
                if debug:
                    st.write("🧬 df.columns:", df.columns.tolist())
                plot_price_with_signals(df, symbol=symbol)

    else:
        st.sidebar.info("Press 'Run & Simulate' to begin.")
