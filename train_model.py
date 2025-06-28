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
    v = "SW version: 0.0.8"
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


def train_model(df, plot_type=0):

    # # 🔧 Flatten columns if MultiIndex (e.g., ('Price', 'Close') → 'Price_Close')
    # if isinstance(df.columns, pd.MultiIndex):
    #     df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]

    # if debug:
    #     st.write("✅ After flattening:", df.columns.tolist())

    expected_features = [
        "MA20", "MA50", "Volatility", "RSI", "MACD", "MACD_Signal", "MACD_Hist",
        "RSI_lag1", "MACD_lag1", "MACD_Signal_lag1", "Close_lag1", "Return_lag1",
        "Bullish_Engulfing", "Hammer", "Pct_from_20d_high", "Pct_from_20d_low",
        "OBV", "OBV_10_MA", "Correl_with_SPY_10"
    ]
    features = [f for f in expected_features if f in df.columns]

    if debug:
        st.write("📌 Using features:", features)

    X = df[features]
    y = df["Target"]

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
                labels={"Importance": "Importance Score", "Feature": "Input Feature"},
                height=600
            )

            fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

    # Predict and assign scalar values
    df["Predicted"] = pd.Series(model.predict(X), index=df.index).astype(int)
    df["Prob"] = pd.Series(model.predict_proba(X)[:, 1], index=df.index).astype(float)

    if debug:
        st.write("🎯 Sample predictions:", df[["Predicted", "Prob"]].head())

    return model, df


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

    st.plotly_chart(fig, use_container_width=True)


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

    st.plotly_chart(fig, use_container_width=True)


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

    st.plotly_chart(fig, use_container_width=True)


def page_config(page_title, page_icon=":desert_island:"):
    st.title(page_title)

    # Streamlit page config
    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )


def simulate_trades(df, symbol, take_profit_pct=0.07, stop_loss_pct=0.03,
                    max_hold_days=15, min_confidence=0.5,
                    show_plot=True):
    if debug:
        st.write("🚧 simulate_trades started")
        st.write("Columns in df:", df.columns.tolist())

    if "Predicted" not in df.columns or "Prob" not in df.columns:
        st.warning("Missing required prediction columns for trade simulation.")
        return pd.DataFrame()

    trades = []
    df = df.copy()
    df = df.sort_index()
    df.reset_index(drop=False, inplace=True)
    df.sort_values(by="Date", inplace=True)
    df.set_index("Date", inplace=True)

    close_col = f"Close_{symbol}"

    for i in range(len(df)):
        row = df.iloc[i]
        entry_idx = df.index[i]

        try:
            pred = row["Predicted"].item() if hasattr(row["Predicted"], "item") else float(row["Predicted"])
            prob = row["Prob"].item() if hasattr(row["Prob"], "item") else float(row["Prob"])
        except Exception as e:
            if debug:
                st.warning(f"❌ Row {i} - conversion issue: {e}")
                st.text(f"🔬 Raw Predicted: {row['Predicted']} | Prob: {row['Prob']}")
            continue

        if debug and i < 3:
            st.text(f"🧪 Row {i} → Pred: {pred}, Prob: {prob}")

        if pred != 1 or prob < min_confidence:
            continue

        try:
            entry_price = row[close_col]
        except KeyError:
            st.warning(f"❗️ Column {close_col} not found in row {i}")
            continue

        entry_date = entry_idx
        exit_price = None
        exit_date = None
        outcome = "Open"

        for hold_day in range(1, max_hold_days + 1):
            try:
                next_idx = df.index.get_loc(entry_idx) + hold_day
                if next_idx >= len(df):
                    break
                future_row = df.iloc[next_idx]
                future_price = future_row[close_col]

                entry_price = entry_price.item() if hasattr(entry_price, "item") else entry_price
                future_price = future_price.item() if hasattr(future_price, "item") else future_price

                change_pct = (future_price - entry_price) / entry_price

                if change_pct >= take_profit_pct:
                    exit_price = future_price
                    exit_date = future_row.name
                    outcome = "TP"
                    break
                elif change_pct <= -stop_loss_pct:
                    exit_price = future_price
                    exit_date = future_row.name
                    outcome = "SL"
                    break
            except Exception as e:
                if debug:
                    st.warning(f"❗️ Exit scan error at row {i}: {e}")
                continue

        if exit_price is None:
            try:
                final_idx = df.index.get_loc(entry_idx) + max_hold_days
                if final_idx < len(df):
                    final_row = df.iloc[final_idx]
                    exit_price = final_row[close_col]
                    exit_date = final_row.name
                    outcome = "Timed"
                else:
                    continue
            except Exception as e:
                if debug:
                    st.warning(f"🧯 Final exit error at row {i}: {e}")
                continue

        ret_pct = (exit_price - entry_price) / entry_price
        holding_days = (exit_date - entry_date).days

        trades.append({
            "Entry Date": entry_date,
            "Exit Date": exit_date,
            "Entry Price": round(entry_price, 2),
            "Exit Price": round(exit_price, 2),
            "Return %": round(ret_pct * 100, 2),
            "Days Held": holding_days,
            "Outcome": outcome
        })

    trades_df = pd.DataFrame(trades)

    st.subheader("📄 Trade Log")
    st.dataframe(trades_df)

    if show_plot and not trades_df.empty:
        trades_df["Cumulative Return"] = (1 + trades_df["Return %"] / 100).cumprod()
        fig = px.line(
            trades_df,
            x="Exit Date",
            y="Cumulative Return",
            title="Cumulative Return from Simulated Trades",
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)

    st.session_state.df = df

    if debug:
        st.success(f"✅ Simulated {len(trades_df)} trades")

    return trades_df


if "model" not in st.session_state:
    st.session_state.model = None
if "df" not in st.session_state:
    st.session_state.df = None

if __name__ == "__main__":
    page_config("Stock Model Dashboard")

    # st.sidebar.write("📊 Available Plot Modes:")
    # st.sidebar.write("- All = Feature Importance, Confidence Heatmap, Signal Accuracy")
    # st.sidebar.write("- Feature Importance = XGBoost feature weights")
    # st.sidebar.write("- Confidence Heatmap = Model probability over time")
    # st.sidebar.write("- Signal Accuracy = Price overlay with signals")

    if debug:
        plot_option = st.sidebar.radio("Choose a Plot", ["All", "Feature Importance", "Confidence Heatmap", "Signal Accuracy"])
    else:
        plot_option = st.sidebar.radio("Choose a Plot", ["All", "Feature Importance"])

    st.sidebar.subheader("🧠 Trade Simulation Settings")
    take_profit = st.sidebar.slider("Take Profit (%)", 1, 20, value=7,
                                    help="Sell when price gains this much from entry") / 100
    stop_loss = st.sidebar.slider("Stop Loss (%)", 1, 20, value=3,
                                  help="Sell when price drops this much from entry") / 100
    max_hold = st.sidebar.slider("Max Hold Days", 5, 30, value=15,
                                 help="Sell after this many days if no target or stop is hit")
    min_conf = st.sidebar.slider("Minimum Confidence", 0.0, 1.0, value=0.5, step=0.05,
                                 help="Only simulate trades with prediction confidence above this")

    col1, col2 = st.sidebar.columns(2)
    symbol = col1.text_input("Enter stock symbol", value="AAPL")
    run_simulate = st.sidebar.button("Run & Simulate")


    if run_simulate:
        df = load_and_prepare_data(symbol)
        df_copy = df.copy()
        model, df_updated = train_model(df_copy, plot_type=plot_option)
        st.session_state.df = df_updated
        st.session_state.model = model

    if st.session_state.df is not None and st.session_state.model is not None:
        df = st.session_state.df

        if plot_option in ["All", "Confidence Heatmap"]:
            with st.expander("Prediction Confidence", expanded=False):
                plot_prediction_confidence(df)

        if plot_option in ["All", "Signal Accuracy"]:
            with st.expander("Price With Signals", expanded=False):
                if debug:
                    st.write("🧬 df.columns:", df.columns.tolist())
                plot_price_with_signals(df, symbol=symbol)


        if run_simulate:
            with st.expander("Simulated Trades", expanded=True):
                simulate_trades(
                    df,
                    symbol=symbol,
                    take_profit_pct=take_profit,
                    stop_loss_pct=stop_loss,
                    max_hold_days=max_hold,
                    min_confidence=min_conf,
                    show_plot=True
                )
    else:
        st.sidebar.info("Press 'Run' to load data and train the model first.")

