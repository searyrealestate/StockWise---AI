import streamlit as st
import pandas as pd
import joblib
import os
import plotly.graph_objects as go
from datetime import datetime, timedelta


def validate_feature_files(directory):
    import pandas as pd
    import os

    broken_files = []
    valid_files = []

    st.markdown("### 🧪 Validating Feature Files")

    for file in os.listdir(directory):
        if file.endswith("_features.pkl"):
            path = os.path.join(directory, file)
            symbol = file.replace("_features.pkl", "")
            try:
                df = pd.read_pickle(path)
                if df.empty:
                    broken_files.append((symbol, "❌ Empty file"))
                elif not pd.to_datetime(df.index, errors="coerce").notna().all():
                    broken_files.append((symbol, "❌ Invalid index"))
                else:
                    valid_files.append(symbol)
                    st.success(f"✅ {symbol}: OK ({df.shape[0]} rows)")
            except Exception as e:
                broken_files.append((symbol, f"❌ Load error: {e}"))

    if broken_files:
        st.warning("⚠️ Found broken feature files:")
        for sym, reason in broken_files:
            st.error(f"{sym}: {reason}")

    return valid_files, broken_files


# --- CONFIG ---
MODEL_DIR = "models"
TICKER_LIST = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "INTC", "AMD"]  # Replace with your list
GOOGLE_SHEET_URL = "https://docs.google.com/spreadsheets/d/1HE1vXOnT7dyWEqQ1zxlNzZ7LVcOFj1nnhhGhHp0P7ZE/edit#gid=0"

st.set_page_config(layout="wide", page_title="📈 StockWise AI")

# 🔍 Run feature file validation
# pkl validation - verify that the PKLs file have at list 1 row
valid_syms, broken_syms = validate_feature_files(MODEL_DIR)

if not valid_syms:
    st.error("❌ No usable feature files found. Please regenerate them before running the app.")
    st.markdown(f"**🧮 Total broken feature files: {len(broken_syms)}**")
    st.stop()

# ✅ Use only valid symbols in dropdown
TICKER_LIST = sorted(valid_syms)

# --- SIDEBAR ---
st.sidebar.title("🔧 Settings")

symbol = st.sidebar.selectbox("📌 Select Stock Symbol", options=sorted(TICKER_LIST))
# date_option = st.sidebar.date_input("📅 Data Range", value=datetime.today())
date_option = st.sidebar.selectbox("📅 Data Range", [
    "Today", "3 Days Ago", "7 Days Ago", "1 Month Ago", "1 Year Ago", "View All"
])

run_button = st.sidebar.button("▶️ Run Prediction")

                                   # options=[
    # "Today", "3 Days Ago", "7 Days Ago", "1 Month Ago", "1 Year Ago", "View All"])

with st.sidebar.container():
    st.markdown("🧠 Trade Simulation Settings")
    take_profit = st.sidebar.slider("Take Profit (%)", 1, 20, value=7) / 100
    stop_loss = st.sidebar.slider("Stop Loss (%)", 1, 20, value=3) / 100
    max_hold = st.sidebar.slider("Max Hold Days", 5, 30, value=15)
    min_conf = st.sidebar.slider("Minimum Confidence", 0.0, 1.0, value=0.5, step=0.05)


save_data = st.sidebar.checkbox("💾 Save Trade to Google Sheet")
if save_data:
    st.sidebar.markdown("**Trade Entry**")
    trade_price = st.sidebar.number_input("Price", min_value=0.0)
    trade_date = st.sidebar.date_input("Date", value=datetime.today())
    trade_type = st.sidebar.selectbox("Buy / Sell", ["Buy", "Sell"])
    state_tax = st.sidebar.number_input("State Tax", min_value=0.0)
    commission = st.sidebar.number_input("Trading Commission", min_value=0.0)
    notes = st.sidebar.text_input("Notes")

show_history = st.sidebar.checkbox("📜 Show Transaction History")
if show_history:
    import datetime

    today = datetime.date.today()
    start_default = today - datetime.timedelta(days=30)
    end_default = today

    try:

        history_range = st.sidebar.date_input(
            "🗓️ Select Transaction History Range",
            (start_default, end_default),
            min_value=datetime.date(2000, 1, 1),
            max_value=today,
            format="YYYY-MM-DD"
        )


        # Unpack the range:
        from_date, to_date = history_range

    except Exception as e:
        st.sidebar.warning("Select End data!!")

# --- MAIN PANEL ---
st.title(f"📊 StockWise AI Dashboard — {symbol}")

if run_button:
    try:
        st.info(f"🚀 Running prediction for: `{symbol}` | Range: **{date_option}**")

        # 🔍 Load model and features
        model_path = os.path.join(MODEL_DIR, f"{symbol}_model.pkl")
        data_path = os.path.join(MODEL_DIR, f"{symbol}_features.pkl")
        model = joblib.load(model_path)
        st.success("📂 Model and data loaded successfully")

        # 📦 Load and clean data
        df = pd.read_pickle(data_path).dropna()
        st.write("📄 Raw data shape:", df.shape)

        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()]
        st.write("🧹 Cleaned data shape:", df.shape)
        st.write("📅 Data index range:", df.index.min(), "→", df.index.max())

        # ⏱️ Filter by date range
        if date_option != "View All":
            days_map = {
                "Today": 0,
                "3 Days Ago": 3,
                "7 Days Ago": 7,
                "1 Month Ago": 30,
                "1 Year Ago": 365
            }
            days = days_map.get(date_option, 0)
            cutoff = df.index.max() - pd.Timedelta(days=days)
            lookback = cutoff - pd.Timedelta(days=5)  # Show extra context days
            df = df[df.index >= lookback]
            st.write("📉 Data after filtering:", df.shape)

        if df.empty:
            st.warning("⚠️ No data available for the selected date range.")
            st.stop()

        # ✅ Run predictions
        features = ["Volume_Relative", "Volume_Delta", "Turnover", "Volume_Spike"]
        st.write("🔑 Features used for prediction:", features)
        st.write("🧪 Sample features row:", df[features].tail(1))

        df["Prediction"] = model.predict(df[features])
        df["Confidence"] = model.predict_proba(df[features])[:, 1]

        st.write("🔍 Prediction value counts:")
        st.write(df["Prediction"].value_counts())

        st.write("📈 Confidence stats:")
        st.write("Min:", df["Confidence"].min(), "| Max:", df["Confidence"].max())

        # 📊 Display Performance Metrics (placeholders for now)
        total_trades = int(df["Prediction"].sum())
        st.markdown("### 📈 Performance Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📌 Total Trades", total_trades)
        col2.metric("📈 Win Rate (%)", "65.0")
        col3.metric("💰 Net Return (%)", "9.8")
        col4.metric("🧾 Tax Paid", "$120")

        # 📉 Candlestick Chart
        st.markdown("### 📉 Candlestick Chart with Predictions")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df[f"Open_{symbol}"],
            high=df[f"High_{symbol}"],
            low=df[f"Low_{symbol}"],
            close=df[f"Close_{symbol}"],
            name="Price"
        ))
        buy_signals = df[df["Prediction"] == 1]
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_signals[f"Close_{symbol}"],
            mode="markers",
            marker=dict(symbol="arrow-bar-down", color="yellow", size=10),
            name="Buy Signal"
        ))
        fig.update_layout(xaxis_rangeslider_visible=False, height=500)
        st.plotly_chart(fig, use_container_width=True)

        # 📌 Prediction Summary
        st.markdown("### 🤖 Prediction Summary")
        latest = df.iloc[-1]
        buy_price = latest[f"Close_{symbol}"]
        sell_price = buy_price * (1 + take_profit)
        stop_price = buy_price * (1 - stop_loss)

        st.write("🧾 Latest prediction row:")
        st.write(latest)

        st.markdown(f"- **Buying price (Confidence)** = ${buy_price:.2f} ({latest['Confidence'] * 100:.1f}%)")
        st.markdown(f"- **Selling price (TP)** = ${sell_price:.2f}")
        st.markdown(f"- **Stop loss price (SL)** = ${stop_price:.2f}")

        # 📝 Placeholder for saving trade
        if save_data:
            st.success("✅ Trade saved to Google Sheet (placeholder)")

        # 📜 Transaction History (placeholder)
        if show_history:
            st.markdown("### 📜 Transaction History")
            st.markdown("**Total Profit: $1,234.56**")
            st.dataframe(pd.DataFrame({
                "Symbol": [symbol] * 3,
                "Date": [datetime.today().date()] * 3,
                "Buy/Sell": ["Buy", "Sell", "Buy"],
                "Price": [100, 110, 105],
                "Profit": [10, -5, 8]
            }))

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")