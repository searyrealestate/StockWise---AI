import streamlit as st
import pandas as pd
import joblib
import os
import glob
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import itertools

"""
📊 NASDAQ Model Evaluation Dashboard
────────────────────────────────────

This Streamlit app allows users to evaluate trained machine learning models
on NASDAQ stock data. It supports:

- Selecting a trained model and test stock
- Running predictions and viewing classification metrics
- Visualizing predicted vs. actual signals on a price chart
- Comparing multiple models on the same stock
- Inspecting the last 20 predictions

🔧 Key Functions:
─────────────────
- load_model_map(): Loads all trained models from disk
- load_test_symbols(): Lists available test stocks
- load_test_stock(): Loads test data from parquet
- evaluate_model(): Runs predictions and returns metrics
- plot_confusion_matrix(): Displays confusion matrix
- plot_prediction_chart(): Shows price chart with signals
- plot_multi_model_overlay(): Compares multiple models visually
- display_prediction_table(): Shows last 20 predictions

📁 Inputs:
──────────
- Trained models (.pkl) in TRAIN_DIR
- Feature datasets (.parquet) in TEST_DIR

📤 Outputs:
───────────
- Interactive charts and metrics in Streamlit
"""


# =======================
# CONFIGURATION & SETUP
# =======================
st.set_page_config(layout="wide")
st.title("📊 NASDAQ Model Evaluation Dashboard")

# Directory paths
TRAIN_DIR = r"C:\Users\user\PycharmProjects\StockWise\models\NASDAQ-training set"
TEST_DIR = r"C:\Users\user\PycharmProjects\StockWise\models\NASDAQ-testing set"
FEATURE_COLS = ["Volume_Relative", "Volume_Delta", "Turnover", "Volume_Spike"]

# =======================
# FUNCTION DEFINITIONS
# =======================


@st.cache_data
def load_model_map():
    """
    Load available trained models from directory.
    📚 Loads all trained model files from the training directory.
    📥 Parameters: None
    📤 Returns: Dict[str, str] — mapping of model names to file paths
    """
    model_files = sorted(glob.glob(os.path.join(TRAIN_DIR, "*_model_*.pkl")))
    return {os.path.basename(f).replace(".pkl", ""): f for f in model_files}


@st.cache_data
def load_test_symbols():
    """Load available test stock symbols from parquet filenames.
    📚 Loads all available test stock symbols from feature files.
    📥 Parameters: None
    📤 Returns:
    List[str] — list of file paths
    List[str] — list of stock symbols extracted from filenames
    """

    feature_files = sorted(glob.glob(os.path.join(TEST_DIR, "*_features_*.parquet")))
    return feature_files, [os.path.basename(f).split("_")[0] for f in feature_files]


def load_test_stock(parquet_file):
    """
    📚 Loads a test stock’s feature DataFrame from a .parquet file.
    📥 Parameters:
    parquet_file: str — path to the parquet file
    📤 Returns:
    pd.DataFrame — test stock features
    """

    return pd.read_parquet(parquet_file)


def evaluate_model(model, df):
    """
    📚 Generate predictions and return classification report and predictions.
    📥 Parameters:
    model: sklearn-compatible model
    df: pd.DataFrame — test data with features and Target
    📤 Returns:
    y_true: np.ndarray — true labels
    y_pred: np.ndarray — predicted labels
    report: dict — classification report as a dictionary
    """
    X = df[FEATURE_COLS]
    y_true = df["Target"]
    y_pred = model.predict(X)
    report = classification_report(y_true, y_pred, output_dict=True)
    return y_true, y_pred, report


def plot_confusion_matrix(y_true, y_pred):
    """
    Display a seaborn confusion matrix.
    📥 Parameters:
    y_true: array-like
    y_pred: array-like
    📤 Returns: None (renders chart in Streamlit)
    """

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", cbar=False,
                xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
    st.subheader("🔍 Confusion Matrix")
    st.pyplot(fig)


def plot_prediction_chart(df, y_pred, stock_symbol):
    """

    📚 Display stock price, predicted & actual signals, and volume.
    📥 Parameters:

    df: pd.DataFrame — test data

    y_pred: array-like — model predictions

    stock_symbol: str — ticker symbol

    📤 Returns: None (renders Plotly chart)
    """

    df = df.copy()
    df["Predicted"] = y_pred

    price_col = [col for col in df.columns if col.startswith("Close_")][0]
    volume_col = [col for col in df.columns if col.startswith("Volume_")][0]

    fig = go.Figure()

    # Price line
    fig.add_trace(go.Scatter(x=df.index, y=df[price_col],
        mode="lines", name="Stock Price", line=dict(color="royalblue")))

    # Predicted signal points
    pred_indices = df[df["Predicted"] == 1].index
    fig.add_trace(go.Scatter(
        x=pred_indices,
        y=df.loc[pred_indices, price_col],
        mode="markers",
        name="🔮 Predicted Signal",
        marker=dict(symbol="x", color="orange", size=8)
    ))

    # Actual signal points
    actual_indices = df[df["Target"] == 1].index
    fig.add_trace(go.Scatter(
        x=actual_indices,
        y=df.loc[actual_indices, price_col],
        mode="markers",
        name="✅ Actual Signal",
        marker=dict(symbol="circle", color="green", size=7, opacity=0.5)
    ))

    # Volume bars
    fig.add_trace(go.Bar(
        x=df.index,
        y=df[volume_col],
        name="Volume",
        marker=dict(color="lightgray"),
        yaxis="y2",
        opacity=0.4
    ))

    fig.update_layout(
        height=600,
        xaxis=dict(title="Date", rangeslider=dict(visible=True)),
        yaxis=dict(title="Price"),
        yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False),
        legend=dict(x=0, y=1.1, orientation="h")
    )

    st.subheader(f"📈 {stock_symbol} Chart with Model Signals")
    st.plotly_chart(fig, use_container_width=True)


def plot_multi_model_overlay(df, model_map, feature_cols):
    """
    📚 Plots all model predictions on one chart with checkboxes per model.
    📥 Parameters:
    df: pd.DataFrame — test data
    model_map: Dict[str, str] — model name to path
    feature_cols: List[str] — feature column names
    📤 Returns: None (renders Plotly chart)
    """

    price_col = [c for c in df.columns if c.startswith("Close_")][0]
    volume_col = [c for c in df.columns if c.startswith("Volume_")][0]

    st.subheader("🎯 Multi-Model Signal Overlay")

    # Generate unique colors for models
    color_cycle = itertools.cycle([
        "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
        "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff"
    ])

    # Predict with all models
    multi_model_preds = {}
    for model_name, path in model_map.items():
        try:
            model_obj = joblib.load(path)
            y_hat = model_obj.predict(df[feature_cols])
            multi_model_preds[model_name] = y_hat
        except Exception as e:
            st.warning(f"⚠️ Skipping {model_name}: {e}")

    # Create Plotly figure
    fig = go.Figure()

    # Price line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[price_col],
        mode="lines",
        name="Stock Price",
        line=dict(color="royalblue", width=2)
    ))

    # Volume bars
    fig.add_trace(go.Bar(
        x=df.index,
        y=df[volume_col],
        name="Volume",
        yaxis="y2",
        marker=dict(color="lightgray"),
        opacity=0.3
    ))

    # Overlay predictions for selected models
    for model_name, y_hat in multi_model_preds.items():
        if st.sidebar.checkbox(f"👁 Show {model_name}", value=False):
            df_pred = df.copy()
            df_pred["Predicted"] = y_hat
            hits = df_pred[df_pred["Predicted"] == 1]
            fig.add_trace(go.Scatter(
                x=hits.index,
                y=hits[price_col],
                mode="markers",
                name=f"{model_name}",
                marker=dict(
                    color=next(color_cycle),
                    size=7,
                    symbol="circle-open-dot",
                    opacity=0.8
                )
            ))

    # Final layout
    fig.update_layout(
        height=700,
        showlegend=True,
        xaxis=dict(title="Date", rangeslider=dict(visible=True)),
        yaxis=dict(title="Price ($)", side="left"),
        yaxis2=dict(
            title="Volume",
            side="right",
            overlaying="y",
            anchor="x",
            showgrid=False
        ),
        legend=dict(x=0, y=1.1, orientation="h")
    )

    st.plotly_chart(fig, use_container_width=True)


def display_prediction_table(df, y_pred):
    """
    📚 Displays the last 20 rows of actual vs. predicted target values.
    📥 Parameters:
    df: pd.DataFrame — test data
    y_pred: array-like — model predictions
    📤 Returns: None (renders table)
    """

    df = df.copy()
    df["Predicted"] = y_pred
    st.subheader("🔬 Prediction vs. Actual (last 20 rows)")
    st.dataframe(df[["Target", "Predicted"] + FEATURE_COLS].tail(20))

# =======================
# MAIN DASHBOARD LOGIC
# =======================

# Sidebar controls
model_map = load_model_map()
feature_files, symbol_list = load_test_symbols()

model_choice = st.sidebar.selectbox("🧠 Choose a trained model", list(model_map.keys()))
symbol_choice = st.sidebar.selectbox("📈 Choose a test stock", symbol_list)
parquet_file = [f for f in feature_files if symbol_choice in f][0]
model = joblib.load(model_map[model_choice])

# Run inference and display
try:
    df = load_test_stock(parquet_file)
    y_true, y_pred, report = evaluate_model(model, df)

    # Show metrics table
    st.subheader(f"📌 Metrics for {symbol_choice} using {model_choice}")
    st.write(pd.DataFrame(report).T.style.format("{:.2f}"))

    # Show confusion matrix
    plot_confusion_matrix(y_true, y_pred)

    # Show plot
    plot_prediction_chart(df, y_pred, stock_symbol=symbol_choice)

    # Show last 20 rows
    display_prediction_table(df, y_pred)

except Exception as e:
    st.error(f"🚨 Failed to evaluate {symbol_choice} using {model_choice}: {e}")
