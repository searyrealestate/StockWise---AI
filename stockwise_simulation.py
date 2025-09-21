import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import joblib
import json
import os
import glob
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
import sys
import traceback
from plotly.subplots import make_subplots
from data_source_manager import DataSourceManager

# --- Page Configuration ---
st.set_page_config(
    page_title="StockWise AI Trading Advisor",
    page_icon="🏢",
    layout="wide"
)

# Add this helper function to your script

def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes a multi-indexed DataFrame from yfinance to a simple single-level index.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten the multi-level column index
        df.columns = df.columns.droplevel(0)
    return df


# --- Feature Engineering Pipeline (for Gen-3 Model) ---
class FeatureCalculator:
    """A dedicated class to handle all feature calculations for the Gen-3 model."""

    def get_dominant_cycle(self, data, min_period=3, max_period=100) -> float:
        data = pd.Series(data).dropna()
        if len(data) < min_period: return 0.0
        detrended = data - np.poly1d(np.polyfit(np.arange(len(data)), data.values, 1))(np.arange(len(data)))
        fft_result = np.fft.fft(detrended.values)
        frequencies = np.fft.fftfreq(len(detrended))
        power = np.abs(fft_result) ** 2
        positive_freq_mask = frequencies > 0
        if not np.any(positive_freq_mask): return 0.0
        periods = 1 / frequencies[positive_freq_mask]
        period_mask = (periods >= min_period) & (periods <= max_period)
        if not np.any(period_mask): return 0.0
        dominant_idx = power[positive_freq_mask][period_mask].argmax()
        return periods[period_mask][dominant_idx]

    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if df.empty or len(df) < 90:
            return pd.DataFrame()

        # --- DEBUG PRINTS ---
        print("\n--- STARTING DEBUG FOR FeatureCalculator ---")
        print(f"1. Initial columns: {df.columns.tolist()}")

        # MODIFIED: Standardize column names to lowercase for pandas-ta compatibility
        df.columns = [col.lower() for col in df.columns]
        print(f"2. Columns after converting to lowercase: {df.columns.tolist()}")

        try:

            # Use pandas-ta for all indicators to match the training pipeline
            df.ta.bbands(length=20, append=True, col_names=("bb_lower", "bb_middle", "bb_upper", "bb_width", "bb_position"))
            df.ta.atr(length=14, append=True, col_names="atr_14")
            df.ta.rsi(length=14, append=True, col_names="rsi_14")
            df.ta.rsi(length=28, append=True, col_names="rsi_28")
            df.ta.macd(append=True, col_names=("macd", "macd_histogram", "macd_signal"))
            df.ta.adx(length=14, append=True, col_names=("adx", "adx_pos", "adx_neg"))
            df.ta.mom(length=5, append=True, col_names="Momentum_5")
            df.ta.obv(append=True)
            print(f"3. Columns after all pandas-ta indicators: {df.columns.tolist()}")
        except Exception as e:
            print(f"--- ERROR during pandas-ta calculations ---")
            print(e)
            print("--- END DEBUG ---")
            return pd.DataFrame()

        df['daily_return'] = df['close'].pct_change()
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        df['volatility_20d'] = df['daily_return'].rolling(20).std()
        df['z_score_20'] = (df['close'] - df['bb_middle']) / df['close'].rolling(20).std()

        try:
            qqq_df = yf.download("QQQ", start=df.index.min() - timedelta(days=70), end=df.index.max(), progress=False,
                                 auto_adjust=True)
            if not qqq_df.empty:
                qqq_df.columns = [col.lower() for col in qqq_df.columns]  # Also standardize QQQ columns
                qqq_close = qqq_df['close'].reindex(df.index, method='ffill')
                df['correlation_50d_qqq'] = df['close'].rolling(50).corr(qqq_close)

            else:
                print("qqq_df.empty in calculate_all_features")
                df['correlation_50d_qqq'] = 0.0
        except Exception as e:
            print(f"--- ERROR qqq_df.empty ---")
            print(e)
            df['correlation_50d_qqq'] = 0.0

        # Mocking the cluster labels for live prediction
        df['volatility_90d'] = df['daily_return'].rolling(90).std()
        # In a real system, thresholds would be loaded, but for the UI we can use dynamic quantiles.
        low_thresh, high_thresh = df['volatility_90d'].quantile([0.33, 0.66])
        df['volatility_cluster'] = pd.cut(df['volatility_90d'], bins=[-np.inf, low_thresh, high_thresh, np.inf],
                                          labels=['low', 'mid', 'high'])

        # # Rename the core columns back to TitleCase before returning the DataFrame
        # df.rename(columns={
        #     'open': 'Open', 'high': 'High', 'low': 'Low',
        #     'close': 'Close', 'volume': 'Volume'
        # }, inplace=True)

        df.bfill(inplace=True)
        df.ffill(inplace=True)
        return df


# --- Main Application Class (with Gen-3 Architecture) ---
class ProfessionalStockAdvisor:
    def __init__(self, model_dir: str, data_source_manager=None, debug=False, testing_mode=False, download_log=False):
        self.log_entries = []
        self.debug = debug  # Use the passed value
        self.model_dir = model_dir
        self.download_log = download_log # Use the passed value

        self.testing_mode = testing_mode

        if data_source_manager:
            self.data_source_manager = data_source_manager
        elif self.testing_mode:
            self.data_source_manager = None
        else:
            self.data_source_manager = DataSourceManager(use_ibkr=True)

        # --- GEN-3: Load the entire suite of specialist models ---
        self.models, self.feature_names = self._load_gen3_models()
        self.calculator = FeatureCalculator()
        self.tax = 0.25
        self.broker_fee = 0.004
        self.position = {}  # New: Tracks the current open position for the state machine
        self.model_version_info = f"Gen-3: {os.path.basename(model_dir)}"
        if self.download_log: self.log_file = self.setup_log_file()
        self.log("Application Initialized.", "INFO")
        if data_source_manager:
            self.data_source_manager = data_source_manager
        elif self.testing_mode:
            self.data_source_manager = None  # Correctly handle testing mode
        else:
            self.data_source_manager = DataSourceManager(use_ibkr=False)


    def _load_gen3_models(self):
        """
        Loads the entire suite of specialist models for Gen-3.
        """
        models = {}
        feature_names = {}
        try:
            model_files = glob.glob(os.path.join(self.model_dir, "*.pkl"))
            if not model_files:
                self.log(f"No models found in {self.model_dir}. Please run the model trainer.", "ERROR")
                return None, None

            for model_path in model_files:
                model_name = os.path.basename(model_path).replace(".pkl", "")
                features_path = model_path.replace(".pkl", "_features.json")
                models[model_name] = joblib.load(model_path)
                with open(features_path, 'r') as f:
                    feature_names[model_name] = json.load(f)

            self.log(f"✅ Successfully loaded {len(models)} specialist models for Gen-3.", "INFO")
            return models, feature_names
        except Exception as e:
            self.log(f"Error loading Gen-3 models: {e}", "ERROR")
            return None, None

    # log, setup_log_file, validate_symbol_professional methods are correct and do not need changes...
    def log(self, message, level="INFO"):
        timestamp = datetime.now().strftime('%H:%M:%S')
        entry = f"[{timestamp}] [{level}] {message}"
        if not hasattr(self, 'log_entries'):
            self.log_entries = []
        self.log_entries.append(entry)
        if self.download_log and hasattr(self, 'log_file') and self.log_file:
            try:
                with open(self.log_file, "a", encoding='utf-8') as f:
                    f.write(entry + "\n")
            except Exception as e:
                st.error(f"Failed to write to log file: {e}")

    def setup_log_file(self):
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        return os.path.join(log_dir, f"stockwise_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    def validate_symbol_professional(self, symbol):
        self.log(f"Using yfinance for validation of {symbol}", "INFO")
        try:
            ticker = yf.Ticker(symbol)
            if 'regularMarketPrice' in ticker.info and ticker.info['regularMarketPrice'] is not None:
                return True
            if 'currentPrice' in ticker.info and ticker.info['currentPrice'] is not None:
                return True
            return False
        except Exception:
            return False

    def is_market_in_uptrend(self, days=200):
        """
        The Conductor: Checks if the general market (SPY) is in an uptrend using the robust data manager.
        """
        try:
            # FIX: Use the new robust data manager to get SPY data
            spy_data = self.data_source_manager.get_stock_data("SPY", period=f"{days + 50}d")
            if spy_data is None or spy_data.empty:
                self.log("Could not download SPY data for market trend analysis.", "WARNING")
                return True  # Failsafe

            spy_data[f'sma_{days}'] = ta.trend.sma_indicator(spy_data['close'], window=days)
            latest_price = spy_data['close'].iloc[-1]
            moving_average = spy_data[f'sma_{days}'].iloc[-1]
            return latest_price > moving_average
        except Exception as e:
            self.log(f"Error during market trend analysis: {e}", "WARNING")
            return True  # Failsafe

    # --- GEN-3: The core state machine logic for prediction ---
    def run_analysis(self, ticker_symbol, analysis_date):
        try:
            # Step 1: The Conductor. Assess overall market health.
            if not self.is_market_in_uptrend():
                stock_data = self.data_source_manager.get_stock_data(ticker_symbol)
                price_on_date = stock_data.loc[pd.to_datetime(analysis_date)]['close'] if pd.to_datetime(
                    analysis_date) in stock_data.index else 0
                return stock_data, {'action': "WAIT / AVOID", 'confidence': 99.9, 'current_price': price_on_date,
                                    'reason': "Market Downtrend", 'buy_date': None, 'agent': "Market Regime Agent"}

            full_stock_data = self.data_source_manager.get_stock_data(ticker_symbol)
            if full_stock_data is None or full_stock_data.empty: return None, None

            # Normalize the DataFrame columns immediately after fetching
            full_stock_data = normalize_dataframe_columns(full_stock_data)

            # Step 2: Feature Engineering
            featured_data = self.calculator.calculate_all_features(
                full_stock_data[full_stock_data.index <= pd.to_datetime(analysis_date)])
            if featured_data.empty: return full_stock_data, {'action': "WAIT",
                                                             'reason': "Insufficient data for analysis."}

            latest_row = featured_data.iloc[-1]
            cluster = latest_row['volatility_cluster']

            # Step 3: State Check (Is there an open position?)
            if self.position.get(ticker_symbol):
                # State: Open Position
                current_position = self.position[ticker_symbol]

                # Dynamic Stop-Loss Check (as a primary exit condition)
                if latest_row['close'] <= current_position['stop_loss_price']:
                    del self.position[ticker_symbol]
                    return full_stock_data, {'action': "CUT LOSS", 'reason': "Stop-loss hit."}

                # Call the specialized agents (The Players)
                profit_model_name = f"profit_take_model_{cluster}_vol"
                loss_model_name = f"cut_loss_model_{cluster}_vol"

                profit_model = self.models.get(profit_model_name)
                loss_model = self.models.get(loss_model_name)

                # Ensure models exist and get features
                if not profit_model or not loss_model:
                    self.log(f"Missing models for cluster {cluster}. Cannot make decision.", "WARNING")
                    return full_stock_data, {'action': "HOLD", 'reason': "Missing Models."}

                features = latest_row[self.feature_names[profit_model_name]].astype(float).to_frame().T

                profit_pred = profit_model.predict(features)[0]
                loss_pred = loss_model.predict(features)[0]

                if loss_pred == 1:
                    del self.position[ticker_symbol]
                    return full_stock_data, {'action': "CUT LOSS",
                                             'confidence': loss_model.predict_proba(features)[0][1],
                                             'agent': f"High-Volatility Risk-Management Agent"}
                elif profit_pred == 1:
                    del self.position[ticker_symbol]
                    return full_stock_data, {'action': "SELL", 'confidence': profit_model.predict_proba(features)[0][1],
                                             'agent': f"High-Volatility Profit-Taking Agent"}
                else:
                    return full_stock_data, {'action': "HOLD", 'reason': "No exit signal."}

            else:
                # State: No Position
                entry_model_name = f"entry_model_{cluster}_vol"
                entry_model = self.models.get(entry_model_name)

                if not entry_model:
                    self.log(f"Missing Entry model for cluster {cluster}. Cannot make decision.", "WARNING")
                    return full_stock_data, {'action': "WAIT", 'reason': "Missing Models."}

                features = latest_row[self.feature_names[entry_model_name]].astype(float).to_frame().T

                entry_pred = entry_model.predict(features)[0]
                entry_prob = entry_model.predict_proba(features)[0]

                if entry_pred == 1:
                    # A BUY signal is detected!
                    stop_loss_price = latest_row['close'] - (latest_row['atr_14'] * 2.5)  # Dynamic ATR stop-loss
                    self.position[ticker_symbol] = {'entry_price': latest_row['close'],
                                                    'stop_loss_price': stop_loss_price}

                    return full_stock_data, {'action': "BUY", 'confidence': entry_prob[1] * 100,
                                             'current_price': float(latest_row['close']),
                                             'buy_date': latest_row.name.date(),
                                             'agent': f"{cluster.capitalize()}-Volatility Entry Agent",
                                             'stop_loss_price': stop_loss_price}
                else:
                    return full_stock_data, {'action': "WAIT", 'confidence': entry_prob[0] * 100,
                                             'current_price': float(latest_row['close']),
                                             'agent': f"{cluster.capitalize()}-Volatility Entry Agent"}

        except Exception as e:
            st.code(traceback.format_exc())
            return None, None

    def calculate_dynamic_profit_target(self, confidence):
        # This function is now deprecated in favor of model-driven 'SELL' signals.
        # It is kept for legacy UI compatibility if needed.
        if confidence > 90:
            return 8.0
        elif confidence > 75:
            return 6.5
        elif confidence > 60:
            return 5.0
        else:
            return 3.5

    def apply_israeli_fees_and_tax(self, gross_profit_dollars, num_shares):
        per_share_fee = 0.008 * num_shares
        minimum_fee = 2.50
        single_transaction_fee = max(per_share_fee, minimum_fee)
        total_fees_dollars = single_transaction_fee * 2
        profit_after_fees_dollars = gross_profit_dollars - total_fees_dollars
        tax_dollars = (profit_after_fees_dollars * self.tax) if profit_after_fees_dollars > 0 else 0
        net_profit_dollars = profit_after_fees_dollars - tax_dollars
        total_deducted_dollars = total_fees_dollars + tax_dollars
        return net_profit_dollars, total_deducted_dollars

    def create_chart(self, stock_symbol, stock_data, result, analysis_date):
        if stock_data.empty: return None
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.8, 0.2])
        fig.add_trace(
            go.Candlestick(x=stock_data.index, open=stock_data['open'], high=stock_data['high'], low=stock_data['low'],
                           close=stock_data['close'], name='Price'), row=1, col=1)
        for period, color in [(5, 'orange'), (20, 'blue'), (50, 'red')]:
            if len(stock_data) >= period:
                ma = stock_data['close'].rolling(window=period).mean()
                fig.add_trace(go.Scatter(x=stock_data.index, y=ma, mode='lines', name=f'MA{period}',
                                         line=dict(color=color, width=1)), row=1, col=1)
        if len(stock_data) >= 20:
            try:
                # Create a temporary DataFrame for the calculation
                bb_df = ta.bbands(close=stock_data['close'], length=20)
                fig.add_trace(go.Scatter(x=stock_data.index, y=bb_df['BBU_20_2.0'], mode='lines', name='BB Upper',
                                         line=dict(color='gray', dash='dot', width=1), showlegend=False), row=1, col=1)
                fig.add_trace(go.Scatter(x=stock_data.index, y=bb_df['BBL_20_2.0'], mode='lines', name='BB Lower',
                                         line=dict(color='gray', dash='dot', width=1), fill='tonexty',
                                         fillcolor='rgba(128,128,128,0.1)', showlegend=False), row=1, col=1)
            except Exception as e:
                self.log(f"Could not calculate Bollinger Bands: {e}", "WARNING")
        fig.add_trace(
            go.Bar(x=stock_data.index, y=stock_data['volume'], name='Volume', marker_color='rgba(100,110,120,0.6)'),
            row=2, col=1)
        fig.add_vline(x=analysis_date, line_width=1, line_dash="dash", line_color="white", name="Analysis Date",
                      row=1)
        action = result['action']
        current_price = result['current_price']
        if "BUY" in action:
            buy_date = result['buy_date']
            stop_loss = result.get('stop_loss_price')
            fig.add_trace(go.Scatter(
                x=[buy_date], y=[current_price], mode='markers',
                marker=dict(color='cyan', size=12, symbol='circle-open', line=dict(width=2)), name='Target Buy'
            ), row=1, col=1)
            if stop_loss:
                fig.add_hline(y=float(stop_loss), line_dash="dash", line_color="red",
                              name="Stop-Loss Price", row=1, annotation_text=f"Stop-Loss: ${stop_loss:.2f}",
                              annotation_position="bottom right")

        zoom_start_date = pd.to_datetime(analysis_date) - timedelta(days=10)
        zoom_end_date = pd.to_datetime(analysis_date) + timedelta(days=120)
        fig.update_layout(title_text=f'{stock_symbol} Price & Volume Analysis', xaxis_rangeslider_visible=False,
                          showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig.update_xaxes(range=[zoom_start_date, zoom_end_date], row=1, col=1)
        fig.update_xaxes(range=[zoom_start_date, zoom_end_date], row=2, col=1)
        fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        return fig


# In stockwise_simulation.py

def create_enhanced_interface():
    st.title("🏢 StockWise AI Trading Advisor")

    AGENT_CONFIGS = {
        'Dynamic Profit (Recommended)': "models/NASDAQ-gen3-dynamic",
        '2% Net Profit': "models/NASDAQ-gen3-2pct",
        '3% Net Profit': "models/NASDAQ-gen3-3pct",
        '4% Net Profit': "models/NASDAQ-gen3-4pct"
    }

    st.sidebar.header("🎯 Trading Analysis")

    selected_agent_name = st.sidebar.selectbox(
        "🧠 Select AI Agent",
        options=list(AGENT_CONFIGS.keys())
    )
    selected_model_dir = AGENT_CONFIGS[selected_agent_name]

    if st.session_state.advisor.model_dir != selected_model_dir:
        with st.spinner(f"Loading '{selected_agent_name}' agent..."):
            st.session_state.advisor = ProfessionalStockAdvisor(model_dir=selected_model_dir)

    advisor = st.session_state.advisor
    st.markdown(f"### Now using `{selected_agent_name}` Agent")
    st.markdown("---")

    stock_symbol = st.sidebar.text_input("📊 Stock Symbol", value="NVDA").upper().strip()
    analysis_date = st.sidebar.date_input("📅 Analysis Date", value=datetime.now().date())

    # --- THIS IS THE FIX ---
    # Define the button only ONCE
    analyze_btn = st.sidebar.button("🚀 Run Professional Analysis", type="primary", use_container_width=True)
    # --- END FIX ---

    if not analyze_btn:
        st.info("Enter a stock symbol and date in the sidebar, then click 'Run Analysis' to begin.")
        return

    if not stock_symbol:
        st.warning("Please enter a stock symbol.")
        return

    if not advisor.models:
        st.error("AI models could not be loaded. Please check the logs.")
        return

    with st.spinner(f"Running analysis for {stock_symbol}..."):
        stock_data, result = advisor.run_analysis(stock_symbol, analysis_date)

    if not result:
        st.error("Analysis failed. Please check the debug logs for more information.")
        return

    # --- Display Successful Results ---
    action = result['action']
    confidence = result.get('confidence', 0)
    current_price = result.get('current_price', 0)
    agent = result.get('agent', "Unknown Agent")

    col1, col2, col3 = st.columns([3, 2, 2])
    with col1:
        if "BUY" in action:
            st.success(f"🟢 **RECOMMENDATION: {action}**")
        elif "SELL" in action or "CUT LOSS" in action:
            st.error(f"🔴 **RECOMMENDATION: {action}**")
        else:
            st.warning(f"🟡 **RECOMMENDATION: {action}**")
    with col2:
        st.info(f"🧠 **Agent**: {agent}")
    with col3:
        st.metric("Model Confidence", f"{confidence:.1f}%")

    st.subheader("💰 Price Information & Analysis")
    price_col, target_buy_col, stop_col = st.columns(3)
    price_col.metric("Current Price", f"${current_price:.2f}")

    if "BUY" in action:
        buy_price = result.get('current_price')
        target_buy_col.metric("🎯 Target Buy", f"${buy_price:.2f}")
        stop_loss_price = result.get('stop_loss_price')
        stop_col.metric("🔴 Stop-Loss", f"${stop_loss_price:.2f}")

    st.subheader("📊 Price Chart")
    fig = advisor.create_chart(stock_symbol, stock_data, result, analysis_date)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📝 Action Summary")
    if "BUY" in action and result.get('buy_date'):
        st.success(f"""
        - **Action:** The model recommends **BUYING** {stock_symbol} at **${current_price:.2f}**.
        - **Agent:** The decision was made by the `{agent}`.
        - **Risk Management:** A dynamic stop-loss is suggested at **${result.get('stop_loss_price'):.2f}**.
        """)
    elif "SELL" in action or "CUT LOSS" in action:
        st.error(f"""
        - **Action:** The model recommends **{action}** the position in {stock_symbol}.
        - **Agent:** The exit signal was triggered by the `{agent}`.
        """)
    else:
        st.warning(f"""
        - **Action:** The model recommends to **WAIT or AVOID** buying {stock_symbol}.
        - **Agent:** The decision was made by the `{agent}`.
        """)


# --- Main Execution ---
if __name__ == "__main__":
    # Define the default agent to load on the very first run
    DEFAULT_AGENT_MODEL_DIR = "models/NASDAQ-gen3-dynamic"

    # Initialize the advisor in the session state ONCE if it doesn't exist
    if 'advisor' not in st.session_state:
        st.session_state.advisor = ProfessionalStockAdvisor(model_dir=DEFAULT_AGENT_MODEL_DIR)

    # Check if models were loaded successfully before running the UI
    if st.session_state.advisor.models:
        create_enhanced_interface()
    else:
        st.error(f"FATAL: Default models could not be loaded from '{DEFAULT_AGENT_MODEL_DIR}'.")