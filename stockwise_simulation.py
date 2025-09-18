import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
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
        if df.empty or len(df) < 50:
            return pd.DataFrame()
        close, high, low, volume = df['Close'].squeeze(), df['High'].squeeze(), df['Low'].squeeze(), df[
            'Volume'].squeeze()

        # --- Existing and New Gen-3 Feature Calculations ---
        df['Daily_Return'] = close.pct_change()
        df['ATR_14'] = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
        df['Volume_MA_20'] = ta.trend.sma_indicator(volume, window=20)
        df['RSI_14'] = ta.momentum.rsi(close, window=14)
        df['Momentum_5'] = close.diff(5)
        macd = ta.trend.MACD(close=close)
        df['MACD'], df['MACD_Signal'], df['MACD_Histogram'] = macd.macd(), macd.macd_signal(), macd.macd_diff()
        bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
        df['BB_Upper'], df['BB_Lower'], df[
            'BB_Middle'] = bb.bollinger_hband(), bb.bollinger_lband(), bb.bollinger_mavg()
        df['BB_Position'] = bb.bollinger_pband()
        df['Volatility_20D'] = df['Daily_Return'].rolling(window=20).std()
        adx_indicator = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14)
        df['ADX'], df['ADX_pos'], df['ADX_neg'] = adx_indicator.adx(), adx_indicator.adx_pos(), adx_indicator.adx_neg()
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
        df['RSI_28'] = ta.momentum.rsi(close, window=28)
        df['Dominant_Cycle_126D'] = close.rolling(window=126, min_periods=50).apply(self.get_dominant_cycle, raw=False)

        # New Statistical Features
        df['Z_Score_20'] = (df['Close'] - df['BB_Middle']) / df['Close'].rolling(20).std()
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']

        # New Contextual Feature
        try:
            qqq_df = yf.download("QQQ", start=df.index.min(), end=df.index.max(), progress=False, auto_adjust=True)
            qqq_close = qqq_df['Close'].reindex(df.index, method='ffill')
            df['Correlation_50D_QQQ'] = df['Close'].rolling(50).corr(qqq_close)
        except Exception:
            df['Correlation_50D_QQQ'] = np.nan

        # Smoothed Features
        df['Smoothed_Close_5D'] = df['Close'].ewm(span=5, adjust=False).mean()
        df['RSI_14_Smoothed'] = df['RSI_14'].ewm(span=5, adjust=False).mean()

        # Volatility Clustering (NOTE: This is done globally in data pipeline, mocking here for live analysis)
        df['Volatility_90D'] = df['Daily_Return'].rolling(90).std()
        # Mocking the cluster labels for live prediction; real system would load this from data.
        df['Volatility_Cluster'] = pd.cut(df['Volatility_90D'], bins=3, labels=['low', 'mid', 'high'])

        df.dropna(inplace=True)
        return df


# --- Main Application Class (with Gen-3 Architecture) ---
class ProfessionalStockAdvisor:
    def __init__(self, debug=False, download_log=False, data_source_manager=None, testing_mode=False):
        self.log_entries = []
        self.debug = debug
        self.download_log = download_log
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
        self.model_dir = "models/NASDAQ-gen3"
        self.model_version_info = "Gen-3: Orchestra of Specialists"
        if self.download_log: self.log_file = self.setup_log_file()
        self.log("Application Initialized.", "INFO")

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

            spy_data[f'SMA_{days}'] = ta.trend.sma_indicator(spy_data['Close'], window=days)
            latest_price = spy_data['Close'].iloc[-1]
            moving_average = spy_data[f'SMA_{days}'].iloc[-1]
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
                price_on_date = stock_data.loc[pd.to_datetime(analysis_date)]['Close'] if pd.to_datetime(
                    analysis_date) in stock_data.index else 0
                return stock_data, {'action': "WAIT / AVOID", 'confidence': 99.9, 'current_price': price_on_date,
                                    'reason': "Market Downtrend", 'buy_date': None, 'agent': "Market Regime Agent"}

            full_stock_data = self.data_source_manager.get_stock_data(ticker_symbol)
            if full_stock_data is None or full_stock_data.empty: return None, None

            # Step 2: Feature Engineering
            featured_data = self.calculator.calculate_all_features(
                full_stock_data[full_stock_data.index <= pd.to_datetime(analysis_date)])
            if featured_data.empty: return full_stock_data, {'action': "WAIT",
                                                             'reason': "Insufficient data for analysis."}

            latest_row = featured_data.iloc[-1]
            cluster = latest_row['Volatility_Cluster']

            # Step 3: State Check (Is there an open position?)
            if self.position.get(ticker_symbol):
                # State: Open Position
                current_position = self.position[ticker_symbol]

                # Dynamic Stop-Loss Check (as a primary exit condition)
                if latest_row['Close'] <= current_position['stop_loss_price']:
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
                    stop_loss_price = latest_row['Close'] - (latest_row['ATR_14'] * 2.5)  # Dynamic ATR stop-loss
                    self.position[ticker_symbol] = {'entry_price': latest_row['Close'],
                                                    'stop_loss_price': stop_loss_price}

                    return full_stock_data, {'action': "BUY", 'confidence': entry_prob[1] * 100,
                                             'current_price': float(latest_row['Close']),
                                             'buy_date': latest_row.name.date(),
                                             'agent': f"{cluster.capitalize()}-Volatility Entry Agent",
                                             'stop_loss_price': stop_loss_price}
                else:
                    return full_stock_data, {'action': "WAIT", 'confidence': entry_prob[0] * 100,
                                             'current_price': float(latest_row['Close']),
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
            go.Candlestick(x=stock_data.index, open=stock_data['Open'], high=stock_data['High'], low=stock_data['Low'],
                           close=stock_data['Close'], name='Price'), row=1, col=1)
        for period, color in [(5, 'orange'), (20, 'blue'), (50, 'red')]:
            if len(stock_data) >= period:
                ma = stock_data['Close'].rolling(window=period).mean()
                fig.add_trace(go.Scatter(x=stock_data.index, y=ma, mode='lines', name=f'MA{period}',
                                         line=dict(color=color, width=1)), row=1, col=1)
        if len(stock_data) >= 20:
            try:
                bb = ta.volatility.BollingerBands(stock_data['Close'])
                fig.add_trace(go.Scatter(x=stock_data.index, y=bb.bollinger_hband(), mode='lines', name='BB Upper',
                                         line=dict(color='gray', dash='dot', width=1), showlegend=False), row=1, col=1)
                fig.add_trace(go.Scatter(x=stock_data.index, y=bb.bollinger_lband(), mode='lines', name='BB Lower',
                                         line=dict(color='gray', dash='dot', width=1), fill='tonexty',
                                         fillcolor='rgba(128,128,128,0.1)', showlegend=False), row=1, col=1)
            except Exception as e:
                self.log(f"Could not calculate Bollinger Bands: {e}", "WARNING")
        fig.add_trace(
            go.Bar(x=stock_data.index, y=stock_data['Volume'], name='Volume', marker_color='rgba(100,110,120,0.6)'),
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


def create_enhanced_interface():
    # --- Part 1: Setup and Sidebar ---
    st.title("🏢 StockWise AI Trading Advisor")
    if 'advisor' not in st.session_state: st.session_state.advisor = ProfessionalStockAdvisor()
    advisor = st.session_state.advisor
    if 'analysis_date' not in st.session_state: st.session_state.analysis_date = datetime.now().date()
    st.markdown(f"### Powered by `{advisor.model_version_info}`")
    st.markdown("---")

    def set_date_to_today():
        st.session_state.analysis_date = datetime.now().date()
        st.session_state.advisor.position = {}

    st.sidebar.header("🎯 Trading Analysis")
    stock_symbol = st.sidebar.text_input("📊 Stock Symbol", value="NVDA").upper().strip()
    st.sidebar.date_input("📅 Analysis Date", key='analysis_date')
    st.sidebar.button("Today", on_click=set_date_to_today, use_container_width=True)
    investment_amount = st.sidebar.number_input("💰 Investment Amount ($)", min_value=1.0, value=1000.0, step=100.0)

    if stock_symbol:
        if 'last_validated_symbol' not in st.session_state or st.session_state.last_validated_symbol != stock_symbol:
            with st.spinner(f"Validating {stock_symbol}..."):
                st.session_state.is_valid_symbol = advisor.validate_symbol_professional(stock_symbol)
                st.session_state.last_validated_symbol = stock_symbol
        if st.session_state.is_valid_symbol:
            st.sidebar.success(f"✅ {stock_symbol} is a valid symbol.")
        else:
            st.sidebar.error(f"❌ {stock_symbol} is not a valid symbol.")
    analyze_btn = st.sidebar.button("🚀 Run Professional Analysis", type="primary", use_container_width=True)

    # --- Part 2: Main Display Logic ---
    if not analyze_btn or not stock_symbol:
        st.info("Enter a stock symbol and date in the sidebar, then click 'Run Analysis' to begin.")
        return
    if not st.session_state.get('is_valid_symbol', False):
        st.error(f"Cannot analyze {stock_symbol} as it is not a valid symbol.")
        return
    if not advisor.models:
        st.error("AI models could not be loaded. Please check the logs.")
        return

    with st.spinner(f"Running analysis for {stock_symbol}..."):
        stock_data, result = advisor.run_analysis(stock_symbol, st.session_state.analysis_date)
    if not result:
        st.error("Analysis failed. Please check the debug logs for more information.")
        return

    # --- Part 3: Display Successful Results ---
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
        if confidence < 55:
            color = "red"
        elif 55 <= confidence < 60:
            color = "yellow"
        elif 60 <= confidence < 70:
            color = "orange"
        elif 70 <= confidence < 80:
            color = "green"
        else:
            color = "blue"
        st.markdown(f"""<div style="text-align: right;"><span style="font-size: 1em;">Model Confidence</span><br><span 
        style="font-size: 2.5em; color: {color}; font-weight: bold;">{confidence:.1f}%</span></div>""",
                    unsafe_allow_html=True)

    st.subheader("💰 Price Information & Analysis")
    price_col, target_buy_col, target_sell_col, stop_col, profit_col = st.columns(5)
    price_col.metric("Current Price", f"${current_price:.2f}")

    if "BUY" in action:
        buy_price = result.get('current_price')
        target_buy_col.metric("🎯 Target Buy", f"${buy_price:.2f}", delta="Signal", delta_color="normal")
        stop_loss_price = result.get('stop_loss_price')
        stop_col.metric("🔴 Stop-Loss", f"${stop_loss_price:.2f}")
        # Profit target is now dynamic based on a trailing exit, so no fixed value to display
        target_sell_col.metric("🟢 Target Sell", "Model-Driven", delta="Dynamic")
    elif "SELL" in action or "CUT LOSS" in action:
        # For exit signals, we don't have a new entry price, but we can show the last known values
        # Assume the last open position is tracked for display purposes
        stop_col.metric("🔴 Stop-Loss", "N/A")
        target_sell_col.metric("🟢 Target Sell", f"Exit Signal", delta="Action")

    st.subheader("📊 Price Chart")
    fig = advisor.create_chart(stock_symbol, stock_data, result, st.session_state.analysis_date)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Chart Legend Explained"):
            st.markdown("""
            - **Dashed Vertical Line**: The date you selected for analysis.
            - **Cyan Circle Marker**: The buying price on the analysis date.
            - **Green Dashed Horizontal Line**: The target selling price.
            - **Red Dashed Horizontal Line**: The stop-loss price.
            - **Colored Lines**: 5-day (Orange), 20-day (Blue), and 50-day (Red) Simple Moving Averages.
            - **Shaded Area**: Bollinger Bands, indicating market volatility.
            """)
    else:
        st.warning("Could not display chart.")

    st.subheader("📝 Action Summary")
    if "BUY" in action and result.get('buy_date'):
        buy_date_str = result['buy_date'].strftime('%B %d, %Y')
        stop_loss_price = result.get('stop_loss_price')
        st.markdown("#### For Those Looking to Buy:")
        st.success(f"""
        - **Action:** The model recommends **BUYING** {stock_symbol}.
        - **When:** A high-probability entry point was detected on or around **{buy_date_str}**.
        - **Price:** The suggested entry price is **${current_price:.2f}**.
        """)
        st.markdown("#### For Existing Stock Holders:")
        st.info(f"""
        - **Action:** The model recommends **HOLDING** your position.
        - **Next Action:** The system will monitor for a model-driven `SELL` or `CUT LOSS` signal.
        - **Stop-Loss:** To manage risk, a dynamic ATR-based stop-loss would be placed at **${stop_loss_price:.2f}**.
        """)
    elif "SELL" in action or "CUT LOSS" in action:
        st.markdown("#### For Existing Stock Holders:")
        st.error(f"""
        - **Action:** The model recommends **{action}** the position.
        - **Reason:** An exit signal was triggered by the `{agent}`.
        """)

    else:
        st.markdown("#### For Those Looking to Buy:")
        st.warning(f"""
        - **Action:** The model recommends to **WAIT or AVOID** buying {stock_symbol} at its current price.
        - **Reason:** The decision was made by the `{agent}`.
        """)
        st.markdown("#### For Existing Stock Holders:")
        st.info(f"""
        - **Action:** The model recommends **HOLDING** your position.
        - **Recommendation:** The system will continue to monitor for an exit signal.
        """)


# --- Main Execution ---
if __name__ == "__main__":
    if 'advisor' not in st.session_state or st.session_state.advisor.model_version_info == "Gen-2 Optimized Model":
        st.session_state.advisor = ProfessionalStockAdvisor()

    if st.session_state.advisor.models:
        create_enhanced_interface()
    else:
        st.error("AI models could not be loaded. The application cannot start.")