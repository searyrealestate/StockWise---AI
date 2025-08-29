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


# --- Feature Engineering Pipeline (for Gen-2 Model) ---
class FeatureCalculator:
    """A dedicated class to handle all feature calculations for the Gen-2 model."""

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
        if df.empty or len(df) < 200:
            return pd.DataFrame()
        close, high, low, volume = df['Close'].squeeze(), df['High'].squeeze(), df['Low'].squeeze(), df[
            'Volume'].squeeze()

        # --- Calculate all indicators ---
        df['Volume_MA_20'] = ta.trend.sma_indicator(volume, window=20)
        df['RSI_14'] = ta.momentum.rsi(close, window=14)
        df['Momentum_5'] = close.diff(5)
        macd = ta.trend.MACD(close=close)
        df['MACD'], df['MACD_Signal'], df['MACD_Histogram'] = macd.macd(), macd.macd_signal(), macd.macd_diff()
        bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
        df['BB_Upper'], df['BB_Lower'], df['BB_Middle'], df[
            'BB_Position'] = bb.bollinger_hband(), bb.bollinger_lband(), bb.bollinger_mavg(), bb.bollinger_pband()
        df['Daily_Return'] = close.pct_change()
        df['Volatility_20D'] = df['Daily_Return'].rolling(window=20).std()
        df['ATR_14'] = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
        adx_indicator = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14)
        df['ADX'], df['ADX_pos'], df['ADX_neg'] = adx_indicator.adx(), adx_indicator.adx_pos(), adx_indicator.adx_neg()
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
        df['RSI_28'] = ta.momentum.rsi(close, window=28)
        df['Dominant_Cycle_126D'] = close.rolling(window=126, min_periods=50).apply(self.get_dominant_cycle, raw=True)

        # --- THIS IS THE KEY FIX ---
        # Instead of dropping rows with ANY NaN, we only drop rows where critical
        # model inputs are NaN. This makes the process much more robust.
        critical_features = ['RSI_14', 'MACD_Histogram', 'BB_Position', 'ADX', 'OBV', 'Dominant_Cycle_126D']
        df.dropna(subset=critical_features, inplace=True)

        return df


# --- Main Application Class (with DataSourceManager integrated) ---
class ProfessionalStockAdvisor:
    def __init__(self, debug=False, download_log=False, data_source_manager=None, testing_mode=False):
        self.log_entries = []
        self.debug = debug
        self.download_log = download_log
        self.testing_mode = testing_mode

        # Logic to handle testing vs. live mode for the data source
        if data_source_manager:
            self.data_source_manager = data_source_manager
        elif self.testing_mode:
            self.data_source_manager = None
        else:
            self.data_source_manager = DataSourceManager(use_ibkr=True, debug=self.debug)

        # In testing mode, we do NOT load the model from disk.
        # The test will provide a fake model manually.
        if self.testing_mode:
            self.model, self.feature_names, self.model_filename = None, [], None
        else:
            self.model, self.feature_names, self.model_filename = self.load_model()

        self.calculator = FeatureCalculator()
        self.tax = 0.25
        self.broker_fee = 0.004

        if self.download_log:
            self.log_file = self.setup_log_file()
        self.log("Application Initialized.", "INFO")

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

    @st.cache_resource
    def load_model(_self):
        _self.log("Loading Gen-2 model...", "INFO")
        model_dir = "models/NASDAQ-training set"
        model_search_pattern = os.path.join(model_dir, "nasdaq_gen2_optimized_model_*.pkl")
        model_files = glob.glob(model_search_pattern)
        if not model_files: return None, None, None
        latest_model_path = max(model_files, key=os.path.getctime)
        features_path = latest_model_path.replace(".pkl", "_features.json")
        try:
            model = joblib.load(latest_model_path)
            with open(features_path, 'r') as f:
                feature_names = json.load(f)
            model_filename = os.path.basename(latest_model_path)
            return model, feature_names, model_filename
        except Exception as e:
            st.error(f"Error loading model: {e}")
            _self.log(f"ERROR: Failed to load model - {e}", "ERROR")
            return None, None, None

    # def validate_symbol_professional(self, symbol):
    #     """
    #     What's new:
    #     - It no longer calls the non-existent 'validate_symbol' from DataSourceManager.
    #     - The yfinance validation logic is now correctly placed here as a fallback.
    #     - It correctly checks the IBKR connection status via the 'ibkr_connected' attribute,
    #       fixing the 'AttributeError: ... has no attribute 'is_ibkr_connected''.
    #     """
    #     # First, try to validate using the IBKR connection if available
    #     if self.data_source_manager.use_ibkr and self.data_source_manager.ibkr_connected:
    #         try:
    #             # This part now correctly assumes your DataSourceManager might have a
    #             # method for IBKR validation, but it will gracefully fail if not.
    #             if hasattr(self.data_source_manager, 'validate_symbol'):
    #                 is_valid = self.data_source_manager.validate_symbol(symbol)
    #                 self.log(f"🔍 IBKR validation for {symbol}: {is_valid}", "INFO")
    #                 return is_valid
    #         except Exception as e:
    #             self.log(f"❌ IBKR validation error for {symbol}: {e}", "ERROR")
    #
    #     # If IBKR is not used or fails, fallback to yfinance validation here
    #     self.log(f"Using yfinance for validation of {symbol}", "INFO")
    #     try:
    #         ticker = yf.Ticker(symbol)
    #         info = ticker.info
    #         # A reliable check for a valid ticker is to see if it has price data
    #         if 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
    #             return True
    #         if 'currentPrice' in info and info['currentPrice'] is not None:
    #             return True
    #         self.log(f"yfinance validation failed for {symbol}: No price data in info dict.", "WARNING")
    #         return False
    #     except Exception:
    #         self.log(f"yfinance lookup failed for {symbol}.", "ERROR")
    #         return False

    def validate_symbol_professional(self, symbol):
        self.log(f"Using yfinance for validation of {symbol}", "INFO")
        try:
            ticker = yf.Ticker(symbol)
            # Check if the info dictionary contains a price, a reliable way to validate a ticker
            if 'regularMarketPrice' in ticker.info and ticker.info['regularMarketPrice'] is not None:
                return True
            if 'currentPrice' in ticker.info and ticker.info['currentPrice'] is not None:
                return True
            return False
        except Exception:
            return False

    def run_analysis(self, ticker_symbol, analysis_date):
        self.log(f"Starting analysis for {ticker_symbol} on {analysis_date}...", "INFO")
        try:
            stock_data = self.data_source_manager.get_stock_data(ticker_symbol)
            if stock_data is None or stock_data.empty:
                self.log(f"WARNING: No data downloaded for {ticker_symbol}.", "WARNING")
                return None, None

            stock_data_filtered = stock_data[stock_data.index <= pd.to_datetime(analysis_date)].copy()
            if stock_data_filtered.empty:
                self.log(f"WARNING: No historical data for {ticker_symbol} on or before {analysis_date}.", "WARNING")
                return None, None

            self.log(f"Downloaded and filtered {len(stock_data_filtered)} rows of data.", "INFO")
            featured_data = self.calculator.calculate_all_features(stock_data_filtered)

            if featured_data.empty:
                self.log("WARNING: Not enough data for prediction.", "WARNING")
                return stock_data_filtered, None

            self.log("Feature calculation complete. Making prediction...", "INFO")
            latest_features_df = featured_data[self.feature_names]
            prediction = self.model.predict(latest_features_df)[-1]
            probability = self.model.predict_proba(latest_features_df)[-1]
            confidence = probability[1] if prediction == 1 else probability[0]

            current_price = float(stock_data_filtered['Close'].iloc[-1])
            action = "BUY" if prediction == 1 else "WAIT / AVOID"

            result = {
                'action': action,
                'confidence': confidence * 100,
                'current_price': current_price,
            }

            if action == "BUY":
                # Calculate profit breakdown
                sell_price = current_price * 1.05
                gross_profit_pct = (sell_price / current_price - 1) * 100
                net_profit_pct, total_deducted_pct = self.apply_israeli_fees_and_tax(gross_profit_pct)

                result['gross_profit_pct'] = gross_profit_pct
                result['net_profit_pct'] = net_profit_pct
                result['total_deducted_pct'] = total_deducted_pct

                # Add the full list of latest features for the UI display
            feature_list_for_model = [
                'Volume_MA_20', 'ADX', 'OBV', 'Volatility_20D', 'ADX_pos', 'ADX_neg',
                'ATR_14', 'MACD_Histogram', 'Daily_Return', 'RSI_28', 'BB_Position',
                'Momentum_5', 'MACD_Signal', 'RSI_14', 'MACD', 'BB_Lower', 'BB_Upper',
                'BB_Middle', 'Dominant_Cycle_126D'
            ]

            # Ensure only available features are selected
            available_features = [f for f in feature_list_for_model if f in featured_data.columns]
            result['latest_features'] = latest_features_df[available_features].to_dict('records')[0]

            self.log(f"Prediction successful: {result['action']} with {result['confidence']:.1f}% confidence.",
                     "SUCCESS")
            return stock_data, result
        except Exception as e:
            self.log(f"ERROR during analysis: {e}", "ERROR")
            st.error(f"An error occurred during analysis: {e}")
            st.code(traceback.format_exc())
            return None, None

    def apply_israeli_fees_and_tax(self, gross_profit_pct):
        """
        Applies Israeli broker fees and capital gains tax.
        Returns the net profit percentage and the total percentage deducted.
        """
        fees_pct = self.broker_fee * 100
        profit_after_fees = gross_profit_pct - fees_pct

        if profit_after_fees > 0:
            tax_amount_pct = profit_after_fees * self.tax
            net_profit_pct = profit_after_fees - tax_amount_pct
        else:
            tax_amount_pct = 0
            net_profit_pct = profit_after_fees

        total_deducted_pct = fees_pct + tax_amount_pct
        self.log(f"Profit calculation: {gross_profit_pct:.2f}% (gross) -> {net_profit_pct:.2f}% (net)")
        return net_profit_pct, total_deducted_pct

    def create_chart(self, stock_symbol, stock_data, result, analysis_date):
        """Creates a comprehensive Plotly chart with price, volume, indicators, and annotations."""
        self.log(f"Generating chart for {stock_symbol}...", "INFO")

        if stock_data.empty:
            self.log("WARNING: No data available to create chart.", "WARNING")
            return None

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.8, 0.2])

        # Plot the candlestick chart
        fig.add_trace(go.Candlestick(
            x=stock_data.index, open=stock_data['Open'], high=stock_data['High'],
            low=stock_data['Low'], close=stock_data['Close'], name='Price'
        ), row=1, col=1)

        # Add Moving Averages
        for period, color in [(5, 'orange'), (20, 'blue'), (50, 'red')]:
            if len(stock_data) >= period:
                ma = stock_data['Close'].rolling(window=period).mean()
                fig.add_trace(go.Scatter(
                    x=stock_data.index, y=ma, mode='lines', name=f'MA{period}',
                    line=dict(color=color, width=1)
                ), row=1, col=1)

        # Add Bollinger Bands
        if len(stock_data) >= 20:
            try:
                bb = ta.volatility.BollingerBands(stock_data['Close'])
                fig.add_trace(go.Scatter(
                    x=stock_data.index, y=bb.bollinger_hband(), mode='lines', name='BB Upper',
                    line=dict(color='gray', dash='dot', width=1), showlegend=False
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=stock_data.index, y=bb.bollinger_lband(), mode='lines', name='BB Lower',
                    line=dict(color='gray', dash='dot', width=1),
                    fill='tonexty', fillcolor='rgba(128,128,128,0.1)', showlegend=False
                ), row=1, col=1)
            except Exception as e:
                self.log(f"Could not calculate Bollinger Bands: {e}", "WARNING")

        # Plot the volume chart
        fig.add_trace(go.Bar(
            x=stock_data.index, y=stock_data['Volume'], name='Volume',
            marker_color='rgba(100,110,120,0.6)'
        ), row=2, col=1)

        # Add markers and lines for a "BUY" signal
        action = result['action']
        current_price = result['current_price']
        if "BUY" in action:
            sell_price = current_price * 1.05
            stop_loss = current_price * 0.97

            # Analysis Date Marker (Vertical Line)
            fig.add_vline(x=analysis_date, line_width=1, line_dash="dash", line_color="white", name="Analysis Date",
                          row=1)
            # Buy Price Marker (Circle)
            fig.add_trace(go.Scatter(
                x=[analysis_date], y=[current_price], mode='markers',
                marker=dict(color='cyan', size=12, symbol='circle-open', line=dict(width=2)), name='Buy Price'
            ), row=1, col=1)
            # Sell Price Marker (Horizontal Line)
            fig.add_hline(y=float(sell_price), line_dash="dash", line_color="lightgreen",
                          name="Target Sell Price", row=1, annotation_text=f"Target: ${sell_price:.2f}",
                          annotation_position="top right")
            # Stop-Loss Marker (Horizontal Line)
            fig.add_hline(y=float(stop_loss), line_dash="dash", line_color="red",
                          name="Stop-Loss Price", row=1, annotation_text=f"Stop-Loss: ${stop_loss:.2f}",
                          annotation_position="bottom right")

        # Define the initial zoom range
        zoom_start_date = pd.to_datetime(analysis_date) - timedelta(days=10)  # Updated from 2 to 10 days
        zoom_end_date = pd.to_datetime(analysis_date) + timedelta(days=120)

        # Update layout
        fig.update_layout(
            title_text=f'{stock_symbol} Price & Volume Analysis',
            xaxis_rangeslider_visible=False, showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_xaxes(range=[zoom_start_date, zoom_end_date], row=1, col=1)
        fig.update_xaxes(range=[zoom_start_date, zoom_end_date], row=2, col=1)
        fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)

        return fig


def create_enhanced_interface():
    # --- Part 1: Setup and Sidebar ---
    st.title("🏢 StockWise AI Trading Advisor")
    if 'advisor' not in st.session_state:
        st.session_state.advisor = ProfessionalStockAdvisor()
    advisor = st.session_state.advisor

    st.markdown(f"### Powered by Gen-2 Model: `{advisor.model_filename or 'N/A'}`")
    st.markdown("---")

    st.sidebar.header("🎯 Trading Analysis")
    stock_symbol = st.sidebar.text_input("📊 Stock Symbol", value="NVDA", help="Enter any stock ticker").upper().strip()
    analysis_date = st.sidebar.date_input("📅 Analysis Date", value=datetime.now().date())
    num_shares = st.sidebar.number_input("📦 Number of Shares", min_value=1, value=10, step=1, help="Enter the number of shares to calculate profit in dollars.")

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

    # Debug Panel Setup
    st.sidebar.markdown("---")
    st.sidebar.header("🔧 Settings & Debug")
    debug_mode = st.sidebar.checkbox("Show Debug Logs", value=False)
    advisor.download_log = st.sidebar.checkbox("Create Downloadable Log File", value=False)
    if advisor.download_log and (not hasattr(advisor, 'log_file') or not advisor.log_file):
        advisor.log_file = advisor.setup_log_file()


    # --- Part 2: Main Display Logic (REFACTORED) ---

    # Guard Clause: Handle the initial welcome screen
    if not analyze_btn or not stock_symbol:
        st.info("Enter a stock symbol and date in the sidebar, then click 'Run Analysis' to begin.")
        return

    # Guard Clause: Handle invalid symbol
    if not st.session_state.get('is_valid_symbol', False):
        st.error(f"Cannot analyze {stock_symbol} as it is not a valid symbol. Please try another.")
        return

    # Run Analysis
    with st.spinner(f"Running analysis for {stock_symbol}..."):
        stock_data, result = advisor.run_analysis(stock_symbol,
                                                  datetime.combine(analysis_date, datetime.min.time()))

    # Guard Clause: Handle analysis failure (e.g., no data)
    if not result:
        st.error("Analysis failed. Please check the debug logs for more information.")
        return

    # --- Part 3: Display Successful Results ---
    # If the code reaches this point, the analysis was successful.
    st.success(f"✅ Analysis complete for {stock_symbol}")
    action = result['action']
    confidence = result['confidence']
    current_price = result['current_price']

    # Display top-level recommendation metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model Confidence", f"{confidence:.1f}%")
    with col2:
        if "BUY" in action:
            st.success(f"🟢 **RECOMMENDATION: {action}**")
        else:
            st.warning(f"🟡 **RECOMMENDATION: {action}**")

    # Display EITHER the profit panel OR the wait message
    if "BUY" in action:
        st.subheader("💰 Price Information & Profit Analysis")
        price_col, target_col, stop_col, gross_profit_col = st.columns(4)

        total_investment = current_price * num_shares
        gross_profit_dollars = total_investment * (result['gross_profit_pct'] / 100)
        net_profit_dollars = total_investment * (result['net_profit_pct'] / 100)
        fees_and_tax_dollars = gross_profit_dollars - net_profit_dollars
        sell_price = current_price * 1.05
        stop_loss = current_price * 0.97

        price_col.metric("Current Price", f"${current_price:.2f}")
        target_col.metric("🟢 Target Sell", f"${sell_price:.2f}", help="Example 5% profit target")
        stop_col.metric("🔴 Stop-Loss", f"${stop_loss:.2f}", help="Example 3% stop-loss")
        gross_profit_col.metric("💰 Gross Profit", f"${gross_profit_dollars:.2f} ({result['gross_profit_pct']:.2f}%)")
        st.markdown(f"**Net Profit Breakdown:** **`${net_profit_dollars:.2f}`** (`{result['net_profit_pct']:.2f}%`) (after deducting `${fees_and_tax_dollars:.2f}` (`{result['total_deducted_pct']:.2f}%`) for Israeli tax & fees)")
    else:
        st.info(f"The model recommends to wait or avoid taking a new position in **{stock_symbol}** at this time. The signals are not strong enough to meet the criteria for a high-confidence 'BUY' signal.")

    # Display the Chart (this now runs for ALL successful outcomes)
    st.subheader("📊 Price Chart")
    fig = advisor.create_chart(stock_symbol, stock_data, result, analysis_date)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Chart Legend Explained"):
            st.markdown("""
            - **Dashed Vertical Line**: The date you selected for analysis.
            - **Cyan Circle Marker**: The buying price on the analysis date.
            - **Green Dashed Horizontal Line**: The target selling price for a 5% gross profit.
            - **Red Dashed Horizontal Line**: The stop-loss price, set at 3% below the buying price.
            - **Colored Lines (Orange, Blue, Red)**: These are the 5-day, 20-day, and 50-day Simple Moving Averages (SMA), respectively.
            - **Gray Dotted Lines / Shaded Area**: These are the Bollinger Bands, which indicate market volatility.
            """)
    else:
        st.warning("Could not display chart: No data available for the selected date range.")

    # Display Debug Logs if toggled
    if debug_mode:
        st.sidebar.markdown("---")
        st.sidebar.header("🐛 Debug Information")
        if st.sidebar.button("Clear Log"):
            st.session_state.advisor.log_entries = []
        with st.sidebar.expander("Show Live Log", expanded=True):
            log_container = st.empty()
            log_text = "\n".join(st.session_state.advisor.log_entries)
            log_container.code(log_text, language='log')
        if advisor.download_log and hasattr(advisor, 'log_file') and advisor.log_file and os.path.exists(advisor.log_file):
            with open(advisor.log_file, "rb") as f:
                st.sidebar.download_button(
                    label="📥 Download Full Log",
                    data=f,
                    file_name=os.path.basename(advisor.log_file),
                    mime="text/plain"
                )


# --- Main Execution ---
if __name__ == "__main__":
    if 'advisor' not in st.session_state:
        st.session_state.advisor = ProfessionalStockAdvisor()
    if st.session_state.advisor.model:
        create_enhanced_interface()
    else:
        st.error("Model could not be loaded. The application cannot start.")