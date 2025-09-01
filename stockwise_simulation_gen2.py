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
        critical_features = ['RSI_14', 'MACD_Histogram', 'BB_Position', 'ADX', 'OBV', 'Dominant_Cycle_126D']

        # First, check which of these critical features actually exist in the DataFrame
        existing_features_to_check = [f for f in critical_features if f in df.columns]

        # Now, drop NaNs only from the columns that were successfully created
        df.dropna(subset=existing_features_to_check, inplace=True)

        return df


# --- Main Application Class (with DataSourceManager integrated) ---
class ProfessionalStockAdvisor:
    def __init__(self, debug=False, download_log=False, data_source_manager=None, testing_mode=False):
        self.log_entries = []
        self.debug = debug
        self.download_log = download_log
        self.testing_mode = testing_mode

        # This logic prevents live connections during tests
        if data_source_manager:
            self.data_source_manager = data_source_manager
        elif self.testing_mode:
            self.data_source_manager = None
        else:
            self.data_source_manager = DataSourceManager(use_ibkr=True, debug=self.debug)

        # This logic prevents loading a real model from disk during tests
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
        try:
            full_stock_data = self.data_source_manager.get_stock_data(ticker_symbol)
            if full_stock_data is None or full_stock_data.empty: return None, None

            # Search for a BUY signal up to 5 days from the user's selected date
            for i in range(5):
                current_analysis_date = pd.to_datetime(analysis_date) + timedelta(days=i)
                stock_data_filtered = full_stock_data[full_stock_data.index <= current_analysis_date].copy()
                if stock_data_filtered.empty or len(stock_data_filtered) < 50: continue

                featured_data = self.calculator.calculate_all_features(stock_data_filtered)
                if featured_data.empty: continue

                latest_features_df = featured_data[self.feature_names]
                prediction = self.model.predict(latest_features_df)[-1]

                if prediction == 1:
                    probability = self.model.predict_proba(latest_features_df)[-1]
                    result = {
                        'action': "BUY",
                        'confidence': probability[1] * 100,
                        'current_price': float(stock_data_filtered['Close'].iloc[-1]),
                        'buy_date': current_analysis_date.date(),
                        'latest_features': latest_features_df.iloc[-1].to_dict()
                    }
                    result['gross_profit_pct'] = self.calculate_dynamic_profit_target(result['confidence'])
                    return full_stock_data, result

            # If no BUY signal, generate a WAIT signal with a Target Buy Price
            stock_data_filtered = full_stock_data[full_stock_data.index <= pd.to_datetime(analysis_date)].copy()
            featured_data = self.calculator.calculate_all_features(stock_data_filtered)
            latest_features_df = featured_data[self.feature_names]
            probability = self.model.predict_proba(latest_features_df)[-1]

            # Calculate a target buy price (e.g., the lower Bollinger Band or 50-day MA)
            lower_bb = featured_data['BB_Lower'].iloc[-1]
            ma_50 = ta.trend.sma_indicator(stock_data_filtered['Close'], window=50).iloc[-1]
            target_buy_price = min(lower_bb, ma_50)  # Use the lower of the two as a support target

            return full_stock_data, {
                'action': "WAIT / AVOID",
                'confidence': probability[0] * 100,
                'current_price': float(stock_data_filtered['Close'].iloc[-1]),
                'buy_date': None,
                'target_buy_price': target_buy_price,  # Add the new target
                'latest_features': latest_features_df.iloc[-1].to_dict()
            }
        except Exception as e:
            st.code(traceback.format_exc())
            return None, None

    def apply_israeli_fees_and_tax(self, gross_profit_dollars, num_shares):
        """
        Applies a complex broker fee (per-share vs. minimum) and Israeli capital gains tax.
        Works with dollar amounts for precision.
        """
        # 1. Calculate the fee for a single transaction (a buy or a sell)
        per_share_fee = 0.008 * num_shares
        minimum_fee = 2.50
        single_transaction_fee = max(per_share_fee, minimum_fee)

        # 2. Calculate the total fee for a round-trip trade (buy + sell)
        total_fees_dollars = single_transaction_fee * 2

        # 3. Calculate profit after fees
        profit_after_fees_dollars = gross_profit_dollars - total_fees_dollars

        # 4. Apply tax only if there's a profit after fees
        tax_dollars = (profit_after_fees_dollars * self.tax) if profit_after_fees_dollars > 0 else 0
        net_profit_dollars = profit_after_fees_dollars - tax_dollars

        # 5. Return the final dollar amounts
        total_deducted_dollars = total_fees_dollars + tax_dollars
        return net_profit_dollars, total_deducted_dollars

    def calculate_dynamic_profit_target(self, confidence):
        """
        Calculates a dynamic gross profit target based on model confidence.
        """
        if confidence > 90:
            gross_profit_pct = 8.0  # Very High Confidence -> Higher Target
        elif confidence > 75:
            gross_profit_pct = 6.5  # High Confidence
        elif confidence > 60:
            gross_profit_pct = 5.0  # Moderate Confidence
        else:
            gross_profit_pct = 3.5  # Low Confidence -> More Conservative Target

        self.log(f"Dynamic profit target set to: {gross_profit_pct:.2f}% (Confidence: {confidence:.1f}%)")
        return gross_profit_pct

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

        # Add the marker for the user's selected analysis date
        fig.add_vline(x=analysis_date, line_width=1, line_dash="dash", line_color="white", name="Analysis Date",
                      row=1)

        action = result['action']
        current_price = result['current_price']
        if "BUY" in action:
            buy_date = result['buy_date']  # Get the actual buy date
            sell_price = current_price * (1 + result['gross_profit_pct'] / 100)
            stop_loss = current_price * 0.97

            # Add the marker for the actual buying date
            fig.add_trace(go.Scatter(
                x=[buy_date], y=[current_price], mode='markers',
                marker=dict(color='cyan', size=12, symbol='circle-open', line=dict(width=2)), name='Target Buy'
            ), row=1, col=1)
            fig.add_hline(y=float(sell_price), line_dash="dash", line_color="lightgreen",
                          name="Target Sell Price", row=1, annotation_text=f"Target: ${sell_price:.2f}",
                          annotation_position="top right")

            # --- THIS IS THE CORRECTED LINE ---
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
    # --- Part 1: Setup and Sidebar (No changes here) ---
    st.title("🏢 StockWise AI Trading Advisor")
    if 'advisor' not in st.session_state: st.session_state.advisor = ProfessionalStockAdvisor()
    advisor = st.session_state.advisor
    if 'analysis_date' not in st.session_state: st.session_state.analysis_date = datetime.now().date()
    st.markdown(f"### Powered by Gen-2 Model: `{advisor.model_filename or 'N/A'}`")
    st.markdown("---")

    def set_date_to_today():
        st.session_state.analysis_date = datetime.now().date()

    st.sidebar.header("🎯 Trading Analysis")
    stock_symbol = st.sidebar.text_input("📊 Stock Symbol", value="NVDA").upper().strip()
    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        st.date_input("📅 Analysis Date", key='analysis_date')
    with col2:
        st.write(" ")
        st.button("Today", on_click=set_date_to_today, use_container_width=True)
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
    with st.spinner(f"Running analysis for {stock_symbol}..."):
        stock_data, result = advisor.run_analysis(stock_symbol, st.session_state.analysis_date)
    if not result:
        st.error("Analysis failed. Please check the debug logs for more information.")
        return

    # --- Part 3: Display Successful Results ---
    action = result['action']
    confidence = result['confidence']
    current_price = result['current_price']

    # Display top-level recommendation metrics
    col1, col2, col3 = st.columns([3, 2, 2])
    with col1:
        if "BUY" in action:
            st.success(f"🟢 **RECOMMENDATION: {action}**")
        else:
            st.warning(f"🟡 **RECOMMENDATION: {action}**")
    with col2:
        st.success(f"✅ Analysis complete for {stock_symbol}")

    with col3:
        if confidence < 55: color = "red"
        elif 55 <= confidence < 60: color = "yellow"
        elif 60 <= confidence < 70: color = "orange"
        elif 70 <= confidence < 80: color = "green"
        else: color = "blue"
        st.markdown(f"""<div style="text-align: right;"><span style="font-size: 1em;">Model Confidence</span><br><span 
        style="font-size: 2.5em; color: {color}; font-weight: bold;">{confidence:.1f}%</span></div>""",
                    unsafe_allow_html=True)

    # Display EITHER the profit panel OR the wait message
    if "BUY" in action:
        st.subheader("💰 Price Information & Profit Analysis")

        # --- All calculations for the metrics ---
        # (This part is unchanged)
        num_shares = investment_amount / current_price
        gross_profit_dollars = investment_amount * (result['gross_profit_pct'] / 100)
        net_profit_dollars, total_deducted_dollars = advisor.apply_israeli_fees_and_tax(
            gross_profit_dollars,
            num_shares
        )
        net_profit_pct = (net_profit_dollars / investment_amount) * 100 if investment_amount > 0 else 0
        total_deducted_pct = (total_deducted_dollars / investment_amount) * 100 if investment_amount > 0 else 0

        # --- THIS IS THE KEY CHANGE ---
        # 1. Change the layout to 5 columns
        # price_col, target_col, stop_col, gross_profit_col, net_profit_col = st.columns(5)
        price_col, target_col, stop_col, profit_col4 = st.columns(4)

        sell_price = current_price * (1 + result['gross_profit_pct'] / 100)
        stop_loss = current_price * 0.97

        # 2. Display the 5 metrics
        price_col.metric("Current Price", f"${current_price:.2f}")
        target_col.metric("🟢 Target Sell", f"${sell_price:.2f}")
        stop_col.metric("🔴 Stop-Loss", f"${stop_loss:.2f}")
        profit_col4.metric("💰 Gross Profit", f"${gross_profit_dollars:.2f} ({result['gross_profit_pct']:.2f}%)")

        # 3. Add the new Net Profit metric
        profit_col4.metric("✨ Net Profit", f"${net_profit_dollars:.2f} ({net_profit_pct:.2f}%)",
                              help="Profit after all fees and taxes.")

        # Keep the detailed breakdown below, but simplified
        st.caption(
            f"_(Deducted ${total_deducted_dollars:.2f} ({total_deducted_pct:.2f}%) for Israeli tax & fixed broker fees)_")
    else:
        st.subheader("⏳ Market Position & Future Opportunity")
        col1, col2 = st.columns(2)
        col1.metric("Current Price", f"${current_price:.2f}")
        if result.get('target_buy_price'):
            col2.metric("🎯 Target Buy Price", f"${result['target_buy_price']:.2f}",
                        help="A price at which the stock may become a better buying opportunity based on technical support levels.")

    # Display the Chart
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

    # Display Debug Logs if toggled
    # ... (Your debug display logic can go here) ...

        # --- NEW: Redesigned Action Summary Section ---
        st.subheader("📝 Action Summary")
        if "BUY" in action and result.get('buy_date'):
            buy_date_str = result['buy_date'].strftime('%B %d, %Y')
            sell_price = current_price * (1 + result['gross_profit_pct'] / 100)
            stop_loss = current_price * 0.97

            st.markdown("#### For Those Looking to Buy:")
            st.success(f"""
            - **Action:** The model recommends **BUYING** {stock_symbol}.
            - **When:** A high-probability entry point was detected on or around **{buy_date_str}**.
            - **Price:** The suggested entry price is **${current_price:.2f}**.
            """)

            st.markdown("#### For Existing Stock Holders:")
            st.info(f"""
            - **Action:** The model recommends **HOLDING** your position.
            - **Target Sell Price:** Consider taking profits around **${sell_price:.2f}**.
            - **Stop-Loss:** To manage risk, consider placing a stop-loss order near **${stop_loss:.2f}**.
            """)
        else:  # This is the WAIT / AVOID case
            target_buy_price_str = f"${result.get('target_buy_price', current_price * 0.95):.2f}"  # Added a fallback

            st.markdown("#### For Those Looking to Buy:")
            st.warning(f"""
            - **Action:** The model recommends to **WAIT or AVOID** buying {stock_symbol} at its current price.
            - **Reason:** No high-probability "BUY" signal was detected in the near term.
            - **Target Buy Price:** The stock may become a more attractive opportunity if it pulls back to the key support level around **{target_buy_price_str}**.
            """)

            st.markdown("#### For Existing Stock Holders:")
            st.error(f"""
            - **Action:** The model recommends **CAUTION**.
            - **Reason:** The current trend does not show strong bullish momentum.
            - **Recommendation:** Consider protecting your profits by setting a **trailing stop-loss** to lock in gains if the price begins to fall.
            """)

# --- Main Execution ---
if __name__ == "__main__":
    if 'advisor' not in st.session_state:
        st.session_state.advisor = ProfessionalStockAdvisor()
    if st.session_state.advisor.model:
        create_enhanced_interface()
    else:
        st.error("Model could not be loaded. The application cannot start.")