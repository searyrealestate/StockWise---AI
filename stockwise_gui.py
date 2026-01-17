import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import time
import pandas_ta as ta # Ensure pandas_ta is installed
import json
import subprocess
from datetime import datetime, timedelta                

# --- SYSTEM IMPORTS ---
sys.path.append(os.getcwd())
try:
    from data_source_manager import DataSourceManager
    from feature_engine import RobustFeatureCalculator
    from strategy_engine import StrategyOrchestra, MarketRegimeDetector
    from stockwise_ai_core import StockWiseAI
    from logging_manager import LoggerSetup
    import system_config as cfg
except ImportError as e:
    st.error(f"Critical Error: Could not import system modules. {e}")
    st.stop()                        

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="StockWise Gen-9 Pro", 
    page_icon="🏢", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- INITIALIZATION ---
if 'dsm' not in st.session_state:
    with st.spinner("Initializing Core Systems..."):                                                
        st.session_state.dsm = DataSourceManager()
        st.session_state.fe = RobustFeatureCalculator()
        st.session_state.ai = StockWiseAI()

# --- SIDEBAR CONFIGURATION ---
st.sidebar.title("🏢 StockWise Gen-9")

# --- 1. LIVE ENGINE STATUS INDICATOR (NEW) ---
status_file = "logs/live_status.json"
if os.path.exists(status_file):
    with open(status_file, 'r') as f:
        status_data = json.load(f)
    st.sidebar.success(f"Engine: {status_data['status']}")
    st.sidebar.caption(f"Last Heartbeat: {status_data['last_heartbeat']}")
else:
    st.sidebar.error("Engine: STOPPED")
    st.sidebar.warning("Run 'python live_trading_engine.py' in terminal")

st.sidebar.markdown("---")
# st.sidebar.caption("Professional AI Trading Advisor")

# 2. GLOBAL CONFIG
st.sidebar.header("⚙️ Global Config")
investment_amount = st.sidebar.number_input("Initial Investment ($)", value=1000, step=100)
risk_per_trade_pct = st.sidebar.slider("Risk per Trade (%)", 0.5, 5.0, 2.0, step=0.1)

st.sidebar.markdown("---")

# --- 3. NAVIGATION (UPDATED) ---
mode = st.sidebar.radio("Navigation", ["📡 Live Trade Monitor", "🎯 Analysis Dashboard", "🛠️ System Simulation"])

st.sidebar.markdown("---")

# --- 4. STRATEGY SETTINGS ---
with st.sidebar.expander("🛠️ Strategy Config", expanded=True):
    st.caption("Active Filters")
    use_regime = st.checkbox("Market Regime Filter", value=True)
    use_volatility = st.checkbox("Volatility (Falling Knife)", value=True)
    use_ai = st.checkbox("Deep Learning Core", value=True)
    
    st.divider()
    st.caption("Exit Strategy")
    stop_atr_mult = st.slider("Stop Loss (ATR Mult)", 1.5, 4.0, 2.0)
    target_ratio = st.slider("Profit Target (Risk Ratio)", 1.0, 3.0, 1.5)

st.sidebar.markdown("---")

# --- 5. MODE SELECTION ---
mode = st.sidebar.radio("Operation Mode", ["🎯 Single Analysis", "🚀 Market Screener (Sim)", "🛠️ System Simulation"])

# --- CHARTING FUNCTION (Gen-3 Style) ---
def create_professional_chart(df, symbol, decision, stop=None, target=None):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])

    # 1. Candlesticks
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], 
        name='Price'
    ), row=1, col=1)

    # 2. Moving Averages
    if 'sma_50' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['sma_50'], line=dict(color='blue', width=1), name='SMA 50'), row=1, col=1)
    if 'sma_200' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['sma_200'], line=dict(color='purple', width=2), name='SMA 200'), row=1, col=1)

    # 3. Bollinger Bands (Calculated on the fly for display if missing)
    if 'bb_upper' not in df.columns:
        # Quick Calc using pandas_ta if not in dataframe
        bb = df.ta.bbands(length=20, std=2)
        if bb is not None:
            df = pd.concat([df, bb], axis=1)
            # Rename for safety if needed, or just use default names
            # Pandas TA default names: BBL_20_2.0, BBU_20_2.0
            
    # Try to plot bands if columns exist (flexible naming)
    bbu = [c for c in df.columns if 'BBU' in c or 'bb_upper' in c]
    bbl = [c for c in df.columns if 'BBL' in c or 'bb_lower' in c]
    
    if bbu and bbl:
        fig.add_trace(go.Scatter(x=df.index, y=df[bbu[0]], line=dict(color='gray', width=1, dash='dot'), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df[bbl[0]], line=dict(color='gray', width=1, dash='dot'), fill='tonexty', fillcolor='rgba(128,128,128,0.1)', name='Bollinger Bands'), row=1, col=1)

    # 4. Trade Markers
    last_date = df.index[-1]
    last_price = df['close'].iloc[-1]
    
    if decision == "BUY":
        fig.add_trace(go.Scatter(
            x=[last_date], y=[last_price], mode='markers+text', 
            marker=dict(color='#00FF00', size=15, symbol='triangle-up', line=dict(width=2, color='black')),
            text=["BUY"], textposition="bottom center", name='Signal'
        ), row=1, col=1)
        
        if stop: 
            fig.add_hline(y=stop, line_dash="dash", line_color="#FF4B4B", annotation_text=f"Stop: ${stop:.2f}")
        if target: 
            fig.add_hline(y=target, line_dash="dash", line_color="#00FF00", annotation_text=f"Target: ${target:.2f}")

    # 5. Volume
    fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color='rgba(200, 200, 200, 0.5)'), row=2, col=1)

    # Layout
    fig.update_layout(
        title=f"{symbol} Professional Analysis",
        xaxis_rangeslider_visible=False,
        height=700,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_chart_with_icons(df, symbol):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25])

    # 1. Price
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)
    
    # 2. Indicators
    if 'sma_200' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['sma_200'], line=dict(color='purple', width=2), name='SMA 200'), row=1, col=1)

    # 3. CURRENT SIGNAL ICONS (Historical Trades)
    trades_file = "logs/portfolio_trades.csv"
    if os.path.exists(trades_file):
        try:
            trades = pd.read_csv(trades_file)
            symbol_trades = trades[trades['Symbol'] == symbol]
            
            for _, trade in symbol_trades.iterrows():
                trade_date = pd.to_datetime(trade['Date'])
                # Icon Color Config
                if trade['Type'] == "BUY":
                    color, symbol_icon = "blue", "triangle-up"
                else:
                    color, symbol_icon = "white", "triangle-down"
                    
                fig.add_trace(go.Scatter(
                    x=[trade_date], y=[trade['Price']], mode='markers',
                    marker=dict(color=color, size=15, symbol=symbol_icon, line=dict(width=2, color='black')),
                    name=f"{trade['Type']} Executed"
                ), row=1, col=1)
        except Exception:
            pass # Skip if trade file is malformed

    # 4. Volume
    fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color='rgba(100, 100, 100, 0.5)'), row=2, col=1)
    
    # Layout styling
    fig.update_layout(title=f"{symbol} Price Action", xaxis_rangeslider_visible=False, height=600, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 1: LIVE TRADE MONITOR (NEW)
# ==========================================
if mode == "📡 Live Trade Monitor":
    st.title("📡 Live Trading Command Center")
    
    # 1. ENGINE STATUS
    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            status_data = json.load(f)
        c1, c2, c3 = st.columns(3)
        c1.metric("Engine Status", status_data['status'])
        c2.metric("Last Scan", status_data.get('last_scan_time', 'N/A'))
        c3.metric("Watchlist Size", len(cfg.WATCHLIST))
        st.info(f"Current Activity: {status_data['message']}")
    else:
        st.error("Live Engine is NOT running. Please start `live_trading_engine.py`.")

    st.markdown("---")

    # 2. LIVE POSITIONS / TRADES
    st.subheader("📜 Live Trade Log")
    trades_file = "logs/portfolio_trades.csv"
    if os.path.exists(trades_file):
        try:
            df_trades = pd.read_csv(trades_file)
            st.dataframe(df_trades.sort_values(by="Date", ascending=False), use_container_width=True)
        except Exception:
            st.warning("Trade log exists but is empty or unreadable.")
    else:
        st.info("No trades executed yet.")

    # 3. LIVE LOGS
    st.subheader("🖨️ Live Logs")
    with st.expander("View Engine Terminal Output", expanded=True):
        if os.path.exists("logs/live_engine.log"):
            with open("logs/live_engine.log", "r") as f:
                lines = f.readlines()
                for line in lines[-10:]:
                    st.text(line.strip())

# --- MAIN LOGIC ---
# ==========================================
# TAB 2: ANALYSIS DASHBOARD (UPDATED)
# ==========================================
elif mode == "🎯 Analysis Dashboard":
    st.title("🎯 Deep Dive Analysis")
    
    # 4.1 ENTER STOCK TICKER (Dropdown from Config)
    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        # Uses the WATCHLIST we added to system_config.py
        symbol = st.selectbox("Select Stock Ticker", cfg.WATCHLIST)
    with col_btn:
        st.write("")
        st.write("")
        run_btn = st.button("🧠 Analyze", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner(f"Analyzing {symbol}..."):
            # Fetch & Calc
            df = st.session_state.dsm.get_stock_data(symbol, days_back=365)
            if df is None or df.empty:
                st.error("No Data Found.")
                st.stop()
                
            df = st.session_state.fe.calculate_features(df)
            latest = df.iloc[-1]
            
            # Predict
            _, prob, _ = st.session_state.ai.predict_trade_confidence(symbol, latest.to_dict(), fundamentals={'Score':80}, df_window=df)
            
            # Decide
            packet = {'AI_Probability': prob, 'Fundamental_Score': 80}
            decision = StrategyOrchestra.decide_action(symbol, latest.to_dict(), packet)
            
            # Get Context
            regime = MarketRegimeDetector.detect_regime(latest.to_dict())
            stop, target, _ = StrategyOrchestra.get_adaptive_targets(latest.to_dict(), latest['close'])

        # --- DASHBOARD LAYOUT ---
        
        # ROW 1: METRICS
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current Price", f"${latest['close']:.2f}")
        
        # 4.3 CONFIDENCE INDICATION
        m2.metric("AI Confidence", f"{prob:.1%}", delta="High" if prob > 0.85 else "Neutral")
        # Visual Bar
        bar_len = int(prob * 10)
        st.caption("Strength: " + "🟦" * bar_len + "⬜" * (10 - bar_len)) 
        
        # 4.2 WHAT IS REGIME?
        m3.metric("Market Regime", regime.replace("_", " "))
        
        if decision == "BUY":
            m4.metric("Action", "BUY 🚀", delta="Signal Active")
        else:
            m4.metric("Action", "WAIT ✋", delta="No Signal", delta_color="off")

        st.divider()

        # ROW 2: CHART (Updated to use Icons)
        st.subheader("Price Action")
        plot_chart_with_icons(df, symbol)
        
        # ROW 3: PnL TABLE (4.4)
        st.subheader(f"💰 {symbol} Position History")
        trades_file = "logs/portfolio_trades.csv"
        if os.path.exists(trades_file):
            try:
                all_trades = pd.read_csv(trades_file)
                symbol_history = all_trades[all_trades['Symbol'] == symbol]
                if not symbol_history.empty:
                    st.dataframe(symbol_history, use_container_width=True)
                else:
                    st.info(f"No historical trades recorded for {symbol}.")
            except:
                st.info("Trade log unavailable.")
        else:
            st.info("No trade history file found.")

        # ROW 4: LOGIC
        st.subheader("🤖 Logic Explanation")
        with st.expander("Why did the system make this decision?", expanded=True):
            logs = LoggerSetup.read_logs()
            # Simple filter to show recent relevant logs
            for line in logs[-15:]:
                st.text(line.strip())

elif mode == "🛠️ System Simulation":
    st.header("🧪 Strategy Simulator")
    if st.button("Run Portfolio Simulation"):
        import subprocess
        with st.spinner("Running Simulation..."):
            result = subprocess.run([sys.executable, "stockwise_simulation.py"], capture_output=True, text=True)
            st.code(result.stdout)