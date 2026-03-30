# stockwise_simulation_v2.py
"""
StockWise Gen-12 Telemetry & Time-Machine Regression
====================================================
Tab 1: Log Telemetry (Parses live execution logs, aggregates errors)
Tab 2: Time-Machine (Before vs. After Code Regression Tester)
"""

import streamlit as st
import pandas as pd
import os
import glob
import re
import logging
from logging.handlers import RotatingFileHandler
import sys
from io import StringIO
from datetime import datetime, timedelta

# Core Engine Imports for Regression Testing
from data_source_manager import DataSourceManager
from feature_engine import FeatureEngine
from strategy_engine import StrategyEngine
import system_config as cfg

# --- CONFIGURATION & DEDICATED SIMULATION LOGGING ---
st.set_page_config(page_title="StockWise Command", page_icon="🦅", layout="wide")
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

# 1. Initialize a DEDICATED logger just for the simulation script
sim_log_path = os.path.join(LOGS_DIR, "StockWise_Simulation_Memory.log")
sim_logger = logging.getLogger("SimulationEngine")
sim_logger.setLevel(logging.DEBUG)

# Prevent duplicate handlers if Streamlit re-runs
if not sim_logger.handlers:
    file_handler = RotatingFileHandler(sim_log_path, maxBytes=5*1024*1024, backupCount=2, encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | [%(name)s] | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    sim_logger.addHandler(file_handler)

sim_logger.info("=== Streamlit Simulation Interface Initialized ===")

# --- INIT SESSION STATE MEMORY ---
if 'regression_results' not in st.session_state: st.session_state.regression_results = None
											  
if 'debug_logs' not in st.session_state: st.session_state.debug_logs = ""
if 'raw_export_text' not in st.session_state: st.session_state.raw_export_text = ""
if 'export_filename' not in st.session_state: st.session_state.export_filename = "export.txt"

# ==========================================
# CUSTOM STREAMLIT LOG & PRINT HANDLER
# ==========================================
class StreamlitLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.logs = []
        self.formatter = logging.Formatter('%(asctime)s | %(levelname)s | [%(name)s] | %(message)s', datefmt='%H:%M:%S')

    def emit(self, record):
        formatted_msg = self.format(record)
        self.logs.append(formatted_msg)
        if len(self.logs) > 2000: self.logs.pop(0)

class StreamlitPrintCapturer:
    def __init__(self, st_handler, file_handler): 
        self.st_handler = st_handler
        self.file_handler = file_handler
    def write(self, text):
        if text.strip():
            record = logging.LogRecord(name="PRINT", level=logging.DEBUG, pathname="", lineno=0, msg=text.strip(), args=(), exc_info=None)
            self.st_handler.emit(record)
            self.file_handler.emit(record)
    def flush(self): pass

# ==========================================
# LOG PARSING ENGINE (For Tab 1)
# ==========================================
def get_available_logs():
    if not os.path.exists(LOGS_DIR): return []
    # Catches both Master Logs AND Simulation Logs
    logs = glob.glob(os.path.join(LOGS_DIR, "StockWise_*.txt"))
    logs.sort(reverse=True)
    return logs

@st.cache_data(ttl=10)
def process_log_lines(lines):
    vip_list, setups, vetoes, errors = [], [], [], []
    in_vip_table = False
    first_date, last_date = None, None
    date_pattern = r'^(\d{4}-\d{2}-\d{2})'
    
    for line in lines:
        line_clean = line.strip()
        date_match = re.match(date_pattern, line_clean)
        if date_match:
            if not first_date: first_date = date_match.group(1)
            last_date = date_match.group(1)
        
        if " | ERROR | " in line or " | CRITICAL | " in line:
            parts = line_clean.split(" | ")
            if len(parts) >= 4: errors.append({"Type": parts[1].strip(), "Function": parts[2].strip(), "Message": " | ".join(parts[3:]).strip()})
            else: errors.append({"Type": "ERROR", "Function": "Unknown", "Message": line_clean})
            
        if "TOP VIP TARGETS" in line:
            in_vip_table = True; continue
            
        if in_vip_table:
            if line_clean.startswith("#"):
                parts = [p.strip() for p in line_clean.split("|")]
                if len(parts) >= 6:
                    vip_list.append({"Rank": parts[0], "Symbol": parts[1].strip(), "Regime": parts[2].strip(), "Tech Score": float(parts[3]) if parts[3].replace('.','',1).isdigit() else parts[3], "AI Score": float(parts[4]) if parts[4].replace('.','',1).isdigit() else parts[4], "Master Score": float(parts[5]) if parts[5].replace('.','',1).isdigit() else parts[5]})
            elif line_clean == "" or "====" in line_clean:
                if len(vip_list) > 0: in_vip_table = False
                    
        if "SETUPS:" in line:
            match = re.search(r'\[([A-Z0-9]+)\] SETUPS: \[(.*?)\] .*? Master: ([\d\.]+)', line_clean)
            if match: setups.append({"Symbol": match.group(1), "Setup": match.group(2), "Master Score": float(match.group(3))})
                
        if "Trade killed by" in line or "VETO:" in line:
            match = re.search(r'\[([A-Z0-9]+)\] Trade killed by (.*)', line_clean)
            if match: vetoes.append({"Symbol": match.group(1), "Reason": match.group(2).strip('.')})
            else:
                match_veto = re.search(r'\[([A-Z0-9]+)\] Alpha VETO: (.*)', line_clean)
                if match_veto: vetoes.append({"Symbol": match_veto.group(1), "Reason": "Alpha VETO: " + match_veto.group(2).strip()})

    if not first_date: first_date = datetime.now().strftime('%Y-%m-%d')
    if not last_date: last_date = datetime.now().strftime('%Y-%m-%d')
    
    # Safely convert to DataFrames with mandated structure to prevent KeyErrors
    df_vip = pd.DataFrame(vip_list) if vip_list else pd.DataFrame(columns=["Rank", "Symbol", "Regime", "Tech Score", "AI Score", "Master Score"])
    df_setups = pd.DataFrame(setups) if setups else pd.DataFrame(columns=["Symbol", "Setup", "Master Score"])
    df_vetoes = pd.DataFrame(vetoes) if vetoes else pd.DataFrame(columns=["Symbol", "Reason"])
    df_err = pd.DataFrame(errors) if errors else pd.DataFrame(columns=["Type", "Function", "Message"])
    
    if not df_err.empty: df_err = df_err.groupby(['Type', 'Function', 'Message']).size().reset_index(name='Count').sort_values(by='Count', ascending=False)
    return df_vip, df_setups, df_vetoes, df_err, first_date, last_date

def parse_local_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f: return process_log_lines(f.readlines())
def parse_uploaded_file(uploaded_file):
    return process_log_lines(StringIO(uploaded_file.getvalue().decode("utf-8")).readlines())

# ==========================================
# UI RENDER LOGIC
# ==========================================
st.title("🦅 StockWise Gen-12 Command Center")

# --- 1. SIDEBAR: LOG LOADING (Tab 1 Controls) ---
st.sidebar.header("📂 1. Log Telemetry")
uploaded_log = st.sidebar.file_uploader("Upload Master Log (.txt)", type=["txt"])
available_logs = get_available_logs()
selected_local_log = st.sidebar.selectbox("Or select local log:", available_logs, format_func=lambda x: os.path.basename(x)) if available_logs else None

				 
if uploaded_log:
    st.session_state.raw_export_text = uploaded_log.getvalue().decode("utf-8")
    st.session_state.export_filename = f"Exported_{uploaded_log.name}"
    df_vip, df_setups, df_vetoes, df_errors, f_date, l_date = parse_uploaded_file(uploaded_log)
elif selected_local_log:
    with open(selected_local_log, 'r', encoding='utf-8') as f: st.session_state.raw_export_text = f.read()
    st.session_state.export_filename = f"Exported_{os.path.basename(selected_local_log)}"
    df_vip, df_setups, df_vetoes, df_errors, f_date, l_date = parse_local_file(selected_local_log)
else:
    st.warning("No log files found. Please upload a `.txt` log file.")
    st.stop()

if st.session_state.raw_export_text:
    st.sidebar.download_button(label="📥 Download Log to PC", data=st.session_state.raw_export_text, file_name=st.session_state.export_filename, mime="text/plain")

st.sidebar.markdown("---")

# --- 2. SIDEBAR: SIMULATION CONTROLS ---
st.sidebar.header("⏪ 2. Regression Simulator")
																   

# --- CONNECTION FLOW COMMANDS ---
st.sidebar.subheader("🔌 Connection Flow")
sim_provider = st.sidebar.selectbox("Active Data Provider", ["ALPACA", "MASSIVE", "YFINANCE", "IBKR"])

st.sidebar.subheader("📅 Simulation Target")
sim_start = st.sidebar.date_input("Start Date", value=datetime.strptime(f_date, "%Y-%m-%d").date())
sim_end = st.sidebar.date_input("End Date", value=datetime.strptime(l_date, "%Y-%m-%d").date())

# Dynamic Ticker Dropdown
all_symbols = []
if not df_vip.empty: all_symbols.extend(df_vip['Symbol'].tolist())
if not df_setups.empty: all_symbols.extend(df_setups['Symbol'].tolist())
if not df_vetoes.empty: all_symbols.extend(df_vetoes['Symbol'].tolist())
unique_symbols = sorted(list(set(all_symbols)))
log_sym_options = ["-- Custom Input --"] + unique_symbols

selected_log_sym = st.sidebar.selectbox("Target Ticker", log_sym_options, index=1 if unique_symbols else 0)
sim_ticker = st.sidebar.text_input("Custom Ticker", value="SPY").upper() if selected_log_sym == "-- Custom Input --" else selected_log_sym

debug_mode = st.sidebar.checkbox("🐛 Enable Deep Logger", value=False)

# --- THE EXECUTION TRIGGER ---
if st.sidebar.button("🚀 Run Time-Machine Simulation", type="primary"):
    
    # EXPLICIT CONNECTION OVERRIDES
    cfg.DATA_PROVIDER = sim_provider
    if sim_provider == "ALPACA": cfg.EN_ALPACA = True
    if sim_provider == "MASSIVE": cfg.EN_MASSIVE = True
    if sim_provider == "YFINANCE": cfg.EN_YFINANCE = True
    if sim_provider == "IBKR": cfg.EN_IBKR = True

    # Extract "Before" State from logs
    b_score, b_setup, b_veto = 0.0, "None", "None"
    v_match = df_vip[df_vip['Symbol'] == sim_ticker]
    s_match = df_setups[df_setups['Symbol'] == sim_ticker]
    vet_match = df_vetoes[df_vetoes['Symbol'] == sim_ticker]
    
    if not v_match.empty: b_score = float(v_match['Master Score'].iloc[0])
    elif not s_match.empty: b_score = float(s_match['Master Score'].iloc[0])
    if not s_match.empty: b_setup = ", ".join(s_match['Setup'].astype(str).tolist())
    if not vet_match.empty: b_veto = ", ".join(vet_match['Reason'].astype(str).tolist())

    # ========================================================
    # SIMULATION LOGGER HIJACK (Prevents polluting Live Logs)
    # ========================================================
    original_stdout, original_stderr = sys.stdout, sys.stderr
    st_handler = StreamlitLogHandler()
    
    # Create the EXACT replica of the Master log file, but name it 'Sim'
    sim_filename = f"StockWise_Sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    sim_filepath = os.path.join(LOGS_DIR, sim_filename)
    sim_file_handler = logging.FileHandler(sim_filepath, encoding='utf-8')
    sim_file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | [%(name)s] | %(message)s'))
    
    root_logger = logging.getLogger()
    all_loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict] + [root_logger]
    
    suspended_handlers = {} # Stores live Master log connections to restore later
    active_handlers = []
    
    for l in all_loggers:
        suspended_handlers[l] = []
        # Strip out the live master logs
        for h in l.handlers[:]:
            if isinstance(h, logging.FileHandler) or isinstance(h, StreamlitLogHandler):
                l.removeHandler(h)
                if isinstance(h, logging.FileHandler): suspended_handlers[l].append(h)
        
        # Attach our clean simulation logs
        if debug_mode: l.addHandler(st_handler)
        l.addHandler(sim_file_handler)
        if l.level > logging.DEBUG: l.setLevel(logging.DEBUG)
        active_handlers.append((l, st_handler, sim_file_handler))
        
    sys.stdout = StreamlitPrintCapturer(st_handler, sim_file_handler)
    sys.stderr = StreamlitPrintCapturer(st_handler, sim_file_handler)

    # --- EXECUTE THE SIMULATION ---
    try:
        with st.spinner(f"Simulating StrategyEngine for {sim_ticker} on {sim_provider}..."):
            logging.info(f"=== INITIATING TIME-MACHINE SIMULATION FOR {sim_ticker} ===")
            
            # Force the simulation provider
            cfg.DATA_PROVIDER = sim_provider
            dsm = DataSourceManager()
            days_ago = (datetime.now().date() - sim_end).days
            days_to_pull = max(400, days_ago + 300)
            
            # Explicitly pass BOTH start and end bounds based on the UI requested end_date 
            # so the Data Source Manager doesn't get confused by shifting chronological windows
            calc_start_dt = sim_end - timedelta(days=days_to_pull)
            calc_start_str = calc_start_dt.strftime('%Y-%m-%d')
            calc_end_str = sim_end.strftime('%Y-%m-%d')
            
            logging.info(f"[SIMULATION DEBUG] Fetching data for {sim_ticker} from {calc_start_str} to {calc_end_str} (Target: {sim_provider})")
            
            df_raw = dsm.get_stock_data(
                sim_ticker, 
                start_date=calc_start_str,
                end_date=calc_end_str,
                days_back=days_to_pull, 
                interval='1d', 
                source=sim_provider
            )
            
            if df_raw is None or df_raw.empty:
                st.session_state.regression_results = {"error": f"Data Starvation: {sim_provider} API returned no data."}
                logging.error(f"[FAIL] Data Starvation: {sim_provider} returned 0 rows.")
            else:
                if df_raw.index.tz is not None: df_raw.index = df_raw.index.tz_localize(None)
                df_sim = df_raw[df_raw.index <= pd.to_datetime(sim_end)]
                
                fe = FeatureEngine()
                df_features = fe.calculate_features(df_sim)
                
                se = StrategyEngine()
                se_result = se.evaluate_ticker(sim_ticker, df_features)
                
                # Extract "After" State from the exact strings we just logged
                a_score, a_setup, a_veto = 0.0, "None", "None"
                
                if isinstance(se_result, dict):
                    a_score = se_result.get("master_score", 0.0)
                    if "reason" in se_result and se_result["action"] != "BUY":
                        a_veto = se_result["reason"]
                
                # We read the raw file we just generated to get exact triggers
                with open(sim_filepath, 'r', encoding='utf-8') as f:
                    sim_logs = f.readlines()
                    
                for log_line in sim_logs:
                    m_set = re.search(r'\[([A-Z0-9]+)\] SETUPS: \[(.*?)\]', log_line)
                    if m_set and m_set.group(1) == sim_ticker: a_setup = m_set.group(2)

				# Save everything to Session Memory so it survives UI refreshes
                st.session_state.regression_results = {
                    "ticker": sim_ticker, "b_score": b_score, "b_setup": b_setup, "b_veto": b_veto,
                    "a_score": a_score, "a_setup": a_setup, "a_veto": a_veto,
                    "last_row": df_features.iloc[-1:]
                }
                st.session_state.debug_logs = "\n".join(st_handler.logs) if debug_mode else ""
                
    except Exception as e:
        import traceback
        logging.critical(f"Simulation crashed: {e}")
        st.session_state.regression_results = {"error": str(e), "trace": traceback.format_exc()}
    
    finally:
        # ========================================================
        # CLEANUP: Restore Live Trading Configuration
        # ========================================================
        sys.stdout, sys.stderr = original_stdout, original_stderr
        for l, sh, fh in active_handlers:
            l.removeHandler(sh)
            l.removeHandler(fh)
            # Reattach the Live Master Handlers so the Live Engine doesn't break
            for original_h in suspended_handlers.get(l, []):
                l.addHandler(original_h)

# ==========================================
# MAIN TABS DISPLAY
# ==========================================
tab_telemetry, tab_simulation = st.tabs(["📊 Live Telemetry Logs", "⏪ Code Regression Results"])

with tab_telemetry:
    st.subheader(f"System Health Metrics ({f_date} to {l_date})")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("VIP Targets", len(df_vip))
    c2.metric("Setups Detected", len(df_setups))
    c3.metric("Trades Vetoed", len(df_vetoes))
    err_count = int(df_errors['Count'].sum()) if not df_errors.empty else 0
    c4.metric("System Errors", err_count, delta="-CRITICAL" if err_count > 0 else None, delta_color="inverse")

    if err_count > 0:
        with st.expander("⚠️ View Aggregated Error Summary", expanded=True): st.dataframe(df_errors, use_container_width=True)
    
    with st.expander("🎯 VIP Targets (Nightly Scan)", expanded=True):
        if not df_vip.empty: st.dataframe(df_vip, use_container_width=True)
        else: st.info("No VIP targets found.")

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.subheader("📡 Detected Setups")
        if not df_setups.empty: st.dataframe(df_setups, use_container_width=True)
    with col_t2:
        st.subheader("🛡️ Trade Veto Analysis")
        if not df_vetoes.empty:
            def hl_red(val): return 'background-color: #ffcccc; color: #990000'
            st.dataframe(df_vetoes['Reason'].value_counts().reset_index().style.map(hl_red, subset=['Reason']), use_container_width=True)

with tab_simulation:
    st.markdown("### ⚖️ Before vs. After Code Execution")
    
    if st.session_state.regression_results is None:
        st.info("👈 Use the Regression Simulator in the sidebar to run a Time-Machine test.")
    elif "error" in st.session_state.regression_results:
        st.error(f"💥 Simulation Failed: {st.session_state.regression_results['error']}")
        if "trace" in st.session_state.regression_results:
            st.code(st.session_state.regression_results['trace'], language="python")
    else:
        res = st.session_state.regression_results
        
        c1, c2 = st.columns(2)
        with c1:
            st.info(f"📜 **BEFORE** (Original Log State for {res['ticker']})")
            st.metric("Original Master Score", round(res['b_score'], 2))
            st.write(f"**Detected Setups:** `{res['b_setup']}`")
            st.write(f"**Trade Veto Status:** `{res['b_veto']}`")
            
        with c2:
            st.success(f"🔬 **AFTER** (Current Updated Code for {res['ticker']})")
            delta = round(res['a_score'] - res['b_score'], 2)
            st.metric("New Master Score", round(res['a_score'], 2), delta=delta)
            
							   
            if res['a_setup'] != "None" and res['b_setup'] == "None": st.markdown(f"**Detected Setups:** 🟢 `{res['a_setup']}` *(NEW)*")
            elif res['a_setup'] == res['b_setup']: st.markdown(f"**Detected Setups:** ⚪ `{res['a_setup']}` *(Unchanged)*")
            else: st.markdown(f"**Detected Setups:** 🟡 `{res['a_setup']}`")
            
							  
            if res['a_veto'] == "None" and res['b_veto'] != "None": st.markdown(f"**Trade Veto Status:** 🟢 `PASSED` *(Bug Fixed!)*")
            elif res['a_veto'] != "None" and res['b_veto'] == "None": st.markdown(f"**Trade Veto Status:** 🔴 `{res['a_veto']}` *(New Regression!)*")
            elif res['a_veto'] == res['b_veto']: st.markdown(f"**Trade Veto Status:** ⚪ `{res['a_veto']}` *(Unchanged)*")
            else: st.markdown(f"**Trade Veto Status:** 🟡 `{res['a_veto']}` *(Changed)*")

        st.markdown("---\n### 📊 Core Diagnostics on Simulation Date")
        lr = res['last_row']
        d1, d2, d3 = st.columns(3)
        d1.metric("Close Price", round(lr['close'].values[0], 2) if 'close' in lr else "N/A")
        d2.metric("RSI", round(lr.get('rsi', [0])[0], 2) if 'rsi' in lr else "N/A")
        d3.metric("ATR Volatility", round(lr.get('atr', [0])[0], 2) if 'atr' in lr else "N/A")

        with st.expander("View Final Feature Matrix (Last Row)", expanded=False): st.dataframe(lr, use_container_width=True)
																   
            
        if st.session_state.debug_logs:
            with st.expander("🐛 View Live Execution Terminal Dump", expanded=True):
                st.code(st.session_state.debug_logs, language="text")