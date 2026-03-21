# data_source_manager.py

"""
Data Source Manager - Gen-12 Professional
=========================================
The Gateway to the Market.
Implements the "Waterfall" Data Retrieval Strategy:
1. IBKR (Interactive Brokers): Premium, Real-time (if connected).
2. Alpaca: Reliable, Free Historical Data.
3. Massive (formerly Polygon): High-Granularity Backup.
4. YFinance: The "Last Resort" (delayed/unofficial).

Handles:
- connection management (TWS API)
- data normalization (cleaning columns, timestamps)
- fallback logic errors
- real-time streaming simulation

CHANGELOG:
-----------
[2026-03-14] Fix #4: Critical waterfall fix — IBKR AttributeError + MASSIVE silent skip
  - Bug: _download_from_ibkr() referenced self.ibkr which does not exist (should be self.app).
    This caused AttributeError on every IBKR attempt, silently caught → IBKR always skipped.
  - Bug: When massive_client is None, no elif handler existed → silent fallthrough with no log.
  - Fix: self.ibkr → self.app in _download_from_ibkr (all references).
  - Fix: Added elif provider == 'MASSIVE' handler with WARNING log + continue.
  - Added: Provider status summary log at __init__ completion.
  - Changed: Provider attempt log upgraded from DEBUG to INFO for visibility.

[2026-03-14] Fix #1: Escalating Circuit Breaker for Massive (429 lockout)
  - Old: 15-min fixed lockout, re-tried and failed in infinite loop
  - Root cause: _download_from_massive() caught 429 internally → outer handler
    (circuit breaker) never fired. Dead code.
  - New: 429 re-raised from _download_from_massive so outer handler trips breaker.
  - Escalating lockout: 1st hit = 1h, subsequent consecutive hits = 4h.
  - _massive_fail_count class var tracks consecutive failures; resets on success.

[2026-03-14] Fix #2: Waterfall provider diagnostics + silent skip fix
  - Old: When Massive locked out, Alpaca/IBKR/YFinance silently skipped (no log).
  - New: Each provider logs a WARNING when skipped due to missing client/connection.
  - IBKR now logs "connection failed" and continues to next provider explicitly.
  - Data Starvation error already logs full provider failure context.

[2026-03-14] Fix #3: Provider delay optimization
  - MASSIVE: 12.5s → 1.0s (circuit breaker handles 429s; no need for per-request delay)
  - ALPACA: 2.5s → 0.5s (free tier allows ~200/min; 0.5s = 2/sec, safely within limit)
  - YFINANCE: 1.0s → 1.5s (slightly more conservative against scraping protection)
"""

import urllib3
import threading
import time
import logging
import random
import asyncio
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import pytz
import os
import streamlit as st
import numpy as np
# import datetime (Removed to avoid conflict with 'from datetime import datetime')
import system_config as cfg
import logging

logger = logging.getLogger(__name__)

# --- SILENCE NOISY LIBRARIES ---
logging.getLogger("ibapi").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

# --- Increase Global Connection Pool ---
# This prevents "Connection pool is full" when streaming multiple stocks
# Default is 10/10. We increase to 50 to handle the watchlist threads.
http = urllib3.PoolManager(
    num_pools=50, 
    maxsize=50, 
    block=True
)
# Monkey-patch default pool for libraries that don't allow configuration
urllib3.connectionpool.HTTPConnectionPool.QueueCls.maxsize = 50

# --- IBKR IMPORT ---
# Check imports
try:
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper
    from ibapi.contract import Contract
    IBKR_AVAILABLE = True
except ImportError:
    class EClient: pass
    class EWrapper: pass
    class Contract: pass
    logger.critical("[CRITICAL] IBKR API (ibapi) NOT FOUND. Install with: 'pip install ibapi'. System effectively crippled.")
    IBKR_AVAILABLE = False

# --- ALPACA IMPORT ---
# --- ALPACA IMPORT ---
try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit
    ALPACA_AVAILABLE = True
except ImportError:
    class TimeFrame: Day, Minute15, Hour = '1D', '15M', '1H'
    class TimeFrameUnit: Minute = 'Min'
    logger.debug("[!] Alpaca SDK not found. Alpaca fallback disabled.")
    ALPACA_AVAILABLE = False

# --- MASSIVE IMPORT ---
try:
    from massive import RESTClient
    MASSIVE_AVAILABLE = True
except ImportError:
    MASSIVE_AVAILABLE = False
    logger.debug("[!] Massive SDK not found.")

def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizer for Financial DataFrames.
    Ensures input data (regardless of source) adheres to the 'Schema':
    - Lowercase Columns: open, high, low, close, volume
    - Index: DatetimeIndex (Timezone Naive)
    - No duplicates
    """
    if df is None or df.empty: return pd.DataFrame()

    # Ensure the input is a DataFrame before accessing 'columns'
    if not isinstance(df, pd.DataFrame):
        logging.error(f"Input to clean_raw_data is not a DataFrame: {type(df)}")
        return pd.DataFrame()  # Return empty DataFrame to avoid AttributeError

    # 1. Standardize column case (lowercase for pandas_ta)
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten each tuple, drop empty parts, then lowercase
        df.columns = [
            "_".join([str(part) for part in col if part is not None and part != ""]).lower()
            for col in df.columns
        ]
    else:
        df.columns = [str(c).lower() for c in df.columns]

    # CLEANUP: Handle "close_ticker" format from YFinance
    # If we have "close_sbux", rename to "close"
    cleaned_cols = []
    known_cols = {'open', 'high', 'low', 'close', 'volume', 'adj close'}
    for c in df.columns:
        # Check if the column starts with a known OHLCV type
        found = False
        for k in known_cols:
            if c.startswith(k):
                cleaned_cols.append(k)
                found = True
                break
        if not found:
            cleaned_cols.append(c)
    df.columns = cleaned_cols

    # 2. Drop duplicates (crucial after concatenation/merging data sources)
    if isinstance(df.index, pd.MultiIndex):
        df = df.droplevel(0)  # Drop symbol index from multi-index

    df = df[~df.index.duplicated(keep='last')]
    df.index.name = "Date"

    # 3. Ensure index is timezone-naive datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # 4a. Remove duplicate columns if any exist (merged from system_config)
    df = df.loc[:, ~df.columns.duplicated(keep='first')]

    # 4b. Enforce numeric types for OHLCV columns (merged from system_config)
    # Coerce non-numeric values to NaN (e.g. string data from bad provider response)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 4. Filter for required columns (must have OHLCV)
    required = {'open', 'high', 'low', 'close', 'volume'}
    if not required.issubset(df.columns):
        # We try to rename if YF/Alpaca capitalization was lost
        df.rename(columns={'adj close': 'close'}, inplace=True)
        if not required.issubset(df.columns):
            logging.warning("Clean raw data failed: Missing OHLCV columns.")
            return pd.DataFrame()

    # Ensure AI matrix compatibility for providers missing trade_count
    if 'trade_count' not in df.columns:
        df['trade_count'] = 1.0

    # 5a. Drop rows where any OHLCV column is NaN (merged from system_config)
    # This removes rows with corrupt or missing price data that would break indicators
    ohlcv_cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
    if ohlcv_cols:
        df.dropna(subset=ohlcv_cols, inplace=True)

    # Final cleanup (Forward fill then backward fill NaNs, usually from splits)
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    return df


class IBKRDataApp(EWrapper, EClient):
    """
    Thread-safe Wrapper for the Interactive Brokers native API.
    Handles the asynchronous event loop of TWS.
    """

    def __init__(self, client_id):
        EWrapper.__init__(self)
        EClient.__init__(self, self)
        self.client_id = client_id
        self.data = []
        self.data_event = threading.Event()
        self.resolved_contract = None
        self.contract_event = threading.Event()
        self.logger = logging.getLogger(f"IBKR_Client_{client_id}")
        self.error_occurred = False
        self.error_message = ""

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        if errorCode in [2104, 2106, 2158]: return  # Ignore connectivity msgs
        self.logger.error(f"Error {errorCode}: {errorString}")
        if errorCode in [162, 200, 1100, 1101, 1102, 504, 502]:
            self.error_occurred = True
            self.error_message = errorString
            self.data_event.set()
            self.contract_event.set()

    def historicalData(self, reqId, bar):
        """Callback: Receives one bar of historical data."""
        self.data.append({
            'Date': bar.date, 'Open': bar.open, 'High': bar.high,
            'Low': bar.low, 'Close': bar.close, 'Volume': bar.volume
        })

    def historicalDataEnd(self, reqId, start, end):
        """Callback: Batch download complete."""
        self.data_event.set()

    def contractDetails(self, reqId, contractDetails):
        """Callback: Contract ID resolved."""
        self.resolved_contract = contractDetails.contract
        self.contract_event.set()

    def contractDetailsEnd(self, reqId):
        self.contract_event.set()


class DataSourceManager:
    """
    Orchestrates Data Retrieval across multiple providers.
    """
    _client_id_counter = random.randint(1000, 5000) # Start random to avoid collisions
    _client_id_lock = threading.Lock()

    # Global Circuit Breaker State
    _massive_lockout_until = None
    _massive_fail_count = 0  # Tracks consecutive 429 failures for escalating lockout

    # ═══ MASSIVE SESSION KILL FLAG (2026-03-18) ═══════════════════════════
    # DO NOT DELETE: After the first 429 from MASSIVE in a scan session,
    # this flag skips MASSIVE instantly for ALL remaining symbols.
    # Without this flag, each symbol wastes 10s on the timeout before
    # the Circuit Breaker lockout kicks in. With this flag, only the
    # FIRST symbol pays the timeout cost; the rest skip in 0ms.
    # See CHANGELOG entry "2026-03-18 MASSIVE timeout fix" for full context.
    # ═════════════════════════════════════════════════════════════════════
    _massive_session_dead = False

    def __init__(self, use_ibkr=True, allow_fallback=True, host=cfg.IBKR_HOST, port=cfg.IBKR_PORT):
        self.use_ibkr = use_ibkr and IBKR_AVAILABLE
        self.allow_fallback = allow_fallback
        self.host = host
        self.port = port
        self.app = None
        self.ibkr_thread = None
        # self.client_id = random.randint(100, 9999)

        with DataSourceManager._client_id_lock:
            self.client_id = DataSourceManager._client_id_counter
            DataSourceManager._client_id_counter += 1
            
        self.req_id_counter = 0  # Atomic counter for this instance requests
        self._req_id_lock = threading.Lock()

        self._setup_logging()

        # --- ALPACA SETUP (SAFE) ---
        self.stock_client = None
        # Strict User Request: "change Alpaca broker to False, use only IBKR and YFinance"
        # We only initialize Alpaca if explicitly allowed or if DATA_PROVIDER is ALPACA
        is_alpaca_enabled = getattr(cfg, 'DATA_PROVIDER', 'ALPACA') == 'ALPACA'
        
        if ALPACA_AVAILABLE and is_alpaca_enabled:
            try:
                # 1. Try Config
                self.api_key = getattr(cfg, 'ALPACA_KEY', None)
                self.api_secret = getattr(cfg, 'ALPACA_SECRET', None)
                
                # 2. Try Streamlit Secrets (Runtime)
                if not self.api_key:
                    try:
                        self.api_key = st.secrets["APCA_API_KEY_ID"]
                        self.api_secret = st.secrets["APCA_API_SECRET_KEY"]
                    except:
                        pass
                
                # 3. Try Manual TOML Parsing (Script Mode)
                if not self.api_key:
                    try:
                        import toml
                        secrets_path = os.path.join(os.getcwd(), ".streamlit", "secrets.toml")
                        if os.path.exists(secrets_path):
                            secrets = toml.load(secrets_path)
                            self.api_key = secrets.get("APCA_API_KEY_ID")
                            self.api_secret = secrets.get("APCA_API_SECRET_KEY")
                            if self.api_key: self._log("Loaded keys manually from secrets.toml")
                    except Exception as e:
                        # self._log(f"Manual TOML load failed: {e}", "DEBUG")
                        pass

                # 4. Try Environment Variables
                if not self.api_key:
                    self.api_key = os.getenv("APCA_API_KEY_ID")
                    self.api_secret = os.getenv("APCA_API_SECRET_KEY")

                if self.api_key and self.api_secret:
                    # self.stock_client = StockHistoricalDataClient(self.api_key, self.api_secret)
                    self.stock_client = tradeapi.REST(self.api_key, self.api_secret, cfg.ALPACA_BASE_URL, api_version='v2')
                    self._log("Alpaca API (tradeapi) Initialized.", "INFO")
                else:
                    self._log("Alpaca API keys missing. Skipping Alpaca.", "WARNING")
            except Exception as e:
                self._log(f"Alpaca Init Error: {e}", "ERROR")
                self.stock_client = None
        else:
             self._log("Alpaca disabled by configuration (DATA_PROVIDER != ALPACA).", "INFO")

        # --- MASSIVE SETUP ---
        self.massive_client = None
        if MASSIVE_AVAILABLE:
            try:
                # User specified key is strictly "API_KEY" in secrets.toml
                # We reuse the logic that loaded 'secrets' dict above if it exists
                # But that variable 'secrets' was local to the try block. 
                # Let's duplicate the logic or check env.
                
                massive_key = None
                
                # 1. Try Config Object
                massive_key = getattr(cfg, 'MASSIVE_API_KEY', None)
                
                # 2. Try Secrets (Manual TOML)
                if not massive_key:
                    try:
                        import toml
                        secrets_path = os.path.join(os.getcwd(), ".streamlit", "secrets.toml")
                        if os.path.exists(secrets_path):
                            secrets = toml.load(secrets_path)
                            # User explicit instruction: "the API KEY located in ... as API_KEY"
                            massive_key = secrets.get("API_KEY")
                    except:
                        pass
                
                # 3. Env
                if not massive_key:
                    massive_key = os.getenv("API_KEY") # Check generic name too?
                
                if massive_key:
                    self.massive_client = RESTClient(massive_key)
                    self._log("Massive API Initialized.", "INFO")
                else:
                    self._log("Massive API Key (API_KEY) missing.", "WARNING")

            except Exception as e:
                self._log(f"Massive Init Error: {e}", "ERROR")

        self._log(f"--- DataSourceManager initialized (ID: {self.client_id}) ---")
        self._log(
            f"Provider Status: "
            f"MASSIVE={'Ready' if self.massive_client else 'DISABLED'} | "
            f"ALPACA={'Ready' if self.stock_client else 'DISABLED'} | "
            f"IBKR={'Enabled' if self.use_ibkr else 'DISABLED'} | "
            f"YFINANCE=Always Ready",
            "INFO"
        )

    def _setup_logging(self):
        """
        Gen-12 Fix: Routes Data Manager logs into the central Master Log 
        instead of generating rogue dsm_xxx.log files.
        """
        self.logger = cfg.LoggerSetup.setup_logger(f"DSM_{self.client_id}")

    def get_new_req_id(self):
        """Thread-safe request ID generator"""
        with self._req_id_lock:
            self.req_id_counter += 1
            # Base the ID on client_id * 10000 to avoid collision with other clients
            return (self.client_id * 10000) + self.req_id_counter

    def _log(self, message, level="INFO"):
        if level == "INFO":
            self.logger.info(message)
        elif level == "WARNING":
            self.logger.warning(message)
        elif level == "ERROR":
            self.logger.error(message)

    def connect_to_ibkr(self):
        if self.isConnected(): return True
        host = self.host if self.host else '127.0.0.1'
        port = self.port if self.port else 7497

        self._log(f"Connecting to TWS on {host}:{port}...")
        try:
            self.app = IBKRDataApp(self.client_id)
            self.app.connect(host, port, self.client_id)
            self.ibkr_thread = threading.Thread(target=self.app.run, daemon=True)
            self.ibkr_thread.start()

            for _ in range(50):
                if self.app.isConnected():
                    self._log("[OK] Connected to IBKR.")
                    return True
                time.sleep(0.1)
            return False
        except Exception as e:
            self._log(f"Connection failed: {e}", "ERROR")
            return False

    def disconnect(self):
        if self.app is not None and self.app.isConnected():
            self.app.disconnect()
            self._log("Disconnected from IBKR.")

    def isConnected(self):
        return self.app is not None and getattr(self.app, 'isConnected', lambda: False)()

    def get_fundamentals(self, ticker):
        """
        Fetches fundamental data for a given ticker using yfinance.
        Returns a dictionary with key metrics or None if failed.
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Extract key metrics safely
            fundamentals = {
                "trailingPE": info.get("trailingPE"),
                "forwardPE": info.get("forwardPE"),
                "pegRatio": info.get("pegRatio"),
                "revenueGrowth": info.get("revenueGrowth"),
                "profitMargins": info.get("profitMargins"),
                "returnOnEquity": info.get("returnOnEquity"),
                "sector": info.get("sector"),
                "industry": info.get("industry")
            }
            logger.info(f"Fundamentals for {ticker}: {fundamentals}")
            return fundamentals
        except Exception as e:
            logger.error(f"Failed to fetch fundamentals for {ticker}: {e}")
            return None

    def fetch_data_sequential(self, tickers: list,days_back=1825):
        """
        Sequentially fetches data for a list of tickers.
        Returns a dictionary {ticker: dataframe}.
        """
        data_map = {}
        for ticker in tickers:
            logger.info(f"[>] Fetching {ticker}...")
            # df = self.get_stock_data(ticker, days_back=None) # Rely on system config default
            df = self.get_stock_data(ticker, days_back=days_back, interval='1d', min_rows=1000)

            if not df.empty:
                data_map[ticker] = df
            else:
                logger.warning(f"Failed to fetch data for {ticker}")
        
        return data_map

    # --- MAIN DATA METHOD (WATERFALL LOGIC) ---
    def get_stock_data(self, symbol, start_date=None, end_date=None, days_back=None, interval='1d', source='AUTO', trade_type=None, min_rows=0):

        """
        Super-Fetcher.
        Attempt 1: Massive (Fastest).
        Attempt 2: Alpaca (Reliable).
        Attempt 3: IBKR (Real-time).
        Attempt 4: YFinance (Fallback).

        Super-Fetcher with Data Integrity Validation.
        
        :param min_rows: Minimum required data points. If fetched data < min_rows, 
                         it is treated as a failure (triggers fallback).
        
        Returns: pd.DataFrame (Cleaned)
        """
        df = pd.DataFrame()
        clean_symbol = symbol.upper().strip()

        # 0. Handle Trade Type Logic (Overrides)
        if trade_type and hasattr(cfg, 'TRADE_TYPE_CONFIG'):
            tt_config = cfg.TRADE_TYPE_CONFIG.get(trade_type)
            if tt_config:
                interval = tt_config.get('interval', interval)
                days_back = tt_config.get('days_back', days_back)

        # User Preference Override (from Config) if AUTO
        # Note: If source is AUTO, we follow the strict waterfall order requested:
        # Massive -> Alpaca -> IBKR -> YFinance
        
        priority_list = ['MASSIVE', 'ALPACA', 'IBKR', 'YFINANCE']
        # priority_list = ['ALPACA'] # TEST MODE: PHASE 1

        # If specific source requested
        if source != 'AUTO' and source in priority_list:
             priority_list.remove(source)
             priority_list.insert(0, source)
        # elif source == 'AUTO':
        #      pass # keep default order
             
        for provider in priority_list:
            self._log(f"[{clean_symbol}] Trying provider: {provider}...", "INFO")

            # --- GEN-13: CIRCUIT BREAKER PATTERN ---
            if provider == 'MASSIVE':
                # ═══ SESSION KILL: instant skip after first 429 (2026-03-18) ═══
                # DO NOT DELETE: Without this, every symbol retries MASSIVE and
                # waits for the 10s timeout before cascading. This saves ~10s × N symbols.
                if DataSourceManager._massive_session_dead:
                    continue  # Silent skip — already logged on first failure
                # ═══ CIRCUIT BREAKER: escalating lockout (2026-03-14) ═══
                if DataSourceManager._massive_lockout_until and datetime.now() < DataSourceManager._massive_lockout_until:
                    remaining = int((DataSourceManager._massive_lockout_until - datetime.now()).total_seconds() / 60)
                    self._log(f"MASSIVE Circuit Breaker Active. Locked out for {remaining} more min. Cascading to next provider.", "WARNING")
                    continue
            
            # Architectural Fix: Rate Limiting to prevent 429 Errors (Massive API)
            import time
            delay = getattr(cfg, 'PROVIDER_DELAY', {}).get(provider, 0.5)
            if delay > 0:
                time.sleep(delay)
            
            fetched_df = pd.DataFrame()

            try:
                # 1. MASSIVE
                if provider == 'MASSIVE' and self.massive_client:
                    fetched_df = self._download_from_massive(clean_symbol, start_date, end_date, days_back, interval, min_rows=min_rows)
                elif provider == 'MASSIVE':
                    self._log("MASSIVE skipped: client not initialized (check API_KEY in secrets.toml).", "WARNING")
                    continue

                # 2. ALPACA
                elif provider == 'ALPACA' and self.stock_client:
                    fetched_df = self._download_from_alpaca(clean_symbol, start_date, end_date, days_back, interval, min_rows=min_rows)
                elif provider == 'ALPACA':
                    self._log("ALPACA skipped: client not initialized (check API keys or DATA_PROVIDER config).", "WARNING")
                    continue

                # 3. IBKR
                elif provider == 'IBKR' and self.use_ibkr:
                    if not self.isConnected(): self.connect_to_ibkr()
                    if self.isConnected():
                        fetched_df = self._download_from_ibkr(clean_symbol, start_date, end_date, days_back, interval, min_rows=min_rows)
                    else:
                        self._log("IBKR skipped: connection failed. Cascading to next provider.", "WARNING")
                        continue
                elif provider == 'IBKR':
                    self._log("IBKR skipped: use_ibkr=False. Cascading to next provider.", "WARNING")
                    continue

                # 4. YFINANCE
                elif provider == 'YFINANCE':
                    fetched_df = self._download_from_yfinance(clean_symbol, days_back, interval, start_date, end_date, min_rows=min_rows)

                # --- VALIDATION GATE ---
                if not fetched_df.empty:
                    # Clean it first to ensure valid OHLCV
                    clean_df = clean_raw_data(fetched_df)
                    
                    # Row Count Check
                    if len(clean_df) >= min_rows:
                        self._log(f"Success ({provider}): Retrieved {len(clean_df)} rows (Target: {min_rows}+).", "DEBUG")
                        # Reset Circuit Breaker on success
                        if provider == 'MASSIVE':
                            DataSourceManager._massive_lockout_until = None
                            DataSourceManager._massive_fail_count = 0
                        
                        return clean_df
                    else:
                        self._log(f"Insufficient Data ({provider}): Got {len(clean_df)} rows, needed {min_rows}. Trying next...", "WARNING")
                        # We do NOT return here; we continue the loop to try the next broker
                
            except Exception as e:
                self._log(f"{provider} Failure: {e}", "WARNING")
                # --- GEN-13: TRIP THE CIRCUIT BREAKER (Escalating Lockout) ---
                if provider == 'MASSIVE' and ('429' in str(e) or 'timeout' in str(e).lower()):
                    # ═══ SESSION KILL + CIRCUIT BREAKER (2026-03-18) ═══
                    # DO NOT DELETE: _massive_session_dead prevents retrying MASSIVE
                    # for the rest of this scan. Without it, every symbol pays 10s timeout.
                    DataSourceManager._massive_session_dead = True
                    DataSourceManager._massive_fail_count += 1
                    # Escalate lockout: 1st hit = 1 hour, subsequent hits = 4 hours
                    lockout_hours = 4 if DataSourceManager._massive_fail_count > 1 else 1
                    DataSourceManager._massive_lockout_until = datetime.now() + timedelta(hours=lockout_hours)
                    self._log(
                        f"MASSIVE 429 Rate Limit — Circuit Breaker TRIPPED for {lockout_hours}h "
                        f"(consecutive failures: {DataSourceManager._massive_fail_count}). "
                        f"Cascading to Alpaca/IBKR/YFinance.",
                        "WARNING"
                    )

                continue

        # If we exit the loop, all providers failed
        self._log(f"[FAIL] Data Starvation: All providers failed to return {min_rows} rows for {symbol}.", "ERROR")
        return pd.DataFrame()

    # --- INTERNAL DOWNLOADERS ---
    def _download_from_ibkr(self, symbol, start_date, end_date, days_back, interval, min_rows=0):
        """
        Robust IBKR Fetcher with Full Contract Definition.
        """
        try:
            # 1. Connection Check
            if not self.app or not self.isConnected():
                if not self.connect_to_ibkr():
                    self._log(f"IBKR Not Connected. Skipping {symbol}.", "WARNING")
                    return pd.DataFrame()

            # 2. Contract Construction
            contract = Contract()
            contract.currency = 'USD'
            clean_symbol = symbol.strip().upper()

            if clean_symbol.startswith('^') or clean_symbol == 'VIX':
                contract.secType = 'IND'
                contract.symbol = clean_symbol.lstrip('^')
                contract.exchange = 'CBOE'
            else:
                contract.secType = 'STK'
                contract.symbol = clean_symbol
                contract.exchange = 'SMART'
                contract.primaryExchange = 'ISLAND'

            # 3. Duration Logic (Looking back from NOW)
            duration = "1 Y"
            if days_back:
                if days_back > 365: duration = "2 Y"
                elif days_back > 30: duration = f"{int(days_back/30) + 1} M"
                else: duration = f"{days_back} D"

            # 4. Bar Size Mapping
            ib_interval_map = {
                "1m": "1 min", "5m": "5 mins", "15m": "15 mins", "30m": "30 mins",
                "1h": "1 hour", "1d": "1 day", "1wk": "1 week", "1mo": "1 month"
            }
            bar_size = ib_interval_map.get(interval, "1 day")

            # 5. Fetch
            df = self.app.fetch_historical_data(
                contract=contract,
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=1
            )
            
            # 6. Integrity Check
            if not df.empty and len(df) < min_rows:
                 self._log(f"IBKR Insufficient Data: Got {len(df)} rows (Target: {min_rows})", "WARNING")

            return df

        except Exception as e:
            self._log(f"IBKR Failure: {e} (Target: {min_rows})", "WARNING")
            return pd.DataFrame()

    def _download_from_alpaca(self, symbol, start_date, end_date, days_back, interval, min_rows=0):
        """
        Robust Alpaca Fetcher (Gen-12 Compliant).
        """
        try:
            # 1. Client Check
            if not self.stock_client:
                return pd.DataFrame()

            # 2. Timeframe Mapping
            # Alpaca uses TimeFrame objects (1Day, 1Hour, etc.)
            from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit
            
            tf = TimeFrame.Day # Default
            if interval == "1d": tf = TimeFrame.Day
            elif interval == "1wk": tf = TimeFrame.Week
            elif interval == "1h": tf = TimeFrame.Hour
            elif interval == "15m": tf = TimeFrame(15, TimeFrameUnit.Minute)
            elif interval == "5m": tf = TimeFrame(5, TimeFrameUnit.Minute)
            elif interval == "1m": tf = TimeFrame.Minute

            # 3. Date Logic (Alpaca requires ISO format)
            # Alpaca requires strictly formatted RFC-3339 timestamps for robust fetching
            utc_now = datetime.utcnow()
            
            if not end_date:
                end_date = utc_now
            elif isinstance(end_date, str):
                try:
                    end_date = datetime.strptime(end_date, '%Y-%m-%d')
                except ValueError:
                    end_date = utc_now

            # Architectural Fix: SIP Free Tier Bypass
            # Enforce a strict 16-minute delay from UTC now to prevent 403 permission errors during live market hours.
            safe_end_dt = utc_now - timedelta(minutes=16)
            
            if end_date > safe_end_dt:
                effective_end = safe_end_dt
            else:
                effective_end = end_date

            end_iso = effective_end.strftime('%Y-%m-%dT%H:%M:%SZ')

            if not start_date and days_back:
                # CRITICAL CALCULATION: A market year is 252 trading days.
                buffer_days_back = int(days_back * 1.5)
                start_dt = effective_end - timedelta(days=buffer_days_back)
                start_iso = start_dt.strftime('%Y-%m-%dT00:00:00Z')
            else:
                if isinstance(start_date, str):
                    try:
                        start_iso = datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y-%m-%dT00:00:00Z')
                    except ValueError:
                        start_iso = start_date
                else:
                    start_iso = start_date.strftime('%Y-%m-%dT00:00:00Z')
            
            # 4. Fetch Data
            # Note: adjustment='raw' or 'all'. 'all' adjusts for splits/divs.
            bars = self.stock_client.get_bars(
                symbol,
                tf,
                start=start_iso,
                end=end_iso,
                adjustment='all',
                feed='iex',
                limit=10000 
            ).df

            # 5. formatting
            if not bars.empty:
                # Alpaca returns index as timezone-aware timestamp. 
                # We normalize columns to lowercase (open, high, low, close, volume)
                bars.columns = [c.lower() for c in bars.columns]
                
                # Filter for valid rows
                if len(bars) < min_rows:
                    self._log(f"ALPACA Insufficient Data: Got {len(bars)} rows (Target: {min_rows})", "WARNING")
            
            return bars

        except Exception as e:
            # FIX: 'min_rows' is now in scope, preventing the crash
            self._log(f"ALPACA Failure: {e} (Target: {min_rows})", "WARNING")
            return pd.DataFrame()

    # ═══════════════════════════════════════════════════════════════════════
    # CRITICAL PERFORMANCE NOTE — DO NOT DELETE (2026-03-18)
    # ═══════════════════════════════════════════════════════════════════════
    # This function wraps self.massive_client.get_aggs() with a hard timeout.
    #
    # WHY: The Polygon SDK (massive_client) has NO built-in request timeout.
    # When the API returns 429 (rate limit), the SDK retries internally
    # for 30-60 seconds before surfacing the exception. During a 4000-symbol
    # scan, this caused the first symbol to waste 30-60 seconds, and if the
    # session kill flag was missing, EVERY symbol would waste 10+ seconds.
    #
    # WHAT WE DO:
    # 1. ThreadPoolExecutor with future.result(timeout=10s)
    # 2. If timeout fires → raise → outer handler sets _massive_session_dead
    # 3. All subsequent symbols skip MASSIVE instantly (0ms)
    #
    # HISTORY:
    # - Before 2026-03-14: No circuit breaker at all → 12+ hour scans
    # - 2026-03-14: Circuit breaker added, double throttle removed → 30-60 min
    # - 2026-03-18: Timeout wrapper + session kill → first symbol 10s max
    #
    # DO NOT remove the timeout wrapper or session flag without understanding
    # the full history above. The Polygon SDK WILL hang without them.
    # ═══════════════════════════════════════════════════════════════════════
    def _download_from_massive(self, symbol, start_date, end_date, days_back, interval, min_rows=0):
        """
        Download from Massive (formerly Polygon).
        """
        if not self.massive_client: return pd.DataFrame()
        
        # Interval Mapping
        multiplier = 1
        timespan = 'day'
        
        if interval in ['1d', '1 day', 'daily']:
            timespan = 'day'
        elif interval in ['1h', '60m']:
            timespan = 'hour'
            multiplier = 1
        elif interval in ['15m']:
            timespan = 'minute'
            multiplier = 15
        elif interval in ['5m']:
            timespan = 'minute'
            multiplier = 5
        elif interval in ['1m']:
            timespan = 'minute'
            multiplier = 1
            
        # Date String YYYY-MM-DD
        # Note: start_date/end_date might be datetime or str
        def fmt(d):
            if isinstance(d, (datetime, pd.Timestamp)): return d.strftime('%Y-%m-%d')
            return str(d)

        # Handle 'days_back' if start_date is missing
        start = start_date
        end = end_date if end_date else datetime.now()
        
        if not start and days_back:
            start = end - timedelta(days=days_back)
            
        try:
            logging.info(f"Massive Request: {symbol} | {fmt(start)} to {fmt(end)} | {multiplier}{timespan}")

            # ═══ TIMEOUT WRAPPER (2026-03-18) ═══════════════════════════════
            # DO NOT DELETE: The Polygon SDK has NO built-in timeout.
            # Without this wrapper, a 429 response causes 30-60 seconds of
            # hidden SDK retries before the exception surfaces. This caps
            # the wait at 10 seconds and lets the waterfall cascade immediately.
            # See CHANGELOG "2026-03-18 MASSIVE timeout fix" for full analysis.
            # ═══════════════════════════════════════════════════════════════════
            import concurrent.futures
            _massive_timeout = getattr(cfg, 'PROVIDER_DELAY', {}).get('MASSIVE_TIMEOUT', 10)
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self.massive_client.get_aggs,
                    ticker=symbol,
                    multiplier=multiplier,
                    timespan=timespan,
                    from_=fmt(start),
                    to=fmt(end),
                    limit=50000
                )
                try:
                    aggs = future.result(timeout=_massive_timeout)
                except concurrent.futures.TimeoutError:
                    self._log(f"MASSIVE timeout ({_massive_timeout}s) for {symbol}. Cascading to next provider.", "WARNING")
                    raise Exception(f"MASSIVE timeout exceeded {_massive_timeout}s — likely 429 rate limit")
            
            if not aggs:
                return pd.DataFrame()
                
            # Convert to DataFrame
            # Agg properties: open, high, low, close, volume, vwap, timestamp
            data = []
            for a in aggs:
                # timestamp is usually ms
                dt = datetime.fromtimestamp(a.timestamp / 1000.0) if a.timestamp else None
                data.append({
                    'timestamp': dt,
                    'open': a.open,
                    'high': a.high,
                    'low': a.low,
                    'close': a.close,
                    'volume': a.volume,
                    'vwap': a.vwap
                })
                
            df = pd.DataFrame(data)
            if df.empty: return df
            
            df.set_index('timestamp', inplace=True)
            return df
            
        except Exception as e:
            # Re-raise 429 and timeout errors so the outer handler can trip the
            # circuit breaker and session kill flag. All other errors are swallowed.
            if '429' in str(e) or 'timeout' in str(e).lower():
                raise
            self._log(f"MASSIVE Failure: {e}", "WARNING")
            return pd.DataFrame()
        
    def _download_from_yfinance(self, symbol, days_back, interval, start_date=None, end_date=None, min_rows=0):
        """
        Robust YFinance Fetcher: Handles Tuples, MultiIndex headers, and Ticker Translation.
        """
        try:
            # 1. Ticker Translation (Yahoo requires '-' instead of '.' for BRK.B)
            yf_symbol = symbol.replace('.', '-')
            
            # 2. Date Logic
            if not start_date and days_back:
                start_dt = datetime.now() - timedelta(days=days_back)
                start_date = start_dt.strftime('%Y-%m-%d')
            
																			  
            if not start_date:
                start_dt = datetime.now() - timedelta(days=365)
                start_date = start_dt.strftime('%Y-%m-%d')

            # 3. Fetch Data
            # 'group_by' ensures consistent formatting even for single tickers
            df = yf.download(
                yf_symbol, 
                start=start_date, 
                end=end_date, 
                interval=interval, 
                progress=False, 
                auto_adjust=True,
                group_by='ticker' 
            )

            # 4. Critical Fix: Handle Headers SAFELY
            if not df.empty:
                # A) Handle MultiIndex (Tuple Headers) FIRST
															
																				  
                if isinstance(df.columns, pd.MultiIndex):
                    try:
                        # If structure is (Price, Ticker), drop the Ticker level
                        if yf_symbol in df.columns.get_level_values(0):
                             # Access the specific ticker's dataframe directly
                             df = df[yf_symbol]
                        else:
                             # Generic drop level
                             df.columns = df.columns.droplevel(1)
                    except Exception:
                        pass # Fallback if structure is unexpected

                # B) NOW it is safe to lowercase strings
                df.columns = [str(c).lower().strip() for c in df.columns]

            # 5. Integrity Check
            if not df.empty and len(df) < min_rows:
																				 
                self._log(f"YFINANCE Insufficient Data: Got {len(df)} rows (Target: {min_rows})", "WARNING")
                # return pd.DataFrame() # Optional strict mode

            return df

        except Exception as e:
																						  
            self._log(f"YFINANCE Failure: {e} (Target: {min_rows})", "WARNING")
            return pd.DataFrame()

    # --- GEN-12 HELPER METHODS ---
    def fetch_and_process(self, symbol, interval="1d"):
        """
        Fetches raw data and immediately runs the Feature Engine.
        Used by: data_engineer.py
        """
        from feature_engine import RobustFeatureCalculator
        
        # 1. Fetch Raw Data
        df = self.get_stock_data(symbol, days_back=365, interval=interval)
        if df.empty: return pd.DataFrame()
        
        # 2. Add Features
        calc = RobustFeatureCalculator()
        try:
            # Add sector context if needed (mock for now or fetch QQQ)
            df = calc.calculate_features(df)
            return df
        except Exception as e:
            self._log(f"Feature Calc failed for {symbol}: {e}", "ERROR")
            return pd.DataFrame()

    def fetch_data(self, symbol, limit=100, interval="1d"):
         """Simple fetch wrapper for StockHunter"""
         df = self.get_stock_data(symbol, days_back=limit, interval=interval)
         return df

    # --- GEN-7 STREAMING ARCHITECTURE ---
    async def stream_data(self, symbol: str, queue: asyncio.Queue):
        """
        Async Generator that mimics a WebSocket.
        Pushes 'BAR' events to the provided asyncio Queue.
        Real implementation would hook into IBKR's EWrapper.tickPrice.
        Current implementation involves 'Polling' latest bar to behave like a stream.
        """
        self._log(f"Starting Data Stream for {symbol}...", "INFO")
        
        # 1. Start Jitter: Prevent all 10 stocks from hitting the API at the exact same millisecond
        await asyncio.sleep(random.uniform(0, 10))

        try:
            while True:
                # 1. Fetch latest snapshot (Polling as temporary shim for WebSocket)
                # This prevents "Blindness" by ensuring we always have fresh data 
                latest_bar = await self._fetch_latest_async(symbol)
                
                if latest_bar:
                   timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                   event = {
                       "type": "BAR",
                       "symbol": symbol,
                       "price": latest_bar['close'],
                       "volume": latest_bar['volume'],
                       "timestamp": timestamp,
                       "data": latest_bar # Full struct
                   }
                   await queue.put(event)
                
                # 3. Optimization: Throttle the Loop
                # Previous setting was 15s. We increase to 60s + Random to reduce
                # pressure on the connection pool by 75%.
                wait_time = 60 + random.uniform(0, 10)
                await asyncio.sleep(wait_time)
                
        except asyncio.CancelledError:
            self._log(f"Stream cancelled for {symbol}", "INFO")
        except Exception as e:
            self._log(f"Stream error for {symbol}: {e}", "ERROR")
            # Backoff significantly on error to allow pool to recover
            await asyncio.sleep(60)

    async def _fetch_latest_async(self, symbol):
        """
        Non-blocking wrapper around synchronous fetch.
        Runs I/O in a separate thread to keep Async loop unblocked.
        """
        loop = asyncio.get_event_loop()
        try:
            # Enforce minimum 20 rows to calculate basic intraday indicators
            df = await loop.run_in_executor(None, lambda: self.get_stock_data(symbol, days_back=5, interval="15m", min_rows=20))

            if not df.empty:
                return df.iloc[-1].to_dict()
            return None
        except Exception:
            return None

    def get_realtime_quote(self, symbol):
        """
        Fetch Level 2 (Bid/Ask) Data.
        """
        quote = {"bid": 0.0, "ask": 0.0, "spread": 0.0}
        
        # 1. Try IBKR (Real Level 2)
        if self.use_ibkr and self.isConnected():
            # (Simplified IBKR implementation placeholder)
            pass
            
        # 2. Fallback to YFinance (Delayed Quote)
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            quote["bid"] = info.get("bid", info.get("currentPrice", 0.0))
            quote["ask"] = info.get("ask", info.get("currentPrice", 0.0))
            quote["spread"] = quote["ask"] - quote["bid"]
            return quote
        except Exception as e:
            self._log(f"Quote fetch failed: {e}", "WARNING")
            return None

    def regenerate_all_data(self):
        """
        Gen-12 Utility: Forces a refresh of all historical data.
        Downloads Deep History for every symbol in Watchlist + Common Indices (SPY, QQQ).
        Calculates Feature Engineering and saves to 'Gold' parquet.
        """
        self._log("Starting Global Data Regeneration...", "INFO")
        from feature_engine import RobustFeatureCalculator
        
        calc = RobustFeatureCalculator()
        import system_config as cfg
        
        # Combine Watchlist + any extras
        targets = list(set(cfg.WATCHLIST + ["NVDA", "SPY", "QQQ", "IWM"]))
        
        for symbol in targets:
            try:
                self._log(f"Processing {symbol}...")
                # 1. Fetch deep history (2 years for training)
                df = self.get_stock_data(symbol, days_back=730, interval="1d")
                
                if df.empty: 
                    self._log(f"Skipping {symbol}: No data returned from any provider.", "WARNING")
                    continue
                
                # 2. Calculate Features (This runs MasterPatternLib)
                df = calc.calculate_features(df)

                # --- NEW DIAGNOSTIC LOG ---
                if 'master_score' not in df.columns:
                    self._log(f"Warning: {symbol} missing 'master_score'. Columns: {list(df.columns[-5:])}", "WARNING")
                
                self._log(f"{symbol} Data Shape: {df.shape} (Rows, Cols)", "INFO") 
                    
                # 3. Calculate Ground Truth (FIX: Use the logic from feature_engine.py)
                from feature_engine import calculate_ground_truth
                df = calculate_ground_truth(df, lookahead=15) # This uses your new RISK_THR = 0.03
                
                # 4. Save to Gold
                save_path = os.path.join(cfg.DB_DIR, "gold", f"{symbol}.parquet")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                df.to_parquet(save_path)
                self._log(f"Saved {symbol}.parquet with Master Score.")
                
            except Exception as e:
                self._log(f"Failed {symbol}: {e}", "ERROR")


class SectorMapper:
    """Helper to map individual stocks to their Sector benchmarks."""
    SECTOR_MAP = {
        'Technology': 'XLK', 'Semiconductors': 'SMH', 'Financial Services': 'XLF',
        'Financials': 'XLF', 'Healthcare': 'XLV', 'Energy': 'XLE',
        'Consumer Cyclical': 'XLY', 'Consumer Discretionary': 'XLY',
        'Communication Services': 'XLC', 'Industrials': 'XLI',
        'Consumer Defensive': 'XLP', 'Utilities': 'XLU',
        'Real Estate': 'XLRE', 'Basic Materials': 'XLB'
    }

    def get_benchmark_symbol(self, ticker: str) -> str:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            industry = info.get('industry', '')
            if 'Semiconductor' in industry: return 'SMH'
            if 'Software' in industry: return 'IGV'
            if 'Biotech' in industry: return 'XBI'

            sector = info.get('sector', '')
            return self.SECTOR_MAP.get(sector, 'SPY')
        except:
            return 'SPY'

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    # Allow running this file directly to regenerate data
    dsm = DataSourceManager()
    dsm.regenerate_all_data()
