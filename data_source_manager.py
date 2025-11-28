"""
⚙️ Data Source Manager - Advanced TWS API Version
===================================================

This script provides a robust, thread-safe, and feature-rich data manager for
fetching historical stock data. It is designed to use the Interactive Brokers (IBKR)
Trader Workstation (TWS) API as its primary data source, with a controllable
fallback to yfinance.

This advanced version is optimized for use in multi-threaded applications and adds
support for fetching high-frequency intraday data.

Key Features:
-------------
-   **Intraday Data Support**: Capable of downloading historical data at various
    intervals (e.g., '1 day', '15 mins'). For large intraday requests, it
    automatically handles IBKR API limitations by fetching data in sequential
    monthly chunks.
-   **Thread-Safe Architecture**:
    -   Generates a unique, thread-safe client ID for each instance to prevent
        connection conflicts with the TWS API.
    -   Implements a sophisticated logging system where each instance writes to its
        own dedicated log file, making it easy to debug parallel operations.
-   **Controllable Fallback**: An `allow_fallback` flag gives the user precise
    control over whether the system should use `yfinance` if an IBKR connection
    or data request fails.
-   **Robust Connection Management**: Automatically connects to the TWS API and
    reliably handles connection timeouts and errors.
-   **Smart Error Filtering**: The TWS error handler is configured to ignore common
    informational messages and warnings, focusing the logs on critical,
    actionable errors.

"""


import pandas as pd
import yfinance as yf
import logging
import time
import threading
import datetime
import os
import traceback
from utils import clean_raw_data
import streamlit as st

# --- Alpaca SDK Imports ---
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.common.exceptions import APIError
except ImportError:
    # If the user doesn't have alpaca-py installed, we create dummies
    class TimeFrame: Day, Minute15, Hour = '1D', '15M', '1H'
    class StockBarsRequest:
        def __init__(self, **kwargs): pass
    class APIError(Exception): pass
    class StockHistoricalDataClient:
        def __init__(self, **kwargs): pass

    class TradingClient:
        def __init__(self, **kwargs): pass


# --- IBAPI Imports ---
try:
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper
    from ibapi.contract import Contract
except ImportError:
    class EClient: pass
    class EWrapper: pass
    class Contract: pass


class Stock(Contract):
    """A simplified Contract class for requesting stock data."""

    def __init__(self, symbol, exchange='SMART', currency='USD'):
        Contract.__init__(self)
        self.symbol = symbol
        self.secType = 'STK'
        self.exchange = exchange
        self.currency = currency


class DataSourceManager(EWrapper, EClient):
    """
    Manages data retrieval using the direct TWS API (ibapi),
    with a fallback to yfinance. This version includes a thread-safe
    client ID generator and robust logging.
    """
    _client_id_counter = int(time.time() % 1000)
    _client_id_lock = threading.Lock()

    def __init__(self, use_ibkr=False, host='127.0.0.1', port=7497, allow_fallback=True):

        # 1. Set the flags FIRST
        self.use_ibkr = use_ibkr
        self.host = host
        self.port = port
        self.allow_fallback = allow_fallback

        # --- Add Alpaca API Key setup ---
        self.api_key = st.secrets.get('APCA_API_KEY_ID')
        self.api_secret = st.secrets.get('APCA_API_SECRET_KEY')

        # --- Initialize Alpaca REST client for historical data ---
        # We use paper=True for the free IEX data
        # self.stock_client = StockHistoricalDataClient(api_key=self.api_key, secret_key=self.api_secret)
        try:
            self.stock_client = StockHistoricalDataClient(api_key=self.api_key, secret_key=self.api_secret)
        except Exception as e:
            # st.error("Failed to initialize StockHistoricalDataClient")
            logging.error(f"Alpaca client initialization error: {e}", exc_info=True)
            self.stock_client = None

        # 2. NOW, conditionally initialize EClient
        if self.use_ibkr:
            EClient.__init__(self, self)

        with DataSourceManager._client_id_lock:
            self.client_id = DataSourceManager._client_id_counter
            DataSourceManager._client_id_counter += 1

        # --- Correct Logger Setup ---
        self.logger = logging.getLogger(f"{type(self).__name__}.Client_{self.client_id}")
        self.logger.info(f"--- DataSourceManager initialized with Client ID {self.client_id} ---")

        self.connection_event = threading.Event()
        self.data_event = threading.Event()
        self.error_occurred = False
        self.error_message = ""
        self.historical_data = []
        self.next_req_id = int(time.time())

    def data_source(self):
        if getattr(self, 'useibkr', False):
            return "ibkr"
        elif getattr(self, 'stockclient', None) is not None:
            return "alpaca"
        else:
            return "yfinance"

    def _log(self, message, level="INFO", exc_info=False):
        """Internal helper for logging with client ID."""
        extra_data = {'client_id': self.client_id}
        if level.upper() == "INFO":
            self.logger.info(message, extra=extra_data, exc_info=exc_info)
        elif level.upper() == "WARNING":
            self.logger.warning(message, extra=extra_data, exc_info=exc_info)
        elif level.upper() == "ERROR":
            self.logger.error(message, extra=extra_data, exc_info=exc_info)
        elif level.upper() == "SUCCESS":
            self.logger.info(message, extra=extra_data, exc_info=exc_info)  # Treat SUCCESS as INFO

    def set_socket_logging(self, enabled=True):
        """Enables or disables low-level IBAPI socket logging."""
        self._log(f"--- TWS Socket-level message logging {'ENABLED' if enabled else 'DISABLED'} ---", "WARNING")
        # EClient.set_socket_logging(self, enabled) # This method does not exist in the base class

    def connect_to_ibkr(self):
        """Establishes a connection to the TWS/Gateway API."""
        if self.isConnected(): return True

        if not self.host: self.host = '127.0.0.1'
        if not self.port: self.port = 7497

        self._log(f"Connecting to TWS on {self.host}:{self.port}...")
        try:
            self.connect(self.host, self.port, self.client_id)

            # Start the socket thread
            api_thread = threading.Thread(target=self.run, daemon=True)
            api_thread.start()

            self._log("Waiting for connection handshake...")
            if self.connection_event.wait(timeout=20):
                self._log("Connection to TWS is active.", "SUCCESS")
                return True
            else:
                self._log("Connection to TWS timed out.", "ERROR")
                self.disconnect()
                return False
        except Exception as e:
            self._log(f"IBKR connection failed: {e}", "ERROR", exc_info=True)
            return False

    def nextValidId(self, orderId: int):
        """EWrapper callback that confirms the connection is ready."""
        super().nextValidId(orderId)
        self._log("Connection handshake confirmed by TWS.")
        self.connection_event.set()

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        """Handle errors"""
        # --- Filter messages by code ---
        if errorCode == 162:
            self._log(f"Informational message: {errorString}", "WARNING")
            return  # Stop processing this error here.

            # This is a warning about the date format which we will fix in the request itself.
        if errorCode == 2174:
            self._log(f"Time zone warning from IBKR (being addressed): {errorString}", "WARNING")
            return

            # --- Filter for other messages ---
        if errorCode in [2104, 2106, 2158]:  # Market data farm "OK" messages
            self._log(f"Market data warning {errorCode}: {errorString}", "WARNING")
        elif errorCode in [502, 503, 504]:  # Critical connection errors
            self._log(f"Connection error {errorCode}: {errorString}", "ERROR")
            self.error_occurred = True
            self.error_message = errorString
            self.connection_event.set()
        else:
            # Treat all other messages as potential errors
            self._log(f"Error {errorCode}: {errorString}", "ERROR")
            if reqId != -1:  # reqId of -1 is a system message
                self.error_occurred = True
                self.error_message = errorString
                self.data_event.set()

    def disconnect(self):
        if self.isConnected():
            self._log("Disconnecting from TWS...")
            super().disconnect()

    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def get_fundamental_info(_self, symbol: str) -> dict:
        """
        Fetches key fundamental info for a symbol using yfinance.
        This is kept separate as it's not OHLCV data and works on cloud.
        """
        _self._log(f"Fetching fundamental info for {symbol} via yfinance...")
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Return a subset of key ratios
            return {
                'trailingPE': info.get('trailingPE'),
                'priceToSalesTrailing12Months': info.get('priceToSalesTrailing12Months'),
                'debtToEquity': info.get('debtToEquity'),
                'marketCap': info.get('marketCap')
            }
        except Exception as e:
            _self._log(f"Error fetching fundamental info for {symbol}: {e}", "ERROR", exc_info=True)
            return {}

    @st.cache_data(ttl=86400)  # Cache for 1 day
    def get_earnings_calendar(_self, symbol: str) -> pd.Timestamp:
        """
        Fetches the next earnings date for a symbol using yfinance.
        """
        _self._log(f"Fetching earnings calendar for {symbol} via yfinance...")
        try:
            ticker = yf.Ticker(symbol)
            earnings_date = ticker.calendar.get('Earnings Date')

            if earnings_date and isinstance(earnings_date, pd.Timestamp):
                # yfinance often returns a date, not a range. We'll use the start.
                return earnings_date
            elif earnings_date and isinstance(earnings_date, (list, tuple)):
                # If it's a range, return the start of the range
                return earnings_date[0]

            _self._log(f"No earnings date found for {symbol}.", "WARNING")
            return None
        except Exception as e:
            _self._log(f"Error fetching earnings calendar for {symbol}: {e}", "ERROR", exc_info=True)
            return None

    def _download_from_ibkr(self, symbol, days_back, bar_size, start_date=None, end_date=None):
        """
        Internal method to request and collect data from IBKR TWS.
        """
        self._log(f"Requesting data for {symbol} from IBKR (Interval: {bar_size})...")

        contract = Contract()
        contract.currency = 'USD'

        # --- FIX: Intelligent Contract Handling for Indices (e.g., ^VIX) ---
        if symbol.startswith('^'):
            contract.secType = 'IND'
            contract.symbol = symbol.lstrip('^')  # IBKR uses 'VIX', not '^VIX'

            # Specific exchange rules for common indices
            if contract.symbol == 'VIX':
                contract.exchange = 'CBOE'
            else:
                contract.exchange = 'SMART'  # Or CBOE, depending on the index
        else:
            contract.secType = 'STK'
            contract.symbol = symbol
            contract.exchange = 'SMART'

        # 1. Reset accumulation variables for this request
        self.historical_data = []
        self.data_event.clear()
        self.error_occurred = False
        self.error_message = ""

        # 2. Calculate End Date String
        # IBKR format example: "20231122 23:59:59"
        if end_date:
            # Ensure end_date includes time if it's just a date object
            if isinstance(end_date, datetime.date) and not isinstance(end_date, datetime.datetime):
                query_time = end_date.strftime("%Y%m%d 23:59:59")
            else:
                query_time = pd.to_datetime(end_date).strftime("%Y%m%d %H:%M:%S")
        else:
            query_time = ""  # Empty string means "now"

        # 3. Calculate Duration String
        # Logic: IBKR uses specific formats "1 Y", "1 M", "1 D", "1 W", "1 S"
        if days_back > 365:
            years = int(days_back / 365) + 1
            duration_str = f"{years} Y"
        elif days_back > 30:
            months = int(days_back / 30) + 1
            duration_str = f"{months} M"
        else:
            duration_str = f"{days_back} D"

        # 4. Thread-safe Request ID
        req_id = self.next_req_id
        self.next_req_id += 1

        # 5. Send Request
        self.reqHistoricalData(
            req_id,
            contract,
            query_time,  # End Date/Time
            duration_str,  # Duration
            bar_size,  # Bar size
            "TRADES",  # What to show
            1,  # Use RTH (Regular Trading Hours)
            1,  # Date Format (1 = string)
            False,  # Keep up to date
            []  # Options
        )

        # 6. Wait for Data (Timeout 60s)
        if self.data_event.wait(timeout=60):
            if self.error_occurred:
                self._log(f"IBKR returned error: {self.error_message}", "ERROR")
                return None

            if not self.historical_data:
                self._log("IBKR returned no data rows.", "WARNING")
                return None

            # 7. Convert to DataFrame
            df = pd.DataFrame(self.historical_data)

            # 8. Clean and Format
            if not df.empty:
                # Standardize Date Index
                # IBKR returns string dates, sometimes with timezone
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)

                # Ensure numeric types
                cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                for c in cols:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c])

                # Rename columns to lowercase to match system standard (open, high, etc.)
                df.columns = df.columns.str.lower()

            return df
        else:
            self._log("IBKR Request timed out (no response in 60s).", "ERROR")
            return None

    def _download_from_alpaca(self, symbol, days_back, interval, end_date):
        """
        Internal method to download data from Alpaca using StockHistoricalDataClient.
        This is much more reliable for historical data than yfinance.
        """
        if self.stock_client is None:
            self._log(f"Alpaca client not initialized. Cannot download {symbol} from Alpaca.", "ERROR")
            return pd.DataFrame()

        self._log(f"Downloading {symbol} from Alpaca (Interval: {interval})...")

        # --- Clean Symbol for Alpaca (Remove ^ for VIX/Indices) ---
        clean_symbol = symbol.lstrip('^')

        try:
            # 1. Define the time range
            end_date_dt = pd.to_datetime(end_date).tz_localize('UTC') if end_date else datetime.datetime.now(
                tz=datetime.timezone.utc)
            start_date_dt = end_date_dt - pd.Timedelta(days=days_back)

            # --- Robust TimeFrame Mapping ---
            # Handles different versions of alpaca-py SDK
            try:
                if interval in ['1d', '1 day']:
                    tf = TimeFrame.Day
                elif interval in ['1h', '1 hour']:
                    tf = TimeFrame.Hour
                elif interval in ['15m', '15 mins']:
                    if hasattr(TimeFrame, 'Minute15'):
                        tf = TimeFrame.Minute15
                    else:
                        tf = TimeFrame(15, TimeFrameUnit.Minute)
                elif interval in ['5m', '5 mins']:
                    tf = TimeFrame(5, TimeFrameUnit.Minute)
                else:
                    tf = TimeFrame.Day
            except Exception as e:
                # Fallback to Daily if anything goes wrong with TimeFrame construction
                self._log(f"Alpaca TimeFrame construction error: {e}. Defaulting to Daily.", "WARNING")
                tf = TimeFrame.Day

            request_params = StockBarsRequest(
                symbol_or_symbols=[clean_symbol],
                timeframe=tf,
                start=start_date_dt,
                end=end_date_dt,
                feed="iex"  # Use 'iex' for free tier, 'sip' for paid
            )

            # 2. Make the single API call
            data = self.stock_client.get_stock_bars(request_params)

            # 3. Convert to DataFrame and clean
            df_raw = data.df
            if df_raw.empty:
                self._log(f"Alpaca returned no data for {clean_symbol}.", "WARNING")
                return pd.DataFrame()

            # Alpaca returns a MultiIndex. We must flatten it.
            df_raw = df_raw.reset_index(level=0, drop=True)
            df_raw.index.name = 'Date'

            self._log(f"Successfully retrieved {len(df_raw)} bars for {clean_symbol} from Alpaca.", "SUCCESS")
            return df_raw  # Return raw data for cleaning outside

        except APIError as e:
            self._log(f"Alpaca download failed for {clean_symbol}: {e}", "WARNING")
            return pd.DataFrame()

    def get_stock_data(self, symbol: str, days_back: int = 1825, interval: str = '1 day', start_date=None,
                       end_date=None):
        """
        Fetches stock data with strict priority: IBKR -> Alpaca -> YFinance.
        """
        clean_symbol = symbol.upper().strip()

        # --- PRIORITY 1: IBKR ---
        if self.use_ibkr:
            # 1. Auto-Connect if not connected
            if not self.isConnected():
                self._log(f"IBKR enabled but not connected. Attempting auto-connect...", "WARNING")
                self.connect_to_ibkr()

            # 2. If Connected, try download
            if self.isConnected():
                self._log(f"Attempting IBKR download for {clean_symbol}...")

                # IBKR Interval Map
                ibkr_interval_map = {
                    '1d': '1 day', '15 mins': '15 mins', '5 mins': '5 mins',
                    '30 mins': '30 mins', '1 hour': '1 hour'
                }
                ibkr_bar_size = ibkr_interval_map.get(interval, interval)

                # Attempt Download
                df_raw = self._download_from_ibkr(clean_symbol, days_back, ibkr_bar_size,
                                                  start_date=start_date, end_date=end_date)

                if df_raw is not None and not df_raw.empty:
                    self._log(f"✅ Success: Data retrieved from IBKR.", "SUCCESS")
                    return clean_raw_data(df_raw)
                else:
                    self._log(f"⚠️ IBKR returned no data or failed. Proceeding to fallback.", "WARNING")
            else:
                self._log(f"⚠️ IBKR connection failed. Proceeding to fallback.", "WARNING")

        # --- Check if fallback is allowed ---
        if not self.allow_fallback and self.use_ibkr:
            self._log("Fallback is disabled. Returning empty DataFrame.", "ERROR")
            st.error(f"Data Error: IBKR failed and fallback is disabled.")
            return pd.DataFrame()

        # --- PRIORITY 2: ALPACA ---
        # Only runs if IBKR failed/disabled AND we have an Alpaca client
        if self.stock_client is not None:
            self._log(f"Attempting Alpaca fallback for {clean_symbol}...")
            df_raw = self._download_from_alpaca(
                clean_symbol, days_back=days_back, interval=interval, end_date=end_date
            )

            if df_raw is not None and not df_raw.empty:
                self._log(f"✅ Success: Data retrieved from Alpaca.", "SUCCESS")
                return clean_raw_data(df_raw)

        # --- PRIORITY 3: YFINANCE ---
        self._log(f"Attempting YFinance fallback for {clean_symbol}...", "WARNING")
        df_raw = self._download_from_yfinance(
            clean_symbol, days_back=days_back, interval=interval,
            start_date=start_date, end_date=end_date
        )

        if df_raw is not None and not df_raw.empty:
            self._log(f"✅ Success: Data retrieved from YFinance.", "SUCCESS")
            return clean_raw_data(df_raw)

        # --- FAILURE ---
        self._log(f"❌ CRITICAL: All data sources failed for {clean_symbol}.", "ERROR")
        return pd.DataFrame()

    def get_screener_data_bulk(self, symbols_list: list, days_back: int = 250, end_date=None):
        """
        Fetches EOD bars for a list of symbols in a single batch request.
        This is much faster for screening than one-by-one requests.
        """
        if not self.api_key or not self.api_secret:
            self._log("Alpaca API keys not found. Cannot perform bulk download.", "ERROR")
            return pd.DataFrame()

        self._log(f"Alpaca: Performing bulk data request for {len(symbols_list)} symbols...")

        try:
            # 1. Define the time range
            if end_date is None:
                end_date_dt = datetime.datetime.now(tz=datetime.timezone.utc)
            else:
                # Ensure the analysis_date is a timezone-aware datetime
                end_date_dt = pd.to_datetime(end_date).tz_localize('UTC')

            start_date_dt = end_date_dt - pd.Timedelta(days=days_back + 50)  # Get extra data for SMA warmup
            self._log(f"Alpaca: Bulk request from {start_date_dt.date()} to {end_date_dt.date()}", "DEBUG")

            # 2. Build the batch request
            request_params = StockBarsRequest(
                symbol_or_symbols=symbols_list,
                timeframe=TimeFrame.Day,
                start=start_date_dt,
                end=end_date_dt,
                feed="iex"
            )

            # 3. Make the single API call
            data = self.stock_client.get_stock_bars(request_params)

            # 4. Convert to DataFrame and return
            # The .df attribute converts the data into a multi-index DataFrame
            # with 'symbol' and 'timestamp' as the index.
            self._log(f"Alpaca: Successfully retrieved bulk data.", "SUCCESS")
            return data.df

        except APIError as e:
            self._log(f"Alpaca API Error during bulk request: {e}", "ERROR", exc_info=True)
            return pd.DataFrame()
        except Exception as e:
            self._log(f"Unexpected error during bulk request: {e}", "ERROR", exc_info=True)
            return pd.DataFrame()

    # def _download_from_yfinance(self, symbol, days_back, interval='1d', start_date=None, end_date=None,
    #                             retries=3, delay=5):
    #     """Internal method to download data from yfinance."""
    #     self._log(f"Downloading {symbol} from yfinance...")
    #
    #     # --- VERIFIED yfinance Interval Translation Map ---
    #     yfinance_interval_map = {
    #         '1 day': '1d',
    #         '15 mins': '15m',  # Short form with no space is correct for yfinance
    #         '5 mins': '5m',
    #         '30 mins': '30m',
    #         '1 hour': '1h'
    #     }
    #     yfinance_interval = yfinance_interval_map.get(interval, interval)
    #
    #     # --- Handle yfinance intraday limit ---
    #     is_intraday = 'm' in yfinance_interval or 'h' in yfinance_interval
    #     if is_intraday and days_back > 60:
    #         self._log(f"Request for {days_back} days of intraday data exceeds yfinance limit. Capping at 60 days.",
    #                   "WARNING")
    #         days_back = 60
    #
    #     for i in range(retries):
    #         try:
    #             # --- THIS IS THE FIX ---
    #             if end_date:
    #                 # Use start/end dates if end_date is provided
    #                 end_date_dt = pd.to_datetime(end_date)
    #                 # Use pd.Timedelta instead of the missing 'timedelta'
    #                 start_date_dt = end_date_dt - pd.Timedelta(days=days_back)
    #
    #                 # yfinance 'end' is exclusive, so add one day
    #                 df = yf.download(symbol, start=start_date_dt, end=(end_date_dt + pd.Timedelta(days=1)),
    #                                  interval=yfinance_interval, auto_adjust=False, progress=False)
    #             # --- END FIX ---
    #             elif start_date:
    #                 df = yf.download(symbol, start=start_date, end=end_date, interval=yfinance_interval,
    #                                  auto_adjust=False, progress=False)
    #             else:
    #                 # Original logic (for calls that don't need a specific end date)
    #                 df = yf.download(symbol, period=f"{days_back}d", interval=yfinance_interval, auto_adjust=False,
    #                                  progress=False)
    #
    #             if not df.empty:
    #                 self._log(f"Successfully retrieved {len(df)} bars for {symbol} from yfinance.", "SUCCESS")
    #                 return df
    #         except Exception as e:
    #             if "No data found for this period" in str(e) or "No objects to concatenate" in str(e):
    #                 self._log(
    #                     f"yfinance returned no data for {symbol}. It may be delisted or have no data for the period.",
    #                     "WARNING")
    #             else:
    #                 self._log(f"yfinance download attempt {i + 1}/{retries} for {symbol} failed: {e}", "WARNING",
    #                           exc_info=True)
    #                 time.sleep(delay)
    #
    #     self._log(f"All yfinance download attempts failed for {symbol}.", "ERROR")
    #     return pd.DataFrame()

    def _download_from_yfinance(self, symbol, days_back, interval='1d', start_date=None, end_date=None,
                                retries=3, delay=5):
        """Internal method to download data from yfinance."""
        self._log(f"Downloading {symbol} from yfinance...")

        # --- VERIFIED yfinance Interval Translation Map ---
        yfinance_interval_map = {
            '1 day': '1d',
            '15 mins': '15m',  # Short form with no space is correct for yfinance
            '5 mins': '5m',
            '30 mins': '30m',
            '1 hour': '1h'
        }
        yfinance_interval = yfinance_interval_map.get(interval, interval)

        # --- Handle yfinance intraday limit ---
        is_intraday = 'm' in yfinance_interval or 'h' in yfinance_interval
        if is_intraday and days_back > 60:
            self._log(f"Request for {days_back} days of intraday data exceeds yfinance limit. Capping at 60 days.",
                      "WARNING")
            days_back = 60

        for i in range(retries):
            try:
                if end_date:
                    # Use start/end dates if end_date is provided
                    end_date_dt = pd.to_datetime(end_date)
                    start_date_dt = end_date_dt - pd.Timedelta(days=days_back)

                    # yfinance 'end' is exclusive, so add one day
                    self._log(f"yfinance: Requesting {symbol} from {start_date_dt.date()} "
                              f"to {end_date_dt.date()}", "DEBUG")
                    df = yf.download(symbol, start=start_date_dt, end=(end_date_dt + pd.Timedelta(days=1)),
                                     interval=yfinance_interval, auto_adjust=False, progress=False)

                elif start_date:
                    # This is for a different use case, but good to keep
                    df = yf.download(symbol, start=start_date, end=end_date, interval=yfinance_interval,
                                     auto_adjust=False, progress=False)
                else:
                    # Original logic (for calls that don't need a specific end date, e.g., single stock analysis)
                    self._log(f"yfinance: Requesting {symbol} for last {days_back} days.", "DEBUG")
                    df = yf.download(symbol, period=f"{days_back}d", interval=yfinance_interval, auto_adjust=False,
                                     progress=False)

                if not df.empty:
                    self._log(f"Successfully retrieved {len(df)} bars for {symbol} from yfinance.", "SUCCESS")
                    return df
            except Exception as e:
                if "No data found for this period" in str(e) or "No objects to concatenate" in str(e):
                    self._log(
                        f"yfinance returned no data for {symbol}. It may be delisted or have no data for the period.",
                        "WARNING")
                else:
                    self._log(f"yfinance download attempt {i + 1}/{retries} for {symbol} failed: {e}", "WARNING",
                              exc_info=True)
                    time.sleep(delay)

        self._log(f"All yfinance download attempts failed for {symbol}.", "ERROR")
        return pd.DataFrame()

    def historicalData(self, reqId, bar):
        """EWrapper callback that receives each bar of historical data."""
        self.historical_data.append({
            'Date': bar.date,
            'Open': bar.open,
            'High': bar.high,
            'Low': bar.low,
            'Close': bar.close,
            'Volume': bar.volume
        })

    def historicalDataEnd(self, reqId, start, end):
        """EWrapper callback that signals the end of a historical data request."""
        super().historicalDataEnd(reqId, start, end)
        self.data_event.set()


def normalize_and_validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and validates a DataFrame from an external source (like yfinance).

    - Handles empty or None DataFrames.
    - Flattens multi-level column indexes.
    - Converts all column names to lowercase.
    - Validates that essential columns are present.

    Returns a clean, standardized DataFrame or an empty one if input is invalid.
    """
    # 1. Handle empty or None DataFrame gracefully
    if df is None or df.empty:
        return pd.DataFrame()

    # 2. Flatten multi-level column index if present (a common yfinance format)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    # 3. Standardize all column names to lowercase to prevent KeyErrors
    df.columns = df.columns.str.lower()

    # 4. (Best Practice) Validate that the essential columns now exist
    required_columns = {'open', 'high', 'low', 'close', 'volume'}
    if not required_columns.issubset(df.columns):
        # Log a warning or raise an error if critical data is missing
        print(f"Data validation failed. Missing columns: {required_columns - set(df.columns)}")
        return pd.DataFrame()  # Return empty frame if validation fails

    return df
