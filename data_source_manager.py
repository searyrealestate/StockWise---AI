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
try:
    # Try to import the real ibapi
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper
    from ibapi.contract import Contract
except ImportError:
    # If it fails (like on Streamlit Cloud), create dummy classes
    # This stops the app from crashing
    class EClient:
        pass
    class EWrapper:
        pass
    class Contract:
        pass
import datetime
import os
import traceback
from utils import clean_raw_data
import streamlit as st


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
        EClient.__init__(self, self)
        self.use_ibkr = use_ibkr
        self.host = host
        self.port = port
        self.allow_fallback = allow_fallback

        with DataSourceManager._client_id_lock:
            self.client_id = DataSourceManager._client_id_counter
            DataSourceManager._client_id_counter += 1

        # --- Correct Logger Setup ---
        self.logger = logging.getLogger(f"Client_{self.client_id}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False # Prevents duplicate messages in parent loggers

        if not self.logger.handlers:
            # Formatter for both console and file
            formatter = logging.Formatter('%(asctime)s | %(levelname)s | (Client ID: %(client_id)s) %(message)s')
            # 1. File Handler: Saves logs to a unique file for each client ID
            log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"data_manager_client_{self.client_id}.log")
            fh = logging.FileHandler(log_file, encoding='utf-8')
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

            # 2. Console Handler: Prints logs to the console
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

        self.connection_event = threading.Event()
        self.data_event = threading.Event()
        self.error_occurred = False
        self.error_message = ""
        self.historical_data = []
        self.next_req_id = int(time.time())

    def _log(self, message, level="INFO"):
        """Internal helper for logging with client ID."""
        extra = {'client_id': self.client_id}
        if level.upper() == "INFO":
            self.logger.info(message, extra=extra)
        elif level.upper() == "WARNING":
            self.logger.warning(message, extra=extra)
        elif level.upper() == "ERROR":
            self.logger.error(message, extra=extra)
        elif level.upper() == "SUCCESS":
            self.logger.info(f"{message}", extra=extra)

    def set_socket_logging(self, enabled=True):
        """Enables or disables low-level IBAPI socket logging."""
        self._log(f"--- TWS Socket-level message logging {'ENABLED' if enabled else 'DISABLED'} ---", "WARNING")
        # EClient.set_socket_logging(self, enabled) # This method does not exist in the base class

    def connect_to_ibkr(self):
        """Establishes a connection to the TWS/Gateway API."""
        if self.isConnected(): return True
        self._log(f"Connecting to TWS on {self.host}:{self.port}...")
        try:
            self.connect(self.host, self.port, self.client_id)
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
            self._log(f"Could not connect to TWS: {e}", "ERROR")
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

    def get_stock_data(self, symbol: str, days_back: int = 1825, interval: str = '1 day', start_date=None,
                       end_date=None):

        # --- Master Retry Loop as requested by user ---
        max_attempts = 10
        pause_duration = 5  # 5 seconds

        for attempt in range(1, max_attempts + 1):
            # We log the attempt number
            self._log(f"Data attempt {attempt}/{max_attempts} for {symbol}...")

            clean_symbol = symbol.upper().strip()
            df_raw = None

            if self.use_ibkr and self.isConnected():
                    # --- VERIFIED IBKR Interval Translation Map ---
                    ibkr_interval_map = {
                        '1d': '1 day',
                        '15 mins': '15 mins',
                        '5 mins': '5 mins',
                        '30 mins': '30 mins',
                        '1 hour': '1 hour'
                    }
                    ibkr_bar_size = ibkr_interval_map.get(interval, interval)
                    df_raw = self._download_from_ibkr(clean_symbol, days_back, ibkr_bar_size, start_date, end_date)

                    # --- 1. Success on IBKR ---
                    if df_raw is not None and not df_raw.empty:
                        self._log(f"Successfully got data for {symbol} from IBKR on attempt {attempt}.", "SUCCESS")
                        return clean_raw_data(df_raw)  # <-- Successful exit

                    # --- 2. IBKR Failed, Fallback Allowed ---
                    if self.allow_fallback:
                        self._log(f"IBKR data retrieval failed for {clean_symbol}, falling back to yfinance.", "WARNING")

                    # --- 3. IBKR Failed, Fallback NOT Allowed ---
                    else:
                        self._log(
                            f"IBKR failed for {clean_symbol} on attempt {attempt}. Fallback disabled. Retrying...",
                            "ERROR")
                        time.sleep(pause_duration)
                        continue  # <-- This is a retry on IBKR fail

            # --- 4. IBKR Not Connected, Fallback NOT Allowed ---
            elif self.use_ibkr and not self.isConnected():
                if not self.allow_fallback:
                    self._log(f"IBKR not connected (Attempt {attempt}). Fallback disabled. Retrying...",
                              "ERROR")
                    time.sleep(pause_duration)
                    continue  # <-- This is a retry on IBKR connection fail

                self._log(f"IBKR not connected (Attempt {attempt}). "
                          f"Falling back directly to yfinance.", "WARNING")

            # --- 5. Use yfinance (either as primary or fallback) ---
            if self.allow_fallback or not self.use_ibkr:
                df_raw = self._download_from_yfinance(
                    clean_symbol, days_back=days_back, interval=interval,
                    start_date=start_date, end_date=end_date
                )

                # --- 6. Success on yfinance ---
                if df_raw is not None and not df_raw.empty:
                    self._log(f"Successfully got data for {symbol} via yfinance on attempt {attempt}.", "SUCCESS")
                    return clean_raw_data(df_raw)  # <-- Successful exit

            # --- 7. All attempts failed for this loop ---
            self._log(
                f"Data attempt {attempt}/{max_attempts} for {symbol} failed. Retrying in {pause_duration} sec...",
                "WARNING")
            time.sleep(pause_duration)

        # --- If loop finishes, all 10 attempts failed ---
        self._log(f"CRITICAL: All {max_attempts} data attempts for {symbol} failed. Returning empty DataFrame.",
                  "ERROR")

        # This is the alert to the user you requested
        st.error(
            f"Network Failure: Could not download data for {symbol} after {max_attempts} attempts. The script will skip this stock.")

        return pd.DataFrame()

    def _download_from_ibkr(self, symbol: str, days_back: int, barSizeSetting: str = "1 day",
                            start_date=None, end_date=None):
        """
        Final robust version for downloading data from TWS using ibapi.
        Handles chunking for large intraday requests and gracefully manages chunks with no data.
        """
        self._log(f"Requesting {days_back} days of '{barSizeSetting}' data for {symbol} from TWS...")
        try:
            contract = Contract()
            if symbol == '^VIX':
                contract.symbol = 'VIX'
                contract.secType = 'IND'
                contract.exchange = 'CBOE'
                contract.currency = 'USD'
            else:
                contract.symbol = symbol
                contract.secType = 'STK'
                contract.exchange = 'SMART'
                contract.currency = 'USD'

            is_intraday_request = "min" in barSizeSetting or "hour" in barSizeSetting

            if is_intraday_request and days_back > 30:
                all_bars = []
                any_chunk_succeeded = False
                # Use 'ME' for month-end frequency to avoid FutureWarning
                date_range = pd.date_range(end=datetime.datetime.now(), periods=round(days_back / 30), freq='-1ME')
                self._log(f"Large intraday request detected. Fetching in {len(date_range)} monthly chunks...")

                for end_date in reversed(date_range):
                    self.historical_data = []
                    self.data_event.clear()
                    self.error_occurred = False
                    req_id = self.next_req_id
                    self.next_req_id += 1
                    # FIX: Explicitly add the required timezone to the request string
                    end_date_str = end_date.strftime('%Y%m%d %H:%M:%S') + " US/Eastern"
                    self._log(f"Fetching chunk for {symbol} ending {end_date_str}...")

                    self.reqHistoricalData(reqId=req_id, contract=contract, endDateTime=end_date_str, durationStr="1 M",
                                           barSizeSetting=barSizeSetting, whatToShow="TRADES", useRTH=1, formatDate=1,
                                           keepUpToDate=False, chartOptions=[])

                    if self.data_event.wait(timeout=20):
                        if not self.error_occurred and self.historical_data:
                            all_bars.extend(self.historical_data)
                            any_chunk_succeeded = True
                    else:
                        self._log(f"Data chunk request for {symbol} timed out.", "WARNING")
                        self.cancelHistoricalData(req_id)
                    time.sleep(2.1)  # Adhere to API pacing requirements

                if not any_chunk_succeeded:
                    self._log(f"IBKR returned no data for {symbol} across all chunks.", "WARNING")
                    return pd.DataFrame()
                self.historical_data = all_bars
            else:
                duration = f'{round(days_back / 365)} Y' if days_back > 365 else f'{days_back} D'
                self.historical_data = []
                self.data_event.clear()
                self.error_occurred = False
                req_id = self.next_req_id
                self.next_req_id += 1

                self.reqHistoricalData(reqId=req_id, contract=contract, endDateTime="", durationStr=duration,
                                       barSizeSetting=barSizeSetting, whatToShow="TRADES", useRTH=1, formatDate=1,
                                       keepUpToDate=False, chartOptions=[])

                if not self.data_event.wait(timeout=60):
                    self._log(f"Data request for {symbol} timed out.", "ERROR")
                    self.cancelHistoricalData(req_id)
                    return pd.DataFrame()

            if self.error_occurred:
                self._log(f"Failed to get data for {symbol}: {self.error_message}", "ERROR")
                return pd.DataFrame()

            if not self.historical_data:
                self._log(f"No historical data received for {symbol} from IBKR.", "WARNING")
                return pd.DataFrame()

            df = pd.DataFrame(self.historical_data)
            df.drop_duplicates(subset=['Date'], inplace=True)
            # FIX: Correctly parse the Unix timestamp returned by the API
            df['Date'] = pd.to_datetime(pd.to_numeric(df['Date']), unit='s')
            df.set_index('Date', inplace=True)
            self._log(f"Successfully retrieved {len(df)} bars for {symbol} from IBKR.", "SUCCESS")
            return df
        except Exception as e:
            # FIX: Ensure the full traceback is logged for easier debugging
            error_message = f"Exception during IBKR download for {symbol}: {e}\n{traceback.format_exc()}"
            self._log(error_message, "ERROR")
            return pd.DataFrame()

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
                if start_date:
                    df = yf.download(symbol, start=start_date, end=end_date, interval=yfinance_interval,
                                     auto_adjust=True, progress=False)
                else:
                    df = yf.download(symbol, period=f"{days_back}d", interval=yfinance_interval, auto_adjust=True,
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
                    self._log(f"yfinance download attempt {i + 1}/{retries} for {symbol} failed: {e}", "WARNING")
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
