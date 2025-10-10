# """
# ⚙️ Data Source Manager - Client Portal Web API Version
# ======================================================
#
# A shared utility for fetching historical stock data, now configured to use the
# Interactive Brokers Client Portal Web API as the primary source.
# It still falls back to yfinance if IBKR is unavailable.
#
# """


import pandas as pd
import yfinance as yf
import logging
import time
import threading
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import datetime
import os
import traceback


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

    def __init__(self, use_ibkr=True, host='127.0.0.1', port=7497):
        EClient.__init__(self, self)
        self.use_ibkr = use_ibkr
        self.host = host
        self.port = port

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

    def get_stock_data(self, symbol: str, days_back=1825, interval='1 day'):
        clean_symbol = symbol.upper().strip()
        if self.use_ibkr:
            if not self.isConnected() and not self.connect_to_ibkr():
                self._log("IBKR connection failed, falling back to yfinance.", "WARNING")
            else:
                # --- VERIFIED IBKR Interval Translation Map ---
                ibkr_interval_map = {
                    '1d': '1 day',
                    '15 mins': '15 mins',
                    '5 mins': '5 mins',
                    '30 mins': '30 mins',
                    '1 hour': '1 hour'
                }
                ibkr_bar_size = ibkr_interval_map.get(interval, interval)

                df = self._download_from_ibkr(clean_symbol, days_back, ibkr_bar_size)
                if not df.empty: return df
                self._log(f"IBKR data retrieval failed for {clean_symbol}, falling back to yfinance.", "WARNING")

        return self._download_from_yfinance(clean_symbol, days_back, interval)

    def _download_from_ibkr(self, symbol: str, days_back: int, barSizeSetting: str = "1 day"):
        """
        Final robust version for downloading data from TWS using ibapi.
        Handles chunking for large intraday requests and gracefully manages chunks with no data.
        """
        self._log(f"Requesting {days_back} days of '{barSizeSetting}' data for {symbol} from TWS...")
        try:
            contract = Contract()
            if symbol == '^VIX':
                contract.symbol = 'VIX';
                contract.secType = 'IND';
                contract.exchange = 'CBOE';
                contract.currency = 'USD'
            else:
                contract.symbol = symbol;
                contract.secType = 'STK';
                contract.exchange = 'SMART';
                contract.currency = 'USD'

            is_intraday_request = "min" in barSizeSetting or "hour" in barSizeSetting

            if is_intraday_request and days_back > 30:
                all_bars = []
                any_chunk_succeeded = False  # Flag to track if we get any data at all
                # Use 'ME' for month-end frequency to avoid FutureWarning
                date_range = pd.date_range(end=datetime.datetime.now(), periods=round(days_back / 30), freq='-1ME')

                self._log(f"Large intraday request detected. Fetching in {len(date_range)} monthly chunks...")

                for end_date in reversed(date_range):
                    self.historical_data = []
                    self.data_event.clear()
                    self.error_occurred = False
                    req_id = self.next_req_id
                    self.next_req_id += 1

                    end_date_str = end_date.strftime('%Y%m%d %H:%M:%S') + " US/Eastern"
                    self._log(f"Fetching chunk for {symbol} ending {end_date_str}...")

                    self.reqHistoricalData(
                        reqId=req_id, contract=contract, endDateTime=end_date_str,
                        durationStr="1 M", barSizeSetting=barSizeSetting, whatToShow="TRADES",
                        useRTH=1, formatDate=1, keepUpToDate=False, chartOptions=[]
                    )

                    if self.data_event.wait(timeout=20):
                        if not self.error_occurred and self.historical_data:
                            all_bars.extend(self.historical_data)
                            any_chunk_succeeded = True  # Mark that we found at least one good chunk
                    else:
                        self._log(f"Data chunk request for {symbol} timed out.", "WARNING")
                        self.cancelHistoricalData(req_id)

                    time.sleep(2.1)

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

                self.reqHistoricalData(
                    reqId=req_id, contract=contract, endDateTime="",
                    durationStr=duration, barSizeSetting=barSizeSetting, whatToShow="TRADES",
                    useRTH=1, formatDate=1, keepUpToDate=False, chartOptions=[]
                )

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

            try:
                df['Date'] = pd.to_datetime(pd.to_numeric(df['Date']), unit='s')
            except (ValueError, TypeError):
                df['Date'] = pd.to_datetime(df['Date'].str.split(' ').str[:2].str.join(' '), format='%Y%m%d %H:%M:%S')

            df.set_index('Date', inplace=True)
            self._log(f"Successfully retrieved {len(df)} bars for {symbol} from IBKR.", "SUCCESS")
            return df

        except Exception as e:
            error_message = f"Exception during IBKR download for {symbol}: {e}\n{traceback.format_exc()}"
            self._log(error_message, "ERROR")
            return pd.DataFrame()

    def _download_from_yfinance(self, symbol, days_back, interval='1d', retries=3, delay=5):
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
#
# import pandas as pd
# import yfinance as yf
# import logging
# import time
# import threading
# from ibapi.client import EClient
# from ibapi.wrapper import EWrapper
# from ibapi.contract import Contract
# import datetime
# import os
#
#
# class Stock(Contract):
#     """A simplified Contract class for requesting stock data."""
#
#     def __init__(self, symbol, exchange='SMART', currency='USD'):
#         Contract.__init__(self)
#         self.symbol = symbol
#         self.secType = 'STK'
#         self.exchange = exchange
#         self.currency = currency
#
#
# class DataSourceManager(EWrapper, EClient):
#     """
#     Manages data retrieval using the direct TWS API (ibapi),
#     with a fallback to yfinance. This version uses the proven connection logic.
#     """
#
#     def __init__(self, use_ibkr=True, host='127.0.0.1', port=7497, client_id=None):
#         EClient.__init__(self, self)
#         self.use_ibkr = use_ibkr
#         self.host = host
#         self.port = port
#         # self.client_id = client_id
#
#         if client_id is None:
#             self.client_id = (os.getpid() % 100) * 100 + int(time.time() % 100)
#         else:
#             self.client_id = client_id
#
#         # # Use a unique, random client ID for each instance to prevent connection conflicts.
#         # self.client_id = client_id if client_id is not None else int(time.time() % 1000)
#
#         logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | (%(name)s) %(message)s')
#         self.logger = logging.getLogger("DataSourceManager")
#
#         # Threading events for managing async API calls
#         self.connection_event = threading.Event()
#         self.data_event = threading.Event()
#         self.error_occurred = False
#         self.error_message = ""
#         self.historical_data = []
#         self.next_req_id = int(time.time())
#
#     def log(self, message, level="INFO"):
#         """Logs a message to the console."""
#         if level.upper() == "INFO":
#             self.logger.info(message)
#         elif level.upper() == "WARNING":
#             self.logger.warning(message)
#         elif level.upper() == "ERROR":
#             self.logger.error(message)
#         elif level.upper() == "SUCCESS":
#             self.logger.info(f"✅ {message}")
#
#     def connect_to_ibkr(self):
#         """Connects to a running TWS or Gateway instance using ibapi."""
#         if self.isConnected():
#             return True
#         self.log(f"Connecting to TWS on {self.host}:{self.port}...")
#         try:
#             self.connect(self.host, self.port, self.client_id)
#             api_thread = threading.Thread(target=self.run, daemon=True)
#             api_thread.start()
#
#             self.log("Waiting for connection handshake...")
#             if self.connection_event.wait(timeout=20):
#                 self.log("Connection to TWS is active.", "SUCCESS")
#                 return True
#             else:
#                 self.log("Connection to TWS timed out.", "ERROR")
#                 self.disconnect()
#                 return False
#         except Exception as e:
#             self.log(f"Could not connect to TWS: {e}", "ERROR")
#             return False
#
#     def nextValidId(self, orderId: int):
#         """EWrapper callback that confirms the connection is ready."""
#         super().nextValidId(orderId)
#         self.log("Connection handshake confirmed by TWS.")
#         self.connection_event.set()
#
#     def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
#         """EWrapper callback for handling errors."""
#         super().error(reqId, errorCode, errorString, advancedOrderRejectJson)
#         if errorCode < 2000 and errorCode not in [2104, 2106, 2158]:
#             self.log(f"TWS API Error {errorCode}: {errorString}", "ERROR")
#             self.error_occurred = True
#             self.error_message = errorString
#             self.data_event.set()
#
#     def disconnect(self):
#         """Disconnects from TWS."""
#         if self.isConnected():
#             self.log("Disconnecting from TWS...")
#             super().disconnect()
#
#     def get_stock_data(self, symbol: str, days_back=1825, interval="1 day"):
#         """Main method to get data, prioritizing IBKR."""
#         if self.use_ibkr:
#             if not self.isConnected():
#                 self.connect_to_ibkr()
#
#             if self.isConnected():
#                 df = self._download_from_ibkr(symbol, days_back, interval)
#                 if not df.empty:
#                     return df
#                 else:
#                     self.log(f"IBKR data retrieval failed for {symbol}, falling back to yfinance.", "WARNING")
#         # Note: yfinance fallback for long-term intraday is limited.
#         return self._download_from_yfinance(symbol, days_back)
#
#     def historicalData(self, reqId, bar):
#         """EWrapper callback that receives each bar of historical data."""
#         self.historical_data.append({
#             'Date': bar.date, 'Open': bar.open, 'High': bar.high,
#             'Low': bar.low, 'Close': bar.close, 'Volume': bar.volume
#         })
#
#     def historicalDataEnd(self, reqId, start, end):
#         """EWrapper callback that signals the end of a historical data request."""
#         super().historicalDataEnd(reqId, start, end)
#         self.data_event.set()
#
#     def _download_from_ibkr(self, symbol: str, days_back: int, barSizeSetting: str):
#         """Internal method for downloading data from TWS using ibapi."""
#         # print("--- DEBUG:RUNNING v2 of _download_from_ibkr ---")
#         self.log(f"Requesting historical data for {symbol} from TWS...")
#         try:
#             # Create a specific contract for the VIX Index, and a general one for stocks/ETFs.
#             if symbol == '^VIX':
#                 contract = Contract()
#                 contract.symbol = 'VIX'
#                 contract.secType = 'IND'
#                 contract.exchange = 'CBOE'
#                 contract.currency = 'USD'
#             else:
#                 contract = Stock(symbol, 'SMART', 'USD')
#
#             if days_back > 365:
#                 duration = f'{round(days_back / 365)} Y'
#             else:
#                 duration = f'{days_back} D'
#             self.log(f"Requesting duration: {duration}")
#
#             self.historical_data = []
#             self.data_event.clear()
#             self.error_occurred = False
#
#             req_id = self.next_req_id
#             self.next_req_id += 1
#
#             self.reqHistoricalData(
#                 reqId=req_id, contract=contract, endDateTime="",
#                 durationStr=duration, barSizeSetting=barSizeSetting, whatToShow="TRADES",
#                 useRTH=1, formatDate=2, keepUpToDate=False, chartOptions=[]
#             )
#
#             if self.data_event.wait(timeout=300):
#                 if self.error_occurred:
#                     self.log(f"Failed to get data for {symbol}: {self.error_message}", "ERROR")
#                     return pd.DataFrame()
#
#                 df = pd.DataFrame(self.historical_data)
#                 # Replace the entire if/else block with this single line
#                 df['Date'] = pd.to_datetime(pd.to_numeric(df['Date']), unit='s')
#
#                 df.set_index('Date', inplace=True)
#                 self.log(f"Successfully retrieved {len(df)} bars for {symbol} from IBKR.", "SUCCESS")
#                 return df
#             else:
#                 self.log(f"Data request for {symbol} timed out.", "ERROR")
#                 self.cancelHistoricalData(req_id)
#                 return pd.DataFrame()
#         except Exception as e:
#             self.log(f"Exception during IBKR download for {symbol}: {e}", "ERROR")
#             return pd.DataFrame()
#
#     def _download_from_yfinance(self, symbol, days_back, retries=3, delay=5):
#         """Internal method to download data from yfinance."""
#         self.log(f"Downloading {symbol} from yfinance...")
#         for i in range(retries):
#             try:
#                 df = yf.download(symbol, period=f"{days_back}d", auto_adjust=True, progress=False)
#                 if not df.empty:
#                     self.log(f"Successfully retrieved {len(df)} bars for {symbol} from yfinance.", "SUCCESS")
#                     return df
#             except Exception as e:
#                 self.log(f"yfinance download attempt {i+1}/{retries} for {symbol} failed: {e}", "WARNING")
#                 time.sleep(delay)
#         self.log(f"All yfinance download attempts failed for {symbol}.", "ERROR")
#         return pd.DataFrame()