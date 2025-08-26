"""
⚙️ Data Source Manager
===================================================

A shared utility for fetching historical stock data, supporting IBKR (Interactive Brokers)
as the primary source and falling back to yfinance if IBKR is unavailable or fails.
This module centralizes data retrieval logic for use across multiple scripts.

"""

import pandas as pd
import numpy as np
import time
import threading
import os
import sys
import logging
from datetime import datetime, timedelta
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import yfinance as yf
import json

# Setup logging for this specific module
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suppress noisy loggers from imported libraries
logging.getLogger('yfinance').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)


class DataSourceManager(EWrapper, EClient):
    """
    Manages connections and data retrieval from IBKR or yfinance.
    Inherits from IBAPI's EWrapper and EClient for IBKR connectivity.
    """

    def __init__(self, use_ibkr=True, ibkr_host="127.0.0.1", ibkr_port=7497, debug=False,
                 yfinance_max_retries=10, yfinance_retry_delay=1):
        """
        Initializes the DataSourceManager.

        Args:
            use_ibkr (bool): If True, attempts to use IBKR as the primary data source.
            ibkr_host (str): The host IP address for the IBKR TWS or Gateway.
            ibkr_port (int): The port number for the IBKR TWS or Gateway.
            debug (bool): If True, enables more verbose logging.
            yfinance_max_retries (int): The number of times to retry fetching data from yfinance upon failure.
            yfinance_retry_delay (int): The delay in seconds between yfinance retry attempts.
        """

        EClient.__init__(self, self)
        self.use_ibkr = use_ibkr
        self.ibkr_host = ibkr_host
        self.ibkr_port = ibkr_port
        self.debug = debug
        self.yfinance_max_retries = yfinance_max_retries
        self.yfinance_retry_delay = yfinance_retry_delay

        # Synchronization and State
        self.data_ready = threading.Event()
        self.connection_ready = threading.Event()
        self.error_occurred = False
        self.error_message = ""
        self.ibkr_connected = False
        self.thread = None

        # Data storage
        self.historical_data = {}
        self.req_id = 3000

        if self.use_ibkr:
            self.connect_to_ibkr()

    def log(self, message, level="INFO"):
        """Logs a message to the console."""
        if level == "INFO":
            logger.info(message)
        elif level == "SUCCESS":
            logger.info(f"✅ {message}")
        elif level == "ERROR":
            logger.error(f"❌ {message}")
        elif level == "WARNING":
            logger.warning(f"⚠️ {message}")
        else:
            logger.debug(message)

    def connect_to_ibkr(self):
        """Establishes a connection to the Interactive Brokers TWS or Gateway."""
        if self.ibkr_connected:
            return True
        try:
            self.log(f"Connecting to IBKR TWS ({self.ibkr_host}:{self.ibkr_port})...")
            self.connect(self.ibkr_host, self.ibkr_port, clientId=1)

            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()

            if self.connection_ready.wait(timeout=20):
                self.log("Connected to IBKR successfully", "SUCCESS")
                self.ibkr_connected = True
                return True
            else:
                self.log("IBKR connection timeout.", "ERROR")
                self.ibkr_connected = False
                return False

        except Exception as e:
            self.log(f"IBKR connection error: {e}", "ERROR")
            self.ibkr_connected = False
            return False

    def disconnect(self):
        """Cleanly disconnects from the Interactive Brokers TWS or Gateway."""
        if self.ibkr_connected:
            self.log("Disconnecting from IBKR...", "INFO")
            super().disconnect()
            self.ibkr_connected = False
            self.log("Disconnected from IBKR.", "INFO")

    def get_stock_data(self, symbol: str) -> pd.DataFrame:
        """
        Fetches historical stock data for a given symbol.
        This now fetches 5 years of data.
        """
        df = pd.DataFrame()

        # Define the number of days for 5 years
        days_back = 5 * 365

        if self.use_ibkr and self.connect_to_ibkr():
            df = self._download_from_ibkr(symbol)
            if not df.empty:
                self.log(f"Successfully retrieved {len(df)} bars for {symbol} from IBKR.", "SUCCESS")
                return df
            else:
                self.log(f"Failed to get data for {symbol} from IBKR after retries. Trying yfinance...", "WARNING")

        # Fallback to yfinance if IBKR is disabled, fails to connect, or returns no data
        df = self._download_from_yfinance(symbol, days_back=days_back)
        if not df.empty:
            self.log(f"Successfully retrieved {len(df)} bars for {symbol} from yfinance.", "SUCCESS")
        else:
            self.log(f"Failed to get data for {symbol} from yfinance after all retries.", "ERROR")

        return df

    def _download_from_ibkr(self, symbol, retries=3):
        """Internal method to download data from IBKR with a defensive retry mechanism."""
        for attempt in range(retries):
            try:
                self.log(f"Downloading {symbol} from IBKR (attempt {attempt + 1}/{retries})", "INFO")
                time.sleep(2)  # Pacing request

                contract = Contract()
                contract.symbol = symbol
                contract.secType = "STK"
                contract.exchange = "SMART"
                contract.currency = "USD"

                self.req_id += 1
                req_id = self.req_id

                self.historical_data[req_id] = []
                self.data_ready.clear()
                self.error_occurred = False

                end_date_time = datetime.now().strftime("%Y%m%d 17:00:00 US/Eastern")

                # --- CHANGE HERE: Updated from "2 Y" to "5 Y" ---
                self.reqHistoricalData(req_id, contract, end_date_time, "5 Y", "1 day", "TRADES", 1, 1, False, [])

                if self.data_ready.wait(timeout=60):
                    if self.error_occurred:
                        self.log(f"IBKR error for {symbol}: {self.error_message}", "ERROR")
                        continue  # Go to next retry attempt

                    data = self.historical_data.get(req_id, [])
                    if data and len(data) > 252:  # Check for at least a year of data
                        rows = [{'Date': pd.to_datetime(bar.date), 'Open': float(bar.open), 'High': float(bar.high),
                                 'Low': float(bar.low), 'Close': float(bar.close), 'Volume': int(bar.volume)} for bar in
                                data]
                        df = pd.DataFrame(rows).set_index('Date').sort_index()
                        return df
                    else:
                        self.log(f"IBKR: Insufficient data for {symbol} ({len(data)} bars).", "WARNING")
                        return pd.DataFrame()  # Return empty df if insufficient, no need to retry
                else:
                    self.log(f"IBKR: Timeout for {symbol}. Connection may be stalled.", "WARNING")
                    self.log("Attempting to reset the connection...", "INFO")
                    self.disconnect()
                    time.sleep(3)  # Give sockets a moment to close
                    if not self.connect_to_ibkr():
                        self.log("Failed to re-establish IBKR connection. Aborting IBKR for this stock.", "ERROR")
                        return pd.DataFrame()  # Give up on IBKR for this specific stock

            except Exception as e:
                self.log(f"IBKR: Exception for {symbol}: {e}", "ERROR")

        return pd.DataFrame()

    def _download_from_yfinance(self, symbol, days_back):
        """Internal method to download data from yfinance with a robust retry mechanism."""
        for attempt in range(self.yfinance_max_retries):
            try:
                self.log(f"Downloading {symbol} from yfinance (attempt {attempt + 1}/{self.yfinance_max_retries})",
                         "INFO")
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days_back)
                data = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)

                if not data.empty:
                    data.index.name = 'Date'
                    return data[['Open', 'High', 'Low', 'Close', 'Volume']]
                else:
                    self.log(f"yfinance: No data found for {symbol}.", "WARNING")
                    return pd.DataFrame()  # No need to retry if yfinance returns empty
            except Exception as e:
                self.log(f"yfinance: Error downloading {symbol}: {e}", "ERROR")

            if attempt < self.yfinance_max_retries - 1:
                time.sleep(self.yfinance_retry_delay)

        return pd.DataFrame()

    # --- IBKR Callback Methods ---
    def nextValidId(self, orderId):
        """IBKR callback that confirms a successful connection."""
        self.connection_ready.set()

    def historicalData(self, reqId, bar):
        """IBKR callback that receives one bar of historical data at a time."""
        if reqId in self.historical_data:
            self.historical_data[reqId].append(bar)

    def historicalDataEnd(self, reqId, start, end):
        """IBKR callback that signals the end of a historical data request."""
        self.data_ready.set()

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        """IBKR callback that handles errors and informational messages."""
        if errorCode in [2104, 2106, 2158]:  # Ignore market data farm connection messages
            return

        self.error_occurred = True
        self.error_message = f"Error {errorCode}: {errorString}"
        if reqId != -1:  # Unblock the wait if it's a request-specific error
            self.data_ready.set()

