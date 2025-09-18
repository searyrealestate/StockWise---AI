"""
⚙️ Data Source Manager - Client Portal Web API Version
======================================================

A shared utility for fetching historical stock data, now configured to use the
Interactive Brokers Client Portal Web API as the primary source.
It still falls back to yfinance if IBKR is unavailable.

"""

import pandas as pd
import numpy as np
import time
import threading
import os
import sys
import logging
from datetime import datetime, timedelta
import requests
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
logger = logging.getLogger("DataSourceManager")

# Suppress noisy loggers from imported libraries
logging.getLogger('yfinance').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)

# Constants for Client Portal API
# The correct host and API endpoints for the Client Portal API
# Constants for Client Portal API
# CORRECTED: Use the local endpoint for the Client Portal Gateway.
CLIENT_PORTAL_HOST = "https://localhost:7497"
SESSION_STATUS_URL = f"{CLIENT_PORTAL_HOST}/v1/api/iserver/auth/status"
HISTORICAL_DATA_URL = f"{CLIENT_PORTAL_HOST}/iserver/marketdata/history"


class DataSourceManager:
    """
    Manages connections and data retrieval from IBKR Client Portal API or yfinance.
    """

    def __init__(self, use_ibkr=True, debug=False,
                 yfinance_max_retries=10, yfinance_retry_delay=1):
        """
        Initializes the DataSourceManager.

        Args:
            use_ibkr (bool): If True, attempts to use IBKR as the primary data source.
            debug (bool): If True, enables more verbose logging.
            yfinance_max_retries (int): The number of times to retry fetching data from yfinance upon failure.
            yfinance_retry_delay (int): The delay in seconds between yfinance retry attempts.
        """
        self.use_ibkr = use_ibkr
        self.debug = debug
        self.yfinance_max_retries = yfinance_max_retries
        self.yfinance_retry_delay = yfinance_retry_delay

        # Session for the Client Portal API
        self.session = requests.Session()
        self.ibkr_connected = False

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
        """Checks the session status for the Client Portal API."""
        if self.ibkr_connected:
            return True

        self.log("Checking IBKR Client Portal session status...")
        try:
            response = self.session.get(SESSION_STATUS_URL, verify=True, timeout=20)

            # Print the raw response text for debugging
            print("Raw response from IBKR auth status check:")
            print(response.text)

            # Try to parse the JSON response
            try:
                status = response.json()
            except ValueError:
                self.log("IBKR Client Portal connection error: Failed to decode JSON.", "ERROR")
                self.ibkr_connected = False
                return False

            if status.get("authenticated") and status.get("connected"):

                self.log("IBKR Client Portal session is active and connected.", "SUCCESS")
                self.ibkr_connected = True
                return True
            else:
                # Print the raw response text for debugging
                print("Raw response from IBKR auth status check:")
                print(response.text)

                self.log(f"IBKR Client Portal session not authenticated or connected. Status: {status}", "ERROR")
                self.ibkr_connected = False
                return False
        except requests.exceptions.RequestException as e:
            self.log(f"IBKR Client Portal connection error: {e}", "ERROR")
            self.ibkr_connected = False
            return False

    def get_stock_data(self, symbol: str, start_date=None, end_date=None, period="5y", fetch_buffer=40):
        """
        Fetch stock data prioritizing IBKR Client Portal; fallback to yfinance.
        """
        df = pd.DataFrame()

        if self.use_ibkr:
            connected = self.connect_to_ibkr()
            if connected:
                df = self._download_from_ibkr(symbol)
                if not df.empty:
                    self.log(f"Successfully retrieved {len(df)} bars for {symbol} from IBKR", "SUCCESS")
                    return df
                else:
                    self.log(f"IBKR data retrieval failed or returned no data for {symbol}, falling back to yfinance",
                             "WARNING")
            else:
                self.log("IBKR connection unavailable, falling back to yfinance", "WARNING")

        # Fallback: use yfinance
        if df.empty:
            df = self._download_from_yfinance(symbol, days_back=5 * 365)
            if not df.empty:
                df.index = pd.to_datetime(df.index)
                self.log(f"Successfully retrieved {len(df)} bars for {symbol} from yfinance", "SUCCESS")
                return df
            else:
                self.log(f"Failed to retrieve data for {symbol} from yfinance after retries", "ERROR")

        return df

    def _download_from_ibkr(self, symbol, retries=3):
        """Internal method to download data from Client Portal API with a retry mechanism."""
        for attempt in range(retries):
            self.log(f"Downloading {symbol} from IBKR (attempt {attempt + 1}/{retries})", "INFO")
            try:
                params = {
                    "conid": self._get_conid(symbol),
                    "period": "5y",
                    "bar": "1d"
                }
                response = self.session.get(HISTORICAL_DATA_URL, params=params, verify=True, timeout=60)

                # Print the raw response text for debugging
                print("Raw response from IBKR historical data check:")
                print(response.text)

                # Try to parse the JSON response
                try:
                    data = response.json()
                except ValueError:
                    self.log("IBKR: Failed to decode JSON response for historical data.", "ERROR")
                    return pd.DataFrame()

                if "data" in data:
                    rows = [{'Date': pd.to_datetime(d['t'], unit='ms'), 'Open': d['o'], 'High': d['h'],
                             'Low': d['l'], 'Close': d['c'], 'Volume': d['v']} for d in data['data']]
                    df = pd.DataFrame(rows).set_index('Date').sort_index()
                    if not df.empty:
                        return df
                else:
                    self.log(f"IBKR: No data or error for {symbol}. Response: {data}", "WARNING")
                    return pd.DataFrame()

            except requests.exceptions.RequestException as e:
                # Print the raw response text for debugging
                print("Raw response from IBKR historical data check:")
                print(response.text)

                self.log(f"IBKR: Exception for {symbol}: {e}", "ERROR")

        return pd.DataFrame()

    def _get_conid(self, symbol):
        """Dummy method to get a ConID from a symbol. Needs implementation."""
        # This is a placeholder. A real implementation would require another API call.
        # For testing, we'll return a known ConID or a dummy value.
        conids = {"AAPL": "265598", "MSFT": "272093", "GOOGL": "17385956"}
        return conids.get(symbol, "265598") # Fallback to AAPL if not found

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