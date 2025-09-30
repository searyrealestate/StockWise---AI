"""
⚙️ Data Source Manager - Client Portal Web API Version
======================================================

A shared utility for fetching historical stock data, now configured to use the
Interactive Brokers Client Portal Web API as the primary source.
It still falls back to yfinance if IBKR is unavailable.

"""

import pandas as pd
import yfinance as yf
import logging
import time
import threading
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import datetime


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
    with a fallback to yfinance. This version uses the proven connection logic.
    """

    def __init__(self, use_ibkr=True, host='127.0.0.1', port=7497, client_id=None):
        EClient.__init__(self, self)
        self.use_ibkr = use_ibkr
        self.host = host
        self.port = port
        self.client_id = client_id

        # FIX: Use a unique, random client ID for each instance to prevent connection conflicts.
        self.client_id = client_id if client_id is not None else int(time.time() % 1000)

        logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | (%(name)s) %(message)s')
        self.logger = logging.getLogger("DataSourceManager")

        # Threading events for managing async API calls
        self.connection_event = threading.Event()
        self.data_event = threading.Event()
        self.error_occurred = False
        self.error_message = ""
        self.historical_data = []
        self.next_req_id = int(time.time())

    def log(self, message, level="INFO"):
        """Logs a message to the console."""
        if level.upper() == "INFO":
            self.logger.info(message)
        elif level.upper() == "WARNING":
            self.logger.warning(message)
        elif level.upper() == "ERROR":
            self.logger.error(message)
        elif level.upper() == "SUCCESS":
            self.logger.info(f"✅ {message}")

    def connect_to_ibkr(self):
        """Connects to a running TWS or Gateway instance using ibapi."""
        if self.isConnected():
            return True
        self.log(f"Connecting to TWS on {self.host}:{self.port}...")
        try:
            self.connect(self.host, self.port, self.client_id)
            api_thread = threading.Thread(target=self.run, daemon=True)
            api_thread.start()

            self.log("Waiting for connection handshake...")
            if self.connection_event.wait(timeout=20):
                self.log("Connection to TWS is active.", "SUCCESS")
                return True
            else:
                self.log("Connection to TWS timed out.", "ERROR")
                self.disconnect()
                return False
        except Exception as e:
            self.log(f"Could not connect to TWS: {e}", "ERROR")
            return False

    def nextValidId(self, orderId: int):
        """EWrapper callback that confirms the connection is ready."""
        super().nextValidId(orderId)
        self.log("Connection handshake confirmed by TWS.")
        self.connection_event.set()

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        """EWrapper callback for handling errors."""
        super().error(reqId, errorCode, errorString, advancedOrderRejectJson)
        if errorCode < 2000 and errorCode not in [2104, 2106, 2158]:
            self.log(f"TWS API Error {errorCode}: {errorString}", "ERROR")
            self.error_occurred = True
            self.error_message = errorString
            self.data_event.set()

    def disconnect(self):
        """Disconnects from TWS."""
        if self.isConnected():
            self.log("Disconnecting from TWS...")
            super().disconnect()

    def get_stock_data(self, symbol: str, days_back=1825):
        """Main method to get data, prioritizing IBKR."""
        if self.use_ibkr:
            if not self.isConnected():
                self.connect_to_ibkr()

            if self.isConnected():
                df = self._download_from_ibkr(symbol, days_back)
                if not df.empty:
                    return df
                else:
                    self.log(f"IBKR data retrieval failed for {symbol}, falling back to yfinance.", "WARNING")

        return self._download_from_yfinance(symbol, days_back)

    def historicalData(self, reqId, bar):
        """EWrapper callback that receives each bar of historical data."""
        self.historical_data.append({
            'Date': bar.date, 'Open': bar.open, 'High': bar.high,
            'Low': bar.low, 'Close': bar.close, 'Volume': bar.volume
        })

    def historicalDataEnd(self, reqId, start, end):
        """EWrapper callback that signals the end of a historical data request."""
        super().historicalDataEnd(reqId, start, end)
        self.data_event.set()

    def _download_from_ibkr(self, symbol: str, days_back: int):
        """Internal method for downloading data from TWS using ibapi."""
        # print("--- DEBUG:RUNNING v2 of _download_from_ibkr ---")
        self.log(f"Requesting historical data for {symbol} from TWS...")
        try:
            # Create a specific contract for the VIX Index, and a general one for stocks/ETFs.
            if symbol == '^VIX':
                contract = Contract()
                contract.symbol = 'VIX'
                contract.secType = 'IND'
                contract.exchange = 'CBOE'
                contract.currency = 'USD'
            else:
                contract = Stock(symbol, 'SMART', 'USD')

            if days_back > 365:
                duration = f'{round(days_back / 365)} Y'
            else:
                duration = f'{days_back} D'
            self.log(f"Requesting duration: {duration}")

            self.historical_data = []
            self.data_event.clear()
            self.error_occurred = False

            req_id = self.next_req_id
            self.next_req_id += 1

            self.reqHistoricalData(
                reqId=req_id, contract=contract, endDateTime="",
                durationStr=duration, barSizeSetting="1 day", whatToShow="TRADES",
                useRTH=1, formatDate=1, keepUpToDate=False, chartOptions=[]
            )

            if self.data_event.wait(timeout=60):
                if self.error_occurred:
                    self.log(f"Failed to get data for {symbol}: {self.error_message}", "ERROR")
                    return pd.DataFrame()

                df = pd.DataFrame(self.historical_data)
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                self.log(f"Successfully retrieved {len(df)} bars for {symbol} from IBKR.", "SUCCESS")
                return df
            else:
                self.log(f"Data request for {symbol} timed out.", "ERROR")
                self.cancelHistoricalData(req_id)
                return pd.DataFrame()
        except Exception as e:
            self.log(f"Exception during IBKR download for {symbol}: {e}", "ERROR")
            return pd.DataFrame()

    def _download_from_yfinance(self, symbol, days_back, retries=3, delay=5):
        """Internal method to download data from yfinance."""
        self.log(f"Downloading {symbol} from yfinance...")
        for i in range(retries):
            try:
                df = yf.download(symbol, period=f"{days_back}d", auto_adjust=True, progress=False)
                if not df.empty:
                    self.log(f"Successfully retrieved {len(df)} bars for {symbol} from yfinance.", "SUCCESS")
                    return df
            except Exception as e:
                self.log(f"yfinance download attempt {i+1}/{retries} for {symbol} failed: {e}", "WARNING")
                time.sleep(delay)
        self.log(f"All yfinance download attempts failed for {symbol}.", "ERROR")
        return pd.DataFrame()