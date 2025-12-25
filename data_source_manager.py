"""
⚙️ Data Source Manager - Professional Version (Triple Fallback)
=============================================================
Logic: IBKR -> Alpaca -> YFinance
This manager ensures data is fetched from the best available source.
"""

"""
⚙️ Data Source Manager - Professional Version (Triple Fallback)
=============================================================
Logic: IBKR -> Alpaca -> YFinance
This manager ensures data is fetched from the best available source.
"""

import threading
import time
import logging
import random
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import pytz
import os
import streamlit as st
# import datetime (Removed to avoid conflict with 'from datetime import datetime')
import system_config as cfg
import logging

logger = logging.getLogger(__name__)


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
    logger.debug("[X] IBKR API (ibapi) not found. Install with: pip install ibapi")
    IBKR_AVAILABLE = False

# --- ALPACA IMPORT ---
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.common.exceptions import APIError
    ALPACA_AVAILABLE = True
except ImportError:
    class TimeFrame: Day, Minute15, Hour = '1D', '15M', '1H'
    class TimeFrameUnit: Minute = 'Min'
    class StockBarsRequest:
        def __init__(self, **kwargs): pass
    class APIError(Exception): pass
    class StockHistoricalDataClient:
        def __init__(self, **kwargs): pass
    logger.debug("[!] Alpaca SDK not found. Alpaca fallback disabled.")
    ALPACA_AVAILABLE = False


class IBKRDataApp(EWrapper, EClient):
    """Thread-safe IBKR wrapper."""

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
        self.data.append({
            'Date': bar.date, 'Open': bar.open, 'High': bar.high,
            'Low': bar.low, 'Close': bar.close, 'Volume': bar.volume
        })

    def historicalDataEnd(self, reqId, start, end):
        self.data_event.set()

    def contractDetails(self, reqId, contractDetails):
        self.resolved_contract = contractDetails.contract
        self.contract_event.set()

    def contractDetailsEnd(self, reqId):
        self.contract_event.set()


class DataSourceManager(EWrapper, EClient):
    _client_id_counter = int(time.time() % 1000)
    _client_id_lock = threading.Lock()

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

        self._setup_logging()

        # --- ALPACA SETUP (SAFE) ---
        self.stock_client = None
        if ALPACA_AVAILABLE:
            try:
                self.api_key = cfg.ALPACA_KEY
                self.api_secret = cfg.ALPACA_SECRET

                if self.api_key and self.api_secret:
                    self.stock_client = StockHistoricalDataClient(self.api_key, self.api_secret)
                else:
                    self._log("Alpaca API keys missing. Skipping Alpaca.", "WARNING")
            except Exception as e:
                self._log(f"Alpaca Init Error: {e}", "ERROR")
                self.stock_client = None

        self._log(f"--- DataSourceManager initialized (ID: {self.client_id}) ---")

    def _setup_logging(self):
        self.logger = logging.getLogger(f"DSM_{self.client_id}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

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
        if self.app and self.app.isConnected():
            self.app.disconnect()
            self._log("Disconnected from IBKR.")

    def isConnected(self):
        return self.app is not None and self.app.isConnected()

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

    def fetch_data_sequential(self, tickers: list):
        """
        Sequentially fetches data for a list of tickers.
        Returns a dictionary {ticker: dataframe}.
        """
        data_map = {}
        for ticker in tickers:
            logger.info(f"[>] Fetching {ticker}...")
            df = self.get_stock_data(ticker, days_back=None) # Rely on system config default
            if not df.empty:
                data_map[ticker] = df
            else:
                logger.warning(f"Failed to fetch data for {ticker}")
        
        return data_map

    # --- MAIN DATA METHOD (WATERFALL LOGIC) ---
    def get_stock_data(self, symbol, start_date=None, end_date=None, days_back=None, interval='1d'):
        """
        Fetches data using the strategy: IBKR -> Alpaca -> YFinance
        """
        df = pd.DataFrame()
        clean_symbol = symbol.upper().strip()

        # 1. Attempt IBKR
        if self.use_ibkr:
            if not self.isConnected(): self.connect_to_ibkr()
            if self.isConnected():
                try:
                    df = self._download_from_ibkr(clean_symbol, start_date, end_date, days_back, interval)
                    if not df.empty: return normalize_and_validate_data(df)
                except Exception as e:
                    logging.info(f"IBKR Failed for {symbol}: {e}", "WARNING")
            else:
                self._log("[!] IBKR not connected. Skipping IBKR download.", "WARNING")
        else:
            self._log("[!] IBKR disabled. Skipping IBKR download.", "INFO")

        # 2. Attempt Alpaca (Fallback #1)
        logging.info(f"--- IBKR failed or disabled. Proceeding to Alpaca for {symbol} ---")
        if self.allow_fallback and self.stock_client:
            try:
                logging.info(f"[!] Attempting Alpaca fallback for {symbol}...")
                df = self._download_from_alpaca(clean_symbol, start_date, end_date, days_back, interval)
                self._log(f"Alpaca download attempt completed for {symbol}.")
                # self._log(f"DEBUG#12 stock_df range: {df.index.min()} to {df.index.max()}")
                # self._log(f"DEBUG#12 stock_df Data: \n{df.head(3)} ...\n{df.tail(3)}")
                # self._log(df[0:5].to_string())
                if not df.empty:
                    logging.info(f"[OK] Success: Data retrieved from Alpaca.")
                    return df
            except Exception as e:
                self._log(f"Alpaca Failed: {e}", "WARNING")
        else:
            self._log("⚠️ Alpaca client not initialized or fallback disabled. Skipping Alpaca.")

        # 3. Attempt YFinance (Fallback #2 - Last Resort)
        logging.info(f"--- Alpaca failed or disabled. Proceeding to YFinance for {symbol} ---")
        if self.allow_fallback:
            try:
                self._log(f"[!] Attempting YFinance fallback for {symbol}...")
                df = self._download_from_yfinance(clean_symbol, days_back, interval, start_date, end_date)
                self._log(f"YFinance download attempt completed for {symbol}.")
                # self._log(df[0:5].to_string())
                if not df.empty:
                    self._log(f"[OK] Success: Data retrieved from YFinance.")
                    return df
            except Exception as e:
                self._log(f"YFinance Failed: {e}", "ERROR")

        return pd.DataFrame()

    # --- INTERNAL DOWNLOADERS ---
        # --- INTERNAL DOWNLOADERS ---
    def _download_from_ibkr(self, symbol, start_date, end_date, days_back, interval):
        contract = Contract()
        contract.currency = 'USD'

        # --- MAPPING 1: IBKR Contract Logic ---
        if symbol.startswith('^'):
            contract.secType = 'IND'
            contract.symbol = symbol.lstrip('^')
            if contract.symbol == 'VIX':
                contract.exchange = 'CBOE'
            else:
                contract.exchange = 'SMART'
        else:
            contract.secType = 'STK'
            contract.symbol = symbol
            contract.exchange = 'SMART'

        # Resolve Contract
        self.app.resolved_contract = None
        self.app.contract_event.clear()
        self.app.reqContractDetails(self.app.client_id + 1, contract)
        self.app.contract_event.wait(timeout=5)
        if self.app.resolved_contract: contract = self.app.resolved_contract

        # Format Time
        if start_date and end_date:
            end_dt_str = end_date.strftime("%Y%m%d 23:59:59") if hasattr(end_date, 'strftime') else end_date
            if isinstance(start_date, datetime):
                delta = (end_date.date() - start_date.date()).days
            else:
                delta = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
            duration_str = f"{delta + 5} D"
        elif days_back:
            end_dt_str = datetime.now().strftime("%Y%m%d 23:59:59")
            duration_str = f"{days_back} D"
        else:
            return pd.DataFrame()

        # Map Interval
        ibkr_interval_map = {'1d': '1 day', '15 mins': '15 mins', '5 mins': '5 mins', '30 mins': '30 mins',
                             '1 hour': '1 hour'}
        bar_size = ibkr_interval_map.get(interval, interval)

        self.app.data = []
        self.app.data_event.clear()
        self.app.error_occurred = False

        self.app.reqHistoricalData(self.client_id + 2, contract, end_dt_str, duration_str, bar_size, "TRADES", 1, 1,
                                   False, [])

        if self.app.data_event.wait(timeout=20):
            if self.app.data:
                df = pd.DataFrame(self.app.data)
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                return df
        return pd.DataFrame()

    def _download_from_alpaca(self, symbol, start_date, end_date, days_back, interval='1d'):
        """
        Internal method to download data from Alpaca using StockHistoricalDataClient.
        This is much more reliable for historical data than yfinance.
        """
        if not self.stock_client:
            self._log(f"Alpaca client not initialized. Cannot download {symbol} from Alpaca.", "ERROR")
            return pd.DataFrame()

        # --- MAPPING 2: Alpaca Symbol Logic ---
        clean_symbol = symbol.lstrip('^')  # Alpaca uses 'VIX', not '^VIX'

        try:
            # Fix dates
            start = start_date if start_date else datetime.now().date() - timedelta(days=days_back or 365)
            end = end_date if end_date else datetime.now().date()
            # end = pd.to_datetime(end_date).tz_localize('UTC') if end_date else datetime.datetime.now(
            #     tz=datetime.timezone.utc)
            # start = end - pd.Timedelta(days=days_back)

            try:
                # Interval Mapping
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

            req = StockBarsRequest(
                symbol_or_symbols=[clean_symbol],
                timeframe=tf,
                start=start,
                end=end,
                feed="iex"
            )

            # 2. Make the single API call
            bars = self.stock_client.get_stock_bars(req)

            # 3. Convert to DataFrame and clean
            df_raw = bars.df
            if df_raw.empty:
                logging.info(f"Alpaca returned no data for {clean_symbol}.")
                return pd.DataFrame()

            # Alpaca returns a MultiIndex (symbol, timestamp). Flatten to DatetimeIndex.
            df = df_raw.reset_index(level=0, drop=True)
            df.index.name = "Date"

            self._log(f"Successfully retrieved {len(df)} bars for {clean_symbol} from Alpaca.", "SUCCESS")
            return pd.DataFrame()

        except APIError as e:
            self._log(f"Alpaca download failed for {clean_symbol}: {e}", "WARNING")
            return pd.DataFrame()

    def _download_from_yfinance(self, symbol, days_back, interval='1d', start_date=None, end_date=None, retries=3,
                                delay=5):
        """Internal method to download data from yfinance with robust logic."""
        self._log(f"Downloading {symbol} from yfinance...")

        # --- MAPPING 3: YFinance Interval Logic ---
        yfinance_interval_map = {
            '1 day': '1d', '1d': '1d',
            '15 mins': '15m', '5 mins': '5m',
            '30 mins': '30m', '1 hour': '1h'
        }
        yfinance_interval = yfinance_interval_map.get(interval, interval)

        # --- Handle yfinance intraday limit ---
        is_intraday = 'm' in yfinance_interval or 'h' in yfinance_interval
        if days_back and is_intraday and days_back > 60:
            days_back = 60

        for i in range(retries):
            try:
                # --- Force String Dates ---
                if end_date:
                    if not start_date:
                        db = days_back if days_back else 365
                        start_dt = pd.to_datetime(end_date) - timedelta(days=db)
                    else:
                        start_dt = pd.to_datetime(start_date)

                    end_dt = pd.to_datetime(end_date)

                    # Convert to strings to avoid TypeError
                    start_str = start_dt.strftime('%Y-%m-%d')
                    end_str = (end_dt + timedelta(days=1)).strftime('%Y-%m-%d')

                    self._log(f"yfinance: Requesting {symbol} from {start_str} to {end_str}", "DEBUG")

                    df = yf.download(symbol, start=start_str, end=end_str,
                                     interval=yfinance_interval, auto_adjust=True, progress=False)

                elif start_date:
                    start_str = pd.to_datetime(start_date).strftime('%Y-%m-%d')
                    df = yf.download(symbol, start=start_str, interval=yfinance_interval,
                                     auto_adjust=True, progress=False)
                else:
                    # Period based logic
                    db = days_back if days_back else 365
                    self._log(f"yfinance: Requesting {symbol} for last {db} days.", "DEBUG")

                    period = f"{db}d"
                    if not is_intraday and db > 300: period = f"{int(db / 365) + 1}y"

                    if not is_intraday and db > 300:
                        years = int(db / 365) + 1
                        period = f"{years}y"
                        df = yf.download(symbol, period=period, interval=yfinance_interval, auto_adjust=True,
                                         progress=False)
                    else:
                        period = f"{db}d"
                        df = yf.download(symbol, period=period, interval=yfinance_interval,
                                         auto_adjust=True, progress=False)

                if not df.empty:
                    self._log(f"Successfully retrieved {len(df)} bars for {symbol} from yfinance.", "SUCCESS")
                    return df

            except Exception as e:
                self._log(f"yfinance attempt {i + 1}/{retries} failed: {e}", "WARNING")
                time.sleep(delay)

        self._log(f"All yfinance download attempts failed for {symbol}.", "ERROR")
        return pd.DataFrame()


class SectorMapper:
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


def normalize_and_validate_data(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
    df.columns = df.columns.str.lower()
    if not isinstance(df.index, pd.DatetimeIndex): df.index = pd.to_datetime(df.index)
    if df.index.tz is not None: df.index = df.index.tz_localize(None)

    required = {'open', 'high', 'low', 'close', 'volume'}
    if not required.issubset(df.columns): return pd.DataFrame()

    # Capitalize for consistency with other tools
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'},
              inplace=True)
    return df


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures data consistency for all downstream technical analysis.
    This is often needed when switching between YF, Alpaca, and IBKR formats.
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

    # 4. Filter for required columns (must have OHLCV)
    required = {'open', 'high', 'low', 'close', 'volume'}
    if not required.issubset(df.columns):
        # We try to rename if YF/Alpaca capitalization was lost
        df.rename(columns={'adj close': 'close'}, inplace=True)
        if not required.issubset(df.columns):
            logging.warning("Clean raw data failed: Missing OHLCV columns.")
            return pd.DataFrame()

    # Final cleanup (Forward fill then backward fill NaNs, usually from splits)
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    return df