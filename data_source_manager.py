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
import sys # For logging handler setup
import logging # For structured logging
from datetime import datetime, timedelta
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import yfinance as yf # Import yfinance for fallback

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
        EClient.__init__(self, self)
        self.use_ibkr = use_ibkr
        self.ibkr_host = ibkr_host
        self.ibkr_port = ibkr_port
        self.debug = debug
        self.yfinance_max_retries = yfinance_max_retries
        self.yfinance_retry_delay = yfinance_retry_delay

        # Data storage
        self.historical_data = {}
        self.download_progress = {
            'successful': [],
            'failed': [],
            'current_symbol': '',
            'total_symbols': 0,
            'completed': 0
        }

        # Synchronization
        self.data_ready = threading.Event()
        self.connection_ready = threading.Event()
        self.error_occurred = False
        self.error_message = ""
        self.ibkr_connected = False # Track IBKR connection status
        self.thread = None # Keep a reference to the IBKR API thread

        # Request management
        self.req_id = 3000
        self.start_time = time.time()

        # Create output directories
        os.makedirs("nasdaq_data", exist_ok=True)
        os.makedirs("logs", exist_ok=True)

        self.log_file = f"logs/nasdaq_download_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        if self.use_ibkr:
            self.connect_to_ibkr()

    def log(self, message, level="INFO"):
        """Enhanced logging with file output, using Python's logging module."""
        # Print to console using the logger (which also handles stream output)
        if level == "INFO":
            logger.info(message)
        elif level == "SUCCESS":
            logger.info(f"✅ {message}")
        elif level == "ERROR":
            logger.error(f"❌ {message}")
        elif level == "WARNING":
            logger.warning(f"⚠️ {message}")
        else:
            logger.debug(message) # Default to debug for other levels

    def get_all_nasdaq_symbols(self):
        """Get comprehensive list of NASDAQ symbols (curated for this example)"""
        # In a real-world scenario, this would likely involve an API call
        # to NASDAQ or a regularly updated database.
        return [
            # Mega Cap (>$200B)
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA',

            # Large Cap ($10B-$200B)
            'NFLX', 'ADBE', 'CRM', 'PYPL', 'INTC', 'AMD', 'QCOM', 'TXN',
            'AVGO', 'COST', 'CSCO', 'CMCSA', 'PEP', 'TMUS', 'INTU', 'AMAT',
            'ISRG', 'BKNG', 'GILD', 'ADP', 'FISV', 'REGN', 'LRCX', 'KLAC',
            'MRNA', 'MELI', 'ASML', 'CSX', 'ABNB', 'CHTR', 'SNPS', 'CDNS',
            'ORLY', 'WDAY', 'NXPI', 'CTAS', 'DXCM', 'CRWD', 'ZS', 'OKTA',
            'TEAM', 'SNOW', 'DOCU', 'ZM', 'ROKU', 'UBER', 'LYFT', 'PTON',

            # Mid Cap ($2B-$10B)
            'LCID', 'RIVN', 'NKLA', 'PLTR', 'HOOD', 'COIN', 'RBLX', 'U',
            'DDOG', 'NET', 'FSLY', 'ESTC', 'MDB', 'SPLK', 'NOW', 'VEEV',
            'ORCL', 'CZR', 'NCLH', 'CCL', 'WYNN', 'LVS', 'MGM', 'PENN',
            'DIS', 'SBUX', 'MCD', 'NKE', 'LULU', 'ULTA', 'TJX', 'HD',
            'WMT', 'TGT', 'LOW', 'BBY', 'EBAY', 'ETSY', 'W', 'SHOP',

            # Tech/Software
            'MSCI', 'SPGI', 'MCO', 'ICE', 'CME', 'NDAQ', 'CBOE', 'VRSN',
            'TTWO', 'EA', 'RBLX', 'UNITY', 'ZYNG', 'MTCH', 'BMBL', 'SNAP',
            'PINS', 'TWTR', 'SPOT', 'ROKU', 'FUBO', 'DISH', 'SIRI', 'WBD',

            # Biotech/Healthcare
            'BIIB', 'VRTX', 'ILMN', 'ALXN', 'INCY', 'BMRN', 'TECH', 'SRPT',
            'BLUE', 'SAGE', 'IONS', 'ARWR', 'FOLD', 'EDIT', 'CRSP', 'BEAM',
            'NVTA', 'PACB', 'CDNA', 'TWST', 'FATE', 'RGNX', 'RARE', 'PBYI',

            # Semiconductors
            'MU', 'MRVL', 'MCHP', 'SWKS', 'QRVO', 'MPWR', 'POWI', 'CRUS',
            'RMBS', 'SITM', 'DIOD', 'AMBA', 'HIMX', 'ACLS', 'NVMI', 'FORM',

            # Clean Energy/EVs
            'ENPH', 'SEDG', 'RUN', 'NOVA', 'SPWR', 'CSIQ', 'JKS', 'SOL',
            'PLUG', 'FCEL', 'BLDP', 'BE', 'CLNE', 'HYLN', 'RIDE', 'GOEV',

            # Cannabis/Alternatives
            'TLRY', 'CGC', 'CRON', 'ACB', 'HEXO', 'OGI', 'SNDL', 'GRWG',

            # REITs
            'EQIX', 'DLR', 'CCI', 'AMT', 'SBAC', 'CONE', 'REIT', 'O',

            # Financial Services
            'SOFI', 'AFRM', 'UPST', 'LC', 'TREE', 'ENVA', 'EVCM', 'IIPR',

            # Retail/Consumer
            'AMZN', 'COST', 'WMT', 'TGT', 'LOW', 'BBY', 'EBAY', 'ETSY', 'W',
            'CHWY', 'PETS', 'CHEWY', 'OSTK', 'OVLV', 'FVRR', 'UPWK', 'FREELANCER'
        ]

    def connect_to_ibkr(self):
        """Connect using your working configuration for IBKR."""
        if not self.use_ibkr:
            self.log("IBKR connection skipped as use_ibkr is False.", "INFO")
            return False

        try:
            self.log(f"Connecting to IBKR TWS Paper Trading ({self.ibkr_host}:{self.ibkr_port})...")
            self.connect(self.ibkr_host, self.ibkr_port, clientId=1)

            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()

            if self.connection_ready.wait(timeout=30):
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
        """Disconnects from IBKR."""
        if self.ibkr_connected:
            self.log("Disconnecting from IBKR...", "INFO")
            super().disconnect()
            self.ibkr_connected = False
            self.log("Disconnected from IBKR.", "INFO")


    def get_stock_data(self, symbol: str, days_back: int = 730) -> pd.DataFrame:
        """
        Fetches historical stock data, preferring IBKR and falling back to yfinance.
        Returns a DataFrame with OHLCV and basic indicators.
        """
        df = pd.DataFrame()
        source = "N/A"

        if self.use_ibkr and self.ibkr_connected:
            self.log(f"Attempting to fetch {symbol} data from IBKR.", "INFO")
            df_ibkr = self._download_from_ibkr(symbol)
            if not df_ibkr.empty:
                df = df_ibkr
                source = "IBKR"
                self.log(f"Successfully retrieved {len(df)} bars for {symbol} from IBKR.", "SUCCESS")
            else:
                self.log(f"Failed to get data for {symbol} from IBKR after retries. Trying yfinance...", "WARNING")

        if df.empty:
            self.log(f"Fetching {symbol} data from yfinance (fallback).", "INFO")
            df_yf = self._download_from_yfinance(symbol, days_back=days_back)
            if not df_yf.empty:
                df = df_yf
                source = "yfinance"
                self.log(f"Successfully retrieved {len(df)} bars for {symbol} from yfinance.", "SUCCESS")
            else:
                self.log(f"Failed to get data for {symbol} from yfinance.", "ERROR")
                logger.critical(f"❌ CRITICAL ERROR: Could not retrieve data for {symbol} from any source after all retries. Stopping script.")
                sys.exit(1)


        if not df.empty:
            df['SMA_20'] = df['Close'].rolling(20).mean()
            df['SMA_50'] = df['Close'].rolling(50).mean()
            df['Returns'] = df['Close'].pct_change()
            df['Volatility'] = df['Returns'].rolling(20).std()
        return df, source

    def _download_from_ibkr(self, symbol, retries=3):
        """Internal method to download historical data for a single symbol from IBKR with retries."""
        for attempt in range(retries):
            try:
                self.log(f"📊 Downloading {symbol} from IBKR (attempt {attempt + 1}/{retries})", "INFO")
                time.sleep(2)

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
                self.error_message = ""

                end_date_time = datetime.now().strftime("%Y%m%d 17:00:00 US/Eastern")
                self.reqHistoricalData(req_id, contract, end_date_time, "2 Y", "1 day", "TRADES", 1, 1, False, [])

                if self.data_ready.wait(timeout=60):
                    if self.error_occurred and self.error_message and not ("2174" in self.error_message):
                        self.log(
                            f"IBKR error for {symbol} (ReqId {req_id}): {self.error_message}. Falling back to yfinance.",
                            "ERROR")
                        del self.historical_data[req_id]
                        return pd.DataFrame()

                    data = self.historical_data.get(req_id, [])
                    if data and len(data) > 100:
                        rows = []
                        for bar in data:
                            rows.append({
                                'Date': pd.to_datetime(bar.date),
                                'Open': float(bar.open),
                                'High': float(bar.high),
                                'Low': float(bar.low),
                                'Close': float(bar.close),
                                'Volume': int(bar.volume)
                            })
                        df = pd.DataFrame(rows)
                        df.set_index('Date', inplace=True)
                        df.sort_index(inplace=True)
                        return df
                    else:
                        self.log(f"IBKR: Insufficient data for {symbol} ({len(data)} bars).", "WARNING")
                else:
                    self.log(f"IBKR: Timeout for {symbol} on attempt {attempt + 1}.", "WARNING")

            except Exception as e:
                self.log(f"IBKR: Exception for {symbol} on attempt {attempt + 1}: {e}", "ERROR")
        return pd.DataFrame()

    def _download_from_yfinance(self, symbol, days_back=730):
        """
        Internal method to download historical data for a single symbol from yfinance with retries.
        """
        for attempt in range(self.yfinance_max_retries):
            try:
                self.log(f"📊 Downloading {symbol} from yfinance (attempt {attempt + 1}/{self.yfinance_max_retries})", "INFO")
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days_back)
                data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                if not data.empty:
                    # ⭐ FIX: Robustly flatten and standardize column names
                    new_columns = []
                    for col in data.columns:
                        if isinstance(col, tuple): # Handle MultiIndex columns
                            new_columns.append(col[0].capitalize())
                        elif isinstance(col, str): # Handle single-level string columns
                            new_columns.append(col.capitalize())
                        else: # Fallback for unexpected types
                            new_columns.append(str(col))
                    data.columns = new_columns

                    # Ensure 'Adj Close' is renamed to 'Close' if present
                    if 'Adj close' in data.columns:
                        data = data.rename(columns={'Adj close': 'Close'})

                    data.index.name = 'Date'
                    data = data[['Open', 'High', 'Low', 'Close', 'Volume']] # Filter to essential columns
                    return data
                else:
                    self.log(f"yfinance: No data found for {symbol} on attempt {attempt + 1}.", "WARNING")
            except Exception as e:
                self.log(f"yfinance: Error downloading {symbol} on attempt {attempt + 1}: {e}", "ERROR")

            if attempt < self.yfinance_max_retries - 1:
                time.sleep(self.yfinance_retry_delay)

        self.log(f"yfinance: Failed to download {symbol} after {self.yfinance_max_retries} attempts.", "ERROR")
        return pd.DataFrame()


    def save_symbol_data(self, symbol, df, source):
        """Save symbol data to CSV with metadata"""
        try:
            if df.empty:
                self.log(f"Cannot save empty DataFrame for {symbol}.", "WARNING")
                return

            # Save to CSV
            filename = f"nasdaq_data/{symbol}_historical.csv"
            df.to_csv(filename)
            self.download_progress['successful'].append(symbol)
            self.log(f"✅ {symbol}: {len(df)} bars saved from {source}", "SUCCESS")

            # Save metadata
            metadata = {
                'symbol': symbol,
                'download_timestamp': datetime.now().isoformat(),
                'data_points': len(df),
                'date_range': {
                    'start': df.index.min().isoformat(),
                    'end': df.index.max().isoformat()
                },
                'price_range': {
                    'min': float(df['Close'].min()),
                    'max': float(df['Close'].max()),
                    'latest': float(df['Close'].iloc[-1])
                },
                'volume_stats': {
                    'avg_volume': int(df['Volume'].mean()),
                    'max_volume': int(df['Volume'].max()),
                    'latest_volume': int(df['Volume'].iloc[-1])
                },
                'source': source # Indicate data source
            }

            metadata_file = f"nasdaq_data/{symbol}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            self.log(f"❌ Error saving {symbol}: {e}", "ERROR")
            self.download_progress['failed'].append((symbol, f"Save Error: {e}"))


    def download_all_nasdaq(self, limit=None, start_from=0):
        """Download all NASDAQ symbols"""

        symbols = self.get_all_nasdaq_symbols()

        if limit:
            symbols = symbols[start_from:start_from + limit]
        else:
            symbols = symbols[start_from:]

        self.download_progress['total_symbols'] = len(symbols)

        self.log(f"🚀 Starting download of {len(symbols)} NASDAQ symbols")
        self.log(f"📁 Data will be saved to: nasdaq_data/")
        self.log(f"📄 Log file: {self.log_file}")

        for i, symbol in enumerate(symbols, 1):
            self.download_progress['current_symbol'] = symbol
            self.download_progress['completed'] = i - 1

            # Progress update
            progress_pct = ((i-1) / len(symbols)) * 100
            elapsed_time = time.time() - self.start_time
            # Avoid division by zero if it's the very first symbol
            estimated_total = (elapsed_time / (i-1)) * len(symbols) if i > 1 else 0
            remaining_time = estimated_total - elapsed_time if estimated_total > 0 else 0

            self.log(f"📈 [{i}/{len(symbols)}] {symbol} | Progress: {progress_pct:.1f}% | "
                    f"ETA: {remaining_time/60:.1f}min")

            # Download symbol using the enhanced get_stock_data method
            df_symbol, source_used = self.get_stock_data(symbol, days_back=730) # 2 years for yfinance

            if not df_symbol.empty:
                self.save_symbol_data(symbol, df_symbol, source_used)
            else:
                self.log(f"❌ Failed to download any data for {symbol} from either source.", "ERROR")
                self.download_progress['failed'].append((symbol, "No data downloaded"))


            # Save progress every 10 symbols
            if i % 10 == 0:
                self.save_progress_report()

        # Final report
        self.generate_final_report()

    def save_progress_report(self, report_path='nasdaq_data/progress_report.json'):
        """Save intermediate progress report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'progress': self.download_progress.copy(),
            'elapsed_time_minutes': (time.time() - self.start_time) / 60,
            'success_rate': len(self.download_progress['successful']) /
                           max(1, len(self.download_progress['successful']) + len(self.download_progress['failed'])) * 100
        }

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

    def generate_final_report(self):
        """Generate comprehensive final report"""
        total_time = time.time() - self.start_time
        successful = len(self.download_progress['successful'])
        failed = len(self.download_progress['failed'])
        total = successful + failed
        success_rate = (successful / total * 100) if total > 0 else 0

        report = {
            'download_summary': {
                'total_symbols': total,
                'successful_downloads': successful,
                'failed_downloads': failed,
                'success_rate_percent': round(success_rate, 2),
                'total_time_minutes': round(total_time / 60, 2),
                'average_time_per_symbol_seconds': round(total_time / total, 2) if total > 0 else 0,
                'download_speed_symbols_per_minute': round((total / (total_time / 60)), 2) if total_time > 0 else 0
            },
            'successful_symbols': self.download_progress['successful'],
            'failed_symbols': self.download_progress['failed'],
            'completion_timestamp': datetime.now().isoformat(),
            'data_location': 'nasdaq_data/',
            'log_file': self.log_file
        }

        # Save detailed report
        report_file = f"nasdaq_data/final_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        # Print summary
        print("\n" + "="*70)
        print("🏆 NASDAQ DOWNLOAD COMPLETE!")
        print("="*70)
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"📊 Success Rate: {success_rate:.1f}%")
        print(f"⏱️ Total Time: {total_time/60:.1f} minutes")
        print(f"⚡ Speed: {(total/(total_time/60)):.1f} symbols/minute")
        print(f"📁 Data Location: nasdaq_data/")
        print(f"📄 Detailed Report: {report_file}")
        print("="*70)

        if failed > 0:
            print(f"\n❌ Failed symbols ({len(self.download_progress['failed'])}):")
            for symbol, error in self.download_progress['failed'][:20]:  # Show first 20
                print(f"   {symbol}: {error}")
            if len(self.download_progress['failed']) > 20:
                print(f"   ... and {len(self.download_progress['failed']) - 20} more (see report)")

    # IBKR Callback Methods
    def nextValidId(self, orderId):
        self.connection_ready.set()

    def historicalData(self, reqId, bar):
        if reqId in self.historical_data:
            self.historical_data[reqId].append(bar)

    def historicalDataEnd(self, reqId, start, end):
        self.data_ready.set()

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        # Ignore common IBKR informational messages and non-critical warnings
        if errorCode in [2104, 2106, 2158, 2174]:
            self.log(f"IBKR Info/Warning (ReqId: {reqId if reqId != -1 else 'N/A'}): {errorCode} - {errorString}", "INFO")
            return

        self.error_occurred = True
        self.error_message = f"Error {errorCode}: {errorString}"

        # If it's an error related to a specific request, unblock it
        if reqId != -1:
            self.data_ready.set()
        # For general connection errors (reqId -1), these will be caught by timeout in connect_to_ibkr


def main():
    """Main execution"""
    logger.info("🏢 Complete NASDAQ Historical Data Downloader")
    logger.info("=" * 50)
    logger.info("Now with yfinance fallback for enhanced reliability!")
    logger.info("")

    # Initialize DataSourceManager with IBKR enabled by default
    downloader = DataSourceManager(use_ibkr=True)

    try:
        logger.info("\n📊 Download Options:")
        logger.info("1. Quick test (10 symbols)")
        logger.info("2. Medium batch (50 symbols)")
        logger.info("3. Large batch (100 symbols)")
        logger.info("4. Complete NASDAQ (all symbols)")
        logger.info("5. Resume from specific symbol")

        choice = input("\nSelect option (1-5): ").strip()

        if choice == "1":
            downloader.download_all_nasdaq(limit=10)
        elif choice == "2":
            downloader.download_all_nasdaq(limit=50)
        elif choice == "3":
            downloader.download_all_nasdaq(limit=100)
        elif choice == "4":
            logger.warning("⚠️ This will download ALL symbols and may take several hours!")
            confirm = input("Continue? (y/n): ").strip().lower()
            if confirm == 'y':
                downloader.download_all_nasdaq()
            else:
                logger.info("Download cancelled.")
        elif choice == "5":
            start_symbol = input("Enter symbol to start from: ").strip().upper()
            symbols = downloader.get_all_nasdaq_symbols()
            try:
                start_index = symbols.index(start_symbol)
                downloader.download_all_nasdaq(start_from=start_index)
            except ValueError:
                logger.error(f"❌ Symbol {start_symbol} not found in list")
        else:
            logger.info("Invalid choice. Running quick test...")
            downloader.download_all_nasdaq(limit=5)

    except KeyboardInterrupt:
        logger.info("\n⏹️ Download interrupted by user")
        downloader.generate_final_report()
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        downloader.generate_final_report()
    finally:
        # Ensure disconnection is always attempted if IBKR was intended to be used
        downloader.disconnect()
        logger.info("👋 Finished data download operations.")

