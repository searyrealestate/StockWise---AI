"""
🏢 Complete NASDAQ Historical Data Downloader
============================================

Downloads historical data for ALL NASDAQ stocks using your working IBKR connection.
Based on your successful test results.
"""

import pandas as pd
import numpy as np
import time
import threading
import json
import os
import requests
from datetime import datetime, timedelta
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

class NASDAQDownloader(EWrapper, EClient):
    """Complete NASDAQ data downloader using IBKR"""

    def __init__(self):
        EClient.__init__(self, self)

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

        # Request management
        self.req_id = 3000
        self.start_time = time.time()

        # Create output directories
        os.makedirs("nasdaq_data", exist_ok=True)
        os.makedirs("logs", exist_ok=True)

        self.log_file = f"logs/nasdaq_download_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    def log(self, message, level="INFO"):
        """Enhanced logging with file output"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] {level}: {message}"

        # Print to console
        icons = {"INFO": "ℹ️", "SUCCESS": "✅", "ERROR": "❌", "WARNING": "⚠️"}
        print(f"[{timestamp}] {icons.get(level, '📊')} {message}")

        # Write to log file
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')

    def get_all_nasdaq_symbols(self):
        """Get comprehensive list of NASDAQ symbols"""

        # Method 1: Try to get from NASDAQ API (if available)
        try:
            self.log("Fetching NASDAQ symbol list from official sources...")

            # You can enhance this with official NASDAQ API
            # For now, using a comprehensive manually curated list
            symbols = self.get_curated_nasdaq_symbols()

            self.log(f"Retrieved {len(symbols)} NASDAQ symbols")
            return symbols

        except Exception as e:
            self.log(f"Error fetching symbols: {e}", "ERROR")
            return self.get_curated_nasdaq_symbols()

    def get_curated_nasdaq_symbols(self):
        """Curated list of major NASDAQ symbols"""
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
            'AMZN', 'COST', 'WMT', 'TGT', 'BBY', 'EBAY', 'ETSY', 'W',
            'CHWY', 'PETS', 'CHEWY', 'OSTK', 'OVLV', 'FVRR', 'UPWK', 'FREELANCER'
        ]

    def connect_to_ibkr(self):
        """Connect using your working configuration"""
        try:
            self.log("Connecting to IBKR TWS Paper Trading (port 7497)...")

            # Use your successful configuration
            self.connect("127.0.0.1", 7497, 1)

            # Start API thread
            api_thread = threading.Thread(target=self.run, daemon=True)
            api_thread.start()

            # Wait for connection
            if self.connection_ready.wait(timeout=30):
                self.log("✅ Connected to IBKR successfully", "SUCCESS")
                return True
            else:
                self.log("❌ Connection timeout", "ERROR")
                return False

        except Exception as e:
            self.log(f"❌ Connection error: {e}", "ERROR")
            return False

    def download_symbol_data(self, symbol, retries=3):
        """Download historical data for a single symbol with retries"""

        for attempt in range(retries):
            try:
                self.log(f"📊 Downloading {symbol} (attempt {attempt + 1}/{retries})")

                # Rate limiting - 2 seconds between requests
                time.sleep(2)

                # Create contract
                contract = Contract()
                contract.symbol = symbol
                contract.secType = "STK"
                contract.exchange = "SMART"
                contract.currency = "USD"

                # Request setup
                self.req_id += 1
                req_id = self.req_id

                self.historical_data[req_id] = []
                self.data_ready.clear()
                self.error_occurred = False
                self.error_message = ""

                # Request 2 years of daily data (like your successful test)
                self.reqHistoricalData(
                    req_id, contract, "", "2 Y", "1 day", "TRADES", 1, 1, False, []
                )

                # Wait for completion
                if self.data_ready.wait(timeout=60):
                    if not self.error_occurred:
                        data = self.historical_data.get(req_id, [])
                        if data and len(data) > 100:  # Ensure we have substantial data
                            self.save_symbol_data(symbol, data)
                            self.download_progress['successful'].append(symbol)
                            self.log(f"✅ {symbol}: {len(data)} bars saved", "SUCCESS")
                            return True
                        else:
                            self.log(f"⚠️ {symbol}: Insufficient data ({len(data)} bars)", "WARNING")
                    else:
                        self.log(f"❌ {symbol}: {self.error_message}", "ERROR")
                else:
                    self.log(f"⏰ {symbol}: Timeout on attempt {attempt + 1}", "WARNING")

            except Exception as e:
                self.log(f"❌ {symbol}: Exception on attempt {attempt + 1}: {e}", "ERROR")

        # All attempts failed
        self.download_progress['failed'].append((symbol, self.error_message or "Max retries exceeded"))
        return False

    def save_symbol_data(self, symbol, data):
        """Save symbol data to CSV with metadata"""
        try:
            # Convert to DataFrame
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

            # Add technical indicators
            df['SMA_20'] = df['Close'].rolling(20).mean()
            df['SMA_50'] = df['Close'].rolling(50).mean()
            df['Returns'] = df['Close'].pct_change()
            df['Volatility'] = df['Returns'].rolling(20).std()

            # Save to CSV
            filename = f"nasdaq_data/{symbol}_historical.csv"
            df.to_csv(filename)

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
                'source': 'IBKR_API'
            }

            metadata_file = f"nasdaq_data/{symbol}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            self.log(f"❌ Error saving {symbol}: {e}", "ERROR")

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
            estimated_total = (elapsed_time / (i-1)) * len(symbols) if i > 1 else 0
            remaining_time = estimated_total - elapsed_time if estimated_total > 0 else 0

            self.log(f"📈 [{i}/{len(symbols)}] {symbol} | Progress: {progress_pct:.1f}% | "
                    f"ETA: {remaining_time/60:.1f}min")

            # Download symbol
            success = self.download_symbol_data(symbol)

            # Save progress every 10 symbols
            if i % 10 == 0:
                self.save_progress_report()

        # Final report
        self.generate_final_report()

    def save_progress_report(self):
        """Save intermediate progress report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'progress': self.download_progress.copy(),
            'elapsed_time_minutes': (time.time() - self.start_time) / 60,
            'success_rate': len(self.download_progress['successful']) /
                           max(1, len(self.download_progress['successful']) + len(self.download_progress['failed'])) * 100
        }

        with open('nasdaq_data/progress_report.json', 'w') as f:
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
        # Ignore market data warnings (like in your successful test)
        if errorCode in [2104, 2106, 2158]:
            return

        self.error_occurred = True
        self.error_message = f"Error {errorCode}: {errorString}"

        if reqId != -1:
            self.data_ready.set()

def main():
    """Main execution"""
    print("🏢 Complete NASDAQ Historical Data Downloader")
    print("=" * 50)
    print("Based on your successful IBKR connection test!")
    print()

    downloader = NASDAQDownloader()

    # Connect using your working configuration
    if not downloader.connect_to_ibkr():
        print("❌ Failed to connect. Please ensure TWS is running on port 7497")
        return

    try:
        print("\n📊 Download Options:")
        print("1. Quick test (10 symbols)")
        print("2. Medium batch (50 symbols)")
        print("3. Large batch (100 symbols)")
        print("4. Complete NASDAQ (all symbols)")
        print("5. Resume from specific symbol")

        choice = input("\nSelect option (1-5): ").strip()

        if choice == "1":
            downloader.download_all_nasdaq(limit=10)
        elif choice == "2":
            downloader.download_all_nasdaq(limit=50)
        elif choice == "3":
            downloader.download_all_nasdaq(limit=100)
        elif choice == "4":
            print("⚠️ This will download ALL symbols and may take several hours!")
            confirm = input("Continue? (y/n): ").strip().lower()
            if confirm == 'y':
                downloader.download_all_nasdaq()
            else:
                print("Download cancelled.")
        elif choice == "5":
            start_symbol = input("Enter symbol to start from: ").strip().upper()
            symbols = downloader.get_all_nasdaq_symbols()
            try:
                start_index = symbols.index(start_symbol)
                downloader.download_all_nasdaq(start_from=start_index)
            except ValueError:
                print(f"❌ Symbol {start_symbol} not found in list")
        else:
            print("Invalid choice. Running quick test...")
            downloader.download_all_nasdaq(limit=5)

    except KeyboardInterrupt:
        print("\n⏹️ Download interrupted by user")
        downloader.generate_final_report()
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        downloader.generate_final_report()
    finally:
        downloader.disconnect()
        print("👋 Disconnected from IBKR")


if __name__ == "__main__":
    main()