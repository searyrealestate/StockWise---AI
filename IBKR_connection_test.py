"""
🔌 IBKR Connection Test Script - Minimal Version
=====================================================

This script tests IBKR API connection and downloads sample data.
Run this first to identify connection issues.
"""
"""
IBKR API Connection & Data Download Utility
===========================================

This script is a comprehensive diagnostic tool designed to help users establish
and test their connection to the Interactive Brokers (IBKR) TWS or Gateway API.

It serves two primary purposes:
1.  To automatically detect a user's correct TWS/Gateway connection settings.
2.  To perform a simple data download to confirm the API is fully functional.

Features:
---------
-   Auto-Discovery: The script automatically cycles through the most common IBKR
    ports (e.g., 7497 for TWS Paper, 4002 for Gateway Paper) to find a working
    connection, removing guesswork for the user.
-   Connection & Data Validation: Once a connection is established, it requests
    a small amount of historical data for AAPL to ensure the API can handle
    data queries.
-   Troubleshooting Guidance: If all connection attempts fail, it prints
    clear, step-by-step instructions on how to configure the API settings
    within TWS.
-   Optional Data Download: If a connection is successful, the user is prompted
    to optionally download 1 year of historical data for a sample list of
    major NASDAQ stocks, saving them as individual CSV files in a 'data/'
    directory.

Usage:
------
    python <name_of_this_script>.py
"""

import sys
import time
import threading
import pandas as pd
from datetime import datetime, timedelta
from data_source_manager import DataSourceManager


# Check if IBKR API is installed
try:
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper
    from ibapi.contract import Contract
    from ibapi.common import BarData

    print("✅ IBKR API imported successfully")
    IBKR_AVAILABLE = True
except ImportError as e:
    print(f"❌ IBKR API not available: {e}")
    print("📥 Install with: pip install ibapi")
    IBKR_AVAILABLE = False
    sys.exit(1)


class SimpleIBKRTest(EWrapper, EClient):
    """Minimal IBKR client for testing connection"""

    def __init__(self):
        EClient.__init__(self, self)
        self.data = []
        self.data_ready = threading.Event()
        self.connection_ready = threading.Event()
        self.error_occurred = False
        self.error_message = ""
        self.req_id = 1001

    def log(self, message, level="INFO"):
        """Simple logging"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        icons = {"INFO": "ℹ️", "SUCCESS": "✅", "ERROR": "❌", "WARNING": "⚠️"}
        print(f"[{timestamp}] {icons.get(level, '📊')} {message}")

    def nextValidId(self, orderId):
        """Connection confirmed"""
        self.connection_ready.set()
        self.log(f"Connection confirmed. Order ID: {orderId}", "SUCCESS")

    def historicalData(self, reqId, bar):
        """Receive historical data"""
        self.data.append({
            'Date': bar.date,
            'Open': bar.open,
            'High': bar.high,
            'Low': bar.low,
            'Close': bar.close,
            'Volume': bar.volume
        })

    def historicalDataEnd(self, reqId, start, end):
        """Historical data complete"""
        self.log(f"Data complete: {len(self.data)} bars from {start} to {end}", "SUCCESS")
        self.data_ready.set()

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        """Handle errors"""
        # --- Filter messages by code ---
        if errorCode in [2104, 2106, 2158]:  # Market data farm "OK" messages
            self.log(f"Market data warning {errorCode}: {errorString}", "WARNING")
        elif errorCode in [502, 503, 504]:  # Critical connection errors
            self.log(f"Connection error {errorCode}: {errorString}", "ERROR")
            self.error_occurred = True
            self.error_message = errorString
            self.connection_ready.set()  # Also set event to stop waiting
        else:
            # Treat other messages as potential errors
            self.log(f"Error {errorCode}: {errorString}", "ERROR")
            if reqId != -1:  # reqId of -1 is a system message
                self.error_occurred = True
                self.error_message = errorString
                self.data_ready.set()  # Set data event if it's a data-related error


def test_ibkr_connection():
    """Test IBKR connection with common configurations"""

    print("🔌 Testing IBKR Connection")
    print("=" * 50)

    # Common IBKR connection configurations
    configs = [
        {"host": "127.0.0.1", "port": 7497, "name": "TWS Paper Trading"},
        {"host": "127.0.0.1", "port": 4002, "name": "Gateway Paper Trading"},
        {"host": "127.0.0.1", "port": 7496, "name": "TWS Live Trading"},
        {"host": "127.0.0.1", "port": 4001, "name": "Gateway Live Trading"}
    ]

    for config in configs:
        print(f"\n🔄 Trying {config['name']} on port {config['port']}...")

        client = SimpleIBKRTest()

        try:
            # Attempt connection
            client.connect(config["host"], config["port"], 1)

            # Start API thread
            api_thread = threading.Thread(target=client.run, daemon=True)
            api_thread.start()

            # Wait for connection confirmation
            if client.connection_ready.wait(timeout=10):
                client.log(f"Connected via {config['name']}", "SUCCESS")

                # Test data retrieval
                success = test_data_retrieval(client)

                client.disconnect()

                if success:
                    print(f"\n🎉 SUCCESS! Use this configuration:")
                    print(f"   Host: {config['host']}")
                    print(f"   Port: {config['port']}")
                    print(f"   Method: {config['name']}")
                    return config

            else:
                client.log(f"Connection timeout for {config['name']}", "ERROR")
                client.disconnect()

        except Exception as e:
            client.log(f"Exception connecting to {config['name']}: {e}", "ERROR")
            try:
                client.disconnect()
            except:
                pass

    print("\n❌ All connection attempts failed!")
    print_setup_instructions()
    return None


def test_data_retrieval(client):
    """Test historical data retrieval"""
    try:
        client.log("Testing data retrieval for AAPL...", "INFO")

        # Create contract for AAPL
        contract = Contract()
        contract.symbol = "AAPL"
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"

        # Clear previous data
        client.data = []
        client.data_ready.clear()
        client.error_occurred = False

        # Request historical data
        client.reqHistoricalData(
            client.req_id, contract, "", "5 D", "1 day", "TRADES", 1, 1, False, []
        )

        # Wait for data
        if client.data_ready.wait(timeout=30):
            if client.error_occurred:
                client.log(f"Data retrieval error: {client.error_message}", "ERROR")
                return False

            if len(client.data) > 0:
                df = pd.DataFrame(client.data)
                client.log(f"Retrieved {len(df)} days of AAPL data", "SUCCESS")
                client.log(f"Latest close: ${df.iloc[-1]['Close']:.2f}", "INFO")
                return True
            else:
                client.log("No data received", "ERROR")
                return False
        else:
            client.log("Data retrieval timeout", "ERROR")
            return False

    except Exception as e:
        client.log(f"Data test exception: {e}", "ERROR")
        return False


def print_setup_instructions():
    """Print IBKR setup instructions"""
    print("\n📋 IBKR Setup Instructions:")
    print("=" * 40)
    print("1. Download and install TWS or IB Gateway")
    print("2. Create paper trading account (free)")
    print("3. Launch TWS/Gateway")
    print("4. Configure API access:")
    print("   - Go to: Configure → API → Settings")
    print("   - Check: 'Enable ActiveX and Socket Clients'")
    print("   - Add 127.0.0.1 to 'Trusted IPs'")
    print("   - Set Socket Port (7497 for paper)")
    print("   - Click 'OK' and restart TWS")
    print("5. Run this script again")
    print("\n📱 Port Reference:")
    print("   - TWS Paper: 7497")
    print("   - Gateway Paper: 4002")
    print("   - TWS Live: 7496")
    print("   - Gateway Live: 4001")


def get_nasdaq_symbols():
    """Get list of NASDAQ symbols (sample for testing)"""
    # For testing, use a small subset
    return [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
        'NVDA', 'META', 'NFLX', 'ADBE', 'CRM'
    ]


def download_nasdaq_historical_data(config):
    """Download historical data for NASDAQ stocks"""
    print("\n📈 Downloading NASDAQ Historical Data")
    print("=" * 50)

    symbols = get_nasdaq_symbols()
    successful_downloads = 0

    client = SimpleIBKRTest()

    try:
        # Connect using successful config
        client.connect(config["host"], config["port"], 1)
        api_thread = threading.Thread(target=client.run, daemon=True)
        api_thread.start()

        if client.connection_ready.wait(timeout=10):
            client.log(f"Connected for data download", "SUCCESS")

            for i, symbol in enumerate(symbols):
                print(f"\n📊 [{i + 1}/{len(symbols)}] Downloading {symbol}...")

                # Create contract
                contract = Contract()
                contract.symbol = symbol
                contract.secType = "STK"
                contract.exchange = "SMART"
                contract.currency = "USD"

                # Reset for each symbol
                client.data = []
                client.data_ready.clear()
                client.error_occurred = False
                client.req_id += 1

                # Request 1 year of daily data
                client.reqHistoricalData(
                    client.req_id, contract, "", "1 Y", "1 day", "TRADES", 1, 1, False, []
                )

                # Wait for data
                if client.data_ready.wait(timeout=60):
                    if not client.error_occurred and len(client.data) > 0:
                        # Save to CSV
                        df = pd.DataFrame(client.data)
                        df['Date'] = pd.to_datetime(df['Date'])
                        df.set_index('Date', inplace=True)

                        filename = f"data/{symbol}_historical.csv"
                        import os
                        os.makedirs("data", exist_ok=True)
                        df.to_csv(filename)

                        client.log(f"Saved {len(df)} days to {filename}", "SUCCESS")
                        successful_downloads += 1
                    else:
                        client.log(f"Failed to get data for {symbol}: {client.error_message}", "ERROR")
                else:
                    client.log(f"Timeout downloading {symbol}", "ERROR")

                # Rate limiting - wait between requests
                if i < len(symbols) - 1:  # Don't wait after last symbol
                    time.sleep(2)

        client.disconnect()

    except Exception as e:
        print(f"❌ Download error: {e}")
        try:
            client.disconnect()
        except:
            pass

    print(f"\n📊 Download Summary: {successful_downloads}/{len(symbols)} successful")
    return successful_downloads


if __name__ == "__main__":
    print("🚀 IBKR API Connection Test")
    print("=" * 30)

    # Test connection
    working_config = test_ibkr_connection()

    if working_config:
        # Ask user if they want to download data
        response = input("\n❓ Download sample NASDAQ data? (y/n): ").strip().lower()
        if response == 'y':
            download_nasdaq_historical_data(working_config)
    else:
        print("\n💡 Fix connection issues first, then run script again")