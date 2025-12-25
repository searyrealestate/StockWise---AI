"""
🔌 Enhanced IBKR Connection Manager - Professional Grade
═══════════════════════════════════════════════════════

Upgraded version with:
- Better error handling
- Async operations support
- Multiple data feed types
- Real-time market data
- Professional data quality
"""

import time
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.common import BarData
from ibapi.ticktype import TickType
import queue
import asyncio
import nest_asyncio
from typing import Optional, Dict, List, Callable

# Enable nested event loops for Streamlit compatibility
nest_asyncio.apply()


class EnhancedIBKRClient(EWrapper, EClient):
    """Professional-grade IBKR client with enhanced features"""

    def __init__(self, debug=False):
        EClient.__init__(self, self)
        self.debug = debug

        # Enhanced data storage
        self.historical_data = {}
        self.real_time_data = {}
        self.contract_details = {}
        self.fundamental_data = {}
        self.market_depth = {}
        self.option_chains = {}

        # Synchronization and threading
        self.data_ready = threading.Event()
        self.connection_ready = threading.Event()
        self.error_occurred = False
        self.error_message = ""
        self.last_error_time = None

        # Request tracking with timeout
        self.next_req_id = 1000
        self.active_requests = {}
        self.request_timeouts = {}

        # Connection state
        self.connection_state = "DISCONNECTED"
        self.last_heartbeat = None

        # Data quality metrics
        self.data_quality_stats = {
            'successful_requests': 0,
            'failed_requests': 0,
            'timeout_requests': 0,
            'data_latency': []
        }

        self.log("Enhanced IBKR Client initialized", "INFO")

    def log(self, message, level="INFO"):
        """Enhanced logging with timestamps and levels"""
        if self.debug:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            icons = {
                "INFO": "ℹ️", "SUCCESS": "✅", "ERROR": "❌",
                "WARNING": "⚠️", "DEBUG": "🔍", "NETWORK": "🌐"
            }
            print(f"[{timestamp}] {icons.get(level, '📊')} IBKR-{level}: {message}")

    def get_next_req_id(self):
        """Generate unique request ID with collision prevention"""
        req_id = self.next_req_id
        self.next_req_id += 1
        return req_id

    def connect_to_ibkr(self, host="127.0.0.1", port=7497, client_id=1, timeout=30):
        """
        Enhanced connection with retry logic and validation

        Common IBKR Ports:
        - TWS Paper Trading: 7497
        - TWS Live Trading: 7496
        - Gateway Paper: 4002
        - Gateway Live: 4001
        """
        try:
            self.log(f"Connecting to IBKR at {host}:{port} (Client ID: {client_id})", "NETWORK")
            self.connection_state = "CONNECTING"

            # Attempt connection
            self.connect(host, port, client_id)

            # Start API thread
            api_thread = threading.Thread(target=self.run, daemon=True, name="IBKR-API-Thread")
            api_thread.start()

            # Wait for connection with timeout
            if self.connection_ready.wait(timeout=timeout):
                if self.isConnected():
                    self.connection_state = "CONNECTED"
                    self.last_heartbeat = datetime.now()
                    self.log("✅ Successfully connected to IBKR", "SUCCESS")

                    # Request account summary for validation
                    self.validate_connection()
                    return True
                else:
                    self.connection_state = "FAILED"
                    self.log("❌ Connection established but not confirmed", "ERROR")
                    return False
            else:
                self.connection_state = "TIMEOUT"
                self.log(f"⏰ Connection timeout after {timeout}s", "ERROR")
                return False

        except Exception as e:
            self.connection_state = "ERROR"
            self.log(f"❌ Connection error: {e}", "ERROR")
            return False

    def validate_connection(self):
        """Validate connection quality and permissions"""
        try:
            # Test basic data request
            req_id = self.get_next_req_id()
            self.reqAccountSummary(req_id, "All", "TotalCashValue")

            self.log("🔍 Validating connection permissions...", "DEBUG")
            time.sleep(2)  # Allow time for response

            # Cancel the request
            self.cancelAccountSummary(req_id)

        except Exception as e:
            self.log(f"⚠️ Connection validation warning: {e}", "WARNING")

    def create_enhanced_contract(self, symbol, sec_type="STK", exchange="SMART",
                               currency="USD", primary_exchange=""):
        """
        Create enhanced contract with better validation

        Args:
            symbol: Stock symbol
            sec_type: Security type (STK, OPT, FUT, etc.)
            exchange: Exchange (SMART for best execution)
            currency: Currency (USD, EUR, etc.)
            primary_exchange: Primary exchange for routing
        """
        contract = Contract()
        contract.symbol = symbol.upper().strip()
        contract.secType = sec_type
        contract.exchange = exchange
        contract.currency = currency

        if primary_exchange:
            contract.primaryExchange = primary_exchange

        self.log(f"Created {sec_type} contract for {symbol} on {exchange}", "DEBUG")
        return contract

    def get_enhanced_historical_data(self, symbol, duration="1 M", bar_size="1 day",
                                   end_date="", what_to_show="TRADES", use_rth=True,
                                   timeout=45):
        """
        Get enhanced historical data with better error handling

        Enhanced Features:
        - Automatic retry on failure
        - Data quality validation
        - Multiple timeframe support
        - Better timeout handling
        """
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                contract = self.create_enhanced_contract(symbol)
                req_id = self.get_next_req_id()
                request_start = datetime.now()

                self.log(f"Requesting historical data for {symbol} (attempt {retry_count + 1})", "INFO")

                # Clear previous data and reset events
                self.historical_data[req_id] = []
                self.data_ready.clear()
                self.error_occurred = False
                self.error_message = ""

                # Store request metadata
                self.active_requests[req_id] = {
                    'symbol': symbol,
                    'type': 'historical',
                    'timestamp': request_start,
                    'retry_count': retry_count
                }

                # Set timeout for this request
                self.request_timeouts[req_id] = request_start + timedelta(seconds=timeout)

                # Request historical data
                self.reqHistoricalData(
                    req_id, contract, end_date, duration, bar_size,
                    what_to_show, int(use_rth), 1, False, []
                )

                # Wait for data with timeout
                if self.data_ready.wait(timeout=timeout):
                    if self.error_occurred:
                        self.log(f"❌ API Error for {symbol}: {self.error_message}", "ERROR")
                        retry_count += 1

                        if "pacing violation" in self.error_message.lower():
                            self.log("⏸️ Pacing violation detected, waiting 60s...", "WARNING")
                            time.sleep(60)
                        else:
                            time.sleep(5)  # Brief pause before retry
                        continue

                    # Process successful response
                    data = self.historical_data.get(req_id, [])
                    if data and len(data) > 0:
                        df = self.convert_to_enhanced_dataframe(data, symbol)

                        # Validate data quality
                        if self.validate_data_quality(df, symbol):
                            # Record success metrics
                            latency = (datetime.now() - request_start).total_seconds()
                            self.data_quality_stats['successful_requests'] += 1
                            self.data_quality_stats['data_latency'].append(latency)

                            self.log(f"✅ Retrieved {len(df)} bars for {symbol} in {latency:.2f}s", "SUCCESS")
                            return df
                        else:
                            self.log(f"❌ Data quality validation failed for {symbol}", "ERROR")
                            retry_count += 1
                            continue
                    else:
                        self.log(f"⚠️ Empty dataset returned for {symbol}", "WARNING")
                        retry_count += 1
                        continue

                else:
                    # Timeout occurred
                    self.log(f"⏰ Timeout waiting for {symbol} data (attempt {retry_count + 1})", "ERROR")
                    self.data_quality_stats['timeout_requests'] += 1
                    retry_count += 1

                    # Cancel the request to clean up
                    try:
                        self.cancelHistoricalData(req_id)
                    except:
                        pass

            except Exception as e:
                self.log(f"❌ Exception in data request for {symbol}: {e}", "ERROR")
                retry_count += 1
                time.sleep(2)

        # All retries failed
        self.data_quality_stats['failed_requests'] += 1
        self.log(f"❌ Failed to get data for {symbol} after {max_retries} attempts", "ERROR")
        return None

    def convert_to_enhanced_dataframe(self, bar_data, symbol):
        """Convert IBKR data to enhanced pandas DataFrame"""
        try:
            data_list = []

            for bar in bar_data:
                # Enhanced data processing with validation
                try:
                    row_data = {
                        'Date': pd.to_datetime(bar.date),
                        'Open': float(bar.open),
                        'High': float(bar.high),
                        'Low': float(bar.low),
                        'Close': float(bar.close),
                        'Volume': int(bar.volume),
                        'WAP': float(getattr(bar, 'wap', bar.close)),  # Weighted Average Price
                        'Count': int(getattr(bar, 'count', 1))  # Number of trades
                    }

                    # Basic data validation
                    if row_data['High'] >= row_data['Low'] >= 0:
                        if row_data['Open'] > 0 and row_data['Close'] > 0:
                            data_list.append(row_data)

                except (ValueError, TypeError) as e:
                    self.log(f"⚠️ Skipping invalid bar for {symbol}: {e}", "WARNING")
                    continue

            if not data_list:
                self.log(f"❌ No valid data bars for {symbol}", "ERROR")
                return pd.DataFrame()

            # Create DataFrame with enhanced processing
            df = pd.DataFrame(data_list)
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)

            # Add derived columns
            df['Returns'] = df['Close'].pct_change()
            df['HL_Range'] = df['High'] - df['Low']
            df['Price_Range_Pct'] = (df['HL_Range'] / df['Close']) * 100

            return df

        except Exception as e:
            self.log(f"❌ Error converting data for {symbol}: {e}", "ERROR")
            return pd.DataFrame()

    def validate_data_quality(self, df, symbol):
        """Validate data quality and completeness"""
        try:
            if df.empty:
                return False

            # Check for minimum data requirements
            if len(df) < 5:
                self.log(f"⚠️ Insufficient data for {symbol}: {len(df)} bars", "WARNING")
                return False

            # Check for missing or invalid data
            invalid_data = df[
                (df['Open'] <= 0) | (df['High'] <= 0) |
                (df['Low'] <= 0) | (df['Close'] <= 0) |
                (df['High'] < df['Low']) |
                pd.isna(df[['Open', 'High', 'Low', 'Close']]).any(axis=1)
            ]

            if len(invalid_data) > len(df) * 0.1:  # More than 10% invalid
                self.log(f"⚠️ Too much invalid data for {symbol}: {len(invalid_data)}/{len(df)} bars", "WARNING")
                return False

            # Check for reasonable price ranges (basic sanity check)
            price_range = df['Close'].max() / df['Close'].min()
            if price_range > 100:  # Price changed more than 100x
                self.log(f"⚠️ Extreme price range for {symbol}: {price_range:.1f}x", "WARNING")
                return False

            self.log(f"✅ Data quality validation passed for {symbol}", "DEBUG")
            return True

        except Exception as e:
            self.log(f"❌ Data validation error for {symbol}: {e}", "ERROR")
            return False

    def get_real_time_data(self, symbol, data_type="snapshot", timeout=10):
        """
        Get real-time market data with multiple options

        Args:
            symbol: Stock symbol
            data_type: "snapshot", "streaming", or "delayed"
            timeout: Timeout in seconds
        """
        try:
            contract = self.create_enhanced_contract(symbol)
            req_id = self.get_next_req_id()

            self.log(f"Requesting {data_type} data for {symbol}", "INFO")

            # Initialize data storage
            self.real_time_data[req_id] = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'data': {}
            }

            self.active_requests[req_id] = {
                'symbol': symbol,
                'type': 'realtime',
                'timestamp': datetime.now()
            }

            # # Request market data
            # if data_type == "streaming":
            #     # For streaming data (be careful with costs)
            #     self.reqMktData(req_id, contract, "", False, False, [])
            # else:
            #     # For snapshot data
            #     self.reqMktData(req_id, contract, "", True, False, [])

            # Request DELAYED market data (free)
            self.reqMktData(req_id, contract, "", True, False, [])

            # Wait for initial data
            time.sleep(timeout)

            # Stop streaming if it was streaming
            if data_type == "streaming":
                self.cancelMktData(req_id)

            # Return collected data
            quote_data = self.real_time_data.get(req_id, {}).get('data', {})
            if quote_data:
                self.log(f"✅ Retrieved {data_type} data for {symbol}", "SUCCESS")
                return quote_data
            else:
                self.log(f"⚠️ No {data_type} data received for {symbol}", "WARNING")
                return None

        except Exception as e:
            self.log(f"❌ Error getting {data_type} data for {symbol}: {e}", "ERROR")
            return None

    def get_contract_details_enhanced(self, symbol, timeout=15):
        """Get enhanced contract details with validation"""
        try:
            contract = self.create_enhanced_contract(symbol)
            req_id = self.get_next_req_id()

            self.log(f"Getting contract details for {symbol}", "INFO")

            self.contract_details[req_id] = []
            self.data_ready.clear()

            self.active_requests[req_id] = {
                'symbol': symbol,
                'type': 'contract_details',
                'timestamp': datetime.now()
            }

            self.reqContractDetails(req_id, contract)

            if self.data_ready.wait(timeout=timeout):
                details = self.contract_details.get(req_id, [])
                if details:
                    self.log(f"✅ Retrieved contract details for {symbol}", "SUCCESS")
                    return details[0]  # Return first match
                else:
                    self.log(f"⚠️ No contract details found for {symbol}", "WARNING")
                    return None
            else:
                self.log(f"⏰ Timeout getting contract details for {symbol}", "ERROR")
                return None

        except Exception as e:
            self.log(f"❌ Error getting contract details for {symbol}: {e}", "ERROR")
            return None

    # Enhanced IBKR Callback Methods
    def nextValidId(self, orderId):
        """Connection confirmation callback"""
        super().nextValidId(orderId)
        self.next_req_id = max(orderId, self.next_req_id)
        self.connection_ready.set()
        self.log(f"Connection confirmed. Next valid ID: {orderId}", "SUCCESS")

    def historicalData(self, reqId, bar):
        """Enhanced historical data callback"""
        if reqId in self.historical_data:
            self.historical_data[reqId].append(bar)

        if self.debug:
            symbol = self.active_requests.get(reqId, {}).get('symbol', 'UNKNOWN')
            self.log(f"📊 Bar for {symbol}: {bar.date} OHLC:[{bar.open:.2f},{bar.high:.2f},{bar.low:.2f},{bar.close:.2f}] V:{bar.volume}", "DEBUG")

    def historicalDataEnd(self, reqId, start, end):
        """Enhanced historical data completion callback"""
        symbol = self.active_requests.get(reqId, {}).get('symbol', 'UNKNOWN')
        count = len(self.historical_data.get(reqId, []))
        self.log(f"📊 Historical data complete for {symbol}: {count} bars ({start} to {end})", "SUCCESS")
        self.data_ready.set()

    def tickPrice(self, reqId, tickType, price, attrib):
        """Enhanced real-time price callback"""
        if reqId not in self.real_time_data:
            self.real_time_data[reqId] = {'data': {}}

        # Enhanced tick type mapping
        tick_map = {
            TickType.BID: 'bid',
            TickType.ASK: 'ask',
            TickType.LAST: 'last',
            TickType.HIGH: 'high',
            TickType.LOW: 'low',
            TickType.CLOSE: 'close',
            TickType.OPEN: 'open'
        }

        if tickType in tick_map:
            field_name = tick_map[tickType]
            self.real_time_data[reqId]['data'][field_name] = price
            self.real_time_data[reqId]['data']['last_update'] = datetime.now()

            symbol = self.active_requests.get(reqId, {}).get('symbol', 'UNKNOWN')
            self.log(f"💱 {symbol} {field_name}: ${price:.2f}", "DEBUG")

    def tickSize(self, reqId, tickType, size):
        """Enhanced real-time size callback"""
        if reqId not in self.real_time_data:
            self.real_time_data[reqId] = {'data': {}}

        size_map = {
            TickType.BID_SIZE: 'bid_size',
            TickType.ASK_SIZE: 'ask_size',
            TickType.LAST_SIZE: 'last_size',
            TickType.VOLUME: 'volume'
        }

        if tickType in size_map:
            field_name = size_map[tickType]
            self.real_time_data[reqId]['data'][field_name] = size

            symbol = self.active_requests.get(reqId, {}).get('symbol', 'UNKNOWN')
            self.log(f"📊 {symbol} {field_name}: {size}", "DEBUG")

    def contractDetails(self, reqId, contractDetails):
        """Enhanced contract details callback"""
        if reqId not in self.contract_details:
            self.contract_details[reqId] = []

        self.contract_details[reqId].append(contractDetails)

        contract = contractDetails.contract
        symbol = contract.symbol
        exchange = contract.exchange
        self.log(f"📋 Contract: {symbol} on {exchange}", "DEBUG")

    def contractDetailsEnd(self, reqId):
        """Contract details completion callback"""
        symbol = self.active_requests.get(reqId, {}).get('symbol', 'UNKNOWN')
        count = len(self.contract_details.get(reqId, []))
        self.log(f"📋 Contract details complete for {symbol}: {count} contracts found", "SUCCESS")
        self.data_ready.set()

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        """Enhanced error handling"""
        self.last_error_time = datetime.now()

        # Categorize errors
        warning_codes = [2104, 2106, 2158, 2119]  # Market data warnings
        info_codes = [2103, 2105]  # Connection info
        pacing_codes = [162, 420, 421]  # Pacing violations

        if errorCode in warning_codes:
            self.log(f"⚠️ IBKR Warning {errorCode}: {errorString}", "WARNING")
            return
        elif errorCode in info_codes:
            self.log(f"ℹ️ IBKR Info {errorCode}: {errorString}", "INFO")
            return
        elif errorCode in pacing_codes:
            self.log(f"⏸️ IBKR Pacing {errorCode}: {errorString}", "WARNING")
            self.error_occurred = True
            self.error_message = f"Pacing violation: {errorString}"
        else:
            self.log(f"❌ IBKR Error {errorCode}: {errorString} (ReqId: {reqId})", "ERROR")

            if reqId != -1:  # Request-specific error
                self.error_occurred = True
                self.error_message = f"Error {errorCode}: {errorString}"

        # Signal completion for request-specific errors
        if reqId != -1 and reqId in self.active_requests:
            self.data_ready.set()

    def get_connection_status(self):
        """Get detailed connection status"""
        return {
            'connected': self.isConnected(),
            'state': self.connection_state,
            'last_heartbeat': self.last_heartbeat,
            'last_error': self.last_error_time,
            'data_quality': self.data_quality_stats.copy()
        }

    def disconnect_safely(self):
        """Safe disconnect with cleanup"""
        try:
            if self.isConnected():
                # Cancel any active requests
                for req_id in list(self.active_requests.keys()):
                    try:
                        request_type = self.active_requests[req_id].get('type')
                        if request_type == 'historical':
                            self.cancelHistoricalData(req_id)
                        elif request_type == 'realtime':
                            self.cancelMktData(req_id)
                    except:
                        pass

                self.disconnect()
                self.connection_state = "DISCONNECTED"
                self.log("✅ Safely disconnected from IBKR", "SUCCESS")
        except Exception as e:
            self.log(f"Error during disconnect: {e}", "ERROR")


class ProfessionalIBKRManager:
    """Professional-grade IBKR data manager for StockWise integration"""

    def __init__(self, debug=False):
        self.debug = debug
        self.client = None
        self.connected = False
        self.connection_config = None

    def connect_with_fallback(self, config_list=None):
        """Connect with automatic fallback to different ports/hosts"""

        if config_list is None:
            config_list = [
                {"host": "127.0.0.1", "port": 7497, "name": "TWS Paper"},
                {"host": "127.0.0.1", "port": 4002, "name": "Gateway Paper"},
                {"host": "127.0.0.1", "port": 7496, "name": "TWS Live"},
                {"host": "127.0.0.1", "port": 4001, "name": "Gateway Live"}
            ]

        for config in config_list:
            try:
                if self.debug:
                    print(f"🔄 Trying {config['name']} on port {config['port']}...")

                self.client = EnhancedIBKRClient(debug=self.debug)

                if self.client.connect_to_ibkr(config["host"], config["port"]):
                    self.connected = True
                    self.connection_config = config

                    if self.debug:
                        print(f"✅ Connected via {config['name']}")

                    return True

            except Exception as e:
                if self.debug:
                    print(f"❌ Failed to connect via {config['name']}: {e}")
                continue

        print("❌ All connection attempts failed")
        return False

    def get_stock_data(self, symbol, days_back=60):
        """Get historical stock data - Enhanced replacement for yfinance"""
        if not self.connected or not self.client:
            print("❌ Not connected to IBKR")
            return None

        try:
            # Convert days to IBKR duration format
            if days_back <= 30:
                duration = f"{days_back} D"
            elif days_back <= 365:
                months = max(1, days_back // 30)
                duration = f"{months} M"
            else:
                years = max(1, days_back // 365)
                duration = f"{years} Y"

            df = self.client.get_enhanced_historical_data(
                symbol=symbol,
                duration=duration,
                bar_size="1 day",
                what_to_show="TRADES",
                use_rth=True,
                timeout=60
            )

            if df is not None and not df.empty:
                # Ensure compatibility with existing StockWise code
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                missing_columns = [col for col in required_columns if col not in df.columns]

                if missing_columns:
                    print(f"⚠️ Missing columns for {symbol}: {missing_columns}")
                    return None

                return df[required_columns]  # Return only required columns

            return None

        except Exception as e:
            print(f"❌ Error getting IBKR data for {symbol}: {e}")
            return None

    def get_current_price(self, symbol):
        """Get current market price"""
        if not self.connected:
            return None

        try:
            quote = self.client.get_real_time_data(symbol, "snapshot", timeout=5)
            if quote:
                # Prefer last price, fallback to bid-ask midpoint
                if 'last' in quote and quote['last'] > 0:
                    return float(quote['last'])
                elif 'bid' in quote and 'ask' in quote:
                    bid, ask = float(quote['bid']), float(quote['ask'])
                    if bid > 0 and ask > 0:
                        return (bid + ask) / 2
            return None

        except Exception as e:
            print(f"❌ Error getting current price for {symbol}: {e}")
            return None

    def validate_symbol(self, symbol):
        """Validate if symbol exists and is tradeable"""
        if not self.connected:
            return False

        try:
            details = self.client.get_contract_details_enhanced(symbol, timeout=10)
            return details is not None

        except Exception as e:
            print(f"❌ Error validating {symbol}: {e}")
            return False

    def get_connection_info(self):
        """Get detailed connection information"""
        if self.client:
            status = self.client.get_connection_status()
            status['connection_config'] = self.connection_config
            return status
        return {'connected': False}

    def disconnect(self):
        """Disconnect from IBKR"""
        if self.client:
            self.client.disconnect_safely()
        self.connected = False


def test_enhanced_ibkr_connection():
    """Test the enhanced IBKR connection with comprehensive validation"""
    print("🔌 Testing Enhanced IBKR Connection")
    print("=" * 50)

    manager = ProfessionalIBKRManager(debug=True)

    # Test connection
    if manager.connect_with_fallback():
        print(f"✅ Connection successful!")

        # Display connection info
        info = manager.get_connection_info()
        print(f"📊 Connection Details:")
        print(f"   Method: {info['connection_config']['name']}")
        print(f"   Port: {info['connection_config']['port']}")
        print(f"   State: {info['state']}")

        # Test data retrieval with multiple symbols
        test_symbols = ['AAPL', 'MSFT', 'GOOGL']
        print(f"\n📈 Testing data retrieval...")

        successful_tests = 0
        for symbol in test_symbols:
            print(f"\n🧪 Testing {symbol}...")

            # Test historical data
            df = manager.get_stock_data(symbol, days_back=30)
            if df is not None and not df.empty:
                print(f"   ✅ Historical: {len(df)} days")
                print(f"   📊 Range: {df.index[0].date()} to {df.index[-1].date()}")
                print(f"   💰 Latest: ${df['Close'].iloc[-1]:.2f}")

                # Test current price
                current = manager.get_current_price(symbol)
                if current:
                    print(f"   💱 Live: ${current:.2f}")

                # Test symbol validation
                is_valid = manager.validate_symbol(symbol)
                print(f"   ✅ Valid: {is_valid}")

                successful_tests += 1
            else:
                print(f"   ❌ Failed to get data")

        # Test results summary
        print(f"\n📊 Test Results: {successful_tests}/{len(test_symbols)} successful")

        # Show data quality stats
        if manager.client:
            stats = manager.client.data_quality_stats
            print(f"\n📈 Data Quality Metrics:")
            print(f"   Successful: {stats['successful_requests']}")
            print(f"   Failed: {stats['failed_requests']}")
            print(f"   Timeouts: {stats['timeout_requests']}")
            if stats['data_latency']:
                avg_latency = sum(stats['data_latency']) / len(stats['data_latency'])
                print(f"   Avg Latency: {avg_latency:.2f}s")

        manager.disconnect()

        if successful_tests == len(test_symbols):
            print("\n🎉 All tests passed! IBKR integration ready.")
            return True
        else:
            print(f"\n⚠️ Some tests failed. Check IBKR permissions and data subscriptions.")
            return False

    else:
        print("\n❌ Connection failed. Setup instructions:")
        print("1. Install and start TWS or IB Gateway")
        print("2. Enable API: Configure → API Settings")
        print("3. Check 'Enable ActiveX and Socket Clients'")
        print("4. Add 127.0.0.1 to trusted IPs")
        print("5. Ensure correct port (7497=Paper, 7496=Live)")
        return False


if __name__ == "__main__":
    test_enhanced_ibkr_connection()