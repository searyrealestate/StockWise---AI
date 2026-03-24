# alpace_test_connection.py

"""
🚀 Alpaca API Connection Test Script (Updated)
==============================================
Diagnostic script to test connection to Alpaca Markets.
Supports loading keys directly from .streamlit/secrets.toml

Checks:
1. Key Authentication (secrets.toml / Env Vars).
2. Access to Account Info.
3. Access to Real-time Market Data (IEX Feed for Free Plan).
4. Ability to download historical data.
"""

import os
import time
import pandas as pd
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta

# נסיון לייבא toml לקריאת קובץ הסיסמאות
try:
    import toml
except ImportError:
    toml = None

# --- Configuration Loading Logic ---
ALPACA_KEY = None
ALPACA_SECRET = None
BASE_URL = "https://paper-api.alpaca.markets"

# 1. Try loading from .streamlit/secrets.toml (Priority 1)
secrets_path = os.path.join(os.getcwd(), ".streamlit", "secrets.toml")
if os.path.exists(secrets_path) and toml:
    try:
        secrets = toml.load(secrets_path)
        # Note: Keys usually match the variable names in TOML
        ALPACA_KEY = secrets.get("APCA_API_KEY_ID")
        ALPACA_SECRET = secrets.get("APCA_API_SECRET_KEY")
        if ALPACA_KEY:
            print("ℹ️  Loaded credentials from .streamlit/secrets.toml")
    except Exception as e:
        print(f"⚠️  Found secrets.toml but failed to load: {e}")

# 2. Fallback: Environment Variables
if not ALPACA_KEY:
    ALPACA_KEY = os.getenv("APCA_API_KEY_ID")
    ALPACA_SECRET = os.getenv("APCA_API_SECRET_KEY")

# 3. Fallback: Try system_config.py
if not ALPACA_KEY:
    try:
        import system_config
        ALPACA_KEY = system_config.ALPACA_KEY
        ALPACA_SECRET = system_config.ALPACA_SECRET
        print("ℹ️  Loaded credentials from system_config.py")
    except ImportError:
        pass

def print_header():
    print("\n" + "="*50)
    print("🦙 ALPACA API CONNECTION TEST")
    print("="*50)

def test_alpaca_connection():
    """Performs basic connection test and retrieves account details."""
    
    if not ALPACA_KEY or not ALPACA_SECRET:
        print("❌ Error: Missing API Credentials.")
        print("   Could not find keys in .streamlit/secrets.toml, Env Vars, or system_config.")
        return None

    print(f"🔑 Testing Credentials (Key ends in: ...{str(ALPACA_KEY)[-4:]})")
    
    try:
        # Create the API Object
        api = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, BASE_URL, api_version='v2')
        
        # 1. Check Account
        account = api.get_account()
        
        status_icon = "✅" if account.status == 'ACTIVE' else "⚠️"
        print(f"{status_icon} Connection Successful!")
        print(f"   • Status: {account.status}")
        print(f"   • Buying Power: ${float(account.buying_power):,.2f}")
        print(f"   • Cash: ${float(account.cash):,.2f}")
        
        # 2. Check Market Clock
        clock = api.get_clock()
        market_status = "OPEN 🟢" if clock.is_open else "CLOSED 🔴"
        print(f"   • Market Status: {market_status}")
        
        return api

    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        return None

def test_market_data(api, symbol="AAPL"):
    """Tests the ability to fetch real-time market data."""
    print(f"\n📡 Testing Real-Time Data for {symbol}...")
    try:
        # Fetch latest trade
        latest_trade = api.get_latest_trade(symbol)
        print(f"✅ Success! Latest {symbol} Trade:")
        print(f"   • Price: ${latest_trade.price}")
        print(f"   • Time: {latest_trade.timestamp}")
        return True
    except Exception as e:
        print(f"❌ Data Error: {e}")
        return False

def download_historical_data(api, symbols):
    """Downloads historical data - FIXED for Free Plan (IEX Feed)."""
    print(f"\n📥 Downloading Historical Data (1 Year)...")
    
    successful_downloads = 0
    os.makedirs("data", exist_ok=True)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    for symbol in symbols:
        try:
            print(f"   ⏳ Fetching {symbol}...", end="\r")
            
            # --- CRITICAL FIX FOR FREE PLAN: feed='iex' ---
            bars = api.get_bars(
                symbol, 
                tradeapi.rest.TimeFrame.Day, 
                start=start_str, 
                end=end_str,
                adjustment='raw',
                feed='iex'  # <--- MUST use 'iex' for free accounts
            ).df

            if not bars.empty:
                filename = f"data/{symbol}_alpaca_history.csv"
                bars.to_csv(filename)
                print(f"   ✅ Saved {len(bars)} rows -> {filename}")
                successful_downloads += 1
            else:
                print(f"   ⚠️  No data found for {symbol}")
                
        except Exception as e:
            print(f"   ❌ Failed {symbol}: {e}")
            
    return successful_downloads

if __name__ == "__main__":
    print_header()
    
    # 1. Connect
    api = test_alpaca_connection()
    
    if api:
        # 2. Test single data point
        data_ok = test_market_data(api, "SPY")
        
        # 3. Download Data
        if data_ok:
            response = input("\n❓ Download sample NASDAQ data? (y/n): ").strip().lower()
            if response == 'y':
                tickers = ["NVDA", "TSLA", "AAPL", "MSFT", "AMD", "QQQ"]
                download_historical_data(api, tickers)
    
    print("\n🏁 Test Complete.")