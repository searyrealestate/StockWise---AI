# verify_brokers.py
import system_config as cfg
from data_source_manager import DataSourceManager
import pandas as pd
import time
from massive import RESTClient

def test_single_broker(broker_name):
    print(f"\n" + "="*40)
    print(f"[TEST] TESTING BROKER: {broker_name}")
    print("="*40)

    # 1. RESET ALL TO FALSE
    cfg.EN_MASSIVE = False
    cfg.EN_ALPACA = False
    cfg.EN_IBKR = False
    cfg.EN_YFINANCE = False

    # 2. ENABLE TARGET ONLY
    if broker_name == "MASSIVE": cfg.EN_MASSIVE = True
    elif broker_name == "ALPACA": cfg.EN_ALPACA = True
    elif broker_name == "IBKR": cfg.EN_IBKR = True
    elif broker_name == "YFINANCE": cfg.EN_YFINANCE = True

    # 3. INITIALIZE MANAGER
    # Re-instantiate so it picks up the new config
    dm = DataSourceManager()
    
    # 4. RUN TEST
    symbol = "SPY"
    start_t = time.time()
    try:
        print(f"   > Requesting {symbol} via {broker_name}...")
        df = dm.get_stock_data(symbol, interval="1d", days_back=5)
        
        duration = time.time() - start_t
        
        if not df.empty:
            print(f"   [PASS] SUCCESS! Received {len(df)} rows.")
            print(f"   [TIME] Latency: {duration:.2f}s")
            print(f"   [DATA] Last Price: {df.iloc[-1]['close']}")
            return True
        else:
            print(f"   [FAIL] FAILURE: Dataframe is empty.")
            return False
            
    except Exception as e:
        print(f"   [FAIL] CRITICAL ERROR: {e}")
        return False

def run_full_diagnostic():
    results = {}
    
    # Test Order
    brokers = ["MASSIVE", "ALPACA", "IBKR", "YFINANCE"]  
    
    for b in brokers:
        success = test_single_broker(b)
        results[b] = "PASS" if success else "FAIL"
        time.sleep(1) 

    print("\n" + "="*40)
    print("DIAGNOSTIC SUMMARY")
    print("="*40)
    for b, status in results.items():
        print(f"[{status}] {b}")

if __name__ == "__main__":
    run_full_diagnostic()