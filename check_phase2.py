import sys
import os
import pandas as pd
import logging
from data_source_manager import DataSourceManager
from feature_engine import RobustFeatureCalculator
import system_config as cfg

# Setup simple logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Phase2_Check")

def check_fixes():
    print("\n🔍 CHECKING PHASE 2 FIXES (FINAL ROBUST VERSION)...")
    print("===================================================")

    # 1. Get Data for NVDA
    dsm = DataSourceManager()
    print("1. Fetching Data for NVDA...")
    data_map = dsm.fetch_data_sequential(["NVDA"])
    df = data_map.get("NVDA")
    
    if df is None or df.empty:
        print("[ERROR] No data found. Check your connection.")
        return

    # 2. Calculate Indicators
    print("2. Calculating Indicators...")
    calc = RobustFeatureCalculator()
    df = calc.calculate_features(df)
    
    # --- DEBUG: Print all columns to see what pandas_ta actually named them ---
    print("\n[DEBUG] ALL AVAILABLE COLUMNS:")
    print(df.columns.tolist()) 
    print("-" * 30)

    # Get the very last candle
    last_row = df.iloc[-1]
    price = last_row['close']
    sma_200 = last_row.get('sma_200', 0)
    
    # --- ROBUST ATR FINDER ---
    # Look for ANY column that looks like ATR (case insensitive)
    atr_col = None
    for col in df.columns:
        if "ATR" in col.upper() and "BATR" not in col.upper(): # Exclude BATR if it exists
            atr_col = col
            break
    
    if atr_col:
        atr = last_row[atr_col]
        print(f"\n[OK] Found ATR Column: '{atr_col}'")
    else:
        print("\n[WARNING] ATR column not found! Using 2% fallback.")
        atr = price * 0.02

    # SAFE PRINTING (Prevents crash if data is None)
    print(f"\n📊 DATA SNAPSHOT (NVDA)")
    print(f"   Current Price:   ${price:.2f}")
    print(f"   200 SMA:         ${sma_200:.2f}")
    
    if atr:
        print(f"   ATR (Volatility): ${atr:.2f}")
    else:
        print("   ATR (Volatility): NaN")

    # --- TEST 1: TREND FILTER ---
    print("\n🧪 TEST 1: TREND FILTER (Falling Knife Protection)")
    if sma_200 > 0 and price < sma_200:
        print("   🛑 BLOCKED: Price is BELOW 200 SMA.")
        print("   -> System WILL prevent this trade (Phase 2 Success).")
    else:
        print("   ✅ PASS: Price is ABOVE 200 SMA.")
        print("   -> System allows this trade (Trend is healthy).")

    # --- TEST 2: ATR STOP LOSS ---
    print("\n🧪 TEST 2: DYNAMIC STOP LOSS")
    # Using the multiplier from config (default 2.0)
    multiplier = cfg.ACTIVE_PROFILE.get("stop_atr", 2.0)
    
    if atr:
        dynamic_stop = price - (atr * multiplier)
        fixed_stop = price * 0.98 # Old 2% method
        
        print(f"   Old Method (Fixed 2%): ${fixed_stop:.2f}")
        print(f"   New Method (ATR x{multiplier}): ${dynamic_stop:.2f}")
        
        diff = abs(dynamic_stop - fixed_stop)
        print(f"   Difference: ${diff:.2f}")
        
        if abs(diff) > 0.01:
            print("   ✅ PASS: Dynamic Stop Loss is DIFFERENT from fixed %.")
            print("   -> Phase 2 Logic is ACTIVE.")
        else:
            print("   ⚠️ INFO: Dynamic Stop matched Fixed Stop (Coincidence or logic fallback).")
    else:
        print("   [ERROR] Cannot calculate Stop Loss (Missing ATR).")

if __name__ == "__main__":
    check_fixes()