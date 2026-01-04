import pandas as pd
import logging
import sys
import system_config as cfg
from data_source_manager import DataSourceManager
from feature_engine import RobustFeatureCalculator
from strategy_engine import StrategyOrchestra
from stockwise_ai_core import StockWiseAI
import json

import os
import csv
from datetime import datetime

# cfg.TELEGRAM_TOKEN = None

# Setup Logging
os.makedirs("logs", exist_ok=True)
timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"logs/verification_actions_{timestamp_str}.log"

# Force reconfiguration to ensure we capture everything
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_filename, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
print(f"[INIT] Logging to: {log_filename}")
logger = logging.getLogger("SniperVerifier")

def run_verification(symbols=None):
    if symbols is None:
        symbols = ["NVDA", "GOOGL", "AMZN", "META", "AAPL", "INTC", "QCOM"]
        
    logger.info(f"\n[DEBUG] STARTING SNIPER LOGIC VERIFICATION...")
    
    # Global Statistics
    grand_stats = {
        "Total_Signals": 0,
        "Wins": 0,
        "Losses": 0,
        "Pending": 0,
        "Total_PnL_Pct": 0.0
    }

    # Initialize dsm once
    dsm = DataSourceManager()
    
    csv_data = []

    for symbol in symbols:
        logger.info(f"\n{'='*40}")
        logger.info(f"PROCESSING SYMBOL: {symbol}")
        logger.info(f"{'='*40}")
        
        # 1. Fetch Data
        df_raw = dsm.get_stock_data(symbol, days_back=400, interval='1d')
        fundamentals = dsm.get_fundamentals(symbol)
        
        if df_raw is None or df_raw.empty:
            logger.error(f"[ERROR] No data fetched for {symbol}.")
            continue

        # 2. Add Features (REQUIRED for AI)
        calc = RobustFeatureCalculator()
        df = calc.calculate_features(df_raw)
        
        # 3. Initialize AI Core
        ai_core = StockWiseAI()
        
        logger.info(f"[INFO] Analyzing last 252 days (1 Year) of {symbol}...")
        logger.info("=" * 130)
        logger.info(f"{'DATE':<12} | {'PRICE':<8} | {'TARGET':<8} | {'STOP':<8} | {'FUND':<5} | {'AI_CONF':<8} | {'DECISION':<10} | {'OUTCOME'}")
        logger.info("=" * 130)

        # 4. Simulate Day-by-Day (Loop through the last 252 days)
        sim_window = 252
        start_idx = max(60, len(df) - sim_window)
        
        # Per-Symbol Stats
        sym_stats = {"Signals": 0, "Wins": 0, "Losses": 0, "Pending": 0}

        def check_outcome(entry_idx, entry_price, target, stop):
            """Look ahead to see if Target or Stop is hit first."""
            for future_i in range(entry_idx + 1, len(df)):
                future_bar = df.iloc[future_i]
                high = future_bar['high']
                low = future_bar['low']
                
                if low <= stop:
                    return "LOSS", stop
                if high >= target:
                    return "WIN", target
            return "PENDING", df.iloc[-1]['close']

        for i in range(start_idx, len(df)):
            current_slice = df.iloc[:i+1] 
            today = current_slice.iloc[-1]
            date_str = str(today.name)[:10]
            
            features = today.to_dict()
            decision, prob, trace = ai_core.predict_trade_confidence(symbol, features, fundamentals, current_slice)
            
            fund_pass = trace['Checks']['Fundamentals']['Pass']
            fund_score = "PASS" if fund_pass else "FAIL"
            price_val = today['close']
            price_str = f"{price_val:.2f}"
            
            # --- MODIFIED: Use ATR-based dynamic stops ---
            # If 'atr' exists in features, use it. Else default to 2% of price.
            atr = features.get('atr', price_val * 0.02)
            
            # Use the new config values
            # --- NEW ADAPTIVE TARGETS ---
            stop_loss, target_price, vol_regime = StrategyOrchestra.get_adaptive_targets(today, price_val)
            # Log the regime so you can see it working
            logger.info(f"   -> Volatility Regime: {vol_regime}")
            # target_price = price_val * (1 + cfg.SniperConfig.TARGET_PROFIT)
            # stop_loss = price_val * (1 + cfg.SniperConfig.MAX_DRAWDOWN)
            
            threshold = cfg.SniperConfig.MODEL_CONFIDENCE_THRESHOLD
            outcome_str = "-"
            
            # --- HARD TREND FILTER (Falling Knife Protection) ---
            sma_200 = today.get('sma_200', 0)
            is_falling_knife = (sma_200 > 0) and (price_val < sma_200 * 0.90)
            
            if decision == "BUY":
                if is_falling_knife:
                    row_color = "[FILTER]"
                    outcome_str = "BLOCKED_BY_TREND"
                    # Do NOT increment stats
                else:
                    row_color = "[BUY] " 
                    sym_stats["Signals"] += 1
                    grand_stats["Total_Signals"] += 1
                    
                    outcome, exit_price = check_outcome(i, price_val, target_price, stop_loss)
                    outcome_str = outcome
                    
                    if outcome == "WIN":
                        sym_stats["Wins"] += 1
                        grand_stats["Wins"] += 1
                        grand_stats["Total_PnL_Pct"] += cfg.SniperConfig.TARGET_PROFIT
                        row_color = "[WIN] " 
                    elif outcome == "LOSS":
                        sym_stats["Losses"] += 1
                        grand_stats["Losses"] += 1
                        grand_stats["Total_PnL_Pct"] += cfg.SniperConfig.MAX_DRAWDOWN
                        row_color = "[LOSS]" 
                    else:
                        sym_stats["Pending"] += 1
                        grand_stats["Pending"] += 1
                    
            elif prob >= (threshold * 0.8): 
                row_color = "[WATCH]" 
            else:
                row_color = "[WAIT] " 
                
            row_str = f"{row_color} {date_str} | {price_str:<8} | {target_price:<8.2f} | {stop_loss:<8.2f} | {fund_score} | {prob:.2%}   | {decision:<10} | {outcome_str}"
            logger.info(row_str)
            
            csv_data.append({
                "Symbol": symbol,
                "Date": date_str,
                "Entry_Price": price_val,
                "Target_Price": target_price,
                "Stop_Loss": stop_loss,
                "Decision": decision,
                "Confidence": prob,
                "Fundamentals": fund_score,
                "Outcome": outcome_str,
                "Reason": trace.get('Final_Decision', ''),
                "Trace": str(trace)
            })

# Calculate win rate safely
        total_closed = sym_stats['Wins'] + sym_stats['Losses']
        if total_closed > 0:
            symbol_win_rate = (sym_stats['Wins'] / total_closed) * 100
        else:
            symbol_win_rate = 0.0
        
        logger.info("-" * 60)
        logger.info(f"Summary for {symbol}: Wins: {sym_stats['Wins']} | Losses: {sym_stats['Losses']} | Signals: {sym_stats['Signals']} | Win Rate: {symbol_win_rate:.2f}%")
        logger.info("-" * 60)

    # 5. Grand Statistics
    win_rate = 0.0
    if grand_stats["Wins"] + grand_stats["Losses"] > 0:
        win_rate = (grand_stats["Wins"] / (grand_stats["Wins"] + grand_stats["Losses"])) * 100
        
    logger.info("\n")
    logger.info("GLOBAL VERIFICATION STATISTICS")
    logger.info("=" * 40)
    logger.info(f"Total Signals:   {grand_stats['Total_Signals']}")
    logger.info(f"Correct (Wins):  {grand_stats['Wins']}")
    logger.info(f"Losses:          {grand_stats['Losses']}")
    logger.info(f"Pending/Active:  {grand_stats['Pending']}")
    logger.info(f"Win Rate:        {win_rate:.2f}%")
    start_cap = cfg.SniperConfig.SIMULATION_STARTING_CAPITAL
    total_pnl_usd = start_cap * grand_stats["Total_PnL_Pct"]
    final_balance = start_cap + total_pnl_usd
    
    logger.info(f"Start Capital:   ${start_cap:,.2f}")
    logger.info(f"Est. Total PnL:  {grand_stats['Total_PnL_Pct']*100:.2f}% (Simple Sum)")
    logger.info(f"Total PnL ($):   ${total_pnl_usd:,.2f}")
    logger.info(f"Final Balance:   ${final_balance:,.2f}")
    logger.info("=" * 40)
    
    logger.info("[DONE] All Verifications Complete.")
    
    # Save CSV
    csv_path = "logs/verification_data.csv"
    keys = csv_data[0].keys() if csv_data else []
    if keys:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(csv_data)
        logger.info(f"[INFO] Detailed data saved to {csv_path}")
    
    track_system_improvements(grand_stats, win_rate)

# Performance Tracker (Tracks improvements over time)
def track_system_improvements(current_stats, current_win_rate):
    history_file = "logs/history_stats.json"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 1. Prepare Current Record
    current_record = {
        "timestamp": timestamp,
        "win_rate": current_win_rate,
        "total_pnl_pct": current_stats["Total_PnL_Pct"],
        "total_signals": current_stats["Total_Signals"]
    }

    # 2. Load History
    history = []
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
        except:
            history = []

    # 3. Compare with Last Run (if exists)
    logger.info("\n🚀 SYSTEM IMPROVEMENT REPORT")
    logger.info("=" * 40)
    
    if history:
        last_run = history[-1]
        
        # Calculate Deltas
        delta_wr = current_win_rate - last_run['win_rate']
        delta_pnl = current_stats["Total_PnL_Pct"] - last_run['total_pnl_pct']
        delta_sig = current_stats["Total_Signals"] - last_run['total_signals']
        
        # Formatting
        wr_icon = "🟢" if delta_wr >= 0 else "🔴"
        pnl_icon = "🟢" if delta_pnl >= 0 else "🔴"
        
        logger.info(f"Previous Run: {last_run['timestamp']}")
        logger.info(f"Win Rate:     {last_run['win_rate']:.2f}% -> {current_win_rate:.2f}%  ({wr_icon} {delta_wr:+.2f}%)")
        logger.info(f"Total PnL:    {last_run['total_pnl_pct']*100:.2f}% -> {current_stats['Total_PnL_Pct']*100:.2f}%  ({pnl_icon} {delta_pnl*100:+.2f}%)")
        logger.info(f"Signal Count: {last_run['total_signals']} -> {current_stats['Total_Signals']}  (Diff: {delta_sig:+})")
        
        if delta_wr > 0:
            logger.info("\n✅ VERDICT: System Logic has IMPROVED since last run.")
        elif delta_wr < 0:
            logger.info("\n⚠️ VERDICT: System Logic has REGRESSED. Check recent changes.")
        else:
            logger.info("\nℹ️ VERDICT: Performance is STABLE (No Change).")
            
    else:
        logger.info("ℹ️ First run recorded. Improvements will be calculated next time.")

    # 4. Save Updates
    history.append(current_record)
    # Keep only last 50 runs to save space
    if len(history) > 50: 
        history = history[-50:]
        
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=4)

if __name__ == "__main__":
    run_verification()