# verify_sniper_logic.py

# --- 1. IMPORTS ---
import pandas as pd  # Import pandas for handling data tables (DataFrames)
import logging  # Import logging to save execution history to files/console
import sys  # Import sys to interact with the Python interpreter/system
import system_config as cfg  # Import global configuration (paths, thresholds, tickers)
from data_source_manager import DataSourceManager  # Import class to fetch stock prices
from feature_engine import RobustFeatureCalculator  # Import class to calculate indicators (RSI, ADX)
from strategy_engine import StrategyOrchestra  # Import the "Brain" that makes buy/sell decisions
from stockwise_ai_core import StockWiseAI  # Import the AI model wrapper
import json  # Import JSON library for saving history stats
import os  # Import OS library for file path management
import csv  # Import CSV library for saving trade logs
from datetime import datetime  # Import datetime for timestamping

# --- 2. CRITICAL WINDOWS FIX ---
# Windows consoles often crash when printing emojis (🚀, 🔴). This forces UTF-8 encoding.
if sys.platform.startswith('win'):  # Check if the OS is Windows
    try:
        sys.stdout.reconfigure(encoding='utf-8')  # Force standard output to accept Unicode/Emojis
    except AttributeError:
        pass  # If Python version is too old, ignore this line

# --- 3. LOGGING SETUP ---
# Create the 'logs' folder if it doesn't exist yet
os.makedirs("logs", exist_ok=True)
# Generate a unique timestamp for this run (e.g., 2026-01-09_14-30-00)
timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# Define the log file name
log_filename = f"logs/verification_actions_{timestamp_str}.log"

# Remove any existing log handlers to prevent duplicate lines in output
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configure the logging system
logging.basicConfig(
    level=logging.INFO,  # Set logging level to INFO (show everything important)
    format='%(asctime)s | %(levelname)s | %(message)s',  # Set the format: Time | Level | Message
    datefmt='%Y-%m-%d %H:%M:%S',  # Set date format
    handlers=[
        logging.FileHandler(log_filename, mode='w', encoding='utf-8'),  # Save logs to file (UTF-8 enabled)
        logging.StreamHandler(sys.stdout)  # Print logs to the screen/console
    ]
)
# Print the log file location to the console
print(f"[INIT] Logging to: {log_filename}")
# Create a logger instance named 'SniperVerifier'
logger = logging.getLogger("SniperVerifier")

# --- 4. MAIN VERIFICATION FUNCTION ---
def run_verification(symbols=None, start_date=None):
    """
    Runs the backtest simulation.
    :param symbols: List of tickers to test.
    :param start_date: (Optional) Date to start the test from (for strict testing).
    """
    # If no symbols provided, load the default list from config
    if symbols is None:
        symbols = cfg.TRAINING_SYMBOLS
        
    # Log the start of the process
    logger.info(f"\n[DEBUG] STARTING SNIPER LOGIC VERIFICATION (GEN-10)...")
    
    # Log which mode we are running in (Strict vs Rolling)
    if start_date:
        logger.info(f"📅 MODE: Strict Testing (Starting from {start_date})")
    else:
        logger.info(f"📅 MODE: Rolling Window (Last 252 Days)")
    
    # Initialize a dictionary to track total statistics across all symbols
    grand_stats = {"Total_Signals": 0, "Wins": 0, "Losses": 0, "Pending": 0, "Total_PnL_Pct": 0.0}
    # Initialize the Data Manager
    dsm = DataSourceManager()
    
    # --- 5. FETCH MARKET CONTEXT ---
    # We need QQQ data to calculate 'Beta' and 'Correlation' for the features
    logger.info("Fetching Market Context (QQQ) for Beta/Correlation...")
    qqq_df = dsm.get_stock_data("QQQ", days_back=800, interval='1d')
    
    # Validation: Did we get QQQ data?
    if qqq_df is None or qqq_df.empty:
        logger.error("CRITICAL: Failed to fetch QQQ. Context features will be dead.")
        context_data = {}  # Set empty context if failed
    else:
        context_data = {'qqq': qqq_df}  # Pack QQQ into context dictionary
    
    # List to store individual trade details for CSV export
    csv_data = []

    # --- 6. SYMBOL LOOP ---
    # Iterate through every stock symbol in the list
    for symbol in symbols:
        logger.info(f"\n{'='*40}")  # Print separator
        logger.info(f"PROCESSING SYMBOL: {symbol}")  # Log current symbol
        logger.info(f"{'='*40}")  # Print separator
        
        # A. Fetch Price Data for the symbol
        df_raw = dsm.get_stock_data(symbol, days_back=800, interval='1d')
        # Fetch Fundamental Data (PE ratio, etc.)
        fundamentals = dsm.get_fundamentals(symbol)
        
        # Validation: Check if data exists
        if df_raw is None or df_raw.empty:
            logger.error(f"[ERROR] No data fetched for {symbol}.")
            continue  # Skip to next symbol

        # B. Calculate Features (Indicators)
        calc = RobustFeatureCalculator()  # Initialize calculator
        # Compute RSI, ADX, etc., passing the QQQ context we fetched earlier
        df = calc.calculate_features(df_raw, context_data=context_data)
        
        # C. Initialize AI Brain
        # Create an AI instance specifically for this symbol (loads 'NVDA_gen9_model.keras')
        ai_core = StockWiseAI(symbol=symbol)
        
        # Check if the model loaded successfully
        if ai_core.model is None:
             # Try one more time to load
             if not ai_core.load_inference_model():
                 logger.warning(f"⚠️ Model for {symbol} failed to load. AI Probability will be 0.0.")
        
        # --- 7. DETERMINE START DATE ---
        # Logic to decide where the simulation begins
        if start_date:
            try:
                # Convert string date to Timestamp
                target_ts = pd.Timestamp(start_date)
                # Ensure DataFrame index is in Datetime format
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                
                # Filter for data AFTER the start date
                future_mask = df.index >= target_ts
                
                # If no data exists after start_date, skip
                if not future_mask.any():
                    logger.warning(f"Start date {start_date} is beyond available data range.")
                    continue
                    
                # Find the row index of the start date
                start_idx = df.index.get_loc(df[future_mask].index[0])
                # Ensure we start at least 60 days in (to allow indicators to warm up)
                start_idx = max(60, start_idx)
                logger.info(f"[INFO] Simulating Out-of-Sample: {start_date} to Present")
            except Exception as e:
                # If date parsing fails, fall back to default
                logger.error(f"Failed to set start date: {e}. Falling back.")
                start_idx = max(60, len(df) - 252)
        else:
            # Default: Run simulation on the last 252 days (1 Trading Year)
            sim_window = 252
            start_idx = max(60, len(df) - sim_window)
            logger.info(f"[INFO] Simulating Last {sim_window} Days")

        # Log table header for the simulation output
        logger.info("=" * 130)
        logger.info(f"{'DATE':<12} | {'PRICE':<8} | {'TARGET':<8} | {'STOP':<8} | {'FUND':<5} | {'AI_CONF':<8} | {'DECISION':<10} | {'OUTCOME'}")
        logger.info("=" * 130)

        # Reset stats for this specific symbol
        sym_stats = {"Signals": 0, "Wins": 0, "Losses": 0, "Pending": 0}

        # --- 8. HELPER FUNCTION: CHECK OUTCOME ---
        def check_outcome(entry_idx, entry_price, target, stop):
            """Looks into the future data to see if Target or Stop was hit."""
            # Loop from the day AFTER entry to the end of data
            for future_i in range(entry_idx + 1, len(df)):
                future_bar = df.iloc[future_i]
                high = future_bar['high']  # Daily High
                low = future_bar['low']    # Daily Low
                
                # Did price drop below Stop Loss?
                if low <= stop: return "LOSS", stop
                # Did price rise above Target?
                if high >= target: return "WIN", target
            # If neither happened by end of data
            return "PENDING", df.iloc[-1]['close']

        # --- 9. SIMULATION LOOP ---
        # Iterate day by day from start_idx to end
        for i in range(start_idx, len(df)):
            current_slice = df.iloc[:i+1]  # Slice dataframe up to 'today'
            today = current_slice.iloc[-1] # Get 'today's' row
            date_str = str(today.name)[:10] # Get date string
            
            features = today.to_dict() # Convert row to dictionary
            
            # A. ASK AI FOR PREDICTION
            # Calls AI Core to get probability and trace
            decision_raw, prob, trace = ai_core.predict_trade_confidence(symbol, features, fundamentals, current_slice)
            
            # Extract fundamental score from trace
            fund_score_val = trace['Checks']['Fundamentals']['Score']
            
            # B. RUN STRATEGY ENGINE (The Conductor)
            # Package inputs for the Strategy Engine
            analysis_packet = {
                'AI_Probability': prob,
                'Fundamental_Score': fund_score_val
            }
            # Ask Strategy Engine for final decision (BUY / WAIT)
            decision = StrategyOrchestra.decide_action(symbol, today, analysis_packet)
            
            # C. GET PRICE & TARGETS
            price_val = today['close']
            price_str = f"{price_val:.2f}"
            
            # Calculate dynamic Stop Loss and Target Price based on Regime
            stop_loss, target_price, vol_regime = StrategyOrchestra.get_adaptive_targets(today, price_val)
            
            threshold = cfg.SniperConfig.MODEL_CONFIDENCE_THRESHOLD # Get config threshold
            outcome_str = "-" # Default outcome string
            
            # D. EXECUTE TRADE (If Decision is BUY)
            if decision == "BUY":
                row_color = "[BUY] " # Log tag
                sym_stats["Signals"] += 1 # Update symbol stats
                grand_stats["Total_Signals"] += 1 # Update global stats
                
                # Check the future outcome
                outcome, exit_price = check_outcome(i, price_val, target_price, stop_loss)
                outcome_str = outcome
                
                # Update Win/Loss Stats
                if outcome == "WIN":
                    sym_stats["Wins"] += 1
                    grand_stats["Wins"] += 1
                    # Calculate % Profit
                    grand_stats["Total_PnL_Pct"] += (target_price/price_val - 1)
                    row_color = "[WIN] " 
                elif outcome == "LOSS":
                    sym_stats["Losses"] += 1
                    grand_stats["Losses"] += 1
                    # Calculate % Loss
                    grand_stats["Total_PnL_Pct"] += (stop_loss/price_val - 1)
                    row_color = "[LOSS]" 
                else:
                    sym_stats["Pending"] += 1
                    grand_stats["Pending"] += 1
            
            # E. LOGGING NON-BUYS (Optional)
            # If AI liked it (>80%) but Strategy said WAIT, log as WATCH
            elif prob >= (threshold * 0.8): 
                row_color = "[WATCH]" 
            else:
                row_color = "[WAIT] " 
                
            # F. PRINT LOG ROW
            if decision == "BUY":
                # Format the log string
                row_str = f"{row_color} {date_str} | {price_str:<8} | {target_price:<8.2f} | {stop_loss:<8.2f} | {fund_score_val} | {prob:.2%}   | {decision:<10} | {outcome_str}"
                logger.info(row_str) # Print it
                
                # Add to CSV list
                csv_data.append({
                    "Symbol": symbol,
                    "Date": date_str,
                    "Entry_Price": price_val,
                    "Target_Price": target_price,
                    "Stop_Loss": stop_loss,
                    "Decision": decision,
                    "Confidence": prob,
                    "Fundamentals": fund_score_val,
                    "Outcome": outcome_str,
                    "Trace": str(trace)
                })

        # G. END OF SYMBOL SUMMARY
        total_closed = sym_stats['Wins'] + sym_stats['Losses']
        # Calculate Win Rate safely (avoid division by zero)
        symbol_win_rate = (sym_stats['Wins'] / total_closed * 100) if total_closed > 0 else 0.0
        
        logger.info("-" * 60)
        logger.info(f"Summary for {symbol}: Wins: {sym_stats['Wins']} | Losses: {sym_stats['Losses']} | Signals: {sym_stats['Signals']} | Win Rate: {symbol_win_rate:.2f}%")
        logger.info("-" * 60)

    # --- 10. GRAND TOTAL REPORT ---
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
    
    # Calculate Financials
    start_cap = cfg.SniperConfig.SIMULATION_STARTING_CAPITAL
    # Estimate final balance (Simple Sum PnL for approximation)
    final_balance = start_cap * (1 + grand_stats["Total_PnL_Pct"])
    total_pnl_usd = final_balance - start_cap
    
    logger.info(f"Start Capital:   ${start_cap:,.2f}")
    logger.info(f"Est. Total PnL:  {grand_stats['Total_PnL_Pct']*100:.2f}%")
    logger.info(f"Total PnL ($):   ${total_pnl_usd:,.2f}")
    logger.info(f"Final Balance:   ${final_balance:,.2f}")
    logger.info("=" * 40)
    logger.info("[DONE] All Verifications Complete.")
    
    # --- 11. SAVE DATA ---
    csv_path = "logs/verification_data.csv"
    if csv_data:
        keys = csv_data[0].keys()
        # Open CSV file for writing
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader() # Write header
            dict_writer.writerows(csv_data) # Write rows
        logger.info(f"[INFO] Detailed data saved to {csv_path}")
    
    # Track improvement history
    track_system_improvements(grand_stats, win_rate)

# --- 12. IMPROVEMENT TRACKER ---
def track_system_improvements(current_stats, current_win_rate):
    """
    Saves performance metrics to a JSON file to track progress over time.
    """
    history_file = "logs/history_stats.json"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create record for current run
    current_record = {
        "timestamp": timestamp,
        "win_rate": current_win_rate,
        "total_pnl_pct": current_stats["Total_PnL_Pct"],
        "total_signals": current_stats["Total_Signals"]
    }

    # Load existing history
    history = []
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
        except:
            history = []

    logger.info("\n🚀 SYSTEM IMPROVEMENT REPORT")
    logger.info("=" * 40)
    
    # Compare with last run
    if history:
        last_run = history[-1]
        delta_wr = current_win_rate - last_run['win_rate']
        delta_pnl = current_stats["Total_PnL_Pct"] - last_run['total_pnl_pct']
        delta_sig = current_stats["Total_Signals"] - last_run['total_signals']
        
        # Set icons based on performance (Safe icons for Windows)
        wr_icon = "[+]" if delta_wr >= 0 else "[-]"
        pnl_icon = "[+]" if delta_pnl >= 0 else "[-]"
        
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

    # Save updated history
    history.append(current_record)
    if len(history) > 50: history = history[-50:] # Keep last 50 runs
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=4)

# Entry Point
if __name__ == "__main__":
    run_verification()