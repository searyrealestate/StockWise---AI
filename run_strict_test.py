# """
# Strict Out-of-Sample Tester
# ===========================
# Ensures the system performs well on unseen future data.
# """
# import logging # Logging
# import sys # System ops
# import os # File ops
# import time # Timing
# from datetime import datetime # Dates

# # Import modules, catch errors
# try:
#     from train_gen9_model import train_model
#     from verify_sniper_logic import run_verification
# except ImportError as e:
#     print(f"❌ Critical Import Error: {e}")
#     sys.exit(1)

# # CONFIGURATION: Train only on data BEFORE this date
# CUTOFF_DATE = "2025-01-01" 

# # Logging Setup
# os.makedirs("logs", exist_ok=True)
# log_filename = f"logs/strict_test_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
# for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', 
#                     handlers=[logging.FileHandler(log_filename, mode='w'), logging.StreamHandler(sys.stdout)])
# logger = logging.getLogger("StrictTester")

# def run_strict_test():
#     start_time = time.time()
#     logger.info("\n" + "="*60)
#     logger.info("STARTING STRICT 'OUT-OF-SAMPLE' TEST")
#     logger.info(f"CUTOFF DATE: {CUTOFF_DATE}")
#     logger.info("="*60 + "\n")
    
#     # PHASE 1: TRAIN (Past)
#     logger.info("[PHASE 1] Training Model on Past Data...")
#     try:
#         # Train strictly on historical data
#         train_model(cutoff_date=CUTOFF_DATE)
#         logger.info("[PHASE 1] Training Complete.")
#     except Exception as e:
#         logger.error(f"Training Failed: {e}", exc_info=True); sys.exit(1)

#     # PHASE 2: VERIFY (Future)
#     logger.info("\n" + "="*60)
#     logger.info("[PHASE 2] Verifying on Future Data...")
#     try:
#         # Simulate trading strictly on future data
#         run_verification(start_date=CUTOFF_DATE)
#         logger.info("[PHASE 2] Verification Complete.")
#     except Exception as e:
#         logger.error(f"Verification Failed: {e}", exc_info=True); sys.exit(1)

#     logger.info(f"\nSTRICT TEST COMPLETED in {time.time() - start_time:.1f}s.")

# if __name__ == "__main__":
#     run_strict_test()


"""
StockWise Gen-10: Real-Life Portfolio Simulator
===============================================
Simulates a SINGLE portfolio trading multiple stocks simultaneously over time.
Logic:
1. Pre-loads data/models for all symbols.
2. Steps through time (Day 1 -> Day N).
3. On each day:
   - Checks Open Positions: Hit Stop Loss? Hit Target? -> Sell.
   - Checks New Signals: AI > Threshold? Strategy == BUY? -> Buy.
4. Manages Cash & Equity Curve.

Author: StockWise Gen-10
"""

# --- 1. LIBRARY IMPORTS ---
import pandas as pd  # Import Pandas for data manipulation (DataFrames)
import numpy as np  # Import NumPy for numerical calculations
import logging  # Import Logging to print status messages to console/file
import sys  # Import Sys to interact with the system (exit, stdout)
import os  # Import OS to manage file paths and directories
import plotly.graph_objects as go  # Import Plotly for creating interactive charts
from datetime import datetime  # Import datetime to handle date objects

# --- 2. INTERNAL MODULE IMPORTS ---
import system_config as cfg  # Import global settings (tickers, thresholds)
from data_source_manager import DataSourceManager  # Import class to fetch stock prices
from feature_engine import RobustFeatureCalculator  # Import class to calculate indicators
from stockwise_ai_core import StockWiseAI  # Import the AI Brain
from strategy_engine import StrategyOrchestra, MarketRegimeDetector  # Import the Logic Engine

# --- 3. LOGGING CONFIGURATION ---
# Set up the logger to print info to the screen
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("PortfolioSim")  # Name the logger

# --- 4. SIMULATION SETTINGS ---
INITIAL_CAPITAL = 10000.0  # Starting cash in the simulated account
MAX_POSITIONS = 5          # Maximum number of stocks we can hold at once
POSITION_SIZE_PCT = 0.20   # How much of the portfolio to bet on one trade (20%)
MIN_CASH_BUFFER = 500.0    # Minimum cash required to attempt a buy

# --- 5. MAIN SIMULATOR CLASS ---
class PortfolioSimulator:
    def __init__(self):
        """Initializes the simulator state."""
        self.dsm = DataSourceManager()  # Create instance of Data Manager
        self.calc = RobustFeatureCalculator()  # Create instance of Feature Calculator
        self.market_data = {}  # Dictionary to store Price DataFrames for each stock
        self.models = {}       # Dictionary to store loaded AI Models for each stock
        
        # Portfolio State Variables
        self.cash = INITIAL_CAPITAL  # Set current cash to initial capital
        self.positions = {}    # Dictionary to track open trades: { 'NVDA': {'shares': 10...} }
        self.trade_history = []  # List to store closed trades for analysis
        self.equity_curve = []   # List to track account value over time
        
        # Create directory for saving charts
        self.chart_dir = os.path.join("logs", "analysis_charts")
        os.makedirs(self.chart_dir, exist_ok=True)  # Make folder if it doesn't exist

    def preload_assets(self, symbols, days_back):
        """
        Phase 1: Downloads all data and loads all AI models into RAM before starting.
        This makes the simulation loop much faster.
        """
        logger.info("--- PHASE 1: PRE-LOADING ASSETS ---")
        
        # 1. Fetch Market Context (QQQ)
        # We need this to calculate 'Market Correlation' features correctly
        logger.info("Fetching Market Context (QQQ)...")
        qqq_df = self.dsm.get_stock_data("QQQ", days_back=days_back + 100, interval='1d')
        
        # Validate QQQ data
        if qqq_df is None or qqq_df.empty:
            logger.error("CRITICAL: QQQ data missing. Cannot calculate features.")
            return False  # Stop if critical data is missing
        
        # Package context into a dictionary
        context_data = {'qqq': qqq_df}

        # 2. Fetch Symbols & Load Brains
        # Loop through every symbol in our list
        for symbol in symbols:
            logger.info(f"Loading Data & Brain for: {symbol}")
            
            # Fetch Stock Data
            df = self.dsm.get_stock_data(symbol, days_back=days_back, interval='1d')
            # Check if data is sufficient
            if df is None or len(df) < 200:
                logger.warning(f"Skipping {symbol} (Insufficient Data)")
                continue  # Skip to next symbol
                
            # Calculate Technical Indicators (Features)
            # Pass context_data so Beta/Correlation are calculated
            df = self.calc.calculate_features(df, context_data=context_data)
            
            # Initialize AI Model for this specific symbol
            ai = StockWiseAI(symbol=symbol)
            # Try to load the trained model file
            if ai.model is None and not ai.load_inference_model():
                logger.warning(f"Skipping {symbol} (No Model Found)")
                continue  # Skip if no brain exists
            
            # Store Data and Model in memory for the simulation loop
            self.market_data[symbol] = df
            self.models[symbol] = ai
            
        # Return True if we successfully loaded at least one stock
        return len(self.market_data) > 0

    def run_simulation(self):
        """
        Phase 2: The Time Machine.
        Iterates through dates chronologically, simulating a real trading day.
        """
        logger.info("\n--- PHASE 2: RUNNING SIMULATION ---")
        
        # 1. Align Timelines
        # We need a master list of all dates where the market was open
        all_dates = set()
        for df in self.market_data.values():
            all_dates.update(df.index)  # Collect dates from all stocks
        sorted_dates = sorted(list(all_dates))  # Sort them chronologically
        
        # Skip the first 60 days to allow moving averages to calculate correctly
        sim_dates = sorted_dates[60:]
        
        # Pre-fetch fundamental data (like PE ratio) to save time in the loop
        fundamentals_cache = {sym: self.dsm.get_fundamentals(sym) for sym in self.market_data.keys()}

        # --- TIME LOOP START ---
        for current_date in sim_dates:
            date_str = current_date.strftime('%Y-%m-%d')  # Convert date to string
            
            # --- A. MANAGE OPEN POSITIONS (Check Exits) ---
            # Create a list of currently held symbols
            active_symbols = list(self.positions.keys())
            
            for symbol in active_symbols:
                df = self.market_data[symbol] # Get data for this stock
                
                # If market was closed for this stock today, skip
                if current_date not in df.index: continue 
                
                today = df.loc[current_date] # Get today's price row
                pos = self.positions[symbol] # Get our position details
                
                # Check Stop Loss: Did the LOW drop below our stop?
                if today['low'] <= pos['stop']:
                    self._execute_sell(symbol, date_str, pos['stop'], "Stop Loss")
                # Check Profit Target: Did the HIGH rise above our target?
                elif today['high'] >= pos['target']:
                    self._execute_sell(symbol, date_str, pos['target'], "Take Profit")
            
            # --- B. SCAN FOR NEW OPPORTUNITIES (Check Entries) ---
            # Only look for trades if we have space in portfolio AND cash available
            if len(self.positions) < MAX_POSITIONS and self.cash > MIN_CASH_BUFFER:
                
                # Check every stock we are tracking
                for symbol in self.market_data.keys():
                    # Don't buy if we already own it
                    if symbol in self.positions: continue
                    # Double check max positions (in case we bought one in this loop)
                    if len(self.positions) >= MAX_POSITIONS: break
                    
                    df = self.market_data[symbol] # Get data
                    if current_date not in df.index: continue # Skip if no data today
                    
                    # Prepare Data Slice (Simulate "Right Now")
                    # We need the data UP TO today to feed the AI
                    loc_idx = df.index.get_loc(current_date)
                    current_slice = df.iloc[:loc_idx+1]
                    today = current_slice.iloc[-1]
                    
                    # 1. Ask AI for Prediction
                    ai = self.models[symbol]
                    features = today.to_dict()
                    funds = fundamentals_cache.get(symbol)
                    
                    # Get Probability
                    _, prob, trace = ai.predict_trade_confidence(symbol, features, funds, current_slice)
                    
                    # 2. Ask Strategy Engine for Verdict
                    fund_score = trace['Checks']['Fundamentals']['Score']
                    analysis_packet = {'AI_Probability': prob, 'Fundamental_Score': fund_score}
                    
                    # Pass data to the "Conductor" to decide
                    verdict = StrategyOrchestra.decide_action(symbol, today, analysis_packet)
                    
                    # 3. Buy Execution
                    if verdict == "BUY":
                        price = today['close']
                        # Calculate where to put Stop Loss and Target
                        stop, target, desc = StrategyOrchestra.get_adaptive_targets(today, price)
                        # Execute the buy
                        self._execute_buy(symbol, date_str, price, stop, target, desc)

            # --- C. UPDATE EQUITY CURVE ---
            # Record the total portfolio value for today
            self._update_equity(current_date)

        # End of loop: Generate final report
        self._generate_report()

    def _execute_buy(self, symbol, date, price, stop, target, desc):
        """Calculates size and executes a buy."""
        # Calculate Position Size: 20% of Current Total Equity
        current_equity = self.equity_curve[-1]['Equity'] if self.equity_curve else INITIAL_CAPITAL
        allocation = current_equity * POSITION_SIZE_PCT
        
        # We cannot spend more cash than we have
        allocation = min(allocation, self.cash)
        
        # If allocation is too small (e.g., < $500), skip trade
        if allocation < 500: return 
        
        # Calculate number of shares
        shares = int(allocation / price)
        cost = shares * price # Total cost
        
        # Deduct cash
        self.cash -= cost
        
        # Add to portfolio
        self.positions[symbol] = {
            'shares': shares, 'entry': price, 'stop': stop, 'target': target, 'date': date
        }
        
        # Log the trade
        logger.info(f"[{date}] 🟢 BUY {symbol:<5} | {shares} shares @ ${price:.2f} | Mode: {desc}")

    def _execute_sell(self, symbol, date, price, reason):
        """Executes a sell and calculates profit/loss."""
        pos = self.positions[symbol] # Get position info
        revenue = pos['shares'] * price # Calculate revenue from sale
        
        # Calculate Profit/Loss
        pnl = revenue - (pos['shares'] * pos['entry'])
        pnl_pct = (price - pos['entry']) / pos['entry']
        
        # Add revenue back to cash pile
        self.cash += revenue
        # Remove from portfolio
        del self.positions[symbol]
        
        # Log the outcome
        icon = "💰" if pnl > 0 else "🛑"
        logger.info(f"[{date}] {icon} SELL {symbol:<4} | PnL: ${pnl:>6.2f} ({pnl_pct:>6.2%}) | {reason}")
        
        # Save record
        self.trade_history.append({
            'Symbol': symbol, 'Entry_Date': pos['date'], 'Exit_Date': date,
            'Entry': pos['entry'], 'Exit': price, 'PnL_USD': pnl, 'PnL_Pct': pnl_pct, 'Reason': reason
        })

    def _update_equity(self, date):
        """Calculates Mark-to-Market value of the portfolio."""
        holdings_value = 0
        # Loop through open positions
        for sym, pos in self.positions.items():
            df = self.market_data[sym]
            # If market is open, use today's close price
            if date in df.index:
                price = df.loc[date]['close']
                holdings_value += pos['shares'] * price
            else:
                # If market closed (e.g. holiday for this stock), use entry price (fallback)
                holdings_value += pos['shares'] * pos['entry']
                
        # Total Equity = Cash on hand + Value of Stocks
        total_equity = self.cash + holdings_value
        # Add to curve
        self.equity_curve.append({'Date': date, 'Equity': total_equity})

    def _generate_report(self):
        """Prints final stats and saves charts."""
        logger.info("\n" + "="*40)
        logger.info("       PORTFOLIO SIMULATION RESULTS       ")
        logger.info("="*40)
        
        # Calculate final metrics
        final_equity = self.equity_curve[-1]['Equity']
        total_return = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        
        wins = [t for t in self.trade_history if t['PnL_USD'] > 0]
        
        win_rate = len(wins) / len(self.trade_history) * 100 if self.trade_history else 0
        
        # Print Summary
        logger.info(f"Start Capital:   ${INITIAL_CAPITAL:,.2f}")
        logger.info(f"Final Equity:    ${final_equity:,.2f}")
        logger.info(f"Total Return:    {total_return:.2f}%")
        logger.info(f"Total Trades:    {len(self.trade_history)}")
        logger.info(f"Win Rate:        {win_rate:.2f}%")
        logger.info("="*40)
        
        # Save Trade Log to CSV
        pd.DataFrame(self.trade_history).to_csv(os.path.join(self.chart_dir, "portfolio_trades.csv"))
        
        # Generate the Equity Curve Chart
        self._plot_equity_curve()

    def _plot_equity_curve(self):
        """Uses Plotly to draw the account growth chart."""
        df = pd.DataFrame(self.equity_curve)
        df.set_index('Date', inplace=True)
        
        fig = go.Figure()
        
        # Add the Equity Line
        fig.add_trace(go.Scatter(x=df.index, y=df['Equity'], mode='lines', name='Portfolio Value',
                                 line=dict(color='green', width=2)))
        
        # Add a baseline (Initial Capital)
        fig.add_hline(y=INITIAL_CAPITAL, line_dash="dash", line_color="gray")
        
        # Style the layout
        fig.update_layout(title="Gen-10 Portfolio Performance", yaxis_title="Equity ($)", height=600)
        
        # Save to HTML file
        path = os.path.join(self.chart_dir, "portfolio_equity_curve.html")
        fig.write_html(path)
        logger.info(f"Equity Chart saved to: {path}")

# --- 6. ENTRY POINT ---
if __name__ == "__main__":
    # Create simulator instance
    sim = PortfolioSimulator()
    # Load default symbols from config
    symbols = cfg.TRAINING_SYMBOLS
    
    # Run: Preload -> Simulate
    if sim.preload_assets(symbols, days_back=400):
        sim.run_simulation()