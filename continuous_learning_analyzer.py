# """
# StockWise Gen-10: Continuous Learning Analyzer
# ==============================================
# Runs a detailed backtest/simulation using the "Specialist" architecture.
# Generates interactive charts to visualize AI decisions, Regimes, and PnL.

# Updates for Gen-10:
# 1. Loads Specialist Models (Per-Symbol .keras).
# 2. Injects Market Context (QQQ) for Beta/Correlation features.
# 3. Uses StrategyOrchestra for Adaptive Logic.
# """

# import pandas as pd
# import numpy as np
# import logging
# import sys
# import os
# import json
# from datetime import datetime
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

# # --- Imports ---
# import system_config as cfg
# from data_source_manager import DataSourceManager
# from feature_engine import RobustFeatureCalculator
# from stockwise_ai_core import StockWiseAI
# from strategy_engine import StrategyOrchestra, MarketRegimeDetector

# # --- Logging Setup ---
# logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
# logger = logging.getLogger("ContinuousAnalyzer")

# class ContinuousLearningAnalyzer:
#     def __init__(self):
#         self.dsm = DataSourceManager()
#         self.calc = RobustFeatureCalculator()
#         self.results = []
        
#         # Output directory for charts
#         self.chart_dir = os.path.join("logs", "analysis_charts")
#         os.makedirs(self.chart_dir, exist_ok=True)

#     def run_analysis(self, symbols=None, days_back=400):
#         """
#         Main execution loop.
#         """
#         if symbols is None:
#             symbols = cfg.TRAINING_SYMBOLS

#         logger.info("==========================================")
#         logger.info("   STARTING GEN-10 CONTINUOUS ANALYSIS    ")
#         logger.info("==========================================")

#         # --- 1. FETCH MARKET CONTEXT (CRITICAL FIX) ---
#         logger.info("Fetching Market Context (QQQ)...")
#         qqq_df = self.dsm.get_stock_data("QQQ", days_back=days_back + 100, interval='1d')
        
#         if qqq_df is None or qqq_df.empty:
#             logger.error("CRITICAL: Failed to fetch QQQ. Analysis features will be broken (0 correlation).")
#             context_data = {}
#         else:
#             context_data = {'qqq': qqq_df}
#         # ----------------------------------------------

#         # 2. PROCESS SYMBOLS
#         for symbol in symbols:
#             self._analyze_single_symbol(symbol, days_back, context_data)

#     def _analyze_single_symbol(self, symbol, days_back, context_data):
#         logger.info(f"\n>> ANALYZING: {symbol}")

#         # A. Fetch Data
#         df = self.dsm.get_stock_data(symbol, days_back=days_back, interval='1d')
#         fundamentals = self.dsm.get_fundamentals(symbol)

#         if df is None or len(df) < 200:
#             logger.warning(f"Skipping {symbol}: Insufficient data.")
#             return

#         # B. Calculate Features (With Context)
#         # This ensures Beta and Correlation are computed
#         df = self.calc.calculate_features(df, context_data=context_data)

#         # C. Load Specialist Brain
#         # IMPORTANT: Passing 'symbol' forces loading of '{symbol}_gen9_model.keras' and '{symbol}_scaler.pkl'
#         ai = StockWiseAI(symbol=symbol)
        
#         if ai.model is None:
#             if not ai.load_inference_model():
#                 logger.warning(f"Skipping {symbol}: Specialist Model not found. Did you run train_gen9_model.py?")
#                 return

#         # D. Simulation Loop
#         trades = []
#         equity = [10000] # Start with $10k
        
#         # Simulation window (skip warmup)
#         start_idx = 60
        
#         # Prepare columns for visualization
#         df['regime'] = "NEUTRAL"
#         df['buy_signal'] = np.nan
#         df['sell_signal'] = np.nan
#         df['ai_conf'] = 0.0

#         active_trade = None

#         for i in range(start_idx, len(df)):
#             current_slice = df.iloc[:i+1]
#             today = current_slice.iloc[-1]
#             idx = today.name
            
#             # 1. Get AI & Strategy Decision
#             features = today.to_dict()
#             _, prob, trace = ai.predict_trade_confidence(symbol, features, fundamentals, current_slice)
            
#             # 2. Detect Regime (For Charting)
#             regime = MarketRegimeDetector.detect_regime(today)
#             df.at[idx, 'regime'] = regime
#             df.at[idx, 'ai_conf'] = prob

#             # 3. Strategy Verdict
#             fund_score = trace['Checks']['Fundamentals']['Score']
#             analysis_packet = {'AI_Probability': prob, 'Fundamental_Score': fund_score}
            
#             verdict = StrategyOrchestra.decide_action(symbol, today, analysis_packet)
            
#             # 4. Trade Logic
#             price = today['close']
            
#             # Check Exit first
#             if active_trade:
#                 # Check Stop or Target
#                 if today['low'] <= active_trade['stop']:
#                     # Stopped out
#                     exit_price = active_trade['stop']
#                     pnl = (exit_price - active_trade['entry']) / active_trade['entry']
#                     equity.append(equity[-1] * (1 + pnl))
#                     df.at[idx, 'sell_signal'] = exit_price
#                     trades.append({'Date': idx, 'Type': 'SELL (Stop)', 'Price': exit_price, 'PnL': pnl})
#                     active_trade = None
                
#                 elif today['high'] >= active_trade['target']:
#                     # Take Profit
#                     exit_price = active_trade['target']
#                     pnl = (exit_price - active_trade['entry']) / active_trade['entry']
#                     equity.append(equity[-1] * (1 + pnl))
#                     df.at[idx, 'sell_signal'] = exit_price
#                     trades.append({'Date': idx, 'Type': 'SELL (Target)', 'Price': exit_price, 'PnL': pnl})
#                     active_trade = None
                    
#             # Check Entry
#             if not active_trade and verdict == "BUY":
#                 stop, target, desc = StrategyOrchestra.get_adaptive_targets(today, price)
#                 active_trade = {
#                     'entry': price,
#                     'stop': stop,
#                     'target': target,
#                     'date': idx
#                 }
#                 df.at[idx, 'buy_signal'] = price
#                 logger.info(f"[{idx.date()}] BUY {symbol} @ {price:.2f} | AI: {prob:.2%} | Mode: {desc}")

#         # E. Generate Chart
#         self._generate_chart(symbol, df, trades)
        
#         # F. Report
#         wins = len([t for t in trades if t['PnL'] > 0])
#         total = len(trades)
#         wr = (wins / total * 100) if total > 0 else 0
#         final_eq = equity[-1]
#         logger.info(f"RESULT: {symbol} | Trades: {total} | Win Rate: {wr:.1f}% | End Equity: ${final_eq:,.2f}")

#     def _generate_chart(self, symbol, df, trades):
#         """
#         Creates a Plotly HTML chart showing Price, Regimes, and Buy/Sell signals.
#         """
#         fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
#                             vertical_spacing=0.05, row_heights=[0.7, 0.3],
#                             subplot_titles=(f"{symbol} Price Action & Signals", "AI Confidence & Regime"))

#         # 1. Candlesticks
#         fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'],
#                                      low=df['low'], close=df['close'], name='Price'), row=1, col=1)

#         # 2. Buy Signals (Green Triangle Up)
#         buys = df[df['buy_signal'].notna()]
#         fig.add_trace(go.Scatter(x=buys.index, y=buys['buy_signal'] * 0.98, mode='markers',
#                                  marker=dict(symbol='triangle-up', color='green', size=12),
#                                  name='Buy Signal'), row=1, col=1)

#         # 3. Sell Signals (Red Triangle Down)
#         sells = df[df['sell_signal'].notna()]
#         fig.add_trace(go.Scatter(x=sells.index, y=sells['sell_signal'] * 1.02, mode='markers',
#                                  marker=dict(symbol='triangle-down', color='red', size=12),
#                                  name='Exit'), row=1, col=1)

#         # 4. Regime Background Colors (Optional - simplified visualization)
#         # We plot EMA 200 to show Trend Baseline
#         fig.add_trace(go.Scatter(x=df.index, y=df['sma_200'], line=dict(color='orange', width=2), name='SMA 200'), row=1, col=1)

#         # 5. AI Confidence (Bottom Panel)
#         fig.add_trace(go.Scatter(x=df.index, y=df['ai_conf'], fill='tozeroy', 
#                                  line=dict(color='purple', width=1), name='AI Confidence'), row=2, col=1)
        
#         # Add Threshold Line
#         fig.add_hline(y=cfg.SniperConfig.MODEL_CONFIDENCE_THRESHOLD, line_dash="dash", line_color="green", row=2, col=1)

#         # Layout
#         fig.update_layout(title=f"Gen-10 Analysis: {symbol}", xaxis_rangeslider_visible=False, height=800)
        
#         # Save
#         filename = os.path.join(self.chart_dir, f"{symbol}_gen10_analysis.html")
#         fig.write_html(filename)
#         logger.info(f"Chart saved to {filename}")

# if __name__ == "__main__":
#     analyzer = ContinuousLearningAnalyzer()
#     analyzer.run_analysis(days_back=400)


"""
StockWise Gen-10: Continuous Learning Analyzer (Portfolio Simulator)
==================================================================
A realistic simulation engine that trades a portfolio of stocks over time.
Generates interactive Red/Green stock charts for analysis.
"""

# --- 1. IMPORTS ---
import pandas as pd  # Library for data manipulation (DataFrames)
import numpy as np  # Library for numerical operations
import logging  # Library for logging events to console/file
import sys  # System-specific parameters and functions
import os  # Operating system interface (file paths)
import plotly.graph_objects as go  # Library for creating interactive charts
from plotly.subplots import make_subplots  # Helper to create multi-panel charts
from datetime import datetime  # Date and time handling

# --- 2. INTERNAL MODULE IMPORTS ---
import system_config as cfg  # Import global settings (paths, thresholds)
from data_source_manager import DataSourceManager  # Module to fetch stock data
from feature_engine import RobustFeatureCalculator  # Module to calculate indicators
from stockwise_ai_core import StockWiseAI  # The AI Brain
from strategy_engine import StrategyOrchestra, MarketRegimeDetector  # The Logic Brain

# --- 3. LOGGING SETUP ---
# Configure the logging format to show Time, Level, and Message
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
# Create a logger instance for this specific script
logger = logging.getLogger("PortfolioSim")

# --- 4. SIMULATION CONFIGURATION ---
INITIAL_CAPITAL = 10000.0  # Starting cash balance ($10,000)
MAX_POSITIONS = 5          # Maximum number of stocks to hold at once
POSITION_SIZE_PCT = 0.20   # Percent of equity to allocate per trade (20%)
MIN_CASH_BUFFER = 500.0    # Minimum cash required to open a new trade

class PortfolioSimulator:
    """
    Main class that orchestrates the chronological trading simulation.
    """
    def __init__(self):
        # Initialize the Data Manager to fetch prices
        self.dsm = DataSourceManager()
        # Initialize the Feature Calculator for technical indicators
        self.calc = RobustFeatureCalculator()
        
        # Dictionaries to store loaded data in memory
        self.market_data = {}  # Format: {'NVDA': DataFrame, 'AMD': DataFrame}
        self.models = {}       # Format: {'NVDA': StockWiseAI_Instance}
        
        # Portfolio State Variables
        self.cash = INITIAL_CAPITAL  # Current available cash
        self.positions = {}    # Tracks open trades: {'Symbol': {entry, shares, stop...}}
        self.trade_history = [] # Logs closed trades for reporting
        self.equity_curve = []  # Tracks total portfolio value over time
        
        # Output Directory Configuration
        # CHANGED: Charts now go to 'debug_charts' as requested
        self.chart_dir = os.path.join("debug_charts")
        # Create the directory if it does not exist
        os.makedirs(self.chart_dir, exist_ok=True)

    def preload_assets(self, symbols, days_back):
        """
        Phase 1: Downloads all data and loads all AI models before simulation starts.
        """
        logger.info("--- PHASE 1: PRE-LOADING ASSETS ---")
        
        # 1. Fetch Market Context (QQQ) for Beta/Correlation calculations
        logger.info("Fetching Market Context (QQQ)...")
        # Get QQQ data slightly further back to ensure moving averages are ready
        qqq_df = self.dsm.get_stock_data("QQQ", days_back=days_back + 100, interval='1d')
        
        # Validation: Did QQQ download correctly?
        if qqq_df is None or qqq_df.empty:
            logger.error("CRITICAL: QQQ data missing. Cannot calculate context features.")
            return False # Stop simulation
        
        # Store context in a dictionary
        context_data = {'qqq': qqq_df}

        # 2. Fetch Symbols & Load Models
        for symbol in symbols:
            logger.info(f"Loading Data & Brain for: {symbol}")
            
            # Download stock price history
            df = self.dsm.get_stock_data(symbol, days_back=days_back, interval='1d')
            
            # Validation: Is the data sufficient?
            if df is None or len(df) < 200:
                logger.warning(f"Skipping {symbol} (Insufficient Data)")
                continue # Skip this symbol
                
            # Calculate Indicators (RSI, ADX, Beta, etc.) using QQQ context
            df = self.calc.calculate_features(df, context_data=context_data)
            
            # Initialize the AI Brain specifically for this symbol
            ai = StockWiseAI(symbol=symbol)
            
            # Attempt to load the trained model file (.keras)
            if ai.model is None and not ai.load_inference_model():
                logger.warning(f"Skipping {symbol} (No Trained Model Found)")
                continue # Skip if no brain exists
            
            # Save data and model to memory for the simulation loop
            self.market_data[symbol] = df
            self.models[symbol] = ai
            
        # Return True if we successfully loaded at least one stock
        return len(self.market_data) > 0

    def run_simulation(self):
        """
        Phase 2: The Time Machine. Steps through history day-by-day.
        """
        logger.info("\n--- PHASE 2: RUNNING SIMULATION ---")
        
        # 1. Align Timelines
        # We need a master list of all trading dates across all stocks
        all_dates = set()
        for df in self.market_data.values():
            all_dates.update(df.index) # Collect all dates
        
        # Sort dates chronologically
        sorted_dates = sorted(list(all_dates))
        
        # Start simulation 60 days in (allow indicators to warm up)
        sim_dates = sorted_dates[60:]
        
        # Optimization: Cache fundamental data once (it doesn't change daily in this sim)
        fundamentals_cache = {sym: self.dsm.get_fundamentals(sym) for sym in self.market_data.keys()}

        # --- MAIN TIME LOOP ---
        for current_date in sim_dates:
            # Convert timestamp to string for logging
            date_str = current_date.strftime('%Y-%m-%d')
            
            # --- A. MANAGE OPEN POSITIONS (Check Exits) ---
            # Create a copy of keys because we might delete positions while iterating
            active_symbols = list(self.positions.keys())
            
            for symbol in active_symbols:
                df = self.market_data[symbol]
                
                # If market was closed for this stock today, skip
                if current_date not in df.index: continue 
                
                # Get today's price row
                today = df.loc[current_date]
                pos = self.positions[symbol] # Get position details
                
                # Rule 1: Check Stop Loss (Did price drop too low?)
                if today['low'] <= pos['stop']:
                    self._execute_sell(symbol, date_str, pos['stop'], "Stop Loss")
                # Rule 2: Check Profit Target (Did price jump high enough?)
                elif today['high'] >= pos['target']:
                    self._execute_sell(symbol, date_str, pos['target'], "Take Profit")
            
            # --- B. SCAN FOR NEW OPPORTUNITIES (Check Entries) ---
            # Only scan if we have empty slots AND enough cash
            if len(self.positions) < MAX_POSITIONS and self.cash > MIN_CASH_BUFFER:
                
                # Check every symbol available
                for symbol in self.market_data.keys():
                    # Skip if we already own it
                    if symbol in self.positions: continue
                    # Stop if we hit max positions mid-loop
                    if len(self.positions) >= MAX_POSITIONS: break
                    
                    df = self.market_data[symbol]
                    # Skip if no data for today
                    if current_date not in df.index: continue
                    
                    # PREPARE DATA FOR AI
                    # Find the integer index of today to slice the window correctly
                    loc_idx = df.index.get_loc(current_date)
                    # Slice dataframe from start up to today (Simulating real-time view)
                    current_slice = df.iloc[:loc_idx+1]
                    today = current_slice.iloc[-1]
                    
                    # 1. AI Prediction
                    ai = self.models[symbol]
                    features = today.to_dict()
                    funds = fundamentals_cache.get(symbol)
                    
                    # Ask AI for confidence score
                    _, prob, trace = ai.predict_trade_confidence(symbol, features, funds, current_slice)
                    
                    # 2. Strategy Logic (Orchestra)
                    fund_score = trace['Checks']['Fundamentals']['Score']
                    analysis_packet = {'AI_Probability': prob, 'Fundamental_Score': fund_score}
                    
                    # Ask Strategy Engine for Verdict (BUY/WAIT)
                    verdict = StrategyOrchestra.decide_action(symbol, today, analysis_packet)
                    
                    # 3. Execution Logic
                    if verdict == "BUY":
                        price = today['close']
                        # Calculate adaptive targets based on volatility
                        stop, target, desc = StrategyOrchestra.get_adaptive_targets(today, price)
                        # Execute the buy
                        self._execute_buy(symbol, date_str, price, stop, target, desc)

            # --- C. UPDATE PORTFOLIO VALUE ---
            self._update_equity(current_date)

        # End of simulation loop: Generate reports
        self._generate_report()

    def _execute_buy(self, symbol, date, price, stop, target, desc):
        """Calculates position size and updates cash/holdings."""
        # Calculate Equity (Cash + value of current stocks)
        current_equity = self.equity_curve[-1]['Equity'] if self.equity_curve else INITIAL_CAPITAL
        # Allocate a percentage of equity (e.g., 20%)
        allocation = current_equity * POSITION_SIZE_PCT
        
        # Ensure we don't spend more cash than we have
        allocation = min(allocation, self.cash)
        
        # Don't make tiny trades (<$500)
        if allocation < 500: return 
        
        # Calculate number of shares
        shares = int(allocation / price)
        cost = shares * price
        
        # Deduct cash
        self.cash -= cost
        # Record position
        self.positions[symbol] = {
            'shares': shares, 'entry': price, 'stop': stop, 'target': target, 'date': date
        }
        
        logger.info(f"[{date}] 🟢 BUY {symbol:<5} | {shares} shares @ ${price:.2f} | Mode: {desc}")

    def _execute_sell(self, symbol, date, price, reason):
        """Closes a position and updates cash."""
        pos = self.positions[symbol]
        # Calculate proceeds
        revenue = pos['shares'] * price
        # Calculate Profit/Loss (PnL)
        pnl = revenue - (pos['shares'] * pos['entry'])
        pnl_pct = (price - pos['entry']) / pos['entry']
        
        # Add cash back to balance
        self.cash += revenue
        # Remove from active positions
        del self.positions[symbol]
        
        # Log result with icons
        icon = "💰" if pnl > 0 else "🛑"
        logger.info(f"[{date}] {icon} SELL {symbol:<4} | PnL: ${pnl:>6.2f} ({pnl_pct:>6.2%}) | {reason}")
        
        # Add to history for reporting
        self.trade_history.append({
            'Symbol': symbol, 'Entry_Date': pos['date'], 'Exit_Date': date,
            'Entry': pos['entry'], 'Exit': price, 'PnL_USD': pnl, 'PnL_Pct': pnl_pct, 'Reason': reason
        })

    def _update_equity(self, date):
        """Calculates total portfolio value for the day."""
        holdings_value = 0
        for sym, pos in self.positions.items():
            # Get today's price to mark-to-market
            df = self.market_data[sym]
            if date in df.index:
                price = df.loc[date]['close']
                holdings_value += pos['shares'] * price
            else:
                # Fallback to entry price if no data for today
                holdings_value += pos['shares'] * pos['entry']
                
        total_equity = self.cash + holdings_value
        # Record for the equity curve chart
        self.equity_curve.append({'Date': date, 'Equity': total_equity})

    def _generate_report(self):
        """Prints summary stats and saves charts."""
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
        
        # Save Trade Log CSV
        pd.DataFrame(self.trade_history).to_csv(os.path.join(self.chart_dir, "portfolio_trades.csv"))
        
        # Generate Charts
        self._plot_equity_curve()
        # Generate individual stock charts for symbols we traded
        traded_symbols = set([t['Symbol'] for t in self.trade_history])
        for sym in traded_symbols:
            self._generate_stock_chart(sym)

    def _plot_equity_curve(self):
        """Draws the main portfolio growth chart."""
        df = pd.DataFrame(self.equity_curve)
        df.set_index('Date', inplace=True)
        
        fig = go.Figure()
        
        # Draw Equity Line (Green)
        fig.add_trace(go.Scatter(x=df.index, y=df['Equity'], mode='lines', name='Portfolio Value',
                                 line=dict(color='#00FF00', width=2))) # Bright Green
        
        # Draw Baseline (Gray dashed)
        fig.add_hline(y=INITIAL_CAPITAL, line_dash="dash", line_color="gray")
        
        fig.update_layout(
            title="Gen-10 Portfolio Growth", 
            yaxis_title="Equity ($)", 
            template="plotly_dark", # Dark mode like trading terminal
            height=600
        )
        
        path = os.path.join(self.chart_dir, "portfolio_equity_curve.html")
        fig.write_html(path)
        logger.info(f"Equity Chart saved to: {path}")

    def _generate_stock_chart(self, symbol):
        """Creates a detailed Red/Green candlestick chart for a specific stock."""
        df = self.market_data[symbol]
        
        # Filter trades for this symbol
        symbol_trades = [t for t in self.trade_history if t['Symbol'] == symbol]
        
        # Create Subplots: Price on top, AI Confidence on bottom
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05, row_heights=[0.7, 0.3],
                            subplot_titles=(f"{symbol} Price Action", "AI Confidence"))

        # 1. Candlestick Chart (Red/Green)
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
            increasing_line_color='#00FF00', # Green for Up
            decreasing_line_color='#FF0000', # Red for Down
            name='Price'
        ), row=1, col=1)

        # 2. Plot Buy/Sell Markers
        # Extract Buy points
        buy_dates = [t['Entry_Date'] for t in symbol_trades]
        buy_prices = [t['Entry'] for t in symbol_trades]
        
        fig.add_trace(go.Scatter(
            x=buy_dates, y=buy_prices, mode='markers',
            marker=dict(symbol='triangle-up', color='#00FF00', size=15), # Green Up Triangle
            name='BUY'
        ), row=1, col=1)

        # Extract Sell points
        sell_dates = [t['Exit_Date'] for t in symbol_trades]
        sell_prices = [t['Exit'] for t in symbol_trades]
        
        fig.add_trace(go.Scatter(
            x=sell_dates, y=sell_prices, mode='markers',
            marker=dict(symbol='triangle-down', color='#FF0000', size=15), # Red Down Triangle
            name='SELL'
        ), row=1, col=1)

        # 3. AI Confidence Line (Purple)
        # We need to re-run prediction or assume we stored it. 
        # For simplicity in this chart, we will just plot the close price line in the bottom for context
        # Or ideally, we store AI prob in the dataframe during simulation.
        # Let's plot the 200 SMA as context instead if AI prob wasn't stored in DF
        if 'sma_200' in df.columns:
             fig.add_trace(go.Scatter(x=df.index, y=df['sma_200'], line=dict(color='orange'), name='SMA 200'), row=1, col=1)

        # Layout styling
        fig.update_layout(
            title=f"Trade Analysis: {symbol}",
            xaxis_rangeslider_visible=False,
            template="plotly_dark", # Dark Theme
            height=800
        )
        
        path = os.path.join(self.chart_dir, f"{symbol}_chart.html")
        fig.write_html(path)
        logger.info(f"Stock Chart saved to: {path}")

if __name__ == "__main__":
    sim = PortfolioSimulator()
    # Load default symbols from config
    symbols = cfg.TRAINING_SYMBOLS
    
    # Run preload and then simulation if successful
    if sim.preload_assets(symbols, days_back=400):
        sim.run_simulation()