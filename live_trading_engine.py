
"""
StockWise Gen-9 Live Trading Engine
===================================
The "Heartbeat" of the system.
Continuously monitors the market, fetches real-time data, and queries the AI "Sniper" Agent.

Modes:
- PAPER: Logs trades to console/file only. (DEFAULT)
- LIVE: Sends orders to Broker (Alpaca/IBKR).

Usage:
    python live_trading_engine.py --symbol NVDA --interval 1h
"""

import time
import logging
import argparse
import pandas as pd
import datetime
import sys
import os

# Ensure project root is in path
sys.path.append(os.getcwd())

from data_source_manager import DataSourceManager
from feature_engine import RobustFeatureCalculator
from strategy_engine import StrategyOrchestra, MarketRegimeDetector
from stockwise_ai_core import StockWiseAI
import system_config as cfg

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler("logs/live_trading.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("LiveTrader")

class LiveTrader:
    def __init__(self, symbols, interval='1d', mode='PAPER'):
        # Ensure symbols is a list
        if isinstance(symbols, str):
            self.symbols = [s.strip().upper() for s in symbols.split(',')]
        else:
            self.symbols = [s.upper() for s in symbols]
            
        self.interval = interval
        self.mode = mode.upper()
        self.dsm = DataSourceManager()
        self.feature_calc = RobustFeatureCalculator(params={})
        self.ai = StockWiseAI() # GEN-9 CORE
        self.is_running = True
        
        logger.info(f"--- StockWise Gen-9 Live Engine Started ---")
        logger.info(f"Targets: {self.symbols} | Interval: {self.interval} | Mode: {self.mode}")
        logger.info("Strategies: Gen-9 Fusion Sniper (Deep Learning + Hard Filters)")

    def fetch_and_process(self, symbol):
        """Fetch latest data and generate features for a specific symbol."""
        logger.info(f"[>] Fetching live data for {symbol}...")
        
        # 1. Fetch Data
        days_back = cfg.HISTORY_LENGTH_DAYS
        if self.interval == '1m': days_back = 5 # Speed opt
        
        # Ensure we define start_date relative to now for reliable fetching
        # end_date is implicitly 'now'
        
        df = self.dsm.get_stock_data(symbol, days_back=days_back, interval=self.interval)
        
        if df is None or df.empty:
            logger.error(f"No data received for {symbol}.")
            return None, None
            
        # FIX: Flatten MultiIndex (YFinance)
        if hasattr(df, 'columns') and isinstance(df.columns, pd.MultiIndex):
             df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df.columns.values]
             
        # Standardize Names
        df.columns = [str(c).lower() for c in df.columns]
        
        # Clean names
        new_cols = {}
        for c in df.columns:
            if 'close' in c: new_cols[c] = 'close'
            elif 'open' in c: new_cols[c] = 'open'
            elif 'high' in c: new_cols[c] = 'high'
            elif 'low' in c: new_cols[c] = 'low'
            elif 'volume' in c: new_cols[c] = 'volume'
        df.rename(columns=new_cols, inplace=True)
            
        # 2. Add Features
        try:
            df = self.feature_calc.calculate_features(df)
            
            # Fetch Fundamentals
            fund_data = self.dsm.get_fundamentals(symbol)
            if not fund_data: fund_data = {} # Handle empty
                
            return df, fund_data
            
        except Exception as e:
            logger.error(f"Feature Engineering Failed for {symbol}: {e}")
            return None, None

    def analyze_market(self, symbol, df, fundamentals):
        """Run Gen-9 Sniper AI on the latest candle."""
        if len(df) < 60:
            logger.warning(f"Not enough data for {symbol} (Need 60+ bars). Skipping...")
            return None

        # Latest Candle Context
        # We pass the WHOLE df window (sliced inside AI)
        # And the LATEST features
        latest_bar = df.iloc[-1]
        features = latest_bar.to_dict()
        
        # AI PREDICTION
        decision, prob, trace = self.ai.predict_trade_confidence(
            symbol, 
            features, 
            fundamentals, 
            df # Pass full DF history
        )
        
        price = features.get('close')
        logger.info(f"[{latest_bar.name}] {symbol} | Price: {price:.2f} | Decision: {decision} | Conf: {prob:.2%}")
        
        if decision == "BUY":
            logger.info(f"    >>> SIGNAL TRIGGERED for {symbol}: {trace}")
            return decision, prob, price, trace
            
        return None

    def execute_trade(self, symbol, action, price, details):
        """Execute or Log Trade."""
        if self.mode == 'PAPER':
            logger.info(f"!!! PAPER TRADE SIGNAL: {action} {symbol} !!!")
            logger.info(f"Price: {price} | Reason: {details}")
            # Log to dedicated signals file
            with open("logs/signals.csv", "a") as f:
                f.write(f"{datetime.datetime.now()},{symbol},{action},{price},{details}\n")
        elif self.mode == 'LIVE':
            logger.warning("LIVE TRADING NOT YET ENABLED. Switch to Paper Mode.")
            # Future: self.dsm.place_order(...)

    def run(self):
        """Main Loop."""
        try:
            while self.is_running:
                logger.info("\n--- Scanning Watchlist ---")
                
                for symbol in self.symbols:
                    df, fund_data = self.fetch_and_process(symbol)
                    
                    if df is not None:
                        # NEW GEN-9 LOGIC
                        result = self.analyze_market(symbol, df, fund_data)
                        
                        if result:
                            decision, prob, price, trace = result
                            
                            if decision == "BUY":
                                 self.execute_trade(symbol, "BUY", price, f"AI_Conf: {prob:.2%} | {trace}")
                        else:
                            # Non-result (WAIT or FILTERED) handles itself in logs
                            pass
                    
                    # Small delay between symbols to avoid rate limits
                    time.sleep(2)
                
                # Sleep between Full Scans
                sleep_sec = 60 if self.interval == '1m' else 900 
                if self.interval == '1d': sleep_sec = 3600 # 1 hour check for daily candles
                
                logger.info(f"Scan Complete. Sleeping for {sleep_sec} seconds...")
                time.sleep(sleep_sec)
                
        except KeyboardInterrupt:
            logger.info("Stopping Live Engine...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='StockWise Gen-9 Live Trader')
    
    # Default basket
    default_basket = "NVDA,AMD,MSFT,GOOGL,AAPL,META,TSLA,QCOM,AMZN"
    
    parser.add_argument('--symbols', type=str, default=default_basket, help='Comma-separated ticker symbols')
    parser.add_argument('--interval', type=str, default='1d', help='Candle interval (1m, 1h, 1d)')
    parser.add_argument('--mode', type=str, default='PAPER', help='PAPER or LIVE')
    
    args = parser.parse_args()
    
    trader = LiveTrader(args.symbols, args.interval, args.mode)
    trader.run()
