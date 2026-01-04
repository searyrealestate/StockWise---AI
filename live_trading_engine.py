
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
# NEW IMPORTS
from portfolio_manager import PortfolioManager
from auditor import DailyAuditor
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

import pytz
from datetime import datetime, timedelta

class LiveTrader:
    def __init__(self, symbols, interval='1d', mode='PAPER'):
        # Ensure symbols is a list
        if isinstance(symbols, str):
            self.symbols = [s.strip().upper() for s in symbols.split(',')]
        else:
            self.symbols = [s.upper() for s in symbols]
            
        self.interval = interval
        self.mode = mode.upper()
        self.dsm = DataSourceManager(use_ibkr=cfg.EN_IBKR)
        self.feature_calc = RobustFeatureCalculator(params={})
        self.ai = StockWiseAI() # GEN-9 CORE
        
        # --- SMART ASSISTANT LAYER ---
        self.pm = PortfolioManager()
        self.auditor = DailyAuditor(self.pm, self.dsm, self.ai.notifier)
        
        self.is_running = True
        self.eod_run_today = False # Track if EOD ran for current date
        
        logger.info(f"--- StockWise Gen-9 Live Engine Started ---")
        logger.info(f"Targets: {self.symbols} | Interval: {self.interval} | Mode: {self.mode}")
        logger.info("Strategies: Gen-9 Fusion Sniper (Deep Learning + Hard Filters)")
        logger.info("Smart Assistant: Enabled (Portfolio Shadowing + Logic Auditing)")

    def is_trading_day(self, date_obj):
        """Check if date is a valid trading day (Mon-Fri, not a holiday)."""
        # 1. Check Weekend (5=Sat, 6=Sun)
        if date_obj.weekday() >= 5:
            return False
            
        # 2. Check Holidays
        date_str = date_obj.strftime('%Y-%m-%d')
        if date_str in cfg.SchedulerConfig.MARKET_HOLIDAYS:
             return False
             
        return True

    def get_next_run_time(self, now_dt):
        """Calculate the next valid start time (Open - PreBuffer)."""
        tz = now_dt.tzinfo
        
        # Define today's end boundary
        today_session_end = datetime.combine(now_dt.date(), cfg.SchedulerConfig.CLOSE_TIME).replace(tzinfo=tz) + timedelta(hours=cfg.SchedulerConfig.POST_BUFFER_HOURS)
        
        target_date = now_dt.date()
        
        # If today is invalid OR we are past today's session, start checking from tomorrow
        if not self.is_trading_day(target_date) or now_dt > today_session_end:
            target_date += timedelta(days=1)
            
        # Find next valid trading day
        while not self.is_trading_day(target_date):
            target_date += timedelta(days=1)
            
        # Construct start time
        next_open = datetime.combine(target_date, cfg.SchedulerConfig.OPEN_TIME).replace(tzinfo=tz)
        next_start = next_open - timedelta(hours=cfg.SchedulerConfig.PRE_BUFFER_HOURS)
        
        return next_start

    def fetch_and_process(self, symbol):
        """Pipeline: Fetch -> Features -> Fundamentals."""
        # Determine strict days_back based on interval to minimize API load but ensure indicators
        days_back = 200 # Safe default for indicators
        
        try:
            # 1. Fetch Data
            df = self.dsm.get_stock_data(symbol, days_back=days_back, interval=self.interval)
            
            if df is None or df.empty:
                logger.warning(f"No data for {symbol}")
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
            
            # 2. Fundamentals
            fund_data = self.dsm.get_fundamentals(symbol)
            if not fund_data: fund_data = {} # Handle empty
            
            # 3. Add Features
            df = self.feature_calc.calculate_features(df)
            
            return df, fund_data
            
        except Exception as e:
            logger.error(f"Pipeline Error {symbol}: {e}")
            return None, None

    def update_stop_loss(self, ticker, new_stop):
        """Updates the stop_loss for an open SHADOW trade."""
        updated = False
        for trade in self.shadow_portfolio.get("trades", []):
            if trade["status"] == "OPEN" and trade["ticker"] == ticker:
                trade["stop_loss"] = float(new_stop)
                updated = True
                logger.info(f"[Shadow] Stop Loss Updated for {ticker}: ${new_stop:.2f}")
        
        if updated:
            self._save_json(self.shadow_file, self.shadow_portfolio)

    def close_shadow_trade(self, ticker, exit_price, reason):
        """Closes a SHADOW trade (Automated System)."""
        closed = False
        for trade in self.shadow_portfolio.get("trades", []):
            if trade["status"] == "OPEN" and trade["ticker"] == ticker:
                trade["status"] = "CLOSED"
                trade["exit_price"] = float(exit_price)
                trade["exit_reason"] = reason
                trade["exit_timestamp"] = datetime.now().isoformat()
                
                # Calc PnL
                pct_change = (float(exit_price) - trade["entry_price"]) / trade["entry_price"]
                trade["pnl"] = pct_change * trade["allocation"]
                closed = True
                
        if closed:
            self._save_json(self.shadow_file, self.shadow_portfolio)
            logger.info(f"[Shadow] Trade Closed: {ticker} @ {exit_price} ({reason})")

    def analyze_market(self, symbol, df, fund_data):
        """Ask the AI for a decision."""
        if len(df) < 60:
            logger.warning(f"Not enough data for {symbol} (Need 60+ bars). Skipping...")
            return None

        # Latest Candle Context
        latest_bar = df.iloc[-1]
        features = latest_bar.to_dict()
        
        # --- Dynamic ATR Stop Loss ---
        # Try to get ATR (usually 'atrr_14' from pandas_ta, fallback to 2% if missing)
        atr = features.get('atr_14') or features.get('atrr_14') or (features.get('close') * 0.02)

        # Use the multiplier from system_config (Conservative = 2.0)
        atr_multiplier = cfg.ACTIVE_PROFILE["stop_atr"] 
        
        stop_price = features.get('close') - (atr * atr_multiplier)

        current_params = {
            "price": features.get('close'),
            "target": features.get('close') * (1 + cfg.SniperConfig.TARGET_PROFIT),
            "stop_loss": stop_price,  # <--- NEW DYNAMIC STOP
            "timestamp": str(latest_bar.name)
        }
        
        # Manually injection of params into AI (hacky but effective without refactoring AI signature completely)
        # We handle Smart Alert inside Notifier via send_alert params
        
        # AI PREDICTION
        decision, prob, trace = self.ai.predict_trade_confidence(
            symbol, 
            features, 
            fundamentals=fund_data, 
            df_window=df # Pass full DF history
        )
        
        price = features.get('close')
        logger.info(f"[{latest_bar.name}] {symbol} | Price: {price:.2f} | Decision: {decision} | Conf: {prob:.2%}")
        
        # --- SWING TRADE MANAGEMENT LOGIC ---
        
        # 1. Check if we already have an OPEN trade for this ticker
        active_position = self.pm.get_active_position(symbol)
        
        if active_position:
            # WE ARE IN A TRADE -> MANAGE IT (Quiet Mode)
            entry_price = active_position['entry_price']
            current_stop = active_position['stop_loss']
            current_target = active_position['target_price']
            price = features.get('close')
            
            # A. Check for EXIT (Target Hit)
            if price >= current_target:
                self.pm.close_shadow_trade(symbol, price, active_position['qty'])  # Use PM to close
                self.ai.notifier.send_sell_alert(symbol, price, (price - entry_price)/entry_price, "TARGET 🎯")
                return # Done for this cycle

            # B. Check for EXIT (Stop Loss Hit)
            elif price <= current_stop:
                self.pm.close_shadow_trade(symbol, price, active_position['qty'])
                self.ai.notifier.send_sell_alert(symbol, price, (price - entry_price)/entry_price, "STOP 🛑")
                return # Done for this cycle

            # C. Trailing Stop Logic (Only update if significant move)
            # Only send an alert if we move the stop up by at least 1%
            new_suggested_stop = current_params['stop_loss']
            if new_suggested_stop > (current_stop * 1.01): 
                # Update the position in the file
                self.pm.update_stop_loss(symbol, new_suggested_stop)
                # Send Quiet Update
                self.ai.notifier.send_risk_update(symbol, new_suggested_stop, current_target, price, "Trailing Stop 🛡️")
            
            # DO NOT SEND "BUY" SIGNALS IF WE ARE ALREADY IN!
            
        else:
            # NO POSITION -> LOOK FOR ENTRY
            if decision == "BUY" and confidence > 0.75:
                # Send the Buy Signal (First time only)
                self.ai.notifier.send_buy_alert(
                    symbol, 
                    price, 
                    current_params['stop_loss'], 
                    current_params['target'], 
                    prob, 
                    fund_data.get('Score', 50)
                )
                # 2. RETURN the signal to the Main Loop
                return (decision, prob, price, trace, current_params)

        return None

    def execute_trade(self, symbol, action, price, details, params):
        """Execute or Log Trade."""
        if self.mode == 'PAPER':
            logger.info(f"!!! PAPER TRADE SIGNAL: {action} {symbol} !!!")
            logger.info(f"Price: {price} | Reason: {details}")
            
            # --- SHADOW PORTFOLIO TRACKING ---
            if action == "BUY":
                self.pm.add_shadow_trade(
                    ticker=symbol, 
                    entry_price=price,
                    stop_loss=params['stop_loss'],
                    target_price=params['target']
                )
            
            # Log to dedicated signals file
            with open("logs/signals.csv", "a") as f:
                f.write(f"{datetime.now()},{symbol},{action},{price},{details}\n")
                
        elif self.mode == 'LIVE':
            logger.warning("LIVE TRADING NOT YET ENABLED. Switch to Paper Mode.")
            # Future: self.dsm.place_order(...)

    def smart_sleep(self, seconds):
        """
        Sleeps for `seconds` but polls Telegram every 2 seconds.
        Keeps bot interactive during idle times.
        """
        end_time = time.time() + seconds
        while time.time() < end_time:
            # Poll Telegram
            self.ai.notifier.check_for_updates(self.pm)
            
            # Short sleep to prevent CPU spin
            time.sleep(2)

    def run(self):
        """Main Loop."""
        try:
            while self.is_running:
                # 0. Startup Notification (First Run Only)
                if not hasattr(self, '_startup_sent'):
                    msg = (f"🚀 **StockWise Gen-9 Engine Started**\n"
                           f"Targets: {len(self.symbols)}\n"
                           f"Mode: {self.mode}\n"
                           f"Interval: {self.interval}")
                    self.ai.notifier.send_alert(msg)
                    self._startup_sent = True

                # 1. Timezone Awareness
                tz = pytz.timezone(cfg.SchedulerConfig.MARKET_TIMEZONE)
                now = datetime.now(tz)
                
                # 2. Define Window for Today
                today_open = datetime.combine(now.date(), cfg.SchedulerConfig.OPEN_TIME).replace(tzinfo=tz)
                today_close = datetime.combine(now.date(), cfg.SchedulerConfig.CLOSE_TIME).replace(tzinfo=tz)
                
                active_start = today_open - timedelta(hours=cfg.SchedulerConfig.PRE_BUFFER_HOURS)
                active_end = today_close + timedelta(hours=cfg.SchedulerConfig.POST_BUFFER_HOURS)
                
                # 3. Check Scheduler Status
                is_active = False
                if self.is_trading_day(now.date()):
                    if active_start <= now <= active_end:
                        is_active = True
                        self.eod_run_today = False # Reset flag if market is open/active
                
                # --- EOD AUDIT TRIGGER ---
                # Run once after market close
                if now > today_close and not self.eod_run_today and self.is_trading_day(now.date()):
                    logger.info("🏁 Market Closed. Running EOD Maintenance...")
                    
                    # 1. Run Maintenance (Audit + Retrain)
                    # We import it here to avoid circular imports at top level if possible
                    from daily_maintenance import AutoCorrector
                    ac = AutoCorrector()
                    ac.run_routine()
                    
                    # 2. CRITICAL: HOT RELOAD THE BRAIN
                    # We must re-initialize the AI to load the NEW .keras file we just trained
                    logger.info("🧠 RELOADING AI MODEL from Disk...")
                    self.ai = StockWiseAI() 
                    logger.info("✅ Brain Reloaded.")

                    self.eod_run_today = True

                # 4. Handle Inactive State
                if not is_active:
                    next_run = self.get_next_run_time(now)
                    sleep_seconds = (next_run - now).total_seconds()
                    
                    # Safety check for negative sleep (shouldn't happen with logic above)
                    if sleep_seconds < 0: sleep_seconds = 60 
                    
                    hours = sleep_seconds / 3600
                    logger.info(f"[SCHEDULER] Market Closed. Sleeping for {hours:.2f} hours (until {next_run.strftime('%Y-%m-%d %H:%M:%S %Z')})...")
                    # Use Smart Sleep to stay interactive
                    self.smart_sleep(sleep_seconds)
                    continue
                
                # --- MARKET OPEN ---
                logger.info("\n--- Scanning Watchlist ---")
                
                for symbol in self.symbols:
                    # Check Updates Intermittently
                    self.ai.notifier.check_for_updates(self.pm)
                    
                    df, fund_data = self.fetch_and_process(symbol)
                    
                    if df is not None:
                        # NEW GEN-9 LOGIC
                        result = self.analyze_market(symbol, df, fund_data)
                        
                        if result:
                            decision, prob, price, trace, params = result
                            
                            if decision == "BUY":
                                 # We pass params to execute for shadow tracking
                                 self.execute_trade(symbol, "BUY", price, f"AI_Conf: {prob:.2%} | {trace}", params)
                                 
                                 # Smart Alert (Pass Ticker/Params to update history)
                                 # Note: The 'msg' construction logic is inside ai.predict (which sends alert).
                                 # Ideally, we should refactor that to send here, but to avoid large diffs:
                                 # We let AI send the alert. BUT we call a separate helper to update history?
                                 # Or better: We assume AI sent it. We just manually update history here to keep them in sync?
                                 # Actually `send_alert` inside AI core calls `notifier.send_alert`.
                                 # We need to update `StockWiseAI` to pass params to `send_alert`.
                                 # To do this safely without verifying AI Core file again, let's just trigger a "Shadow Update" here.
                                 self.ai.notifier.signal_history[symbol] = params
                                 self.ai.notifier._save_history()

                    
                    # Small delay between symbols
                    time.sleep(2)
                
                # Sleep between Full Scans
                sleep_sec = 60 if self.interval == '1m' else 900 
                if self.interval == '1d': sleep_sec = 3600 # 1 hour check for daily candles
                
                logger.info(f"Scan Complete. Sleeping for {sleep_sec} seconds...")
                self.smart_sleep(sleep_sec)
                
        except KeyboardInterrupt:
            logger.info("Stopping Live Engine...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='StockWise Gen-9 Live Trader')
    
    # Default basket
    default_basket = "NVDA,AMD,MSFT,GOOGL,AAPL,META,TSLA,QCOM,AMZN"
    
    parser.add_argument('--symbols', type=str, default=default_basket, help='Comma-separated ticker symbols')
    # Force Daily Interval in the code if you don't use CLI args
parser.add_argument('--interval', type=str, default='1d', help='Candle interval')
    parser.add_argument('--mode', type=str, default='PAPER', help='PAPER or LIVE')
    
    args = parser.parse_args()
    
    trader = LiveTrader(args.symbols, args.interval, args.mode)
    trader.run()
