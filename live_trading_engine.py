# live_trading_engine.py

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
import json
import pytz
from datetime import datetime, timedelta
import traceback
import numpy as np

# Ensure project root is in path
sys.path.append(os.getcwd())

# --- FIX WINDOWS EMOJI CRASH ---
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

from data_source_manager import DataSourceManager
from feature_engine import RobustFeatureCalculator
from strategy_engine import StrategyOrchestra, MarketRegimeDetector
from stockwise_ai_core import StockWiseAI
from portfolio_manager import PortfolioManager
from auditor import DailyAuditor
import system_config as cfg
import notification_manager as nm

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler("logs/live_trading.log", encoding='utf-8'),
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
        self.dsm = DataSourceManager(use_ibkr=cfg.EN_IBKR)
        self.feature_calc = RobustFeatureCalculator(params={})
        self.ai = StockWiseAI() # GEN-9 CORE
        
        # --- SMART ASSISTANT LAYER ---
        self.pm = PortfolioManager()
        self.auditor = DailyAuditor(self.pm, self.dsm, self.ai.notifier)
        
        self.is_running = True
        self.eod_run_today = False # Track if EOD ran for current date

        self.status_file = "logs/live_status.json"
        self.update_status("Initializing", "Engine starting up...")
        
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

        # --- 🔥 CRASH FIX: ROBUST DATA SANITIZATION ---
        try:
            # 1. Identify Numeric Columns (Avoid processing strings/objects)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            # 2. Replace Infinity with NaN (Only in numeric columns)
            df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
            
            # 3. Forward Fill (Fix FutureWarning: use ffill() instead of method='ffill')
            df.ffill(inplace=True)
            
            # 4. Fill remaining NaNs with 0
            df.fillna(0, inplace=True)

            # 5. Final Safety Check (Check ONLY numeric columns)
            check_data = df[numeric_cols].values
            if np.isnan(check_data).any() or np.isinf(check_data).any():
                 logger.error(f"[{symbol}] CORRUPT DATA DETECTED (NaN/Inf). Skipping.")
                 return None
                 
        except Exception as e:
            logger.error(f"[{symbol}] Data Sanitization Failed: {e}")
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
            "stop_loss": stop_price,
            "timestamp": str(latest_bar.name)
        }
        
        # --- GET AI CONFIDENCE (Raw Prob) ---
        _, prob, trace = self.ai.predict_trade_confidence(
            symbol, features, fundamentals=fund_data, df_window=df
        )

        # --- ASK STRATEGY ORCHESTRA (The Brain) ---
        analysis_packet = {
            'AI_Probability': prob,
            'Fundamental_Score': fund_data.get('Score', 50)
        }
        
        # Use the updated Strategy Logic
        decision = StrategyOrchestra.decide_action(symbol, features, analysis_packet)
        
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
                # --- DYNAMIC EXIT (Let Winners Run) ---
                ema_20 = features.get('ema_20') or features.get('sma_20') # Fallback to SMA if EMA missing
                
                if ema_20 and price > ema_20:
                    # Trend is still strong! Don't sell yet.
                    # Instead, move Stop Loss to Breakeven or slightly below current price to lock gains
                    new_sl = max(current_stop, price * 0.95) # Lock in some profit but keep room
                    if new_sl > current_stop:
                        self.pm.update_stop_loss(symbol, new_sl)
                        logger.info(f"Target Hit but Trend Strong (Price > EMA20). Holding & Trailing SL to {new_sl:.2f}")
                    return 
                else:
                    # Trend broken OR no EMA data -> Take Profit
                    self.pm.close_shadow_trade(symbol, price, active_position['qty'])  
                    self.ai.notifier.send_sell_alert(symbol, price, (price - entry_price)/entry_price, "TARGET 🎯")
                    return

            # B. Check for EXIT (Stop Loss Hit)
            # Use LOW and OPEN to simulate reality better than CLOSE
            day_low = features.get('low', price)
            day_open = features.get('open', price)
            
            if day_low <= current_stop:
                # Calculate REALISTIC Exit Price
                if day_open < current_stop:
                    exit_price = day_open # Gap Down Reality
                    reason = "STOP 🛑 (GAP DOWN)"
                else:
                    exit_price = current_stop # Standard Stop Hit
                    reason = "STOP 🛑"

                self.pm.close_shadow_trade(symbol, exit_price, active_position['qty'])
                self.ai.notifier.send_sell_alert(symbol, exit_price, (exit_price - entry_price)/entry_price, reason)
                return 

            # C. Trailing Stop Logic (Only update if significant move)
            # Only send an alert if we move the stop up by at least 1%
            new_suggested_stop = current_params['stop_loss']
            if new_suggested_stop > (current_stop * 1.01): 
                # Update the position in the file
                self.pm.update_stop_loss(symbol, new_suggested_stop)
                # Send Quiet Update
                self.ai.notifier.send_risk_update(symbol, new_suggested_stop, current_target, price, "Trailing Stop 🛡️")
            
        else:
            # NO POSITION -> LOOK FOR ENTRY
            # We now use the 'decision' from StrategyOrchestra (which handles the Falling Knife check)
            if decision == "BUY":
                # Send the Buy Signal
                self.ai.notifier.send_buy_alert(
                    symbol, price, current_params['stop_loss'], current_params['target'], 
                    prob, fund_data.get('Score', 50)
                )
                return (decision, prob, price, trace, current_params)

        return None

    def execute_trade(self, symbol, action, price, details, params):
        """Execute or Log Trade."""

        # 1. Get the Fixed Amount from Config (The line you asked about)
        max_dollars = cfg.INVESTMENT_AMOUNT

        # --- VOLATILITY POSITION SIZING ---
        # 2. Calculate Risk-Based Sizing
        stop_loss_price = params['stop_loss']
        risk_per_share = price - stop_loss_price
        
        # Account Settings (Hardcoded for now, move to config later)
        account_balance = 100000 # Example: $100k account
        risk_per_trade_pct = 0.02 # Risk 2% per trade ($2000)
        
        if risk_per_share > 0:
            qty_risk = int((account_balance * risk_per_trade_pct) / risk_per_share)
        else:
            qty_risk = 1 
            
        # 3. Calculate Quantity based on Dollar Cap
        qty_cap = int(max_dollars / price)
        
        # 4. FINAL DECISION: Take the SMALLER of the two
        qty = min(qty_risk, qty_cap)
            
        logger.info(f"Calculated Position Size: {qty} shares (Risk: ${risk_per_share*qty:.2f})")

        if self.mode == 'PAPER':
            logger.info(f"!!! PAPER TRADE SIGNAL: {action} {symbol} !!!")
            logger.info(f"Price: {price} | Reason: {details}")
            
            # --- SHADOW PORTFOLIO TRACKING ---
            if action == "BUY":
                self.pm.add_shadow_trade(
                    ticker=symbol, 
                    entry_price=price,
                    stop_loss=params['stop_loss'],
                    target_price=params['target'],
                    qty=qty
                )
            
            self.log_trade_csv(symbol, "BUY", price, params['stop_loss'], params['target'])
            
            # Log to dedicated signals file
            with open("logs/signals.csv", "a") as f:
                f.write(f"{datetime.now()},{symbol},{action},{price},{details}\n")
                
        elif self.mode == 'LIVE':
            logger.warning("LIVE TRADING NOT YET ENABLED. Switch to Paper Mode.")

    def smart_sleep(self, seconds):
        """
        Sleeps for `seconds` but polls Telegram every 2 seconds.
        Keeps bot interactive during idle times.
        """
        end_time = time.time() + seconds
        while time.time() < end_time:
            # Poll Telegram
            try:
                self.ai.notifier.check_for_updates(self.pm)
            except Exception as e:
                # Log error but DO NOT CRASH
                logger.warning(f"Telegram Poll Error: {e} (Retrying...)")
            
            # Short sleep to prevent CPU spin
            time.sleep(2)

    def update_status(self, state, message, last_scan=None):
        """Saves heartbeat for the GUI to read."""
        status = {
            "status": state,
            "message": message,
            "last_heartbeat": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "last_scan_time": last_scan
        }
        with open(self.status_file, 'w') as f:
            json.dump(status, f)

    def log_trade_csv(self, symbol, type, price, stop, target):
        """Saves trade to CSV for the GUI PnL table."""
        file = "logs/portfolio_trades.csv"
        new_row = pd.DataFrame([{
            "Date": datetime.now(), "Symbol": symbol, "Type": type, 
            "Price": price, "Stop": stop, "Target": target, "Status": "OPEN"
        }])
        
        if os.path.exists(file):
            new_row.to_csv(file, mode='a', header=False, index=False)
        else:
            new_row.to_csv(file, mode='w', header=True, index=False)

    def run(self):
        """Main Loop."""
        try:
            while self.is_running:
                self.update_status("Active", "Checking Scheduler...", datetime.now().strftime("%H:%M"))

                # 0. Startup Notification (First Run Only)
                if not hasattr(self, '_startup_sent'):
                    msg = (f"**StockWise Gen-9 Engine Started**\n"
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
                    logger.info("Market Closed. Running EOD Maintenance...")
                    
                    # 1. Run Maintenance (Audit + Retrain)
                    # We import it here to avoid circular imports at top level if possible
                    from daily_maintenance import AutoCorrector
                    ac = AutoCorrector()
                    ac.run_routine()
                    
                    # 2. CRITICAL: HOT RELOAD THE BRAIN
                    # We must re-initialize the AI to load the NEW .keras file we just trained
                    logger.info("RELOADING AI MODEL from Disk...")
                    self.ai = StockWiseAI() 
                    logger.info("Brain Reloaded.")

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

                # --- Update GUI Status ---
                self.update_status("Scanning", f"Analyzing {len(self.symbols)} items", datetime.now().strftime("%H:%M"))
                
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
                                 # (See reasoning in original)
                                 self.ai.notifier.signal_history[symbol] = params
                                 self.ai.notifier._save_history()

                    
                    # Small delay between symbols
                    time.sleep(2)
                
                # Sleep between Full Scans
                sleep_sec = 60 if self.interval == '1m' else 900 
                if self.interval == '1d': sleep_sec = 3600 # 1 hour check for daily candles
                
                logger.info(f"Scan Complete. Sleeping for {sleep_sec} seconds...")
                # ---Update GUI Status ---
                self.update_status("Idle", f"Sleeping ({sleep_sec}s)", datetime.now().strftime("%H:%M"))
                
                self.smart_sleep(sleep_sec)
                
        except KeyboardInterrupt:
            logger.info("Stopping Live Engine...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='StockWise Gen-9 Live Trader')
    
    # Default basket
    # Default basket
    if hasattr(cfg, 'WATCHLIST') and isinstance(cfg.WATCHLIST, list):
        default_basket = ",".join(cfg.WATCHLIST)
    else:
        default_basket = "NVDA,AMD,MSFT" # Fallback    
    
    parser.add_argument('--symbols', type=str, default=default_basket, help='Comma-separated ticker symbols')
    parser.add_argument('--interval', type=str, default='1d', help='Candle interval')
    parser.add_argument('--mode', type=str, default='PAPER', help='PAPER or LIVE')
    
    args = parser.parse_args()
    
    # Robust Handling: Ensure we have a list of symbols
    if isinstance(args.symbols, list):
        symbols = args.symbols
    else:
        symbols = args.symbols.split(',')
    
    print(f"\n{'='*50}")
    print(f"STOCKWISE LIVE ENGINE | Mode: {args.mode}")
    print(f"{'='*50}\n")
    
    try:
        # Initialize and Run
        trader = LiveTrader(symbols=symbols, interval=args.interval, mode=args.mode)
        trader.run()

    except KeyboardInterrupt:
        # User manually stopped the script (Ctrl+C)
        logger.info("🛑 Engine stopped by user request.")
        sys.exit(0)

    except Exception as e:
        # --- CRASH HANDLER ---
        error_msg = str(e)
        stack_trace = traceback.format_exc()
        
        # 1. Log Critical Error locally
        logger.critical(f"CRITICAL ENGINE FAILURE: {error_msg}", exc_info=True)
        
        # 2. Send Telegram Alert
        try:
            logger.info("Attempting to send Telegram Crash Alert...")
            notifier = nm.NotificationManager()
            
            # Format message (Markdown)
            telegram_msg = (
                f"**SYSTEM CRASH ALERT**\n\n"
                f"**Engine:** StockWise Gen-10\n"
                f"**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                f"**Error:** `{error_msg}`\n\n"
                f"**Traceback (Last 10 lines):**\n"
                f"`{stack_trace[-1000:]}`"  # Truncate to avoid Telegram limit
            )
            
            # Send (assuming .send_message or .send_telegram_message exists)
            if hasattr(notifier, 'send_message'):
                notifier.send_message(telegram_msg)
            elif hasattr(notifier, 'send_telegram_message'):
                notifier.send_telegram_message(telegram_msg)
            else:
                logger.error("Could not find 'send_message' method in NotificationManager")
                
            logger.info("Crash alert sent successfully.")
            
        except Exception as notify_err:
            logger.error(f"Failed to send Telegram alert: {notify_err}")
        
        # 3. Exit with Error Code 1
        # This tells the .bat file that a crash occurred (triggering the restart loop)
        sys.exit(1)
