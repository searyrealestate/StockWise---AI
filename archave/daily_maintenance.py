# daily_maintenance.py

import logging
from datetime import datetime, date
import pandas as pd
import sys

# Import your system modules
from portfolio_manager import PortfolioManager
from auditor import DailyAuditor
from data_source_manager import DataSourceManager
from notification_manager import NotificationManager
from training_manager import TrainingManager # Unified Manager
from watchlist_manager import WatchlistManager
from stock_hunter import StockHunter


# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("DailyMaintenance")

class AutoCorrector:
    def __init__(self):
        self.pm = PortfolioManager()
        self.dsm = DataSourceManager()
        self.notifier = NotificationManager()
        self.auditor = DailyAuditor(self.pm, self.dsm, self.notifier)
        
        # --- CONFIGURATION ---
        self.MIN_TRADES_FOR_VERDICT = 3   # Don't retrain if we only made 1 trade
        self.WIN_RATE_THRESHOLD = 55.0    # % Win Rate required to stay "Synced"
        
    def run_routine(self, simulation_mode=False, mock_trades=None):
        """
        Main Workflow: Check -> Verdict -> Correction.
        :param simulation_mode: If True, uses mock_trades instead of real files (for verification script).
        """
        logger.info("STARTING NIGHTLY MAINTENANCE ROUTINE...")

        # 1. THE CHECK (Audit)
        # -------------------
        recent_trades = []                  
        
        if not simulation_mode:
            logger.info("Running EOD Audit (Updating Closing Prices)...")
            self.auditor.generate_eod_report()
            
            # Fetch closed trades history
            all_trades = self.pm.shadow_portfolio.get('trades', [])
            
                                                               
                                                                 
            today_str = datetime.now().strftime('%Y-%m-%d')
            
            # Filter for trades closed TODAY
            todays_trades = [
                t for t in all_trades 
                if t.get('status') == 'CLOSED' and t.get('exit_timestamp', '').startswith(today_str)
 
                # In a real scenario, check if exit_date == today. 
                # For simplicity, we check if they are in the list.
            ]
            if len(todays_trades) > 0:
                logger.info(f"Analyzing {len(todays_trades)} trades closed TODAY.")
                recent_trades = todays_trades
            else:
                # Fallback: If no trades today, look at last 10 to check general health
                logger.info("No trades closed today. Analyzing last 10 historical trades.")
                recent_trades = all_trades[-10:]                                                                            
                                                                                     
        else:
            # SIMULATION MODE (For verify_sniper_logic.py)
            logger.info("🧪 RUNNING IN SIMULATION MODE")
            recent_trades = mock_trades

        # 2. THE VERDICT (Grade)
        # ----------------------
        total = len(recent_trades)
        win_rate = 0.0
        if total > 0:
            wins = sum(1 for t in recent_trades if t['pnl'] > 0)
            win_rate = (wins / total) * 100
            logger.info(f"Recent Performance: {wins}/{total} Wins ({win_rate:.1f}%)")
        else:
            logger.info("Not enough data to grade.")

        # wins = sum(1 for t in recent_trades if t['pnl'] > 0)					
        # win_rate = (wins / total) * 100
        
                  
                                                                
                                           
             
                                                           
        # 3. THE CORRECTION (Continuous Learning)
        # ----------------------------
        # We retrain EVERY night to include today's new data into the brain.
        logger.info("CONTINUOUS LEARNING: Initiating nightly model update...")
        status = "IDLE"                       

        try:
            if not simulation_mode:
                # 1. Train the model (This fetches all history + TODAY's new data)
                # Notify User
                msg = f"🧠 <b>NIGHTLY TRAINING</b>\nWin Rate: {win_rate:.1f}%\nAbsorbing today's market data..."
                self.notifier.send_message(msg)
                
                # EXECUTE TRAINING (The heavy lifting)                                                                                                  
                                               
                
                                                      
                # EXECUTE TRAINING (The heavy lifting)
                tm = TrainingManager()
                tm.trainer.train()
                
                logger.info("Daily Improvement Complete. Model Updated.")
                self.notifier.send_message("✅ <b>UPGRADE COMPLETE</b>\nSystem is ready for tomorrow.")
                status = "UPDATED"                  
            else:
                logger.info("🧪 [SIMULATION] Model would be retrained here.")
                status = "SIMULATED_UPDATE"

        except Exception as e:
            logger.error(f"NIGHTLY TRAINING FAILED: {e}")
            if not simulation_mode:
                self.notifier.send_message(f"❌ <b>TRAINING ERROR</b>\n{e}")
            status = "ERROR"

        # 4. THE RESET (Now Reachable!)
        # -----------------------------
        logger.info("Performing System Cleanup...")
        # Add any cleanup logic here if needed in the future
        
        logger.info("Routine Complete.")
        
        return status

    def run_nightly_expansion(self):
        """
        The 'Iterative Growth' Routine.
        1. Find 10 new stocks.
        2. Add to Watchlist.
        3. Train models ONLY for the new 10 (Fast).
        """
        logger.info("🌱 Starting Nightly Expansion...")
        
        # 1. Discovery
        hunter = StockHunter(self.dsm)
        new_candidates = hunter.run_discovery_scan()
        
        if not new_candidates:
            logger.info("No new candidates found.")
            return

        # 2. Add to Watchlist
        wm = WatchlistManager()
        added_stocks = wm.add_new_candidates(new_candidates)
        
        # 3. Surgical Training (Train ONLY the new ones)
        if added_stocks:
            logger.info(f"🧠 Training models for {len(added_stocks)} new stocks...")
            
            # We use the Unified TrainingManager to recruit and train
            tm = TrainingManager()
            for symbol in added_stocks:
                # recruit_new_stock handles: Sync Gold -> Data Engineer -> Train
                # We can just call it, or if we want to batch train, we do data engineer first then train once.
                # But recruit_new_stock is designed for JIT.
                tm.recruit_new_stock(symbol)
                
            logger.info("✅ Expansion Complete. System IQ Increased.")

if __name__ == "__main__":
    ac = AutoCorrector()
    # ac.run_routine()
    ac.run_nightly_expansion()
