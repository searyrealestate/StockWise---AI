import logging
from datetime import datetime, date
import pandas as pd
import sys

# Import your system modules
from portfolio_manager import PortfolioManager
from auditor import DailyAuditor
from data_source_manager import DataSourceManager
from notification_manager import NotificationManager
import train_gen9_model  # <--- The ability to retrain itself

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
                                               
                
                                                      
                train_gen9_model.train_model()
                
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

if __name__ == "__main__":
    ac = AutoCorrector()
    ac.run_routine()