
"""
StockWise Daily Auditor
=======================
Runs at EOD (End of Day) to:
1. Reconcile Shadow Portfolio (Check for Stops/Targets).
2. Calculate Real User Performance.
3. Compare System vs User (Gap Analysis).
4. Generate and send the Daily Report.
"""

import logging
import pandas as pd
from datetime import datetime

logger = logging.getLogger("DailyAuditor")

class DailyAuditor:
    def __init__(self, portfolio_manager, data_source_manager, notifier):
        self.pm = portfolio_manager
        self.dsm = data_source_manager
        self.notifier = notifier

    def generate_eod_report(self):
        """
        Main EOD Process.
        Fetches live prices, updates ledgers, and generates report.
        """
        logger.info("--- Starting EOD Audit ---")
        
        # 1. Fetch Active Symbols
        shadow_trades = self.pm.get_active_positions('SHADOW')
        user_trades = self.pm.get_active_positions('USER')
        
        all_tickers = list(set([t['ticker'] for t in shadow_trades] + [t['ticker'] for t in user_trades]))
        
        if not all_tickers:
            logger.info("No active positions to audit.")
            # Still send a basic "No active trades" report? 
            # Or just skip. Let's send a summary so user knows system is alive.
            if self.notifier.enabled:
                self.notifier.send_alert("📊 EOD REPORT\n\nNo active positions.")
            return

        # 2. Fetch Current Prices (Batch or Loop)
        current_prices = {}
        for ticker in all_tickers:
            # We fetch 1 day to get latest Close
            # Using interval='1d' to get today's close
            df = self.dsm.get_stock_data(ticker, days_back=5, interval='1d')
            if not df.empty:
                current_prices[ticker] = df.iloc[-1]['close']
            else:
                logger.warning(f"Could not fetch closing price for {ticker}")
                current_prices[ticker] = 0.0

        # 3. Shadow Audit (Update System Status)
        shadow_closed_today = []
        shadow_pnl_unrealized = 0.0
        shadow_pnl_realized = 0.0 # From closed today
        
        # Update Status based on today's price action
        for ticker in all_tickers:
             if ticker in current_prices:
                 closed = self.pm.update_position_status(ticker, current_prices[ticker])
                 shadow_closed_today.extend(closed)
        
        # Recalculate Active List after updates
        active_shadow = self.pm.get_active_positions('SHADOW')
        
        # Calc Unrealized PnL (Shadow)
        for t in active_shadow:
            cp = current_prices.get(t['ticker'], t['entry_price'])
            u_pnl_pct = (cp - t['entry_price']) / t['entry_price']
            shadow_pnl_unrealized += (u_pnl_pct * t['allocation'])
            
        # Calc Realized PnL (Shadow - Historical + Closed Today)
        # Note: PM doesn't strictly track date-based realized PnL in simple JSON, 
        # so we just sum EVERYTHING for "Total System PnL"
        total_shadow_pnl = 0.0
        shadow_won = 0
        shadow_lost = 0
        for t in self.pm.shadow_portfolio.get('trades', []):
            if t['status'] == 'CLOSED':
                total_shadow_pnl += t['pnl']
                if t['pnl'] > 0: shadow_won += 1
                else: shadow_lost += 1
        
        shadow_wr = (shadow_won / (shadow_won + shadow_lost)) if (shadow_won + shadow_lost) > 0 else 0.0

        # 4. User Audit (Real Performance)
        user_pnl_unrealized = 0.0
        user_holdings_str = ""
        
        for t in user_trades:
            cp = current_prices.get(t['ticker'], t['entry_price'])
            if cp > 0:
                val_entry = t['entry_price'] * t['qty']
                val_now = cp * t['qty']
                pnl = val_now - val_entry
                user_pnl_unrealized += pnl
                
                pct = (cp - t['entry_price']) / t['entry_price'] * 100
                user_holdings_str += f"- {t['ticker']}: ${pnl:+.2f} ({pct:+.1f}%)\n"
        
        # 5. Gap Analysis (System vs User)
        # Simple comparison of Active Count
        missed_opps = len(active_shadow) - len(user_trades)
        
        # 6. Generate Report
        report = "📊 EOD REPORT\n"
        report += f"Date: {datetime.now().strftime('%Y-%m-%d')}\n\n"
        
        # Section 1: Real Money
        report += "👤 **MY PORTFOLIO**\n"
        if user_trades:
            report += f"Open PnL: ${user_pnl_unrealized:+.2f}\n"
            report += "Active Holdings:\n" + user_holdings_str
        else:
            report += "No Active Trades.\n"
            
        report += "\n"
        
        # Section 2: System Potential
        report += "🤖 **SYSTEM PERFORMANCE (Shadow)**\n"
        report += f"Total PnL (All Time): ${total_shadow_pnl:+.2f}\n"
        report += f"Win Rate: {shadow_wr:.1%}\n"
        report += f"Active Shadow Positions: {len(active_shadow)}\n"
        if shadow_closed_today:
            report += f"Closed Today: {len(shadow_closed_today)}\n"

        # Section 3: Gap Analysis
        if missed_opps > 0:
            report += f"\n⚠️ **GAP ANALYSIS**\n"
            report += f"System found {len(active_shadow)} setups. You took {len(user_trades)}.\n"
            report += f"Potentially missed {missed_opps} trades this cycle.\n"

        # Send
        if self.notifier.enabled:
            self.notifier.send_alert(report)
        
        logger.info("EOD Audit Complete & Report Sent.")
