# notification_manager.py

"""
StockWise Gen-12 Notification Manager
=====================================
The Communication Hub.
Responsible for:
1. Sending real-time alerts to the User (Telegram).
2. Handling incoming commands (Bidirectional Control).
3. Managing the Interactive Wizard logic for manual trades.
4. Resilience: Queuing messages when offline.
"""

import logging
import time
import requests
import system_config as cfg
from datetime import datetime
import os
import json
from safe_json_io import safe_json_read, safe_json_write

logger = logging.getLogger("NotificationManager")

class NotificationManager:
    """
    Handles all external communication via Telegram Bot API.
    Includes a Polling mechanism for receiving commands.
    """
    def __init__(self):
        self.token = cfg.TELEGRAM_TOKEN
        self.chat_id = cfg.TELEGRAM_CHAT_ID
        # Enabled only if credentials are provided in secrets.toml
        self.enabled = bool(self.token and self.chat_id)
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.last_update_id = 0
        
        # State machine for the Interactive Wizard (e.g. Buying Process)
        # Format: {chat_id: {"step": "PRICE/QTY/STRATEGY", "data": {...}}}
        self.conversation_state = {} 
        
        # Buffer for messages created while offline
        self.message_queue = []

    def send_ibkr_disconnect_alert(self):
        """
        [Proactive Health Monitoring]
        Broadcasts a critical system alert to Telegram when IB Gateway drops connection 
        or hits the Soft Token=0 weekly reset.
        """
        alert_msg = (
            "SYSTEM ALERT: IBKR Gateway is DISCONNECTED.\n"
            "Weekly reauthentication required (Soft token=0).\n"
            "Please login to IB Gateway on the server."
        )
        # Strictly no icons in the log file to comply with system rules
        logger.error("Triggering Telegram Alert: IB Gateway Disconnected.")
        
        # Assuming you have a base send_message method in this class to hit the Telegram API
        if hasattr(self, 'send_message'):
            self.send_message(alert_msg)
        else:
            logger.warning("send_message method not implemented. Alert printed to log only.")

    def _update_ledger_status(self, ticker, new_status):
        """
        [Execution Vector Synchronization]
        Safely edits the JSON shadow ledger directly from Telegram headless control.
        """
        import json
        import os
        
        journal_path = getattr(cfg, 'TRADE_JOURNAL_PATH', 'data/trade_journal.json')
        try:
            journal = safe_json_read(journal_path, default={})

            # Check if the ticker exists and has history
            if ticker in journal and len(journal[ticker]) > 0:
                # Overwrite the status of the most recent trade entry for this ticker
                journal[ticker][-1]['status'] = new_status
                safe_json_write(journal_path, journal)
                logger.info(f"Shadow Ledger manually synced via Telegram: [{ticker}] marked as {new_status}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to update ledger from Telegram: {str(e)}")
            return False

    def process_incoming_command(self, text):
        """
        [Headless Control Feedback Loop]
        Parses commands from the user to categorize paper trades into reality for ML training.
        Commands:
        /confirm [TICKER] -> Marks trade as executed in reality (2x weight in ML)
        /unfilled [TICKER] -> Marks trade as slipped/unfillable (Trains ML liquidity risk)
        """
        if not text or not text.startswith('/'):
            return
            
        parts = text.strip().upper().split()
        command = parts
        
        if command in ['/CONFIRM', '/UNFILLED'] and len(parts) > 1:
            ticker = parts[1]
            
            # Determine standard ML flag from system configuration
            if command == '/CONFIRM':
                status = getattr(cfg, 'TRADE_STATUS_EXECUTED', 'CONFIRMED')
            else:
                status = getattr(cfg, 'TRADE_STATUS_UNFILLED', 'UNFILLED')
                
            success = self._update_ledger_status(ticker, status)
            
            if hasattr(self, 'send_message'):
                if success:
                    self.send_message(f"System Update: {ticker} marked as {status}. ML Execution Vector synced.")
                else:
                    self.send_message(f"System Error: Failed to update ledger for {ticker}.Check logs.")

    def send_message(self, message):
        """
        Sends a message to the Telegram Chat.
        If offline, adds to queue for retry.
        """
        if not self.enabled:
            logger.info(f"[TG Mock]: {message}") # Mock mode for dev/paper
            return
        
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {"chat_id": self.chat_id, "text": message, "parse_mode": "Markdown"}
            # Short timeout to avoid blocking main thread
            requests.post(url, json=payload, timeout=5)
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            logger.info("Queueing message for retry.")
            self.message_queue.append(message)

    def check_for_updates(self):
        """
        Periodically called by the main loop to keep the bot alive
        and retry any queued messages.
        """
        self._retry_queue()
        
    def _retry_queue(self):
        """Attempts to flush the message queue."""
        if not self.message_queue: return
        
        logger.info(f"Retrying {len(self.message_queue)} queued messages...")
        # Create a copy to iterate safely while modifying
        queue_copy = list(self.message_queue)
        self.message_queue = []
        
        for msg in queue_copy:
            self.send_message(msg)

    def poll_commands(self, system_controller):
        """
        Polls Telegram headers for new updates/commands.
        Handles the "Wizard" state machine for multi-step inputs.
        """
        # Always try to clear queue first
        self._retry_queue()
        
        if not self.enabled: return

        try:
            # Long Polling (getUpdates)
            url = f"{self.base_url}/getUpdates"
            params = {"offset": self.last_update_id + 1, "timeout": 1} # Short timeout for loop responsiveness
            resp = requests.get(url, params=params, timeout=5)
            data = resp.json()
            
            if not data.get("ok"): return
            
            for result in data.get("result", []):
                self.last_update_id = result["update_id"]
                chat_id = result["message"]["chat"]["id"]
                text = result["message"].get("text", "").strip()
                
                # 1. Handle Active Conversation (Wizard)
                if chat_id in self.conversation_state:
                    self._handle_wizard_step(chat_id, text, system_controller)
                    continue

                # 2. Handle New Commands
                self._handle_command(chat_id, text, system_controller)
                
        except Exception as e:
            logger.debug(f"Polling error: {e}")

    def _handle_command(self, chat_id, text, system_controller):
        """Parses and routes slash commands."""
        
        if text.startswith("/buy "):
            # Syntax: /buy NVDA
            try:
                ticker = text.split(" ")[1].upper()
                # Start Conversation state
                self.conversation_state[chat_id] = {"step": "PRICE", "data": {"ticker": ticker}}
                self.send_message(f"Buying **{ticker}**. What is the Limit Price? (e.g., 140.50)")
            except:
                self.send_message("Usage: /buy [TICKER]")
                
        elif text.lower().startswith("sold "):
            # User reports manual sale: "sold AAPL" or "sold NVDA"
            try:
                ticker = text.split()[1].upper()
                # Notify the system controller to close this position
                if hasattr(system_controller, 'mark_position_sold'):
                    system_controller.mark_position_sold(ticker)
                    self.send_message(f"Position {ticker} marked as SOLD. Updated.")
                else:
                    self.send_message(f"Received: {ticker} sold. System will update at next cycle.")
                logger.info(f"User reported sale: {ticker}")
            except (IndexError, Exception) as e:
                self.send_message("Usage: sold TICKER (e.g., sold AAPL)")

        elif text == "/sell":
            self.send_message("Selling functionality not fully interactive yet.")
            
        elif text == "/status":
            # Reports system health/stats
            self.send_message(system_controller.get_status_report())
            
        elif text == "/scan":
            self.send_message("Triggering Manual Scan...")
            # Trigger scanning logic (requires callback implementation in controller)
            # system_controller.force_scan() 
            
        elif text == "/report":
            self.send_message("Generating Report...")
            # Generate PnL report

    def _handle_wizard_step(self, chat_id, text, system_controller):
        """
        Processes one step of the interactive buying wizard.
        Steps: PRICE -> QTY -> STRATEGY -> EXECUTE
        """
        state = self.conversation_state[chat_id]
        step = state["step"]
        data = state["data"]
        
        if step == "PRICE":
            data["price"] = text
            state["step"] = "QTY"
            self.send_message(f"Price set to ${text}. How many shares? (e.g., 10)")
            
        elif step == "QTY":
            data["qty"] = text
            state["step"] = "STRATEGY"
            self.send_message("Shares set. Which Strategy? (SNIPER/TACTICAL/STRATEGIC)")
            
        elif step == "STRATEGY":
            data["strategy"] = text.upper()
            
            # Final Confirmation
            ticker = data["ticker"]
            price = data["price"]
            qty = data["qty"]
            strat = data["strategy"]
            
            self.send_message(f"✅ Order Confirmed: Buy {qty} {ticker} @ ${price} ({strat})")
            
            # Execute
            system_controller.execute_manual_trade(ticker, float(price), int(qty), strat)
            
            # Clear State (End Wizard)
            del self.conversation_state[chat_id]

    def generate_stock_report(self, symbol, action, price, score, agent):

        """
        Generates a formatted Markdown report for trade alerts.
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        emoji = "🟢" if action == "BUY" else "🔴"
        
        report = f"""
        {emoji} **TRADE ALERT: {symbol}**
        ---------------------------
        ⏰ Time: {timestamp}
        🤖 Agent: {agent}
        🎯 Action: {action} @ ${price:.2f}
        🏆 Score: {score}/100

        **Analysis:**
        The {agent} agent identified a high-probability setup based on the 'Master Function' matrix. 
        """
        return report

    def send_eod_summary(self, date_str, portfolio_status, system_pnl, win_rate, active_shadow, setups_found, user_taken):
        """
        Transmits the End of Day summary report to the user.
        Calculates the Gap Analysis (System intent vs User action).
        """
        missed = setups_found - user_taken
        
        report = (
            f"📊 END OF DAY SUMMARY: {date_str}\n"
            f"---------------------------\n"
            f"💰 Portfolio Status: {portfolio_status}\n"
            f"📈 System PnL: {system_pnl}\n"
            f"🎯 Win Rate: {win_rate}%\n"
            f"👻 Active Shadow Trades: {active_shadow}\n"
            f"🤖 Suggested stocks: {setups_found}, Executed (Buy/Sell): {user_taken}\n"
        )
        
        # Strictly no icons in the log file
        logger.info(f"Sending EOD Summary to Telegram. Suggested: {setups_found} | Executed: {user_taken}")
        
        if hasattr(self, 'send_message'):
            self.send_message(report)
            
        return report

    def _log_trade_to_csv(self, ticker, buy_date, buy_price, sell_price, pnl_net):
        """
        [Admin & ML Database] 
        Silently appends closed position data into a centralized CSV file 
        for long-term statistical tracking and ML training enhancement.
        """
        import csv
        import os
        
        csv_path = getattr(cfg, 'TRADE_HISTORY_CSV_PATH', 'data/trade_history.csv')
        
        try:
            # Ensure the directory exists before saving
            directory = os.path.dirname(csv_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
                
            file_exists = os.path.isfile(csv_path)
            
            with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Write header row if the file is completely new
                if not file_exists:
                    writer.writerow(['Ticker', 'Suggested_Buying_Date', 'Buying_Price', 'Selling_Price', 'PnL_Net_Pct'])
                
                writer.writerow([ticker, buy_date, buy_price, sell_price, pnl_net])
                
            logger.debug(f"Trade history natively appended to CSV for Admin: [{ticker}]")
        except Exception as e:
            logger.error(f"Critical error writing trade to CSV history: {str(e)}")

    def send_closed_position_report(self, ticker, buy_date, buy_price, sell_price, pnl_net):
        """
        [Trade Lifecycle Alert] 
        Generates and sends the exact user-requested breakdown when a position is closed 
        (profit or loss), and routes the data to the Admin CSV log.
        """
        report = (
            f"Stock name: {ticker}\n"
            f"Suggested buying date: {buy_date}\n"
            f"Buying Price: ${buy_price:.2f}\n"
            f"Selling Price: ${sell_price:.2f}\n"
            f"PnL Net: {pnl_net:.2f}%\n"
        )
        
        # Logging without icons
        logger.info(f"Closed Position Alert Generated: [{ticker}] | PnL: {pnl_net:.2f}%")
        
        if hasattr(self, 'send_message'):
            self.send_message(report)
            
        # Call the internal CSV logger to satisfy Admin visibility requirements
        self._log_trade_to_csv(ticker, buy_date, buy_price, sell_price, pnl_net)
