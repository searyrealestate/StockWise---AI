# notification_manager.py

"""
StockWise Notification Manager (Telegram Bot)
==============================================
Handles sending automated trading alerts and system status messages via Telegram.
Now includes interactive features: "Smart Alerts" and "User Input Processing".
"""

import requests
import os
import logging
import json
import time
from datetime import datetime
import system_config as cfg

logger = logging.getLogger("NotificationManager")

TELEGRAM_API_BASE_URL = "https://api.telegram.org/bot"


class NotificationManager:
    """
    Manages communication with the Telegram Bot API to send alerts and receive commands.
    """

    def __init__(self):
        self.token = cfg.TELEGRAM_TOKEN
        self.chat_id = cfg.TELEGRAM_CHAT_ID
        
        # Smart Alert History
        self.history_file = "dsm_signal_history.json"
        self.signal_history = self._load_history()
        
        # User Input State Machine
        self.last_update_id = 0
        self.user_states = {} # {chat_id: {"state": "AWAITING_PRICE", "data": {...}}}

        # CRITICAL FIX: Aggressively strip whitespace/hash and clean the ID
        if self.chat_id:
            # Step 1: Strip whitespace and the '#' anchor
            clean_id = str(self.chat_id).strip().lstrip('#')

            # Step 2: Set the chat_id to the clean, hyphenated numeric ID.
            # We explicitly REMOVE the erroneous 'chat_' logic.
            self.chat_id = clean_id

        if not self.token or not self.chat_id:
            self.enabled = False
            logger.warning("⚠️ Telegram credentials (TOKEN/CHAT_ID) are missing. Alerts are disabled.")
        else:
            self.enabled = True
            logger.info("Telegram credentials loaded successfully.")

    def _load_history(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_history(self):
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.signal_history, f)
        except:
            pass

    def check_connection(self):
        """Attempts to ping the Telegram API to verify connection."""
        if not self.enabled:
            return False, "Not enabled (missing credentials)."

        method = "getMe"
        url = f"{TELEGRAM_API_BASE_URL}{self.token}/{method}"

        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            if response.json().get('ok'):
                return True, "Connection successful."
            else:
                return False, f"API returned error: {response.json().get('description')}"
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Telegram Connection Test Failed: {e}")
            return False, f"Connection failed: {e}"

    def send_alert(self, message: str, parse_mode: str = 'Markdown', ticker=None, current_params=None):
        """
        Sends a text message. If ticker/params provided, performs 'Smart Alert' check.
        """
        if not self.enabled:
            return

        final_msg = message
        
        # --- SMART ALERT LOGIC ---
        if ticker and current_params:
            last_signal = self.signal_history.get(ticker)
            
            header = "🎯 SNIPER SIGNAL" # Default
            
            if last_signal:
                # Compare Diffs
                try:
                    new_stop = current_params.get('stop_loss', 0)
                    old_stop = last_signal.get('stop_loss', 0)
                    new_target = current_params.get('target', 0)
                    old_target = last_signal.get('target', 0)
                    
                    if new_stop > old_stop:
                        header = "🔒 SECURING PROFITS (Stop Raised)"
                    elif new_target > old_target:
                        header = "🚀 TARGET LIFTED"
                    elif new_stop < old_stop: # Bad sign usually
                         header = "⚠️ WARNING: RISK ADJUSTED"
                        
                    # Reconstruct message with new header if needed
                    if header != "🎯 SNIPER SIGNAL":
                         # Heuristic replace of first line
                         lines = message.split('\n')
                         if "SNIPER SIGNAL" in lines[0]:
                             lines[0] = f"{header}: {ticker}"
                             final_msg = "\n".join(lines)
                             
                except Exception as e:
                    logger.warning(f"Smart Alert Diff Failed: {e}")
            
            # Save new state
            self.signal_history[ticker] = current_params
            self._save_history()

        method = "sendMessage"
        url = f"{TELEGRAM_API_BASE_URL}{self.token}/{method}"

        payload = {
            'chat_id': str(self.chat_id),
            'text': final_msg,
            'parse_mode': parse_mode
        }

        try:
            # Use requests.post for thread safety and simplicity
            response = requests.post(url, data=payload, timeout=5)
            response.raise_for_status() # Raises HTTPError for bad responses
            # We explicitly log only errors, avoiding unnecessary INFO logs on success
        except requests.exceptions.HTTPError as e:
            logger.error(f"[ERROR] Telegram API Error ({e.response.status_code}): {e.response.text}")
        except requests.exceptions.RequestException as e:
            logger.error(f"[ERROR] Telegram Connection Error: {e}")

    # --- INPUT PROCESSING ---
    
    def check_for_updates(self, portfolio_manager):
        """
        Polls for new messages and processes them.
        Should be called in the main loop.
        """
        if not self.enabled: return

        method = "getUpdates"
        url = f"{TELEGRAM_API_BASE_URL}{self.token}/{method}"
        params = {"offset": self.last_update_id + 1, "timeout": 1}
        
        try:
            resp = requests.get(url, params=params, timeout=3)
            data = resp.json()
            
            if data.get("ok"):
                for update in data.get("result", []):
                    self.last_update_id = update["update_id"]
                    if "message" in update and "text" in update["message"]:
                        self.process_incoming_message(update["message"], portfolio_manager)
            else:
                logger.warning(f"Telegram Poll Error: {data.get('description')}")
                        
        except Exception as e:
            logger.error(f"Telegram Poll Exception: {e}")

    def process_incoming_message(self, message_obj, pm):
        """State Machine & Command Processor"""
        chat_id = message_obj["chat"]["id"]
        text = message_obj["text"].strip()
        user_id = message_obj["from"]["id"]
        
        # Log incoming for debug
        logger.info(f"Incoming Telegram: {text} from {user_id}")
        
        state = self.user_states.get(chat_id, {}).get("state", "IDLE")
        
        # --- SCENARIO 1: PRO MODE (One-Liner) ---
        if text.lower().startswith("/buy "):
            parts = text.split()
            if len(parts) == 4:
                try:
                    ticker = parts[1].upper()
                    price = float(parts[2])
                    qty = float(parts[3])
                    
                    id = pm.add_user_trade(ticker, price, qty)
                    self.send_reply(chat_id, f"✅ Fast execution recorded.\nTrade ID: {id}\n{ticker} @ {price} x {qty}")
                    return
                except:
                    self.send_reply(chat_id, "❌ Valid format: /buy TICKER PRICE QTY")
            # Fallthrough if args are wrong but starts with /buy space
            else:
                 self.send_reply(chat_id, "❌ Usage: /buy TICKER PRICE QTY\nOr type /buy TICKER to start wizard.")
            return

        # --- SCENARIO 2: CONVERSATIONAL FLOW ---
        
        # Start Wizard (e.g. "/buy NVDA" or just "/buy")
        if text.lower() == "/buy" or (text.lower().startswith("/buy") and len(text.split()) == 2):
            parts = text.split()
            if len(parts) == 2:
                ticker = parts[1].upper()
                self.user_states[chat_id] = {"state": "AWAITING_PRICE", "ticker": ticker}
                self.send_reply(chat_id, f"Buying {ticker}. What is the Entry Price?")
            else:
                self.send_reply(chat_id, "ℹ️ To buy, type: /buy TICKER\nExample: /buy NVDA")
            return
            
        # State: AWAITING_PRICE
        if state == "AWAITING_PRICE":
            try:
                price = float(text)
                self.user_states[chat_id]["price"] = price
                self.user_states[chat_id]["state"] = "AWAITING_QTY"
                self.send_reply(chat_id, f"Price ${price}. How many shares (Qty)?")
            except:
                self.send_reply(chat_id, "❌ Please enter a valid number for Price.")
            return

        # State: AWAITING_QTY
        if state == "AWAITING_QTY":
            try:
                qty = float(text)
                data = self.user_states[chat_id]
                
                id = pm.add_user_trade(data["ticker"], data["price"], qty)
                self.send_reply(chat_id, f"✅ Trade Recorded.\n{data['ticker']} @ {data['price']} x {qty}")
                
                # Reset
                self.user_states[chat_id] = {"state": "IDLE"}
            except:
                self.send_reply(chat_id, "❌ Please enter a valid number for Quantity.")
            return
            
        # Reset Command
        if text.lower() == "/cancel":
            self.user_states[chat_id] = {"state": "IDLE"}
            self.send_reply(chat_id, "🚫 Operation cancelled.")
            return

        # --- SCENARIO 3: SELL (Pro Mode + Wizard) ---
        if text.lower().startswith("/sell"):
            parts = text.split()
            
            # Format: /sell TICKER PRICE QTY
            if len(parts) == 4:
                try:
                    ticker = parts[1].upper()
                    price = float(parts[2])
                    qty = float(parts[3])
                    
                    closed_ids = pm.close_user_trade(ticker, price, qty)
                    if closed_ids:
                        self.send_reply(chat_id, f"✅ Trade Closed.\nIds: {closed_ids}\n{ticker} Sold @ {price} x {qty}")
                    else:
                        self.send_reply(chat_id, f"⚠️ No Open Position found for {ticker} to sell.")
                    return
                except:
                    self.send_reply(chat_id, "❌ Valid format: /sell TICKER PRICE QTY")
            
            # Wizard Start
            elif len(parts) == 2:
                ticker = parts[1].upper()
                # Check if position exists first?
                # For now just proceed
                self.user_states[chat_id] = {"state": "SELL_AWAITING_PRICE", "ticker": ticker}
                self.send_reply(chat_id, f"Selling {ticker}. What is the Exit Price?")
            else:
                 self.send_reply(chat_id, "ℹ️ To sell, type: /sell TICKER\nExample: /sell NVDA")
            return

        # State: SELL_AWAITING_PRICE
        if state == "SELL_AWAITING_PRICE":
            try:
                price = float(text)
                self.user_states[chat_id]["price"] = price
                self.user_states[chat_id]["state"] = "SELL_AWAITING_QTY"
                self.send_reply(chat_id, f"Exit Price ${price}. Quantity to sell?")
            except:
                self.send_reply(chat_id, "❌ Please enter a valid number for Price.")
            return

        # State: SELL_AWAITING_QTY
        if state == "SELL_AWAITING_QTY":
            try:
                qty = float(text)
                data = self.user_states[chat_id]
                
                closed_ids = pm.close_user_trade(data["ticker"], data["price"], qty)
                if closed_ids:
                    self.send_reply(chat_id, f"✅ Trade Closed.\n{data['ticker']} @ {data['price']} x {qty}")
                else:
                    self.send_reply(chat_id, f"⚠️ No Open Position found for {data['ticker']} to sell.")
                
                # Reset
                self.user_states[chat_id] = {"state": "IDLE"}
            except:
                self.send_reply(chat_id, "❌ Please enter a valid number for Quantity.")
            return

        # --- REPORTING COMMANDS ---
        
        # /list
        if text.lower() == "/list":
            summary = pm.get_user_position_summary()
            self.send_reply(chat_id, summary)
            return

        # /today_profit
        if text.lower() == "/today_profit":
            pnl, count = pm.calculate_user_pnl('TODAY')
            self.send_reply(chat_id, f"📅 **Today's Realized PnL**\n💰 Profit: ${pnl:.2f}\n🔢 Trades: {count}")
            return

        # /month_profit
        if text.lower() == "/month_profit":
            pnl, count = pm.calculate_user_pnl('MONTH')
            self.send_reply(chat_id, f"🗓️ **Monthly Realized PnL**\n💰 Profit: ${pnl:.2f}\n🔢 Trades: {count}")
            return

        # /profit (Total)
        if text.lower() == "/profit":
            pnl, count = pm.calculate_user_pnl('ALL')
            self.send_reply(chat_id, f"🏦 **Total Realized PnL**\n💰 Profit: ${pnl:.2f}\n🔢 Trades: {count}")
            return

        # Generic Help if confused
        if text.lower() == "/help":
             self.send_reply(chat_id, (
                 "🤖 **StockWise Bot Commands**\n"
                 "--------------------------\n"
                 "🛒 **Trading**\n"
                 "• `/buy TICKER` - Buy Wizard\n"
                 "• `/sell TICKER` - Sell Wizard\n"
                 "• `/buy TKR $ QTY` - Quick Buy\n"
                 "• `/sell TKR $ QTY` - Quick Sell\n\n"
                 "📊 **Reporting**\n"
                 "• `/list` - Active Positions\n"
                 "• `/today_profit` - Daily PnL\n"
                 "• `/month_profit` - Monthly PnL\n"
                 "• `/profit` - Total PnL\n"
                 "• `/cancel` - Cancel Operation"
             ))

    def send_reply(self, chat_id, text):
        """Helper to reply to specific user/chat"""
        if not self.enabled: return
        url = f"{TELEGRAM_API_BASE_URL}{self.token}/sendMessage"
        try:
            requests.post(url, data={'chat_id': chat_id, 'text': text}, timeout=5)
        except Exception as e:
            logger.error(f"Reply Failed: {e}")