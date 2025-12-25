# notification_manager.py

"""
StockWise Notification Manager (Telegram Bot)
==============================================
Handles sending automated trading alerts and system status messages via Telegram.
"""

import requests
import os
import logging
import system_config as cfg

logger = logging.getLogger("NotificationManager")

TELEGRAM_API_BASE_URL = "https://api.telegram.org/bot"


class NotificationManager:
    """
    Manages communication with the Telegram Bot API to send alerts.
    """

    def __init__(self):
        self.token = cfg.TELEGRAM_TOKEN
        self.chat_id = cfg.TELEGRAM_CHAT_ID

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

    def send_alert(self, message: str, parse_mode: str = 'Markdown'):
        """
        Sends a text message to the configured Telegram chat ID.
        """
        if not self.enabled:
            return

        method = "sendMessage"
        url = f"{TELEGRAM_API_BASE_URL}{self.token}/{method}"

        payload = {
            'chat_id': str(self.chat_id),
            'text': message,
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