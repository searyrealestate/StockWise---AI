# system_action_logger.py
import logging
import os
from datetime import datetime

class SystemActionLogger_old:
    """
    Singleton logger for tracking high-level system actions.
    Saves to logs/system_actions.log
    """
    _logger = None

    @classmethod
    def get_logger(cls):
        if cls._logger is None:
            cls._setup()
        return cls._logger

    @classmethod
    def _setup(cls):
        if not os.path.exists("logs"):
            os.makedirs("logs")

        cls._logger = logging.getLogger("SystemActions")
        cls._logger.setLevel(logging.INFO)
        cls._logger.handlers.clear()

        # File Handler
        log_file = os.path.join("logs", "system_actions.log")
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        
        # Format: TIMESTAMP | COMPONENT | ACTION | DETAILS
        formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
        file_handler.setFormatter(formatter)
        
        cls._logger.addHandler(file_handler)

    @staticmethod
    def log_action(component, action, details=""):
        """
        Log a formatted system action.
        Usage: SystemActionLogger.log_action("LiveTrader", "EXECUTION", "Bought AAPL @ 150")
        """
        logger = SystemActionLogger.get_logger()
        msg = f"[{component}] {action}: {details}"
        logger.info(msg)
