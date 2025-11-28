# logging_setup.py
import logging
import sys
import os
from pythonjsonlogger import jsonlogger


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """
    A custom formatter to add and standardize fields in the JSON log.
    """

    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)

        # Standardize timestamp
        if not log_record.get('timestamp'):
            log_record['timestamp'] = record.created

        # Standardize level
        if log_record.get('level'):
            log_record['level'] = log_record['level'].upper()
        else:
            log_record['level'] = record.levelname

        # Add the logger's name
        log_record['logger_name'] = record.name


def setup_json_logging(log_level=logging.INFO):
    """
    Call this function ONCE at the very start of your application
    (e.g., in stockwise_simulation.py).
    """
    formatter = CustomJsonFormatter(
        '%(timestamp)s %(level)s %(logger_name)s [%(funcName)s] %(message)s'
    )

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "stockwise_gen4_log.jsonl")

    # --- 1. File Handler (writes JSON to a file) ---
    # We use ".jsonl" (JSON Lines) as each log is a separate JSON object
    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)

    # --- 2. Console Handler (writes readable text to your console) ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s [%(funcName)s] - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)

    # --- 3. Configure the Root Logger ---
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)  # Set the dynamic level

    # Clear any existing handlers to avoid duplicates
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # --- 4. Silence Noisy Third-Party Loggers ---
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("ibapi").setLevel(logging.WARNING)

    logging.info(f"JSON logging initialized. Writing logs to {log_file_path}")