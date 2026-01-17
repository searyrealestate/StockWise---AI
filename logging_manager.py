import logging
import os
import sys
import re

class EmojiFilter(logging.Filter):
    """Removes emojis and non-ascii characters for clean logs."""
    def filter(self, record):
        # Remove chars outside standard ASCII range (removes emojis)
        record.msg = re.sub(r'[^\x00-\x7F]+', '', str(record.msg)).strip()
        return True

class LoggerSetup:
    @staticmethod
    def setup_logger(name, log_file='system_thoughts.log', level=logging.INFO):
        if not os.path.exists("logs"):
            os.makedirs("logs")
            
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.handlers.clear()
        
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        
        # 1. File Handler (Strict: No Emojis)
        file_handler = logging.FileHandler(os.path.join("logs", log_file), mode='w', encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.addFilter(EmojiFilter()) # <--- Apply Filter
        logger.addHandler(file_handler)
        
        # 2. Console Handler (Allow Emojis for readability if desired, or filter too)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        # console_handler.addFilter(EmojiFilter()) # Uncomment to remove from console too
        logger.addHandler(console_handler)
        
        return logger

    @staticmethod
    def read_logs(log_file='system_thoughts.log'):
        try:
            with open(os.path.join("logs", log_file), 'r', encoding='utf-8') as f:
                return f.readlines()
        except:
            return []