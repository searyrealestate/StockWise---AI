import logging
import sys
import os
from datetime import datetime
from train_gen9_model import train_model
from verify_sniper_logic import run_verification

# Setup Logging
os.makedirs("logs", exist_ok=True)
timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"logs/strict_test_{timestamp_str}.log"

# Force reconfiguration
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_filename, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("StrictTester")
print(f"[INIT] Logging to: {log_filename}")

def run_strict_test():
    CUTOFF = "2025-01-01"
    
    logger.info("\n🧪 STARTING STRICT 'OUT-OF-SAMPLE' TEST")
    logger.info("=======================================")
    logger.info(f"1. Training Model ONLY on data BEFORE {CUTOFF}...")
    logger.info("   (The AI will be blind to everything that happened in 2025)")
    
    # 1. Train on "The Past"
    train_model(cutoff_date=CUTOFF)
    
    logger.info("\n✅ Training Complete. AI is now living in 2024.")
    logger.info(f"2. Running Verification on data AFTER {CUTOFF}...")
    
    # 2. Test on "The Future"
    # (verify_sniper_logic automatically loops through the last 252 days)
    # Since today is roughly 2026-01, the last 252 days ARE the 2025 data we excluded!
    run_verification()

if __name__ == "__main__":
    run_strict_test()