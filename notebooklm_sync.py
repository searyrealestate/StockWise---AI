# notebooklm_sync.py
import sys
print("[NotebookLM Sync] DISABLED — Google Drive (G:) removed from system. Exiting.")
sys.exit(0)

"""
StockWise RAG Sync Pipeline
===========================
Automated Data Ingestion for NotebookLM.
Combines core files, state, and recent logs into structured TXT files.
"""

import os
import json
from safe_json_io import safe_json_read
from datetime import datetime
import time

# --- CONFIGURATION ---
# Define source and destination directories
SOURCE_DIR = os.getcwd()  # Assumes script runs from \StockWise - AI
DEST_DIR = r"G:\My Drive\StockWise_AI_Trading_System\Code"

# Define the exact 12 Core Engine files to inject
CORE_FILES = [
    "system_config.py",
    "live_trading_engine.py",
    "strategy_engine.py",
    "market_intelligence.py",
    "stock_hunter.py",
    "feature_engine.py",
    "notification_manager.py",
    "portfolio_manager.py",
    "master_validator.py",
    "train_model.py",
    "data_source_manager.py",
    "stockwise_simulation.py"
]

JSON_STATE_FILES = ["shadow_portfolio.json", "best_params.json"]

def ensure_dest_dir():
    """Ensure the Google Drive directory exists."""
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)

def sync_codebase():
    """Concatenates all Python files into a single structured prompt-friendly TXT."""
    dest_file = os.path.join(DEST_DIR, "StockWise_01_Codebase.txt")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(dest_file, "w", encoding="utf-8") as outfile:
        outfile.write(f"=== STOCKWISE AI CODEBASE SNAPSHOT ===\nGenerated at: {timestamp}\n\n")
        
        for filename in CORE_FILES:
            filepath = os.path.join(SOURCE_DIR, filename)
            if os.path.exists(filepath):
                outfile.write(f"\n{'='*60}\n")
                outfile.write(f"FILE: {filename}\n")
                outfile.write(f"{'='*60}\n\n")
                
                with open(filepath, "r", encoding="utf-8") as infile:
                    outfile.write(infile.read())
                outfile.write("\n\n")
            else:
                outfile.write(f"\n[WARNING] File not found: {filename}\n")
    return dest_file

def sync_live_state():
    """Converts JSON state files to readable text format."""
    dest_file = os.path.join(DEST_DIR, "StockWise_02_Live_State.txt")
    
    with open(dest_file, "w", encoding="utf-8") as outfile:
        outfile.write("=== STOCKWISE SYSTEM STATE (JSON) ===\n\n")
        
        for filename in JSON_STATE_FILES:
            filepath = os.path.join(SOURCE_DIR, filename)
            if os.path.exists(filepath):
                outfile.write(f"--- STATE FILE: {filename} ---\n")
                try:
                    data = safe_json_read(filepath, default={})
                    # Pretty print JSON to readable text
                    outfile.write(json.dumps(data, indent=4))
                except Exception as e:
                    outfile.write(f"Error parsing JSON: {e}")
                outfile.write("\n\n")
    return dest_file


def sync_recent_logs(lines_to_keep=500):
    """Tails the most recent log file to provide context without context-bloat."""
    logs_dir = os.path.join(SOURCE_DIR, "logs")
    dest_file = os.path.join(DEST_DIR, "StockWise_03_Recent_Logs.txt")
    
    if not os.path.exists(logs_dir):
        return

    # Find the most recently modified text/log file
    log_files = [os.path.join(logs_dir, f) for f in os.listdir(logs_dir) if f.endswith(('.log', '.txt'))]
    if not log_files:
        return
        
    latest_log = max(log_files, key=os.path.getctime)
    
    with open(dest_file, "w", encoding="utf-8") as outfile:
        outfile.write(f"=== STOCKWISE RECENT SYSTEM LOGS ===\n")
        outfile.write(f"Source: {os.path.basename(latest_log)}\n\n")
        
        try:
            with open(latest_log, "r", encoding="utf-8") as infile:
                lines = infile.readlines()
                # Take only the last N lines
                tail = lines[-lines_to_keep:] if len(lines) > lines_to_keep else lines
                outfile.writelines(tail)
        except Exception as e:
            outfile.write(f"Could not read log file: {e}")
            
    print(f"[Sync] Recent logs saved to Drive.")

if __name__ == "__main__":
    print(f"\n[Ingestion Pipeline] Initiating RAG Sync...")
    print(f"[Ingestion Pipeline] Target Directory: {DEST_DIR}\n")
    
    start_time = time.time()
    ensure_dest_dir()
    
    file1 = sync_codebase()
    print(f"  [+] Unified Codebase created: {os.path.basename(file1)}")
    
    file2 = sync_live_state()
    print(f"  [+] Live State JSON created: {os.path.basename(file2)}")
    
    duration = time.time() - start_time
			  
