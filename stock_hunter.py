# stock_hunter.py

"""
StockWise Gen-12 Stock Hunter (The Scout)
=========================================
The Stateful Discovery Engine.
Implements a Multi-Level Feedback Queue (MLFQ) to efficiently scan thousands of 
equities, prioritizing high Signal-to-Noise (DSP) waveforms while ensuring 
no stock is left behind.
"""

import random
import os
import json
import logging
import pandas as pd
from datetime import datetime, timedelta
import system_config as cfg
from feature_engine import FeatureEngine
from strategy_engine import StrategyEngine # <-- שינוי 1: ייבוא המוח האסטרטגי
import time
import numpy as np

# הגדרת שם לוג נפרד לסורק
cfg.LOG_PREFIX = "StockWise_Scanner" 

# Initialize Logger
logger = logging.getLogger("StockHunter")

class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON Encoder specifically designed to handle numpy data types.
    Prevents serialization crashes when dumping AI and Technical scores.
    """
    def default(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def setup_system_logger():
    """Initializes the dual-tier logging architecture based on system configuration."""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG) # Base level must be debug to catch everything
    
    # 1. Console Handler (Always ON for INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, cfg.LOG_LEVEL_CONSOLE, logging.INFO))
    console_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | [%(name)s] | %(message)s'))
    logger.addHandler(console_handler)
    
    # 2. File Handler (Controlled by Kill-Switch)
    if getattr(cfg, 'ENABLE_DEBUG_FILE_LOGGING', True):
        file_handler = logging.FileHandler(cfg.LOG_FILE_PATH, encoding='utf-8')
        file_handler.setLevel(getattr(logging, cfg.LOG_LEVEL_FILE, logging.DEBUG))
        file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | [%(name)s] | %(message)s'))
        logger.addHandler(file_handler)


class StockHunter:
    def __init__(self, data_manager):
        """
        Initializes the Scout, giving it access to Market Data and its Memory Ledgers.
        """
        self.dm = data_manager
        self.fe = FeatureEngine()
        self.orchestra = StrategyEngine() 

        # === FIXING PATHS: USING CENTRAL CONFIGURATION ===
        self.vip_list_file = cfg.VIP_LIST_PATH  # saves to data/daily_review_list.json
        self.ledger_file = cfg.LEDGER_PATH      # saves to data/scan_ledger.json
        self.watchlist_file = cfg.VIP_LIST_PATH  # Map the old variable to the new config path to fix legacy calls
        
        # # Stateful Ledgers (The Memory)
        # self.ledger_file = os.path.join(cfg.DB_DIR, "scan_ledger.json")
        # self.watchlist_file = os.path.join(cfg.DB_DIR, "daily_review_list.json")
        
        self.ledger = self._load_json(self.ledger_file, default_type={})
        self.watchlist = self._load_json(self.watchlist_file, default_type={"tickers": [], "last_updated": ""})

    def _load_json(self, filepath, default_type):
        if not os.path.exists(filepath):
            return default_type
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except:
            return default_type

    def _save_json(self, file_path, data):
        """
        Safely serializes dictionary data to a JSON file using NumpyEncoder.
        Separates INFO logging for operation success and DEBUG logging for forensic payload details.
        """
        try:
            # Ensure the target directory exists before saving
            directory = os.path.dirname(file_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
                
            # Dump the data with the custom numpy encoder
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, cls=NumpyEncoder, indent=4)
                
            # INFO: Basic operational confirmation
            logger.info(f"Successfully saved scan results to {file_path}")
            
            # DEBUG: Granular details of the saved payload for forensic analysis
            logger.debug(f"JSON Payload saved containing {len(data)} tickers. Target Path: {file_path}")
            
        except Exception as e:
            # INFO/ERROR: Critical failure notification
            logger.error(f"Failed to serialize JSON to {file_path}: {str(e)}")
            # DEBUG: Exact dataset state during crash
            logger.debug(f"JSON Serialization Crash Dump Data: {data}")

    def _classify_trend_direction(self, df):
        """
        Mandatory Template 1: TREND DIRECTION
        Checks Daily SMA alignment and slope to determine macro direction.
        Returns: 'BULLISH', 'BEARISH', or 'SIDEWAYS'
        """
        try:
            if len(df) < 200:
                return "SIDEWAYS"

            last = df.iloc[-1]
            sma_50 = last.get('sma_50', 0)
            sma_200 = last.get('sma_200', 0)
            close = last.get('close', 0)

            if sma_50 == 0 or sma_200 == 0 or close == 0:
                return "SIDEWAYS"

            # Check SMA_50 slope (rising or falling over last 10 days)
            sma_50_10_ago = df['sma_50'].iloc[-10] if 'sma_50' in df.columns and len(df) >= 10 else sma_50
            sma_50_slope = (sma_50 - sma_50_10_ago) / max(sma_50_10_ago, 1)

            # Daily alignment: close > SMA_50 > SMA_200 AND SMA_50 rising
            if close > sma_50 > sma_200 and sma_50_slope > 0:
                return "BULLISH"
            elif close < sma_50 < sma_200 and sma_50_slope < 0:
                return "BEARISH"
            else:
                return "SIDEWAYS"

        except Exception as e:
            logger.debug(f"Trend classification error: {e}")
            return "SIDEWAYS"

    def _classify_structure(self, df):
        """
        Mandatory Template 2: PRICE STRUCTURE
        Identifies if price is near support/resistance or in open field.
        Returns: 'NEAR_SUPPORT', 'NEAR_RESISTANCE', 'OPEN_FIELD'
        """
        try:
            scan_cfg = getattr(cfg, 'MANDATORY_SCAN_CONFIG', {})
            lookback = scan_cfg.get('support_resistance_lookback', 60)
            near_pct = scan_cfg.get('near_level_pct', 0.02)

            if len(df) < lookback:
                return "OPEN_FIELD"

            recent = df.tail(lookback)
            close = df.iloc[-1]['close']

            if close == 0:
                return "OPEN_FIELD"

            # Simple S/R: recent highs and lows
            recent_high = recent['high'].max()
            recent_low = recent['low'].min()

            dist_to_resistance = (recent_high - close) / close
            dist_to_support = (close - recent_low) / close

            if dist_to_resistance <= near_pct:
                return "NEAR_RESISTANCE"
            elif dist_to_support <= near_pct:
                return "NEAR_SUPPORT"
            else:
                return "OPEN_FIELD"

        except Exception as e:
            logger.debug(f"Structure classification error: {e}")
            return "OPEN_FIELD"

    def _classify_volume_health(self, df):
        """
        Mandatory Template 3: VOLUME HEALTH
        Checks if the stock has enough liquidity and whether volume is growing or drying up.
        Returns: 'HEALTHY', 'SURGING', 'DRYING_UP', 'ILLIQUID'
        """
        try:
            scan_cfg = getattr(cfg, 'MANDATORY_SCAN_CONFIG', {})
            min_volume = scan_cfg.get('min_avg_volume', 500000)
            vol_lookback = scan_cfg.get('volume_trend_lookback', 20)

            if len(df) < vol_lookback:
                return "ILLIQUID"

            avg_volume = df['volume'].tail(vol_lookback).mean()

            if avg_volume < min_volume:
                return "ILLIQUID"

            # Volume trend: compare last 5 days avg to 20-day avg
            recent_vol = df['volume'].tail(5).mean()
            vol_ratio = recent_vol / max(avg_volume, 1)

            if vol_ratio > 1.5:
                return "SURGING"
            elif vol_ratio < 0.6:
                return "DRYING_UP"
            else:
                return "HEALTHY"

        except Exception as e:
            logger.debug(f"Volume classification error: {e}")
            return "ILLIQUID"

    def _classify_volatility_state(self, df):
        """
        Mandatory Template 4: VOLATILITY STATE
        Uses Bollinger Band width to determine if the stock is compressed, normal, or volatile.
        Returns: 'COMPRESSED', 'NORMAL', 'VOLATILE'
        """
        try:
            scan_cfg = getattr(cfg, 'MANDATORY_SCAN_CONFIG', {})
            squeeze_threshold = scan_cfg.get('squeeze_bb_width_threshold', 0.10)
            volatile_threshold = scan_cfg.get('volatile_bb_width_threshold', 0.30)

            last = df.iloc[-1]
            bb_width = last.get('bb_width', 0.15)

            if bb_width < squeeze_threshold:
                return "COMPRESSED"
            elif bb_width > volatile_threshold:
                return "VOLATILE"
            else:
                return "NORMAL"

        except Exception as e:
            logger.debug(f"Volatility classification error: {e}")
            return "NORMAL"

    def classify_stock_state(self, df):
        """
        Master classifier: runs all 4 mandatory templates and returns a state dict.
        Called on every stock during morning/evening scan.
        """
        return {
            "trend": self._classify_trend_direction(df),
            "structure": self._classify_structure(df),
            "volume": self._classify_volume_health(df),
            "volatility": self._classify_volatility_state(df),
        }

    def assign_tier(self, master_score):
        """
        Assigns a scan priority tier based on the stock's master score.
        Tier 1 (VIP):   >= 85  -- scanned every 20 min
        Tier 2 (Watch): 75-84  -- scanned 3x/day
        Tier 3 (Pool):  < 75   -- nightly only
        """
        tier_cfg = getattr(cfg, 'SCAN_TIER_CONFIG', {})
        tier1_min = tier_cfg.get('tier1_min_score', 85.0)
        tier2_min = tier_cfg.get('tier2_min_score', 75.0)

        if master_score >= tier1_min:
            return 1
        elif master_score >= tier2_min:
            return 2
        else:
            return 3

    def _get_tonights_scan_queue(self, is_weekend=False):
        """
        Constructs the queue of symbols to scan.
        1. VIP Watchlist (Always scanned first).
        2. The Entire US Equity Market (via Alpaca) up to the daily limit.
        """
        priority_queue = cfg.WATCHLIST
        standard_queue = []
        
        try:
            logger.info("Requesting full US Equity market list from Alpaca...")
            if hasattr(self.dm, 'stock_client') and self.dm.stock_client:
                assets = self.dm.stock_client.list_assets(status='active', asset_class='us_equity')
                for asset in assets:
                    if asset.tradable and asset.marginable and asset.fractionable:
                        standard_queue.append(asset.symbol)
                logger.info(f"Successfully retrieved {len(standard_queue)} tradable assets from Alpaca.")
            else:
                logger.warning("Alpaca API not available for market scan. Scanning priority only.")
        except Exception as e:
            logger.error(f"Failed to fetch market universe: {e}")

        # הסרת כפילויות
        standard_queue = list(set(standard_queue) - set(priority_queue))
        random.shuffle(standard_queue)
        
        # חיתוך לפי המגבלה
        scan_limit = cfg.SCAN_ROUTING_CONFIG.get("daily_scan_limit", 4000)
        standard_queue = standard_queue[:scan_limit]
        
        logger.info(f"Nightly Queue Built: {len(priority_queue)} Priority + {len(standard_queue)} Standard.")
        return priority_queue + standard_queue

    def run_nightly_scan(self):
        """
        The Main Execution Loop. 
        Iterates through the queue, runs FULL STRATEGY ANALYSIS, and saves state.
        """
        logger.info("Initiating Nightly Deep-Scan (Tech + AI + DSP)...")
        
        scan_queue = self._get_tonights_scan_queue()
        
        for symbol in scan_queue:
            try:
                logger.info(f"🔍 Scanning [{symbol}]...")
                
                # Fetch Data
                df = self.dm.get_stock_data(symbol, days_back=730)
                
                # Architectural Fix: Row Gatekeeper to prevent Data Starvation crashes
                # 100 rows is the absolute minimum required to calculate MACD, Keltner, and base SMA safely.
                if df is None or df.empty or len(df) < 100:
                    logger.warning(f"[{symbol}] Row Gatekeeper Veto: Insufficient historical data. Skipping.")
                    continue
                
                # 1. Calculate Features (Math) - CPU SAVING OPMITIZATION
                # Instead of 'all', only request DSP & Volatility arrays to calculate the weight
                df_features = self.fe.calculate_features(df, strategy_config={"active_indicators": ["dsp", "volatility"]})
                
                if 'er_slow' not in df_features.columns:
                    continue
                
                # 2. Identify Regime and extract Fast Math metrics
                regime = self.orchestra.router.classify_regime(df_features)
                latest = df_features.iloc[-1]
                er_score = float(latest.get('er_slow', 0.0))
                atr = float(latest.get('atr', 0.0))
                price = float(latest.get('close', 1.0))
                
                # 3. Quick Heuristic Filter
                # If the DSP trend is completely dead (< 0.3), don't waste CPU running the 
                # TacticalSniper full 85-feature setup. Just rank it on raw DSP/Vol and move on.
                if er_score < 0.3:
                     # Calculate weight manually without full Sniper scan
                     volatility_pct = min((atr / price) * 100, 10.0) / 10.0
                     w_score = cfg.SCAN_ROUTING_CONFIG.get("weight_score_mult", 0.7)
                     w_vol = cfg.SCAN_ROUTING_CONFIG.get("weight_volatility_mult", 0.3)
                     master_score = (er_score * 100 * w_score) + (volatility_pct * 100 * w_vol)
                     
                     tech_score = 0
                     ai_score = 0
                     logger.debug(f"[{symbol}] Quick Reject (Dead Trend). ER: {er_score:.2f}, Weight: {master_score:.1f}")
                     
                else:
                    # Stock has a heartbeat! Now we calculate the heavy candlestick features 
                    # and ask the Sniper to grade it.
                    df_full = self.fe.calculate_features(df, strategy_config={"active_indicators": ["all"]})
                    
                    # RUN FULL STRATEGY (Agent 2 - The Sniper)
                    verdict = self.orchestra.sniper.analyze(symbol, df_full, regime)
                    
                    scores = verdict.get('scores', {})
                    tech_score = scores.get('tech', verdict.get('tech_score', 0))
                    ai_score = scores.get('ai', verdict.get('ai_score', 0))
                    master_score = scores.get('master', verdict.get('master_score', 0))
                
                # 4. Classify stock state using mandatory templates
                stock_state = self.classify_stock_state(df_features)
                tier = self.assign_tier(master_score)

                # 5. Update the Ledger with DETAILED SCORES + STATE + TIER
                self.ledger[symbol] = {
                    "weight": master_score,
                    "er_score": round(er_score, 2),
                    "tech_score": round(tech_score, 1),
                    "ai_score": round(ai_score, 1),
                    "master_score": round(master_score, 1),
                    "regime": regime,
                    "state": stock_state,
                    "tier": tier,
                    "last_scanned": datetime.now().isoformat()
                }

                logger.debug(f"[{symbol}] State: {stock_state} | Tier: {tier}")
                
            except Exception as e:
                logger.error(f"Scan failed for {symbol}. Moving to next. Error: {e}")
            finally:
                # Throttling
                time.sleep(12.5)
                
        # Persist & Update
        self._save_json(self.ledger_file, self.ledger)
        self._update_daily_review_list()
        logger.info("Nightly Scan Complete. Ledger updated.")

    def _update_daily_review_list(self):
        """
        Promotes the top-scoring stocks to the VIP list and prints the FULL LEADERBOARD.
        """
        if not self.ledger:
            logger.warning("Ledger is empty. No VIP list generated.")
            return

        # 1. NEW: Filter the ledger to only include stocks that meet the system's baseline threshold.
        # This prevents waking up the Live Trading engine on 10 bad stocks during a crash market.
        min_threshold = cfg.SCAN_ROUTING_CONFIG.get("min_vip_score_threshold", 50.0)
        
        qualified_items = [
            (sym, data) for sym, data in self.ledger.items() 
            if data.get('master_score', 0.0) >= min_threshold
        ]
        
        if not qualified_items:
            logger.warning(f"No stocks passed the baseline threshold of {min_threshold}. VIP list will be empty tomorrow.")

        # 2. Sort the qualified pool by Master Score (Highest to Lowest)
        sorted_items = sorted(qualified_items, key=lambda x: x[1].get('master_score', 0.0), reverse=True)
        
        limit = cfg.SCAN_ROUTING_CONFIG.get("max_daily_review_stocks", 10)
        top_candidates = sorted_items[:limit]
        
        vip_symbols = [item[0] for item in top_candidates]

        # Build tier-based lists for intraday scanner
        tier1_symbols = [sym for sym, data in self.ledger.items()
                         if data.get('tier') == 1]
        tier2_items = [(sym, data) for sym, data in self.ledger.items()
                       if data.get('tier') == 2]
        tier2_items.sort(key=lambda x: x[1].get('master_score', 0), reverse=True)
        tier2_limit = getattr(cfg, 'SCAN_TIER_CONFIG', {}).get('tier2_max_count', 10)
        tier2_symbols = [item[0] for item in tier2_items[:tier2_limit]]

        # Save tiered lists alongside VIP list
        tiered_data = {
            "tier1_vip": tier1_symbols,
            "tier2_watch": tier2_symbols,
            "total_scanned": len(self.ledger),
            "last_updated": datetime.now().isoformat()
        }
        tiered_path = os.path.join(cfg.DB_DIR, "tiered_watchlist.json")
        self._save_json(tiered_path, tiered_data)

        logger.info(f"Tiered Lists: Tier1(VIP)={len(tier1_symbols)}, Tier2(Watch)={len(tier2_symbols)}")

        # Save VIP list
        self.watchlist = {"tickers": vip_symbols, "last_updated": datetime.now().isoformat()}
        self._save_json(self.watchlist_file, self.watchlist)
        
        # --- 4. BUILD THE NEW DETAILED LEADERBOARD ---
        board = []
        board.append("\n" + "="*85)
        board.append("🏆 TOP VIP TARGETS - FULL ANALYSIS 🏆")
        board.append("="*85)
        # כותרות חדשות ומפורטות
        board.append(f"{'RANK':<5} | {'SYMBOL':<6} | {'REGIME':<6} | {'TREND':<8} | {'TECH':<6} | {'AI':<6} | {'MASTER':<7} | {'TIER':<4}")
        board.append("-" * 85)
        
        for i, (symbol, data) in enumerate(top_candidates, 1):
            regime = data.get('regime', 'N/A')
            tech = data.get('tech_score', 0)
            ai = data.get('ai_score', 0)
            master = data.get('master_score', 0)
            
            # הדגשה ויזואלית לציון מאסטר גבוה
            fire = "🔥" if master > 60 else "  "
            
            trend_dir = data.get('state', {}).get('trend', 'N/A')
            tier = data.get('tier', 3)
            tier_label = f"T{tier}"
            board.append(f"#{i:<4} | {symbol:<6} | {regime:<6} | {trend_dir:<8} | {tech:<6} | {ai:<6} | {master:<7} | {tier_label:<4} {fire}")
            
        board.append("="*85)
        
        leaderboard_str = "\n".join(board)
        logger.info(f"Daily Leaderboard Generated:{leaderboard_str}")
        print(leaderboard_str) # הדפסה למסך למקרה שהלוגר מושתק
        logger.info(f"VIP List Successfully Saved to Disk: {vip_symbols}")
    
    def get_active_vip_watchlist(self):
        """
        Public method for the Live Engine to fetch the active intraday targets.
        Updated to handle List structures and prevent crashes.
        """
        # שימוש במשתנה הנתיב המעודכן (תואם לתיקון שעשינו ב-__init__)
        # אם לא שינית ב-__init__, תשאיר self.watchlist_file
        target_path = getattr(self, 'vip_list_file', getattr(self, 'watchlist_file', None))

        if not target_path or not os.path.exists(target_path):
            logger.warning(f"VIP Watchlist not found at: {target_path}")
            return []

        try:
            # טעינה ישירה ובטוחה
            with open(target_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # === תמיכה בכל סוגי המבנים ===
            
            # אפשרות 1: הקובץ הוא רשימה פשוטה ["TSLA", "NVDA"] (מה שקורה אצלך כרגע)
            if isinstance(data, list):
                return data

            # אפשרות 2: הקובץ הוא מילון {"tickers": ["TSLA", ...]} (Best Practice)
            if isinstance(data, dict):
                if "tickers" in data:
                    return data["tickers"]
                else:
                    # תמיכה לאחור: פורמט ישן {"TSLA": {...}}
                    return list(data.keys())

            return []

        except Exception as e:
            logger.error(f"Critical error loading VIP list: {e}")
            return []

if __name__ == "__main__":
    import logging
    from data_source_manager import DataSourceManager
    
    # 1. Initialize the dual-tier logging architecture FIRST before any other action
    setup_system_logger()
    
    # Re-acquire the logger strictly after setup is complete
    logger = logging.getLogger("StockHunter")
    logger.info("=== MANUAL EXECUTION: NIGHTLY STOCK HUNTER ===")

    try:
        # 2. Initialize Core Infrastructure (Data Manager)
        dm_instance = DataSourceManager()
        
        # 3. Wake up the Scout (Agent 1) and inject data dependency
        hunter = StockHunter(dm_instance)
        
        # 4. Execute the MLFQ Scan
        hunter.run_nightly_scan()
        
        logger.info("=== NIGHTLY SCAN COMPLETE. VIP LIST GENERATED ===")
        
    except Exception as e:
        logger.error(f"Critical System Failure during Nightly Scan: {str(e)}")
        logger.debug("Exception Stack Trace:", exc_info=True)
