# stock_hunter.py

"""
StockWise Gen-12 Stock Hunter (The Scout)
=========================================
The Stateful Discovery Engine.
Implements a Multi-Level Feedback Queue (MLFQ) to efficiently scan thousands of
equities, prioritizing high Signal-to-Noise (DSP) waveforms while ensuring
no stock is left behind.

CHANGELOG:
-----------
[2026-03-14] Fix #1: Removed double throttle
  - Old: time.sleep(12.5) in finally block DOUBLED the per-provider delay already
    applied inside DataSourceManager.get_stock_data() via PROVIDER_DELAY config.
  - Result: 25s per stock × 4000 stocks = 27+ hours for a full scan.
  - New: time.sleep(0.5) — minimal inter-stock courtesy gap only.
    Provider-level throttling is handled entirely by PROVIDER_DELAY in DSM.
  - Impact: ~13 hours eliminated from a 4000-stock scan.

[2026-03-14] Fix #2: Added scan progress logging every 50 stocks
  - Logs: progress %, stocks scanned, elapsed time, rate (stocks/min), ETA.
  - Enables real-time monitoring of scan health during nightly runs.
  - A silent 12-hour scan is now impossible — health is visible every ~50 stocks.
"""

import random
import os
import sys
import json
import logging
import pandas as pd
from safe_json_io import safe_json_write, safe_json_read
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
        """Safely loads a JSON file with retry on parse failure."""
        return safe_json_read(filepath, default=default_type)

    def _save_json(self, file_path, data):
        """Safely serializes dictionary data to a JSON file using atomic write."""
        try:
            safe_json_write(file_path, data, cls=NumpyEncoder)
            logger.info(f"Successfully saved scan results to {file_path}")
            logger.debug(f"JSON Payload saved containing {len(data)} entries. Target Path: {file_path}")
        except Exception as e:
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

        # Clean stale entries before scanning (TTL enforcement)
        self._cleanup_stale_ledger()

        scan_queue = self._get_tonights_scan_queue()
        total = len(scan_queue)
        scan_start = time.time()

        # ═══ BENCHMARK DATA: Fetch SPY once for Relative Strength (2026-03-19) ═══
        # DO NOT DELETE: SPY data is fetched ONCE at scan start and used to
        # calculate Relative Strength for every symbol. This avoids fetching
        # SPY 4000 times (once per symbol).
        # ═══════════════════════════════════════════════════════════════════════
        benchmark_ticker = getattr(cfg, 'BENCHMARK_TICKER', 'SPY')
        benchmark_df = None
        try:
            benchmark_df = self.dm.get_stock_data(benchmark_ticker, days_back=730)
            if benchmark_df is not None and not benchmark_df.empty:
                logger.info(f"Benchmark {benchmark_ticker} loaded: {len(benchmark_df)} rows")
            else:
                logger.warning(f"Failed to load benchmark {benchmark_ticker} — RS will be skipped")
        except Exception as e:
            logger.warning(f"Benchmark fetch failed: {e} — RS will be skipped")

        for idx, symbol in enumerate(scan_queue, 1):
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

                # ═══ VETO GATE (SPEC v13.4 §3) ═══
                vetoed, veto_reason = self.fe.check_veto_gates(df_features, symbol)
                if vetoed:
                    logger.info(f"[{symbol}] VETO GATE: {veto_reason} — skipping")
                    continue
                # ═══════════════════════════════════

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

                    # ═══ VETO GATE (SPEC v13.4 §3) ═══
                    vetoed, veto_reason = self.fe.check_veto_gates(df_full, symbol)
                    if vetoed:
                        logger.info(f"[{symbol}] VETO GATE: {veto_reason} — skipping")
                        continue
                    # ═══════════════════════════════════

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

                # Calculate Relative Strength vs benchmark
                if benchmark_df is not None:
                    rs_data = self._calculate_relative_strength(df, benchmark_df)
                    if rs_data:
                        self.ledger[symbol].update(rs_data)

                logger.debug(f"[{symbol}] State: {stock_state} | Tier: {tier}")

            except Exception as e:
                logger.error(f"Scan failed for {symbol}. Moving to next. Error: {e}")
            finally:
                # Minimal inter-stock gap (courtesy only).
                # Provider-level throttling is handled by PROVIDER_DELAY inside get_stock_data().
                time.sleep(0.5)

            # --- SCAN PROGRESS: log every 50 stocks ---
            if idx % 50 == 0 or idx == total:
                elapsed = time.time() - scan_start
                rate = idx / (elapsed / 60) if elapsed > 0 else 0  # stocks per minute
                remaining = total - idx
                eta_min = remaining / rate if rate > 0 else 0
                pct = (idx / total) * 100
                logger.info(
                    f"[SCAN PROGRESS] {idx}/{total} ({pct:.0f}%) | "
                    f"Elapsed: {elapsed/60:.1f}m | "
                    f"Rate: {rate:.1f} stocks/min | "
                    f"ETA: {eta_min:.0f}m"
                )

        # Persist & Update
        self._save_json(self.ledger_file, self.ledger)
        self._update_daily_review_list()
        logger.info("Nightly Scan Complete. Ledger updated.")

    def _calculate_relative_strength(self, symbol_df, benchmark_df):
        """
        ═══ RELATIVE STRENGTH vs BENCHMARK (2026-03-19) ═══════════════════
        DO NOT DELETE: Calculates how a stock performs relative to SPY.

        RS = (stock_close / stock_close_N_days_ago) / (spy_close / spy_close_N_days_ago)
        RS > 1.0 = outperforming the market
        RS < 1.0 = underperforming the market

        Returns dict: {"rs_20": 1.05, "rs_60": 0.98, "rs_120": 1.12, "rs_label": "OUTPERFORM"}
        ═══════════════════════════════════════════════════════════════════════
        """
        if benchmark_df is None or benchmark_df.empty:
            return {}
        if symbol_df is None or symbol_df.empty:
            return {}

        rs_config = getattr(cfg, 'RELATIVE_STRENGTH_CONFIG', {})
        lookbacks = rs_config.get('lookback_days', [20, 60, 120])
        outperform_thr = rs_config.get('outperform_threshold', 1.05)
        underperform_thr = rs_config.get('underperform_threshold', 0.95)

        result = {}

        try:
            sym_close = symbol_df['close']
            bench_close = benchmark_df['close']

            current_sym = float(sym_close.iloc[-1])
            current_bench = float(bench_close.iloc[-1])

            if current_sym <= 0 or current_bench <= 0:
                return {}

            for lb in lookbacks:
                if len(sym_close) > lb and len(bench_close) > lb:
                    past_sym = float(sym_close.iloc[-lb - 1])
                    past_bench = float(bench_close.iloc[-lb - 1])

                    if past_sym > 0 and past_bench > 0:
                        sym_return = current_sym / past_sym
                        bench_return = current_bench / past_bench
                        rs = round(sym_return / bench_return, 3)
                        result[f"rs_{lb}"] = rs

            # Label based on 60-day RS (or shortest available)
            rs_60 = result.get('rs_60', result.get(f'rs_{lookbacks[0]}', 1.0))
            if rs_60 >= outperform_thr:
                result['rs_label'] = 'OUTPERFORM'
            elif rs_60 <= underperform_thr:
                result['rs_label'] = 'UNDERPERFORM'
            else:
                result['rs_label'] = 'INLINE'

        except Exception as e:
            logger.debug(f"RS calculation failed: {e}")

        return result

    def _cleanup_stale_ledger(self):
        """
        ═══ TTL ENFORCEMENT (2026-03-19) ═══════════════════════════════
        DO NOT DELETE: Removes symbols from the scan ledger that have not
        been scanned for more than max_days_untraded_on_watchlist days
        (default: 210 = 7 months). This prevents the ledger from growing
        infinitely and keeps the VIP list fresh.

        A symbol is considered "stale" if:
        1. Its last_scanned timestamp is older than TTL days
        2. It has no open positions in the trade journal

        Called at the start of run_nightly_scan() before new symbols are added.
        ═══════════════════════════════════════════════════════════════════
        """
        ttl_days = cfg.SCAN_ROUTING_CONFIG.get("max_days_untraded_on_watchlist", 210)
        cutoff = datetime.now() - timedelta(days=ttl_days)

        stale_symbols = []
        for sym, data in list(self.ledger.items()):
            last_scanned = data.get("last_scanned", "")
            if last_scanned:
                try:
                    scan_date = datetime.fromisoformat(last_scanned)
                    if scan_date < cutoff:
                        stale_symbols.append(sym)
                except (ValueError, TypeError):
                    pass  # Can't parse date — keep the symbol

        if stale_symbols:
            for sym in stale_symbols:
                del self.ledger[sym]
            logger.info(f"TTL Cleanup: Removed {len(stale_symbols)} stale symbols "
                       f"(older than {ttl_days} days): {stale_symbols[:10]}{'...' if len(stale_symbols) > 10 else ''}")
        else:
            logger.debug(f"TTL Cleanup: All {len(self.ledger)} ledger entries are fresh.")

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

        # ═══ CUMULATIVE VIP LIST (2026-03-19) ═══════════════════════════
        # DO NOT DELETE: VIP list now MERGES with existing list instead of
        # overwriting. New top-scoring symbols are added; existing symbols
        # stay unless they fail the TTL check (see _cleanup_stale_ledger).
        # This ensures good stocks from previous scans are not lost when
        # they temporarily drop in rank.
        # ═══════════════════════════════════════════════════════════════════

        # Load existing VIP list from disk
        existing_vip = []
        try:
            existing_data = self._load_json(self.watchlist_file, default_type={"tickers": []})
            if isinstance(existing_data, dict):
                existing_vip = existing_data.get("tickers", [])
            elif isinstance(existing_data, list):
                existing_vip = existing_data
        except:
            existing_vip = []

        # Merge: new VIP symbols first, then existing symbols not already in new list
        merged_vip = list(vip_symbols)  # Start with current top scorers
        for sym in existing_vip:
            if sym not in merged_vip:
                # Only keep if still in ledger and above minimum threshold
                if sym in self.ledger:
                    score = self.ledger[sym].get('master_score', 0)
                    if score >= min_threshold:
                        merged_vip.append(sym)

        # Apply max list size (prevent infinite growth)
        max_vip_size = cfg.SCAN_ROUTING_CONFIG.get("max_vip_list_size", 50)
        merged_vip = merged_vip[:max_vip_size]

        # ═══ BENCHMARK ALWAYS IN VIP (2026-03-20) ═══════════════════════
        # DO NOT DELETE: Only SPY is permanently pinned in VIP (benchmark
        # for Relative Strength). All other symbols, including DEFAULT_TRAINING_SYMBOLS,
        # follow normal VIP rules: they enter via scanner score and exit
        # after 210 days without recommendation (TTL).
        # DEFAULT_TRAINING_SYMBOLS is used ONLY as fallback when VIP is empty
        # (first run / fresh install). See system_config.py.
        # ═════════════════════════════════════════════════════════════════
        benchmark = getattr(cfg, 'BENCHMARK_TICKER', 'SPY')
        if benchmark in merged_vip:
            merged_vip.remove(benchmark)
        merged_vip.insert(0, benchmark)

        self.watchlist = {"tickers": merged_vip, "last_updated": datetime.now().isoformat()}
        self._save_json(self.watchlist_file, self.watchlist)

        logger.info(f"VIP List: {len(vip_symbols)} new + {len(existing_vip)} existing → {len(merged_vip)} merged (max {max_vip_size})")
        
        # --- 4. BUILD THE NEW DETAILED LEADERBOARD ---
        board = []
        board.append("\n" + "="*85)
        board.append("🏆 TOP VIP TARGETS - FULL ANALYSIS 🏆")
        board.append("="*85)
        # כותרות חדשות ומפורטות
        board.append(f"{'RANK':<5} | {'SYMBOL':<6} | {'REGIME':<6} | {'TREND':<8} | {'TECH':<6} | {'AI':<6} | {'MASTER':<7} | {'RS60':<6} | {'TIER':<4}")
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
            rs_60 = data.get('rs_60', 'N/A')
            rs_str = f"{rs_60:.2f}" if isinstance(rs_60, (int, float)) else 'N/A'
            board.append(f"#{i:<4} | {symbol:<6} | {regime:<6} | {trend_dir:<8} | {tech:<6.1f} | {ai:<6.1f} | {master:<7.1f} | {rs_str:<6} | {tier_label:<4} {fire}")
            
        board.append("="*85)
        
        leaderboard_str = "\n".join(board)
        logger.info(f"Daily Leaderboard Generated:{leaderboard_str}")
        try:
            print(leaderboard_str)
        except UnicodeEncodeError:
            print(leaderboard_str.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8'))
        logger.info(f"VIP List Successfully Saved to Disk: {merged_vip}")
    
    def get_active_vip_watchlist(self):
        """Returns the current VIP list, with DEFAULT_TRAINING_SYMBOLS fallback."""
        target_path = getattr(self, 'vip_list_file', getattr(self, 'watchlist_file', None))
        if target_path:
            data = self._load_json(target_path, default_type={"tickers": []})
            if isinstance(data, dict):
                tickers = data.get("tickers", [])
            elif isinstance(data, list):
                tickers = data
            else:
                tickers = []

            if tickers:
                return tickers

        # ═══ FALLBACK TO DEFAULT SYMBOLS (2026-03-19) ═══════════════════
        # DO NOT DELETE: If VIP list is empty or file missing, return
        # DEFAULT_TRAINING_SYMBOLS so the live engine has something to work with.
        # ═════════════════════════════════════════════════════════════════
        default = getattr(cfg, 'DEFAULT_TRAINING_SYMBOLS',
            ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AMD', 'NFLX', 'SPY'])
        logger.info(f"VIP list empty — falling back to DEFAULT_TRAINING_SYMBOLS ({len(default)} symbols)")
        return default

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
