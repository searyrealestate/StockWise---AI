# stockwise_scanner.py

"""
StockWise Gen-12: Active Market Scanner & Talent Scout
======================================================
1. "Playbook Mode": Scans the Active Watchlist for BUY signals (Live Trading).
2. "Talent Scout Mode": Scans the broader market (NASDAQ 100) for new best-in-class stocks.
   - If a new stock scores > 75, it is recruited into the Dynamic Watchlist.
   - Triggers "Just-in-Time" training via TrainingManager.
"""

import pandas as pd
from datetime import date, timedelta, datetime
import logging
import os
import json
import itertools
from concurrent.futures import ThreadPoolExecutor
import sys

# --- Core Modules ---
from data_source_manager import DataSourceManager, SectorMapper, clean_raw_data
import system_config as cfg
from feature_engine import RobustFeatureCalculator
from strategy_engine import StrategyOrchestra
from notification_manager import NotificationManager
from watchlist_manager import WatchlistManager
from training_manager import TrainingManager

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", stream=sys.stdout)
logger = logging.getLogger("StockWise_Scanner")

# --- Globals ---
ANALYSIS_DATE = date.today()
dm = DataSourceManager(use_ibkr=cfg.EN_IBKR, allow_fallback=True, port=cfg.IBKR_PORT)
nm = NotificationManager()
wm = WatchlistManager()
tm = TrainingManager()
SECTOR_MAPPER = SectorMapper()
orchestra = StrategyOrchestra()

# --- HELPER: Load Model Params ---
def _load_best_params(ticker):
    path = os.path.join(cfg.MODELS_DIR, f"optimization_results_{ticker}.json")
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                best = data[0]
                params = cfg.STRATEGY_PARAMS.copy()
                for k in params.keys():
                    if k in best: params[k] = best[k]
                return params
        except:
            pass
    return cfg.STRATEGY_PARAMS

# --- SINGLE STOCK ANALYZER (Unified) ---
def run_single_stock_analysis(symbol: str, analysis_date: date, main_advisor_instance) -> dict:
    """
    Runs Strategy Orchestra. Returns dict with Signal, Score, and Metrics.
    """
    # 1. Fetch Data
    start_date = cfg.DATA_START_DATE
    end_date = cfg.DATA_END_DATE # datetime.now() usually

    # NOTE: For "Talent Scout" candidates (not in watchlist), we might rely on Fallback Data (YFinance)
    # first if IBKR is slow. But DM handles this.
    stock_df_raw = clean_raw_data(dm.get_stock_data(symbol, start_date=start_date, end_date=end_date))

    if stock_df_raw is None or stock_df_raw.empty:
        return {'symbol': symbol, 'signal': 'SKIP', 'reason': 'No Data'}

    # Context Data
    sector_symbol = SECTOR_MAPPER.get_benchmark_symbol(symbol)
    qqq_df = clean_raw_data(dm.get_stock_data("QQQ", start_date=start_date, end_date=end_date))
    sec_df = clean_raw_data(dm.get_stock_data(sector_symbol, start_date=start_date, end_date=end_date))
    context_data = {'qqq': qqq_df, 'sector': sec_df}

    df_slice = stock_df_raw[stock_df_raw.index <= pd.to_datetime(analysis_date)].copy()

    if len(df_slice) < cfg.STRATEGY_PARAMS.get('sma_long', 100):
        return {'symbol': symbol, 'signal': 'SKIP', 'reason': 'Insufficient History'}

    # 2. Feature Engineering
    params = _load_best_params(symbol)
    calculator = RobustFeatureCalculator(params=params)
    featured_data = calculator.calculate_features(df_slice, context_data)

    if featured_data.empty:
        return {'symbol': symbol, 'signal': 'SKIP', 'reason': 'Feature Calculation Failed'}

    # 3. Strategy Orchestration
    # We pass a neutral confidence (0.5) because the Scanner runs *before* the nightly AI prediction loop.
    # The AI model will be run inside the Live Engine. The Scanner is for filtering.
    decision = orchestra.decide_action(symbol, featured_data, {}, ai_confidence=0.5)

    if decision:
        # Unpack Tuple: (Action, Probability, Price, AgentName, StopLoss, TargetPrice, Score)
        action, prob, price, agent, sl, tp, score = decision
        
        # Calculate hypotheticals
        risk_per_share = price - sl
        if risk_per_share <= 0: risk_per_share = price * 0.01
        
        shares = 100 # Standard lot for estimation
        if main_advisor_instance:
             # Calculate fees using the passed advisor
             gross_profit = (tp - price) * shares
             net_profit, _ = main_advisor_instance.apply_israeli_fees_and_tax(gross_profit, shares)
        else:
             net_profit = 0

        return {
            'Symbol': symbol,
            'Source': 'Scanner',
            'Signal': 'BUY',
            'Agent': agent,
            'Score': score,
            'Entry Price': price,
            'Stop-Loss': sl,
            'Profit Target ($)': tp,
            'Est. Net Profit ($)': net_profit
        }
    
    # Return Score even if no signal (for Talent Scout filtering)
    # We need to manually calculate score if Orchestra returned None?
    # No, Orchestra returns None if Score < 75. 
    # But for Talent Scout, we want to know if it was close? 
    # Actually, we only Recruit if Score > 75 (which means decision is NOT None).
    # So returning None is fine.
    return None

def run_market_scan(tickers: list, description: str, main_advisor_instance):
    """
    Generic runner for a list of tickers.
    """
    logger.info(f"--- 🟢 Starting {description} Scan ({len(tickers)} symbols) ---")

    def wrapper(sym):
        return run_single_stock_analysis(sym, ANALYSIS_DATE, main_advisor_instance)

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(wrapper, tickers))

    valid_results = [r for r in results if r is not None and r.get('signal') != 'SKIP']
    return valid_results

def main_scan_operations(main_advisor_instance):
    # 1. Get Lists
    active_watchlist = wm.get_active_watchlist()
    
    # Broader Universe for Talent Scout (e.g., NASDAQ 100, but let's load a file or use a static list)
    # For now, let's assume we have a list file "nasdaq_top_100.txt" or similar.
    # Or just use the seed list + some extras if file missing.
    try:
        with open(os.path.join(cfg.PROJECT_ROOT, "nasdaq_top_1000.txt"), "r") as f:
            full_universe = [line.strip() for line in f if line.strip()]
    except:
        full_universe = active_watchlist # Fallback if no file

    # Remove duplicates
    talent_pool = list(set(full_universe) - set(active_watchlist))
    
    # 2. Run Playbook Scan (Current Watchlist)
    playbook_results = run_market_scan(active_watchlist, "PLAYBOOK (Active Watchlist)", main_advisor_instance)
    
    # 3. Run Talent Scout (Broader Market)
    # Limit to top 50 to avoid massive wait times if list is huge
    scout_results = run_market_scan(talent_pool[:50], "TALENT SCOUT (Discovery)", main_advisor_instance)
    
    # 4. Process Scouts (Recruitment)
    recruits = []
    for res in scout_results:
        if res['Score'] >= 75: # Strict quality control
            recruits.append(res['Symbol'])
            tm.recruit_new_stock(res['Symbol'])
            
    # 5. Reporting
    all_results = playbook_results + scout_results
    send_and_log_alerts(all_results, recruits)

def send_and_log_alerts(scan_results: list, recruits: list):
    # System Health
    dm_status = "✅ IBKR (Live)" if (dm.use_ibkr and dm.isConnected()) else "⚠️ Data Fallback Active"
    
    msg = f"泙 **StockWise REPORT ({ANALYSIS_DATE})**\nHealth: {dm_status}\n\n"
    
    # Recruitment Section
    if recruits:
        msg += f"🎓 **NEW RECRUITS ({len(recruits)})**\n"
        msg += f"Added to Watchlist & Retraining AI: {', '.join(recruits)}\n\n"
    
    # Signals Section
    buy_signals = [r for r in scan_results if r.get('Signal') == 'BUY']
    if buy_signals:
         msg += f"Found **{len(buy_signals)}** BUY signals:\n"
         for s in buy_signals:
             icon = "🆕" if s['Symbol'] in recruits else "✅"
             msg += f"{icon} **{s['Symbol']}** (Score: {s['Score']:.0f} | {s['Agent']})\n"
    else:
        msg += "No actionable signals found."
        
    nm.send_message(msg)
    logger.info("Scan & Recruitment Complete.")

if __name__ == "__main__":
    # Mock Advisor for standalone run
    class MockAdvisor:
        def apply_israeli_fees_and_tax(self, gp, shares): return gp * 0.75, 0
    
    try:
        if cfg.EN_IBKR: 
            try: dm.connect_to_ibkr()
            except: pass
            
        main_scan_operations(MockAdvisor())
        
    except Exception as e:
        logger.error(f"Scanner Failed: {e}", exc_info=True)
        nm.send_message(f"🚨 Scanner Failure: {e}")
    finally:
        if dm.isConnected(): dm.disconnect()
