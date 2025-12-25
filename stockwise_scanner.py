# stockwise_scanner.py

"""
StockWise Gen-6: Active Market Scanner (Nightly/Scheduled Job)
============================================================
Consolidated production script for parallel market scanning.
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
from strategy_engine import MarketRegimeDetector, StrategyOrchestra
from notification_manager import NotificationManager

# NOTE: MichaAdvisor, ProfessionalStockAdvisor and RiskManager are imported via globals when needed

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", stream=sys.stdout)
logger = logging.getLogger("StockWise_Scanner")

# --- Globals ---
ANALYSIS_DATE = date.today()
# NOTE: dm and nm are initialized in global scope below main scan functions
dm = DataSourceManager(use_ibkr=cfg.EN_IBKR, allow_fallback=True, port=cfg.IBKR_PORT)
nm = NotificationManager()
SECTOR_MAPPER = SectorMapper()


# --- Helper Functions (From previous turns) ---

def _load_best_params(ticker):
    # ... (function body is unchanged) ...
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


def _calculate_actionable_metrics(latest_features, entry_price, latest_atr):
    # ... (function body is unchanged) ...
    stop_loss_price = entry_price - (latest_atr * cfg.TIGHT_STOP_ATR)
    initial_risk_dollars = entry_price - stop_loss_price

    if initial_risk_dollars <= 0:
        return 0, 0, 0

    profit_target_price = entry_price + (initial_risk_dollars * cfg.RISK_REWARD_RATIO)

    return stop_loss_price, profit_target_price, initial_risk_dollars


# --- CONSOLIDATED SINGLE STOCK ANALYZER (Handles all models in the future) ---

def run_single_stock_analysis(symbol: str, analysis_date: date, main_advisor_instance) -> dict:
    """
    Runs the full Strategy Orchestra logic for one stock and returns an actionable signal.
    Note: This is the core logic from the old screener's helper, simplified for production.
    """

    # 1. Fetch Data
    start_date = cfg.DATA_START_DATE
    end_date = cfg.DATA_END_DATE

    stock_df_raw = clean_raw_data(dm.get_stock_data(symbol, start_date=start_date, end_date=end_date))

    if stock_df_raw.empty:
        return {'symbol': symbol, 'signal': 'SKIP', 'reason': 'No Data'}

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

    # DEBUG: check input columns to feature engine
    # print(f"[DEBUG] {symbol} columns before calculate_features:", df_slice.columns)

    featured_data = calculator.calculate_features(df_slice, context_data)

    # DEBUG: check that WT was added
    # if not featured_data.empty:
    #     print(f"[DEBUG] {symbol} features tail:",
    #           featured_data[['close', 'wt1', 'wt2']].tail())

    if featured_data.empty:
        return {'symbol': symbol, 'signal': 'SKIP', 'reason': 'Feature Calculation Failed'}

    latest_features = featured_data.iloc[-1].to_dict()

    # 3. Strategy and Scoring (Orchestra)
    regime = MarketRegimeDetector.detect_regime(latest_features)
    score = StrategyOrchestra.get_score(latest_features, regime, params)

    # 4. Final Decision
    if score >= 50:
        signal = "BUY"
    elif score <= 30:
        signal = "DOWN"
    else:
        signal = "WAIT"

    # 5. Calculate Metrics for Actionable Signal
    if signal == "BUY":
        next_day_data = stock_df_raw[stock_df_raw.index > pd.to_datetime(analysis_date)]
        entry_price = next_day_data.iloc[0]['open'] if not next_day_data.empty else latest_features['close']

        stop_loss, profit_target, risk = _calculate_actionable_metrics(
            latest_features, entry_price, latest_features.get('atr_14', 0)
        )

        # Calculate hypothetical shares and net profit (using the advisor's function)
        shares = cfg.RISK_REWARD_RATIO / entry_price  # Placeholder amount based on $2 initial risk
        gross_profit = (profit_target - entry_price) * shares
        net_profit_dollars, _ = main_advisor_instance.apply_israeli_fees_and_tax(gross_profit, shares)

        return {
            'Symbol': symbol,
            'Source': 'Orchestra',
            'Signal': 'BUY',
            'Score': score,
            'Regime': regime,
            'Analysis Date': analysis_date,
            'Entry Price': entry_price,
            'Profit Target ($)': profit_target,
            'Stop-Loss': stop_loss,
            'Est. Net Profit ($)': net_profit_dollars,
        }

    return {'symbol': symbol, 'signal': signal, 'score': score, 'regime': regime}


def run_full_market_scan(universe_tickers: list, analysis_date=ANALYSIS_DATE, main_advisor_instance=None):
    """
    Executes the analysis across the entire universe in parallel.
    Returns a list of result dictionaries.
    """
    if main_advisor_instance is None:
        raise ValueError("main_advisor_instance (ProfessionalStockAdvisor) is required for run_full_market_scan.")

    logger.info(f"--- 🟢 Starting Full Market Scan on {analysis_date} for {len(universe_tickers)} stocks ---")

    # Wrap the core function to pass the fixed arguments (dm, analysis_date, advisor)
    def wrapper_analyze(symbol):
        return run_single_stock_analysis(symbol, analysis_date, main_advisor_instance)

    # Use ThreadPoolExecutor for concurrent fetching and analysis
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_results = list(executor.map(wrapper_analyze, universe_tickers))

    scan_results = [r for r in future_results if r is not None and r.get('signal') != 'SKIP']
    # --- GEN-6 TELEGRAM INTEGRATION POINT ---
    # After the scan is complete, send the consolidated results via Telegram
    send_and_log_alerts(scan_results)

    return [r for r in future_results if r is not None and r.get('signal') != 'SKIP']


def send_and_log_alerts(scan_results: list):
    # --- ADD System Health Check to the end-of-day report ---

    # Calculate a simple health score based on the DataSourceManager's status
    dm_status = "✅ IBKR (Live) & Fallbacks Ready"
    if not dm.use_ibkr and dm.stock_client:
        dm_status = "⚠️ Alpaca Fallback Active"
    elif not dm.use_ibkr and not dm.stock_client:
        dm_status = "❌ YFinance ONLY (Alpaca/IBKR Disabled)"

    # --- Filter for BUY signals ---
    buy_signals = [r for r in scan_results if r.get('Signal') == 'BUY']

    # --- Start building the Alert Message ---
    alert_message = f"泙 **StockWise DAILY SCAN REPORT ({ANALYSIS_DATE})**\n"
    alert_message += f"**System Health:** {dm_status}\n\n"  # <-- NEW HEALTH STATUS

    if not buy_signals:
        alert_message += "No high-confidence BUY signals found today. Market conditions require patience."
        nm.send_alert(alert_message)
        logger.info("Daily Scan finished: No BUY signals found.")
        return

    # If signals are found
    alert_message += f"Found **{len(buy_signals)}** high-confidence BUY signals:\n\n"

    for signal in buy_signals:
        # Use Markdown formatting for clean alerts
        msg = (
            f"嶋 **{signal['Symbol']}** (Score: {signal['Score']:.0f} / {signal['Regime']})\n"
            f"   - **Entry:** ${signal['Entry Price']:.2f}\n"
            f"   - **SL/TP:** ${signal['Stop-Loss']:.2f} / ${signal['Profit Target ($)']:.2f}\n"
            f"   - *Est. Net Profit: ${signal['Est. Net Profit ($)']:.2f} (Hypothetical).*\n"
        )
        alert_message += msg

    nm.send_alert(alert_message)
    logger.info(f"Daily Scan finished. Sent {len(buy_signals)} BUY alerts.")


if __name__ == "__main__":
    # ... (execution logic remains for scheduled running) ...
    UNIVERSE = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "SMH", "TLT", "SPY"]

    # MOCK ADVISOR/RISK MANAGER FOR STANDALONE EXECUTION:
    # This requires defining a minimal mock object to access apply_israeli_fees_and_tax
    class MockAdvisor:
        def apply_israeli_fees_and_tax(self, gross_profit_dollars, num_shares):
            # This is hardcoded mock logic, should be replaced by real advisor instance in simulation
            FEE_PER_SHARE = 0.01
            MINIMUM_FEE = 2.50
            total_fees_dollars = max(FEE_PER_SHARE * num_shares, MINIMUM_FEE) * 2
            profit_after_fees_dollars = gross_profit_dollars - total_fees_dollars
            tax_dollars = (profit_after_fees_dollars * 0.25) if profit_after_fees_dollars > 0 else 0
            net_profit_dollars = profit_after_fees_dollars - tax_dollars
            return net_profit_dollars, total_fees_dollars + tax_dollars


    mock_advisor = MockAdvisor()

    if cfg.EN_IBKR:
        try:
            dm.connect_to_ibkr()
        except:
            logger.warning("IBKR connection failed. Proceeding with fallback data sources.")

    try:
        # Pass the mock advisor for fee calculation compatibility
        scan_results = run_full_market_scan(UNIVERSE, main_advisor_instance=mock_advisor)
    except Exception as e:
        logger.error(f"FATAL SCAN ERROR: {e}", exc_info=True)
        nm.send_alert(f"🚨 **StockWise ERROR**\n\nMarket scan failed: {e}. Check server logs.")
        scan_results = []

    if scan_results:
        send_and_log_alerts(scan_results)

    if dm.isConnected():
        dm.disconnect()