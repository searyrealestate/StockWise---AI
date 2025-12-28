# continuous_learning_analyzer.py

"""
StockWise Gen-7: The Orchestra Architecture (With Binary Prediction Chart)
==========================================================================
1. Feature Calculator: Computes SMAs, ADX, RSI, and Donchian Channels.
2. MarketRegimeDetector: Determines if we are in a TREND, RANGE, or BEAR market.
3. StrategyOrchestra: Selects the correct sub-strategy.
4. Visualization: Adds a specific 'Prediction Signal' (0/1) panel synced to T+1.
5. Logic: Adds a 'WAIT' state.
6. Visualization: -1/0/1 scale for clear decision tracking.
7. Logic Update: Adds 'Early Breakout' override to catch fast rallies.
8. Visualization Upgrade: Replaced Line Chart with Japanese Candlesticks
9. Data Logic: Captures OHLC data for precise visualization.
10. Logic Update: Maintains 'Early Breakout' and Tri-State prediction.
11Logic: Sector-Aware Orchestra with Tri-State Decision.
12. Performance: Parallel Data Downloading (Thread Pool).
13. Architecture: Centralized Config, No Duplication, Robust Disconnect.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
import logging
import sys
import os
import json
from datetime import timedelta, datetime
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor

# --- Imports ---
from data_source_manager import DataSourceManager, SectorMapper
from data_source_manager import normalize_and_validate_data
from stockwise_simulation import ProfessionalStockAdvisor, clean_raw_data, calculate_net_pnl_raw
import system_config as cfg  # Centralized Config

from feature_engine import RobustFeatureCalculator
from strategy_engine import MarketRegimeDetector, StrategyOrchestra

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", stream=sys.stdout)
logger = logging.getLogger("Gen7_Complete")

# --- Silence pandas-ta verbose output ---
try:
    # This disables the repetitive print statements from technical analysis libraries.
    ta.core.settings["verbose"] = False
except AttributeError:
    # Handle older library versions if necessary
    pass

LOOK_AHEAD_DAYS = 5


# ==========================================
# 🧠 CORE SIMULATION LOGIC & PNL CALCULATION
# ==========================================


def _calculate_ground_truth_label(df: pd.DataFrame, current_index: int, forward_days: int = 15) -> tuple:
    """
    Calculates the Gen-7 'Ground Truth' label (Local Extrema) for a given bar.
    Labels: 0=STRONG_BUY, 1=STRONG_SELL, 2=NOISE/HOLD.
    The logic aligns with the 15-day forward window definition.
    """
    if current_index + 1 + forward_days >= len(df):
        return 2, 0.0, 0.0 # Not enough forward data for a label (NOISE/HOLD)

    future_data = df.iloc[current_index: current_index + forward_days]
    current_close = df.iloc[current_index]['close']

    # Gen-7 Spec: 15-Day Local Extrema Detection (Tolerance epsilon = 0.5%)
    # Window: Current day t vs [t+1 ... t+15]
    
    epsilon = 0.005 # 0.5% tolerance
    
    # We need to ensure we are strictly checking if P_t is a local extrema relative to the FUTURE window
    # NOTE: Strictly speaking, a local extrema is usually defined by looking at BOTH past and future.
    # However, for PREDICTION, we want to know if today is a good entry point relative to what comes next.
    # Definition: Buy if Price(t) <= Minimum(FutureWindow) * (1 + epsilon)
    
    future_min = future_data['low'].min()
    future_max = future_data['high'].max()
    
    label = 2 # NOISE/HOLD
    
    # Check for Local Minimum (Strong Buy)
    if current_close <= future_min * (1 + epsilon):
        label = 0 # STRONG_BUY
        
    # Check for Local Maximum (Strong Sell)
    elif current_close >= future_max * (1 - epsilon):
        label = 1 # STRONG_SELL
    
    # Calculate MPP and MPD for logging consistency (though not used for label anymore)
    max_potential_profit = (future_max - current_close) / current_close
    max_potential_drawdown = (future_min - current_close) / current_close
        
    return label, max_potential_profit, max_potential_drawdown


# def calculate_net_pnl_raw(gross_profit_dollars, num_shares, entry_price, exit_price):
#     """
#     Calculates Net PnL using IBKR Tiered Pricing Logic.
#     Logic: Fee = (Shares * Rate), subject to Min($0.35) and Max(1% of Trade Value).
#     """
#
#     # --- Helper to calculate fee for one leg (Buy or Sell) ---
#     def get_leg_fee(price, shares):
#         trade_value = price * shares
#         # 1. Base calculation: Max of (Min Fee) or (Per Share Rate)
#         base_fee = max(cfg.MINIMUM_FEE, shares * cfg.FEE_PER_SHARE)
#         # 2. Safety Cap: Fee cannot exceed 1% of trade value
#         final_fee = min(base_fee, trade_value * cfg.MAX_FEE_PCT)
#         return final_fee
#
#     # 1. Calculate Commissions
#     entry_fee = get_leg_fee(entry_price, num_shares)
#     exit_fee = get_leg_fee(exit_price, num_shares)
#     total_fees = entry_fee + exit_fee
#
#     # 2. Net PnL Calculation
#     profit_after_fees = gross_profit_dollars - total_fees
#
#     # 3. Tax (Only on positive net profit)
#     tax = (profit_after_fees * cfg.TAX_RATE) if profit_after_fees > 0 else 0
#
#     net_profit = profit_after_fees - tax
#
#     return net_profit, total_fees + tax

# def calculate_net_pnl(entry_price, exit_price, num_shares):
#     """Calculates Net PnL considering fees and taxes (using simplified hardcoded logic)."""
#     # Use constants from system_config.py
#     FEE_PER_SHARE = cfg.FEE_PER_SHARE
#     MINIMUM_FEE = cfg.MINIMUM_FEE
#     TAX_RATE = cfg.TAX_RATE
#     RISK_REWARD_RATIO = cfg.RISK_REWARD_RATIO
#
#     # 1. Gross Profit
#     gross_profit_dollars = (exit_price - entry_price) * num_shares
#
#     # 2. Calculate net profit using the raw helper
#     net_profit_dollars, _ = calculate_net_pnl_raw(gross_profit_dollars, num_shares)
#     return net_profit_dollars


# def calculate_net_pnl(entry_price, exit_price, num_shares):
#     """Calculates Net PnL considering fees and taxes (using simplified hardcoded logic)."""
#     # Use constants from system_config.py
#     FEE_PER_SHARE = cfg.FEE_PER_SHARE
#     MINIMUM_FEE = cfg.MINIMUM_FEE
#     TAX_RATE = cfg.TAX_RATE
#     RISK_REWARD_RATIO = cfg.RISK_REWARD_RATIO
#
#     # 1. Gross Profit
#     gross_profit_dollars = (exit_price - entry_price) * num_shares
#
#     # 2. Commission Calculation (Two transactions: Buy + Sell)
#     single_transaction_fee = max(FEE_PER_SHARE * num_shares, MINIMUM_FEE)
#     total_fees_dollars = single_transaction_fee * 2
#
#     profit_after_fees_dollars = gross_profit_dollars - total_fees_dollars
#
#     # 3. Tax on positive profits
#     tax_dollars = (profit_after_fees_dollars * TAX_RATE) if profit_after_fees_dollars > 0 else 0
#
#     net_profit_dollars = profit_after_fees_dollars - tax_dollars
#     return net_profit_dollars

def calculate_future_outcomes(df, lookahead=15):
    """
    Ground Truth Labeling for Gen-9 Sniper.
    Determines if a Trade would have been a WIN (Target Hit) or LOSS (Stop Hit).
    Uses System Config Thresholds.
    
    Returns:
    - labels: List of [0=SELL/AVOID, 1=WAIT, 2=BUY]
    """
    labels = []
    
    target_pct = cfg.SniperConfig.TARGET_PROFIT
    stop_pct = cfg.SniperConfig.MAX_DRAWDOWN # Negative, e.g. -0.02
    
    # Pre-calculate arrays for speed
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    
    n = len(df)
    
    for i in range(n):
        if i + lookahead >= n:
            labels.append(1) # Unknown/Wait at end
            continue
            
        entry_price = closes[i]
        
        # Win/Loss State
        result = 1 # Default WAIT
        
        # Scan forward 'lookahead' days
        for j in range(1, lookahead + 1):
            curr_high = highs[i+j]
            curr_low = lows[i+j]
            
            # Check Stop First (Conservative)
            drawdown = (curr_low - entry_price) / entry_price
            if drawdown <= stop_pct:
                result = 0 # LOSS / SELL
                break
                
            # Check Target
            profit = (curr_high - entry_price) / entry_price
            if profit >= target_pct:
                result = 2 # WIN / BUY
                break
        
        labels.append(result)
        
    return labels



# def check_and_calculate_exit(df, current_index, entry_price, entry_date, atr_value, look_ahead_days,
#                              risk_reward_ratio=cfg.RISK_REWARD_RATIO):
def check_and_calculate_exit(df, entry_index, entry_price, entry_date, atr_value, look_ahead_days,
                             risk_reward_ratio=cfg.RISK_REWARD_RATIO):

    """
    Simulates the trade using dynamic stops and targets.
    Returns: (exit_price, exit_date, result_status)
    """

    # 1. Define Stops and Targets
    # 1. Define Stops and Targets
    # STRATEGY: DYNAMIC ATR EXIT (Risk/Reward 1:2)
    # Stop: 1.5 ATR (Tight but fair)
    # Target: 3.0 ATR (Let winners run)
    stop_atr_mult = 1.5
    target_atr_mult = 3.0
    
    risk_dollars_per_share = atr_value * stop_atr_mult
    stop_loss = entry_price - risk_dollars_per_share
    target_price = entry_price + (atr_value * target_atr_mult)
    
    # Dynamic Exit State
    current_stop = stop_loss
    highest_high = entry_price
    
    # 2. Slice future data
    future_data = df.iloc[entry_index : entry_index + look_ahead_days]

    if future_data.empty:
        return None, None, "PENDING"

    # 3. Simulation Day-by-Day
    exit_price = None
    exit_date = None
    status = "PENDING"

    for idx, row in future_data.iterrows():
        current_high = row['high']
        current_low = row['low']
        current_open = row['open']
        
        # 0. Check GAP OPEN
        if current_open < current_stop:
             exit_price = current_open
             exit_date = idx
             status = "STOP_LOSS_GAP"
             break
        if current_open >= target_price:
             exit_price = current_open
             exit_date = idx
             status = "TARGET_HIT_GAP"
             break

        # 1. Check Profit Target (Bank it)
        if current_high >= target_price:
            exit_price = target_price
            exit_date = idx
            status = "TARGET_HIT"
            break

        # Check for Stop-Loss
        if current_low <= current_stop:
             exit_price = current_stop
             exit_date = idx
             status = "STOP_LOSS"
             break
    if status == "PENDING":
        final_day = future_data.iloc[-1]
        exit_price = final_day['close']
        exit_date = final_day.name
        status = "TIME_EXIT"

        #         exit_date = time_exit_date
        #         status = "TIME_EXIT_PROFIT"
        #     else:
        #         exit_price = time_exit_price
        #         exit_date = time_exit_date
        #         status = "TIME_EXIT_LOSS"
        #
        #     # NOTE: The current implementation of the simulation *must* select one definitive exit price
        #     # for PnL calculation. The suggestion for *multiple* exits should be handled in the log.
        #     # We prioritize the Time Exit if no hard stop/target was hit.

    return exit_price, exit_date, status

# ==========================================
# 🚀 EXECUTION LOGIC
# ==========================================


# --- 1. PARAMETER LOADER ---
def load_optimized_params(ticker):
    """Loads best params from JSON or defaults."""

    path = os.path.join(cfg.MODELS_DIR, f"optimization_results_{ticker}.json")
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                best = data[0]  # Take the best one (Micha or Breakout)
                # Map generic keys if needed, or just use what matches
                params = cfg.STRATEGY_PARAMS.copy()
                # Update only existing keys to avoid garbage injection
                for k in params.keys():
                    if k in best: params[k] = best[k]
                logger.info(f"✅ Loaded Optimized Params from JSON")
                return params
        except:
            logger.error("❌ Failed to load or parse optimization JSON. Using defaults.")
            pass

    logger.warning("[!] Using Default Params.")
    return cfg.STRATEGY_PARAMS


def load_data_sequential(dm, ticker):
    """
    Fetches Target, QQQ, and Sector sequentially to avoid ThreadPool errors.
    """
    logger.info("[>] Starting Sequential Data Download...")

    start_date = cfg.DATA_START_DATE
    end_date = cfg.DATA_END_DATE

    mapper = SectorMapper()
    sector_symbol = mapper.get_benchmark_symbol(ticker)
    logger.info(f"Identified Sector Benchmark: {sector_symbol}")

    # 1. Main Stock
    logger.info(f"[>] Fetching {ticker}...")
    stock_df = clean_raw_data(dm.get_stock_data(ticker, start_date=start_date, end_date=end_date, interval=cfg.TIMEFRAME))
    # logger.info(f"DEBUG_CLEAN[{ticker}] range: {stock_df.index.min()} -> {stock_df.index.max()}")
    # logger.info(f"DEBUG_CLEAN[{ticker}] head:\n{stock_df.head(3)}")
    # logger.info(f"DEBUG_CLEAN[{ticker}] tail:\n{stock_df.tail(3)}")

    # 2. Context (QQQ)
    logger.info(f"[>] Fetching QQQ...")
    qqq_df = clean_raw_data(dm.get_stock_data("QQQ", start_date=start_date, end_date=end_date))
    # logger.info(f"DEBUG#10 qqq_df Data:\n{qqq_df.head(3)} ...\n{qqq_df.tail(3)}")

    # 3. Context (Sector)
    logger.info(f"[>] Fetching {sector_symbol}...")
    sec_df = clean_raw_data(dm.get_stock_data(sector_symbol, start_date=start_date, end_date=end_date))
    # logger.info(f"DEBUG#11 sec_df Data:\n{sec_df.head(3)} ...\n{sec_df.tail(3)}")

    context = {'qqq': qqq_df, 'sector': sec_df}
    logger.info(f"[OK] Data Loaded. {cfg.TARGET_TICKER}: {len(stock_df)} | QQQ: {len(qqq_df)} | {sector_symbol}: {len(sec_df)}")
    return stock_df, context


def generate_chart(df, ticker, global_acc, win_rate):
    df = df.sort_values('Date').reset_index(drop=True)
    filename = os.path.join(cfg.CHARTS_DIR, f"{ticker}_Orchestra_Report.html")

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=[0.4, 0.15, 0.15, 0.3],
                        subplot_titles=(f'{ticker} Price & Trades', 'Regime', 'Decision', 'Score'))

    # 1. Price Candlesticks
    fig.add_trace(
        go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'),
        row=1, col=1)

    # --- SPLIT & MERGE DETECTION ---
    df['prev_close'] = df['Close'].shift(1)
    df['pct_change'] = (df['Close'] - df['prev_close']) / df['prev_close']

    # A. Splits (Drop > 50%)
    potential_splits = df[df['pct_change'] < -0.5]
    if not potential_splits.empty:
        logger.info(f"⚠️ Found {len(potential_splits)} potential splits. Marking on chart.")
        fig.add_trace(go.Scatter(x=potential_splits['Date'], y=potential_splits['Close'],
                                 mode='markers', marker=dict(symbol='x', color='white', size=14, line=dict(width=2)),
                                 name='Split (Post)'), row=1, col=1)

        split_indices = potential_splits.index
        prev_indices = [i - 1 for i in split_indices if i > 0]
        if prev_indices:
            split_highs = df.iloc[prev_indices]
            fig.add_trace(go.Scatter(x=split_highs['Date'], y=split_highs['Close'],
                                     mode='markers',
                                     marker=dict(symbol='x', color='white', size=14, line=dict(width=2)),
                                     name='Split (Pre)'), row=1, col=1)

    # B. Merges / Reverse Splits (Jump > 50%) <--- NEW
    potential_merges = df[df['pct_change'] > 0.5]
    if not potential_merges.empty:
        logger.info(f"⚠️ Found {len(potential_merges)} potential merges. Marking on chart.")
        fig.add_trace(go.Scatter(x=potential_merges['Date'], y=potential_merges['Close'],
                                 mode='markers',
                                 marker=dict(symbol='diamond', color='orange', size=14, line=dict(width=2)),
                                 name='Merge (Post)'), row=1, col=1)

        merge_indices = potential_merges.index
        prev_m_indices = [i - 1 for i in merge_indices if i > 0]
        if prev_m_indices:
            merge_lows = df.iloc[prev_m_indices]
            fig.add_trace(go.Scatter(x=merge_lows['Date'], y=merge_lows['Close'],
                                     mode='markers',
                                     marker=dict(symbol='diamond', color='orange', size=14, line=dict(width=2)),
                                     name='Merge (Pre)'), row=1, col=1)
    # -------------------------------

    # wins = df[(df['Prediction'] == 'UP') & (df['Is_Correct'] == True)]
    # losses = df[(df['Prediction'] == 'UP') & (df['Is_Correct'] == False)]
    #
    # fig.add_trace(go.Scatter(x=wins['Date'], y=wins['High'] * 1.02, mode='markers',
    #                          marker=dict(color='green', size=12, symbol='triangle-down'), name='Win'), row=1, col=1)
    # fig.add_trace(go.Scatter(x=losses['Date'], y=losses['High'] * 1.02, mode='markers',
    #                          marker=dict(color='red', size=12, symbol='x'), name='Loss'), row=1, col=1)

    # --- TRADE MARKERS (B/S + EXIT X) ---
    # Long buys
    long_entries = df[(df['Prediction'] == 'UP') & (df['Num_Shares'] > 0)]

    fig.add_trace(
        go.Scatter(
            x=long_entries['Entry_Date'],
            y=long_entries['Entry_Price'],
            mode='markers+text',
            marker=dict(
                symbol='triangle-up',
                color='blue',
                size=14,
                line=dict(width=1, color='black')
            ),
            text=['B'] * len(long_entries),
            textposition='middle center',
            textfont=dict(color='black', size=10),
            name='Buy UP'
        ),
        row=1, col=1
    )

    # Long exits – only rows where Exit_Date is not null
    exits = df[
        (df['Prediction'] == 'UP') &
        (df['Num_Shares'] > 0) &
        (df['Exit_Date'].notna())
        ]

    fig.add_trace(
        go.Scatter(
            x=exits['Exit_Date'],
            y=exits['Exit_Price'],
            mode='markers+text',
            marker=dict(
                symbol='triangle-down',
                color='red',
                size=14,
                line=dict(width=1, color='black')
            ),
            text=['S'] * len(exits),
            textposition='middle center',
            textfont=dict(color='black', size=10),
            name='Sell Exit'
        ),
        row=1, col=1
    )

    # Exit price X – same day & price as sell
    fig.add_trace(
        go.Scatter(
            x=exits['Exit_Date'],
            y=exits['Exit_Price'],
            mode='markers',
            marker=dict(
                symbol='x',
                color='white',
                size=10,
                line=dict(width=1)
            ),
            name='Exit Price'
        ),
        row=1, col=1
    )

    # Include 'RANGING_SHUFFLE' in the regime map for visualization
    regime_map = {'BEARISH': 0, 'RANGING_SHUFFLE': 1, 'RANGING_BULL': 1, 'TRENDING_UP': 2}
    df['Regime_Val'] = df['Regime'].map(regime_map)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Regime_Val'], mode='lines', name='Regime',
                             line=dict(shape='hv', color='purple')), row=2, col=1)
    fig.update_yaxes(tickvals=[0, 1, 2], ticktext=["BEAR", "RANGE", "TREND"], row=2, col=1)

    def get_signal_val(score):
        if score >= 50: return 1
        if score <= 30: return -1
        return 0

    df['Signal_Val'] = df['System_Score'].apply(get_signal_val)

    fig.add_trace(go.Scatter(x=df['Date'], y=df['Signal_Val'], mode='lines', name='Decision',
                             line=dict(shape='hv', color='#00CC96'), fill='tozeroy'), row=3, col=1)
    fig.update_yaxes(tickvals=[-1, 0, 1], ticktext=["DOWN", "WAIT", "UP"], row=3, col=1)

    # df['Rolling_Win'] = df['Is_Correct'].rolling(20).mean() * 100
    fig.add_trace(go.Scatter(x=df['Date'], y=df['System_Score'], name='Score', line=dict(color='cyan')), row=4, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Rolling_Win_20'], name='Win Rate 20',
                             line=dict(color='white', dash='dot')), row=4, col=1)

    fig.add_hline(y=50, line_dash="solid", line_color="red", row=4, col=1)

    start_d = df['Date'].iloc[0].strftime('%Y-%m-%d')
    end_d = df['Date'].iloc[-1].strftime('%Y-%m-%d')
    title_text = f"Gen-5 Orchestra: {ticker} ({start_d} to {end_d}) | Acc: {global_acc:.1f}% | Win: {win_rate:.1f}%"

    fig.update_layout(title=title_text, template="plotly_dark", height=1200, xaxis_rangeslider_visible=False)
    fig.write_html(filename)
    logger.info(f"Chart Saved: {filename}")


def run_simulation(start_date=None, end_date=None, return_stats=False, preloaded_data=None):
    logger.info(f"STARTING GEN-7 SIMULATION FOR {cfg.TARGET_TICKER}")

    # 1. Connect
    dm = DataSourceManager(use_ibkr=cfg.EN_IBKR, allow_fallback=True, port=cfg.IBKR_PORT)
    if cfg.EN_IBKR:
        try:
            dm.connect_to_ibkr()
        except:
            logger.info("⚠️ IBKR Connection Failed. Using Fallback Data Source.")
            pass

    try:
        # 2. Parallel Download (Single Source of Truth)
        if preloaded_data:
            df, context_data = preloaded_data
            # Ensure we are working with a copy to avoid side effects if modifying
            df = df.copy() 
            # Context data might need copying or is read-only sufficient
        else:
            df, context_data = load_data_sequential(dm, cfg.TARGET_TICKER)

        if df.empty:
            logger.error("[X] Main Stock Data is Empty.")
            return {} if return_stats else None

        # 3. Load Optimized Params
        params = load_optimized_params(cfg.TARGET_TICKER)

        # --- GEN-8: Fetch Fundamentals ---
        logger.info(f"Fetching Fundamentals for {cfg.TARGET_TICKER}...")
        fundamentals = dm.get_fundamentals(cfg.TARGET_TICKER) # Use the existing dm for fundamentals

        # Initialize Strategy Orchestra
        orchestra = StrategyOrchestra()
        
        # --- GEN-8: Inject Fundamentals ---
        if fundamentals:
             orchestra.set_fundamentals(fundamentals)
             logger.info(f"Fundamentals injected: RevGrowth={fundamentals.get('revenueGrowth')}, Margins={fundamentals.get('profitMargins')}")

        # # 2. Load Smart Context
        # context_data = load_smart_context(TICKER)

        advisor = ProfessionalStockAdvisor(model_dir=cfg.MODELS_DIR, data_source_manager=dm)

        # --- Initialize the calculator with params, dropping the context_data dependency in __init__ ---
        advisor.calculator = RobustFeatureCalculator(params=params)

        advisor.log.info(f"🛠️ Using Robust Feature Calculator with Sector Data")
        # --- IMPORTANT: Ensure index is clean and data is chronologically sorted ---
        df = df.sort_index().ffill().bfill()

        # Log the full range of data received from the API for debugging the end date
        logger.info(f"💾 Raw Data Range (from API): {df.index.min().date()} -> {df.index.max().date()}")

        # end_date = datetime.now().date()
        # start_date = end_date - timedelta(days=1200)
        # logger.info("📥 Downloading NVDA Data...")
        # df = clean_raw_data(dm.get_stock_data(TICKER, start_date=start_date, end_date=end_date))
        #
        # if df.empty: return

        # 5. Run Loop
        simulation_log = []
        trades_list = [] # SERIAL ENTRY EXPORT
        log_file_path = os.path.join(cfg.LOGS_DIR, 'orchestra_decision_breakdown.jsonl')
        journal_path = os.path.join(cfg.LOGS_DIR, 'trade_journal.log')
        
        # Clear journal
        with open(journal_path, 'w') as f:
            f.write("--- GEN-7 SYSTEM THINKING JOURNAL ---\n")

        if start_date and end_date:
            # We filter the 'valid_indices' list, not the raw DF (to keep history for indicators)
            # Ensure dates are Timestamps
            start_ts = pd.Timestamp(start_date)
            end_ts = pd.Timestamp(end_date)
            
            # --- TZ FIX: Align request TZ with DataFrame TZ ---
            if not df.empty and df.index.tz is not None:
                if start_ts.tz is None:
                    start_ts = start_ts.tz_localize(df.index.tz)
                if end_ts.tz is None:
                    end_ts = end_ts.tz_localize(df.index.tz)

            # Find indices within the requested test range
            # Note: We still need cfg.HISTORY_LENGTH_DAYS before the start_ts for warmup
            valid_indices = [i for i in range(len(df))
                             if df.index[i] >= start_ts and df.index[i] <= end_ts]
        else:
            # --- CALCULATE DYNAMIC START INDEX ---
            # 1. Determine the first date the chart should *display* (cfg.CHART_YEARS)
            chart_start_date = cfg.DATA_END_DATE - timedelta(days=cfg.CHART_YEARS * 365)
            chart_start_ts = pd.Timestamp(chart_start_date)

            # Smart Slicing based on Date
            valid_indices = [i for i in range(len(df)) if df.index[i] >= chart_start_ts]

        # Filter out early indices that don't have enough history for indicators
        valid_indices = [i for i in valid_indices if i >= cfg.HISTORY_LENGTH_DAYS]

        # # 2. Find the index where the data history is long enough (len > 200)
        # # We start the loop from the FIRST index that has sufficient history,
        # # which is len(df) >= 200 days *before* the visible chart range starts.
        #
        # # Ensure 'valid_indices' only includes data points that satisfy the 200-bar history requirement
        # required_warmup = cfg.HISTORY_LENGTH_DAYS  # Minimum bars needed for indicators


        if not valid_indices:
            logger.error("No data found in the requested chart range.")
            logger.error(f"DF Start: {df.index[0] if not df.empty else 'EMPTY'}, DF End: {df.index[-1] if not df.empty else 'EMPTY'}")
            logger.error(f"Req Start: {start_date}, Req End: {end_date}")
            return {} if return_stats else None

        logger.info(f"📊 Simulating range: {df.index[valid_indices[0]].date()} -> {df.index[valid_indices[-1]].date()}")

        logger.info("[>] Conductor Starting...")

        INVESTMENT_AMOUNT = cfg.INVESTMENT_AMOUNT
        
        # --- TIMEFRAME SCALING ---
        # Adjust "Days" to "Bars" based on timeframe
        if cfg.TIMEFRAME == "1h":
             bars_per_day = 7
             LOOK_AHEAD_BARS = cfg.LOOK_AHEAD_DAYS * bars_per_day
             GT_WINDOW = 15 * bars_per_day # Scaled Ground Truth
        elif cfg.TIMEFRAME == "1d":
             bars_per_day = 1
             LOOK_AHEAD_BARS = cfg.LOOK_AHEAD_DAYS # Standard 20 Days
             GT_WINDOW = 15 # Standard 15 Days
        else:
             bars_per_day = 1
             LOOK_AHEAD_BARS = cfg.LOOK_AHEAD_DAYS
             GT_WINDOW = 15
             
        logger.info(f"TIMEFRAME: {cfg.TIMEFRAME} | LookAhead: {LOOK_AHEAD_BARS} bars | GT Window: {GT_WINDOW} bars")

        # --- OPTIMIZATION: Vectorized Feature Calculation ---
        # Calculate features ONCE for the whole dataframe
        logger.info("Pre-calculating features for the entire simulation range...")
        features_full_df = advisor.calculator.calculate_features(df, context_data)
        
        if features_full_df.empty:
             logger.error("❌ Feature calculation failed (Empty DataFrame).")
             return {} if return_stats else None

        # Align features with the main df index
        # (Assuming calculate_features returns a DF with the same index)
        features_full_df = features_full_df.reindex(df.index)
        
        # Pre-convert to dictionary records for O(1) access inside loop is often faster
        # OR just use .iloc inside the loop. Let's use .to_dict('index') for safety/speed balance
        # actually, simply indexing into the dataframe by location or label is fine.
        # Let's keep it simple: access via location inside loop.
        

        with open(log_file_path, 'w') as log_file, open(journal_path, 'a') as journal_file:
            logger.info(f"[>] Writing decision log to {log_file_path}")
            logger.info(f"[>] Writing system thinking to {journal_path}")

            skip_until_index = -1

            for current_idx_pos, i in enumerate(tqdm(valid_indices)):
                # --- SERIAL EXECUTION CHECK (ENABLED) ---
                # If we are in a trade (skip_until_index > i), we skip analysis/entry
                if i <= skip_until_index:
                    continue
                current_date = df.index[i]
                data_slice = df.iloc[:i + 1].copy()

                score = 0
                regime = "INITIALIZING"
                pred = "WAIT"
                features = {'close': df.iloc[i]['close'], 'atr_14': 0.0}  # Ensure atr_14 default is present
                score_details = {"status": "INITIALIZING/ERROR"}

                # --- GLOBAL PNL/EXECUTION VARIABLES INITIALIZATION ---
                num_shares = 0
                net_pnl = 0.0
                pnl_percent = 0.0
                exit_status = "N/A"
                is_correct = False
                entry_price_used = features.get('close')  # Default to Close for logging WAIT/DOWN
                exit_price = entry_price_used
                entry_date_executed = pd.NaT  # Will be overwritten if trade executes
                INVESTMENT_AMOUNT = cfg.INVESTMENT_AMOUNT

                # --- DEFAULTS FOR GROUND TRUTH ---
                gt_label = 2
                mpp = 0.0
                mpd = 0.0
                ground_truth_label = "NOISE/HOLD"

                # --- Skip days where not enough history exists (e.g., first 50 days) ---
                if len(data_slice) < 200:
                    pass
                else:
                    # OPTIMIZED: Use pre-calculated features
                    # Since 'i' is the integer index in 'valid_indices', and valid_indices indexes into 'df'
                    # We can access features_full_df at index 'i' (since it matches df)
                    
                    # Safety check: ensure i is within bounds of features_full_df
                    if i < len(features_full_df):
                         # current_features_series = features_full_df.iloc[i]
                         
                         # Check if the row contains NaNs critical for strategy? 
                         # (Logic assumes calculator handles fillna usually)
                         
                         features = features_full_df.iloc[i].to_dict()
                    else:
                         regime = "DATA_ERROR"
                         features = {}

                    if not features:
                        regime = "DATA_ERROR"
                    else:
                        # features = features_df.iloc[-1].to_dict() # OLD LINE REMOVED

                        # --- REGIME DETECTION (Using imported class) ---
                        regime = MarketRegimeDetector.detect_regime(features)

                        # --- SCORE CALCULATION (Using imported class) ---
                        # GEN-9: Pass 60-day History Window for Deep Learning
                        start_hist = max(0, i - 60)
                        features['history_window'] = df.iloc[start_hist : i + 1] # Include current bar
                        
                        score, score_details = StrategyOrchestra.get_score(features, regime, params)
                        
                        # --- Gen-7 Orchestration Logic: Score is the definitive signal ---
                        if score >= cfg.SCORE_THRESHOLD_BUY:
                            pred = "UP"
                        elif score <= 100 - cfg.SCORE_THRESHOLD_BUY:
                            pred = "DOWN"
                        else:
                            pred = "WAIT"
                        # strong_trend = (regime == "TRENDING_UP")
                        # bullish_pattern = bool(features.get('bullish_pattern', False))
                        #
                        # if score >= cfg.SCORE_THRESHOLD_BUY and strong_trend and bullish_pattern:
                        #     pred = "UP"
                        # elif score <= 100 - cfg.SCORE_THRESHOLD_BUY:
                        #     pred = "DOWN"
                        # else:
                        #     pred = "WAIT"

                        # --- PNL/EXIT CALCULATION (ONLY FOR 'UP' PREDICTION) ---
                        # continuous_learning_analyzer.py (around Line 538)
                        # --- PNL/EXIT CALCULATION (ONLY FOR 'UP' PREDICTION) ---
                            # continuous_learning_analyzer.py (around Line 646)

                        # --- PNL/EXIT & EXECUTION VARIABLES INITIALIZATION (ALWAYS RUNS) ---
                        # num_shares = 0
                        # net_pnl = 0.0
                        # pnl_percent = 0.0
                        # exit_status = "N/A"
                        # entry_price_used = features.get('close')  # Default to Close for logging WAIT/DOWN
                        # exit_price = entry_price_used
                        # entry_date_executed = pd.NaT  # Will be overwritten if trade executes
                        #
                        # INVESTMENT_AMOUNT = cfg.INVESTMENT_AMOUNT

                        # --- Find Next Day's Open Price (The True Entry Point) ---
                        next_index = i + 1
                        if next_index < len(df):
                            next_day_data = df.iloc[next_index]
                            next_day_open = float(next_day_data['open'])
                            entry_date_executed = df.index[next_index]
                        else:
                            # No future data to execute the trade
                            next_day_open = None

                        # --- Gen-7 Trade Execution (Only if next day data exists) ---
                        if pred == "UP" and next_day_open is not None and features.get('atr_14', 0) > 0:
                            entry_price_used = next_day_open  # CRITICAL: Use next day OPEN for realistic entry
                            atr_value = float(features.get('atr_14', 0.0))

                            # 1. Calculate number of shares based on fixed investment
                            num_shares = INVESTMENT_AMOUNT / entry_price_used

                            # 2. Simulate Exit
                            # We use the index *after* the analysis day for the exit check (next_index)
                            exit_price, exit_date, exit_status = check_and_calculate_exit(
                                df, next_index, entry_price_used, entry_date_executed, atr_value, LOOK_AHEAD_BARS
                            )
                            
                            # --- SERIAL EXECUTION UPDATE ---
                            # If trade executed, block new entries until exit date
                            if exit_date is not None:
                                # Find the integer location of the exit date in the main df
                                try:
                                    skip_until_index = df.index.get_loc(exit_date)
                                except KeyError:
                                    # Fallback if specific timestamp match fails (rare)
                                    skip_until_index = next_index + 1
                            else:
                                # Pending trade (ran out of data), skip strictly to end
                                skip_until_index = len(df) + 1

                            if exit_price is not None:
                                # 3. Calculate Net PnL
                                gross_profit = (exit_price - entry_price_used) * num_shares

                                net_pnl, fees_paid = calculate_net_pnl_raw(
                                    gross_profit,
                                    num_shares,
                                    entry_price_used,
                                    exit_price
                                )

                                pnl_percent = (net_pnl / INVESTMENT_AMOUNT) * 100.0
                            else:
                                pnl_percent = 0.0
                        else:
                            # If we predicted UP but couldn't execute (no future data/ATR=0),
                            # reset entry to decision day close
                            entry_price_used = features.get('close')

                    # --- Conditionally calculate Is_Correct based on future data ---
                    # if i + LOOK_AHEAD_DAYS < len(df):
                    #     future_close = df.iloc[i + LOOK_AHEAD_DAYS]['close']
                    #     actual = "UP" if future_close > df.iloc[i]['close'] * 1.005 else "DOWN"
                    #     is_correct = (pred == actual)
                    # else:
                    #     # For the last few days where we cannot look ahead, set dummy values
                    #     actual = "N/A (Pending)"
                    #     is_correct = False

                    # --- GEN-7 Ground Truth Labeling (for AI Training) ---
                    # NOTE: We use 15 days forward as per spec
                    gt_label, mpp, mpd = _calculate_ground_truth_label(df, i, forward_days=GT_WINDOW)
                    # Map integer to string label
                    gt_label_map = {0: "STRONG_BUY", 1: "STRONG_SELL", 2: "NOISE/HOLD"}
                    ground_truth_label = gt_label_map[gt_label]

                    # --- 5. IS_CORRECT CALCULATION ---
                    WIN_THRESHOLD_USD = cfg.WIN_THRESHOLD_USD
                    WIN_THRESHOLD_PCT = cfg.WIN_THRESHOLD_PCT
                    is_correct = (net_pnl > WIN_THRESHOLD_USD) and (pnl_percent > WIN_THRESHOLD_PCT)

                # if pred == "UP" and features.get('atr_14', 0) > 0:
                #     entry_price = float(features.get('close'))
                #     atr_value = float(features.get('atr_14', 0.0))
                #     dc_lower = float(features.get('dc_lower', entry_price))
                #
                #     # Allow limit only if reasonably close to market (<= 1 ATR below)
                #     if entry_price - dc_lower <= atr_value:
                #         entry_price_used = dc_lower
                #     else:
                #         entry_price_used = entry_price
                #
                #     # 2. Calculate number of shares based on fixed investment
                #     # Assuming a fixed investment amount of $1000 for PnL calculation proxy
                #     num_shares = INVESTMENT_AMOUNT / entry_price_used
                #
                #     # 3. Simulate Exit
                #     # NOTE: We use the improved entry price in the exit calculation
                #     exit_price, exit_date, exit_status = check_and_calculate_exit(
                #         df, i, entry_price_used, current_date, atr_value, LOOK_AHEAD_DAYS
                #     )
                #
                #     if exit_price is not None:
                #         # 4. Calculate Net PnL
                #         net_pnl = calculate_net_pnl(entry_price_used, exit_price, num_shares)
                #         pnl_percent = (net_pnl / INVESTMENT_AMOUNT) * 100.0
                #     else:
                #         pnl_percent = 0.0

                # --- Conditionally calculate Is_Correct based on PnL (True if PnL > 0) ---
                # is_correct = net_p 0
                #
                #                 # --- Conditionally calculate Is_Correct based on PnL ---
                #                 WIN_THRESHOLD_USD = cfg.WIN_THRESHOLD_USD
                #                 WIN_THRESHOLD_PCT = cfg.WIN_THRESHOLD_USD
                #
                #                 is_correct = (net_pnl > WIN_THRESHOLD_USD) and (pnl_percent > WIN_THRESHOLD_PCT)nl >

                # Sanitize features for JSON logging
                sanitized_features = {}
                for k, v in features.items():
                    try:
                        if hasattr(v, 'item'): 
                            sanitized_features[k] = v.item() # Convert numpy scalar to python scalar
                        elif isinstance(v, (pd.DataFrame, pd.Series)):
                             sanitized_features[k] = str(v.iloc[0] if not v.empty else "Empty")
                        else:
                            sanitized_features[k] = v
                    except:
                        sanitized_features[k] = str(v)

                log_entry = {
                    'Date': current_date.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
                    'Close': float(df.iloc[i]['close']),
                    'Regime': regime,
                    'System_Score': float(score),
                    'Prediction': pred,
                    'Is_Correct': str(is_correct),
                    'Entry_Price': float(entry_price_used) if entry_price_used is not None else None,
                    'Exit_Price': float(exit_price) if exit_price is not None else None,
                    'Exit_Status': exit_status,
                    'Net_PNL_USD': float(net_pnl),
                    'Net_PNL_Percent': float(pnl_percent),
                    'Ground_Truth_Label': ground_truth_label,
                    'Max_Potential_Profit': float(mpp),
                    'Max_Potential_Drawdown': float(mpd),
                    'Decision_Breakdown': score_details,
                    'Feature_Values': sanitized_features  # Log Sanitized features
                }

                # --- VERBOSE JOURNALING ---
                if pred != "WAIT":
                    journal_file.write(f"\n[{current_date.strftime('%Y-%m-%d')}] DECISION: {pred} (Score: {score})\n")
                    journal_file.write(f"  - REGIME: {regime}\n")
                    journal_file.write(f"  - AGENT: {score_details.get('Active_Agent', 'Unknown')}\n")
                    journal_file.write(f"  - REASONING:\n")
                    for k, v in score_details.items():
                        if k not in ['status', 'Active_Agent'] and isinstance(v, (int, float)) and abs(v) > 0:
                             journal_file.write(f"    * {k}: {v}\n")
                             
                if pred == "UP":
                    journal_file.write(f"  -> ATTEMPTING ENTRY...\n")

                entry_dt = current_date
                exit_dt = exit_date if (pred == "UP" and exit_date is not None) else None

                # continuous_learning_analyzer.py (around Line 596)
                if pred == "UP" and features.get('atr_14', 0) > 0 and num_shares > 0:
                    # ------------------ 2.1 ENTRY LOGGING ------------------
                    risk_dollars_per_share = atr_value * cfg.TRAILLING_STOP_ATR
                    stop_loss_price = entry_price_used - risk_dollars_per_share
                    profit_target = entry_price_used + (risk_dollars_per_share * cfg.RISK_REWARD_RATIO)

                    logger.info(
                        f" ENTRY Signal ({current_date.date()}): Price={entry_price_used:.2f} | "
                        f"Stop={stop_loss_price:.2f} | Target=DYNAMIC | "
                        f"Shares={num_shares:.2f} | Initial Risk={risk_dollars_per_share * num_shares:.2f}"
                    )
                    
                    journal_file.write(f"  -> EXECUTED: Buy {num_shares:.2f} shares @ {entry_price_used:.2f}.\n")
                    journal_file.write(f"     Stop: {stop_loss_price:.2f}, Target: {profit_target:.2f}\n")

                    # ------------------ 2.2 EXIT LOGGING -------------------
                    if exit_price is not None:
                        # Find potential gross profit if target was hit
                        # 3. Calculate Net PnL using the IMPORTED function
                        gross_profit = (exit_price - entry_price_used) * num_shares

                        net_pnl, fees_paid = calculate_net_pnl_raw(
                            gross_profit,
                            num_shares,
                            entry_price_used,
                            exit_price
                        )

                        pnl_percent = (net_pnl / INVESTMENT_AMOUNT) * 100.0

                        logger.info(
                            f" EXIT Status ({exit_date.date()}): {exit_status} | "
                            f"Exit Price={exit_price:.2f} | Profit=${net_pnl:.2f} | "
                            f"Est. Target Profit=${net_pnl:.2f} (Max)"
                        )
                        
                        journal_file.write(f"  -> EXIT CLOSED ({exit_date.strftime('%Y-%m-%d')}): {exit_status}\n")
                        journal_file.write(f"     Px: {exit_price:.2f}. PnL: ${net_pnl:.2f} ({pnl_percent:.2f}%)\n")
                        
                        # --- SERIAL TRADE LIST EXPORT ---
                        trades_list.append({
                            'Entry Date': entry_date_executed,
                            'Entry Price': entry_price_used,
                            'Exit Date': exit_date,
                            'Exit Price': exit_price,
                            'Direction': 'Long',
                            'PnL': net_pnl,
                            'PnL %': pnl_percent,
                            'Exit Reason': exit_status
                        })

                # Write detailed JSONL log
                log_file.write(json.dumps(log_entry) + '\n')  # <-- WRITING THE JSONL ENTRY

                trade_date = exit_date if (
                            pred == "UP" and 'exit_date' in locals() and exit_date is not None) else current_date

                # Check if a trade was attempted/executed
                trade_executed = (pred == "UP" and next_day_open is not None and num_shares > 0)

                # simulation_log.append({
                #     'Date': current_date,  # keep main candle date = entry bar
                #     'Open': df.iloc[i]['open'],
                #     'High': df.iloc[i]['high'],
                #     'Low': df.iloc[i]['low'],
                #     'Close': df.iloc[i]['close'],
                #     'Regime': regime,
                #     'System_Score': score,
                #     'Prediction': pred,
                #     'Entry_Date': entry_dt,
                #     'Exit_Date': exit_dt,
                #     'Entry_Price': entry_price_used,
                #     'Exit_Price': exit_price,
                #     'Num_Shares': num_shares,
                #     'Net_PNL_USD': net_pnl,
                #     'Net_PNL_Percent': pnl_percent,
                #     'Is_Correct': is_correct
                # })
                # Log the trade result against the DECISION BAR (i) for chart alignment
                log_item = {
                    'Date': current_date,
                    'Open': df.iloc[i]['open'],
                    'High': df.iloc[i]['high'],
                    'Low': df.iloc[i]['low'],
                    'Close': df.iloc[i]['close'],
                    'Regime': regime,
                    'System_Score': score,
                    'Prediction': pred,
                    'Num_Shares': num_shares,
                    'Net_PNL_USD': net_pnl,
                    'Net_PNL_Percent': pnl_percent,
                    'Is_Correct': is_correct,
                    'Ground_Truth_Label': ground_truth_label,
                    'Max_Potential_Profit': mpp,
                    'Max_Potential_Drawdown': mpd,

                }

                # Add execution details only if a trade was attempted
                if trade_executed:
                    log_item['Entry_Date'] = entry_date_executed
                    log_item['Exit_Date'] = exit_date
                    log_item['Entry_Price'] = entry_price_used
                    log_item['Exit_Price'] = exit_price
                else:
                    log_item['Entry_Date'] = pd.NaT  # Use pd.NaT for proper timestamp NaN
                    log_item['Exit_Date'] = pd.NaT
                    log_item['Entry_Price'] = np.nan
                    log_item['Exit_Price'] = np.nan

                simulation_log.append(log_item)

        res = pd.DataFrame(simulation_log)
        if res.empty:
            logger.error("❌ Simulation Log is Empty.")
            return {} if return_stats else None

        # 1. SYSTEM WIN RATE
        actionable_trades = res[(res['Prediction'] == 'UP') & (res['Num_Shares'] > 0)]
        win_rate = actionable_trades['Is_Correct'].mean() * 100 if not actionable_trades.empty else 0.0
        total_pnl = actionable_trades['Net_PNL_USD'].sum() if not actionable_trades.empty else 0.0

        # 2. ACTIONABLE TREND ACCURACY (UP Signals Only)
        # We only trade UP signals, so we only care about UP accuracy.
        # Accuracy = 1 if Ground_Truth_Label == 0 (Local Min) OR (MPP > 2%)
        
        up_predictions = res[res['Prediction'] == 'UP'].copy()
        
        # Helper for vector accuracy logic
        def is_accurate(row):
            pred = row['Prediction']
            gt = row['Ground_Truth_Label']
            mpp = row['Max_Potential_Profit']
            
            # Accurate if it was a labeled buy (0) OR if we simply had a decent move up (MPP > 2%)
            return 1 if (gt == 0 or mpp > 0.02) else 0

        if not up_predictions.empty:
            up_predictions['Is_Accurate'] = up_predictions.apply(is_accurate, axis=1)
            acc = up_predictions['Is_Accurate'].mean() * 100
        else:
            acc = 0.0
            
        # Log specific "Trend Accuracy"
        logger.info(f" Actionable Trend Accuracy: {acc:.2f}% (UP Signals > 2% Profit)")


        logger.info(f"Actionable trades: {len(actionable_trades)}")
        logger.info(f"Profitable trades: {actionable_trades['Is_Correct'].sum()}")
        logger.info(f"Total PnL: {total_pnl:.2f}")

        seed_win_rate = win_rate
        
        # --- Add Rolling Stats for Charting ---
        res['Rolling_Win_20'] = res['Is_Correct'].rolling(20).mean() * 100
        res['Rolling_PNL_20'] = res['Net_PNL_USD'].rolling(20).sum()

        if trades_list:
            trades_df = pd.DataFrame(trades_list)
            trades_csv_path = os.path.join(cfg.LOGS_DIR, 'trades_list.csv')
            trades_df.to_csv(trades_csv_path, index=False)
            logger.info(f"Serial Trade List saved to {trades_csv_path}")

        logger.info(f"\n GEN-7 RESULTS")
        logger.info(f" Simulation Interval: {cfg.DATA_START_DATE} to {cfg.DATA_END_DATE}")
        logger.info(f" Global Accuracy: {acc:.2f}%")
        logger.info(f" System Win Rate: {win_rate:.2f}%")

        # Only generate chart if NOT in stats-return mode (avoids popup windows during automated testing)
        if not return_stats:
            generate_chart(res, cfg.TARGET_TICKER, acc, win_rate)

        # --- RETURN STATS FOR ABLATION SCRIPT ---
        if return_stats:
            return {
                "total_pnl": total_pnl,
                "system_win_rate": win_rate,
                "total_trades": len(actionable_trades),
                "actionable_accuracy": acc
            }

    finally:
        if dm.isConnected():
            logger.info("🔌 Disconnecting...")
            dm.disconnect()
    #     # 1) Keep only the user-selected window (CHART_YEARS)
    #     visible_start = pd.Timestamp(cfg.DATA_END_DATE) - pd.Timedelta(days=int(cfg.CHART_YEARS * 365))
    #     res = res[res['Date'] >= visible_start]
    #
    #     # 2) Enforce strict chronological order for the chart
    #     res = res.sort_values('Date').reset_index(drop=True)
    #
    #     res['Rolling_Win_20'] = res['Is_Correct'].rolling(20).mean() * 100
    #     res['Rolling_PNL_20'] = res['Net_PNL_USD'].rolling(20).sum()
    #
    #     # --- Recalculate Acc/Win Rate only using actionable trades (excluding WAIT) ---
    #     # Note: 'Actual_Direction' column is still required by the old Chart function, even if deprecated in the loop
    #     known_outcome_res = res.copy()
    #
    #     # 1. SYSTEM WIN RATE (Execution Performance)
    #     # Actionable trades = trades where we actually entered (UP) and have non-zero position
    #     actionable_trades = known_outcome_res[
    #         (known_outcome_res['Prediction'] == 'UP') &
    #         (known_outcome_res['Num_Shares'] > 0)
    #         ]
    #
    #     # System Win Rate: percentage of executed trades that met the profit threshold (Is_Correct = True)
    #     win_rate = actionable_trades['Is_Correct'].mean() * 100 if not actionable_trades.empty else 0.0
    #
    #     # 2. GLOBAL ACCURACY (Prediction Accuracy against Ground Truth)
    #     # We only check accuracy for days where the system made a directional call (UP or DOWN)
    #     directional_predictions = res[res['Prediction'].isin(['UP', 'DOWN'])]
    #
    #     # Define correctness based on prediction vs. Ground Truth label
    #     # UP is correct if Ground Truth is STRONG_BUY (0)
    #     # DOWN is correct if Ground Truth is STRONG_SELL (1)
    #
    #     # Add a temporary column for calculation
    #     directional_predictions['Is_Accurate'] = (
    #                                                      (directional_predictions['Prediction'] == 'UP') &
    #                                                      (directional_predictions['Ground_Truth_Label'] == 'STRONG_BUY')
    #                                              ) | (
    #                                                      (directional_predictions['Prediction'] == 'DOWN') &
    #                                                      (directional_predictions[
    #                                                           'Ground_Truth_Label'] == 'STRONG_SELL')
    #                                              )
    #
    #     # Global Accuracy: mean of Is_Accurate across all directional predictions
    #     acc = directional_predictions['Is_Accurate'].mean() * 100 if not directional_predictions.empty else 0.0
    #
    #     logger.info(f"Actionable trades: {len(actionable_trades)}")
    #     logger.info(f"Profitable trades: {actionable_trades['Is_Correct'].sum()}")
    #     logger.info(f"Total PnL: {actionable_trades['Net_PNL_USD'].sum():.2f}")
    #
    #     logger.info(f"\n📊 GEN-7 RESULTS")
    #     logger.info(f"🎯 Global Accuracy: {acc:.2f}%")
    #     logger.info(f"🚀 System Win Rate: {win_rate:.2f}%")
    #
    #     generate_chart(res, cfg.TARGET_TICKER, acc, win_rate)
    #
    # finally:
    #     if dm.isConnected():
    #         logger.info("🔌 Disconnecting...")
    #         dm.disconnect()


if __name__ == "__main__":
    run_simulation()