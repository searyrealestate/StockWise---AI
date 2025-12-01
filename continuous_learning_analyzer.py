# continuous_learning_analyzer.py

"""
StockWise Gen-4: The Orchestra Architecture (With Binary Prediction Chart)
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
from stockwise_simulation import ProfessionalStockAdvisor, FeatureCalculator, clean_raw_data
import system_config as cfg  # Centralized Config

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", stream=sys.stdout)
logger = logging.getLogger("Gen5_Complete")

LOOK_AHEAD_DAYS = 5

# ==========================================
# 🧠 REGIME & STRATEGY LOGIC
# ==========================================


# --- 1. PARAMETER LOADER ---
def load_optimized_params(ticker):
    """Loads best params from JSON or defaults."""
    # Default fallback
    # params = {
    #     'sma_short': 20, 'sma_long': 100,  # Optimized values we saw earlier
    #     'rsi_threshold': 75, 'atr_mult_stop': 2.5
    # }

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

    logger.warning("⚠️ Using Default Params.")
    return cfg.STRATEGY_PARAMS


# --- 2. REGIME DETECTOR (Dynamic) ---
class MarketRegimeDetector:
    @staticmethod
    def detect_regime(latest_row):
        close = latest_row['close']

        # Use Optimized SMAs for Regime Definition
        sma_50 = latest_row.get('sma_50', close)  # Actually SMA_Short
        sma_200 = latest_row.get('sma_200', close)  # Actually SMA_Long
        adx = latest_row.get('ADX_14', 0)

        # Bearish: Below Long Term Trend
        if close < sma_200: return "BEARISH"

        # Bullish
        if close > sma_200:
            if adx > cfg.STRATEGY_PARAMS['adx_threshold'] and close > sma_50:
                return "TRENDING_UP"
            else:
                return "RANGING_BULL"
        return "RANGING_BULL"


# --- 3. STRATEGY ORCHESTRA (FULL) ---
class StrategyOrchestra:
    @staticmethod
    def get_score(features, regime, params):
        if regime == "TRENDING_UP":
            return StrategyOrchestra._agent_breakout(features, params)
        elif regime == "RANGING_BULL":
            # Early Breakout Override
            if features['close'] > features.get('recent_high', 99999):
                return StrategyOrchestra._agent_breakout(features, params)
            return StrategyOrchestra._agent_dip_buyer(features, params)
        elif regime == "BEARISH": return StrategyOrchestra._agent_bear_defense(features)
        return 0

    @staticmethod
    def _agent_breakout(f, p):
        score = 0
        if f['close'] > f.get('recent_high', 99999): score += cfg.SCORE_TRIGGER_BREAKOUT

        # Volume
        if f['volume'] > f.get('vol_ma', 0) * 1.1: score += cfg.SCORE_CONFIRM_VOLUME

        # Slope/Velocity
        angle = f.get('slope_angle', 0)
        if angle > p.get('slope_threshold', 20): score += 10

        # Sector
        if f.get('rel_strength_sector', 0) > -0.01:
            score += cfg.SCORE_CONFIRM_SECTOR
        else:
            score -= 15

        # Safety
        if f.get('rsi_14', 50) > 82: score += cfg.SCORE_PENALTY_RSI
        return score

    @staticmethod
    def _agent_dip_buyer(f, p):
        score = 0
        rsi = f.get('rsi_14', 50)
        thresh = p['rsi_threshold']  # Dynamic Threshold

        # Trigger: Oversold relative to trend
        if rsi < 40:
            score += cfg.SCORE_TRIGGER_DIP
        elif rsi < 50:
            score += 20

        # Support
        sma_short = f.get('sma_50', 0)
        if sma_short > 0:
            dist = (f['close'] - sma_short) / sma_short
            if -0.03 < dist < 0.03: score += 25

        # Sector Divergence (Stock down, Sector Up)
        if f.get('rel_strength_sector', 0) < -0.02:
            if f.get('sector_daily_return', 0) > 0 and f.get('daily_return', 0) < 0:
                score += 15

        # IBS Reversion
        if f.get('ibs', 0.5) < 0.2: score += 15

        return score

    @staticmethod
    def _agent_bear_defense(f):
        if f.get('rsi_14', 50) < 20: return 55
        return 0


# --- 4. ROBUST CALCULATOR (With Slope & Params) ---
class RobustFeatureCalculator:
    def __init__(self, data_manager, contextual_data, is_cloud, params):
        self.qqq_data = contextual_data.get('qqq', pd.DataFrame()).copy()
        self.sector_data = contextual_data.get('sector', pd.DataFrame()).copy()
        self.params = params
        self._logged_first_last = False

        # Ensure indices are unique before any calculations to prevent reindex failure
        if not self.qqq_data.empty:
            self.qqq_data = self.qqq_data[~self.qqq_data.index.duplicated(keep='last')]

        if not self.sector_data.empty:
            self.sector_data = self.sector_data[~self.sector_data.index.duplicated(keep='last')]

        # Standardize column names
        for df_ref in (self.qqq_data, self.sector_data):
            if not df_ref.empty:
                df_ref.columns = [c.lower() for c in df_ref.columns]

    # def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
    #     df = df.copy()
    #     if df.empty:
    #         return pd.DataFrame()
    #
    #     try:
    #         df.columns = [col.lower() for col in df.columns]
    #
    #         required = {'open', 'high', 'low', 'close', 'volume'}
    #         if not required.issubset(df.columns):
    #             df = normalize_and_validate_data(df)
    #             if df.empty:
    #                 logger.error("Calc Error: normalize_and_validate_data returned empty df (missing OHLCV).")
    #                 return pd.DataFrame()
    #
    #             df.columns = [c.lower() for c in df.columns]
    #
    #         # Dynamic SMAs
    #         s_short = int(self.params.get('sma_short', 20))
    #         s_long = int(self.params.get('sma_long', 100))
    #
    #         # We map to standardized names so Regime Detector works generically
    #         df['sma_50'] = df['close'].rolling(s_short).mean()
    #         df['sma_200'] = df['close'].rolling(s_long).mean()
    #
    #         # SLOPE (Velocity)
    #         # Calculate Linear Regression Angle of the Short SMA
    #         # This tells us how steep the trend is (0-90 degrees)
    #         try:
    #             slope_df = df.ta.slope(length=10)  # DataFrame with one or more columns
    #             if isinstance(slope_df, pd.DataFrame) and not slope_df.empty:
    #                 df['slope_angle'] = slope_df.iloc[:, 0] * (180 / np.pi)
    #             else:
    #                 df['slope_angle'] = 0.0
    #         except Exception as e:
    #             logger.error(f"Calc Error in slope_angle: {e}")
    #             df['slope_angle'] = 0.0
    #
    #         # Log only once for the very first full feature frame
    #         if not self._logged_first_last and len(df) > 0:
    #             logger.info(f"DEBUG#F1 features df range: {df.index.min()} -> {df.index.max()}")
    #             logger.info(f"DEBUG#F1 features head:\n{df.head(3)}")
    #             self._logged_first_last = True
    #
    #
    #         try:
    #             adx = df.ta.adx(length=14)
    #             if adx is not None:
    #                 df = pd.concat([df, adx], axis=1)
    #         except:
    #             pass
    #
    #         high_series = df['high']
    #         if isinstance(high_series, pd.DataFrame):
    #             high_series = high_series.iloc[:, 0]
    #
    #         df['recent_high'] = high_series.rolling(50).max().shift(1)
    #
    #         # Vol/Mom
    #         # Vol/Mom
    #         try:
    #             df.ta.atr(length=14, append=True, col_names='atr_14')
    #             df.ta.rsi(length=14, append=True, col_names='rsi_14')
    #
    #             # Some pandas_ta volume functions return DataFrames; we avoid that and
    #             # explicitly use a simple rolling mean, which is always a Series.
    #             vol_ma = df['volume'].rolling(20).mean()
    #             if isinstance(vol_ma, pd.DataFrame):
    #                 df['vol_ma'] = vol_ma.iloc[:, 0]
    #             else:
    #                 df['vol_ma'] = vol_ma
    #         except Exception as e:
    #             logger.error(f"Calc Error in vol_ma: {e}")
    #             df['vol_ma'] = df['volume'].rolling(20).mean()
    #
    #         df['daily_return'] = df['close'].pct_change()
    #         df['ibs'] = (df['close'] - df['low']) / ((df['high'] - df['low']) + 1e-9)
    #
    #         # Context: QQQ
    #         if not self.qqq_data.empty:
    #             qqq = self.qqq_data.copy()
    #             qqq.index = pd.to_datetime(qqq.index)
    #             qqq = qqq[~qqq.index.duplicated(keep='last')]
    #             qqq_ret = qqq['close'].pct_change()
    #             aligned_qqq = qqq_ret.reindex(df.index).fillna(0)
    #             df['rel_strength_qqq'] = df['daily_return'] - aligned_qqq
    #         else:
    #             df['rel_strength_qqq'] = 0.0
    #
    #         # Context: SECTOR (Gen-5)
    #         if not self.sector_data.empty:
    #             sec = self.sector_data.copy()
    #             sec.index = pd.to_datetime(sec.index)
    #             sec = sec[~sec.index.duplicated(keep='last')]
    #             sec_ret = sec['close'].pct_change()
    #             aligned_sec = sec_ret.reindex(df.index).fillna(0)
    #             df['rel_strength_sector'] = df['daily_return'] - aligned_sec
    #             df['sector_daily_return'] = aligned_sec
    #         else:
    #             df['rel_strength_sector'] = 0.0
    #             df['sector_daily_return'] = 0.0
    #
    #         df.fillna(0, inplace=True)
    #         return df
    #
    #     except Exception as e:
    #         logger.error(f"Calc Error: {e}")
    #         return pd.DataFrame()

    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if df.empty: return pd.DataFrame()

        try:
            df.columns = [col.lower() for col in df.columns]

            # FIX: Remove Duplicate Columns (e.g. if two 'close' columns exist)
            df = df.loc[:, ~df.columns.duplicated()]

            s_short = int(self.params.get('sma_short', 20))
            s_long = int(self.params.get('sma_long', 100))

            df['sma_50'] = df['close'].rolling(s_short).mean()
            df['sma_200'] = df['close'].rolling(s_long).mean()

            try:
                slope_df = df.ta.slope(length=10)
                if isinstance(slope_df, pd.DataFrame):
                    df['slope_angle'] = slope_df.iloc[:, 0] * (180 / np.pi)
                else:
                    df['slope_angle'] = slope_df * (180 / np.pi)
            except:
                df['slope_angle'] = 0.0

            try:
                adx = df.ta.adx(length=14)
                if adx is not None: df = pd.concat([df, adx], axis=1)
            except:
                pass

            df = df.loc[:, ~df.columns.duplicated()]

            df['recent_high'] = df['high'].rolling(50).max().shift(1)

            df.ta.atr(length=14, append=True, col_names='atr_14')
            df.ta.rsi(length=14, append=True, col_names='rsi_14')

            vol_series = df['volume']
            if isinstance(vol_series, pd.DataFrame): vol_series = vol_series.iloc[:, 0]
            df['vol_ma'] = vol_series.rolling(20).mean()

            close_series = df['close']
            if isinstance(close_series, pd.DataFrame): close_series = close_series.iloc[:, 0]
            df['daily_return'] = close_series.pct_change()

            low_s = df['low'] if isinstance(df['low'], pd.Series) else df['low'].iloc[:, 0]
            high_s = df['high'] if isinstance(df['high'], pd.Series) else df['high'].iloc[:, 0]
            df['ibs'] = (close_series - low_s) / ((high_s - low_s) + 1e-9)

            if not self.qqq_data.empty:
                qqq_ret = self.qqq_data['close'].pct_change()
                aligned_qqq = qqq_ret.reindex(df.index).fillna(0)
                df['rel_strength_qqq'] = df['daily_return'] - aligned_qqq
            else:
                df['rel_strength_qqq'] = 0.0

            if not self.sector_data.empty:
                sec_ret = self.sector_data['close'].pct_change()
                aligned_sec = sec_ret.reindex(df.index).fillna(0)
                df['rel_strength_sector'] = df['daily_return'] - aligned_sec
                df['sector_daily_return'] = aligned_sec
            else:
                df['rel_strength_sector'] = 0.0
                df['sector_daily_return'] = 0.0

            df.fillna(0, inplace=True)
            return df

        except Exception as e:
            logger.error(f"Calc Error: {e}")
            return pd.DataFrame()

# ==========================================
# 🚀 EXECUTION LOGIC
# ==========================================


# --- UTILS ---
# def load_smart_context(ticker):
#     """
#     Uses SectorMapper to identify and fetch the correct Sector ETF.
#     """
#     logger.info("🌐 Loading Contextual Data (QQQ + Sector)...")
#     context = {'qqq': pd.DataFrame(), 'sector': pd.DataFrame()}
#
#     try:
#         mapper = SectorMapper()
#         sector_symbol = mapper.get_benchmark_symbol(ticker)
#         logger.info(f"🌍 Sector Benchmark: {sector_symbol}")
#
#         tickers = ['QQQ', sector_symbol]
#         for t in tickers:
#             df = yf.download(t, period="2y", progress=False, auto_adjust=True)  # Fetch more data
#             if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
#             df.columns = [c.lower() for c in df.columns]
#             context['sector' if t == sector_symbol else 'qqq'] = df
#     except:
#         logger.warning("⚠️ Failed to load contextual data (QQQ/Sector). Proceeding without it.")
#         pass
#     return context

# def load_data_parallel(dm, ticker):
#     """
#     Fetches Target (IBKR) + Sector/QQQ (YF) in parallel.
#     Replaces all other load_context functions.
#     """
#     logger.info("⚡ Starting Parallel Data Download...")
#
#     start_date = cfg.FIXED_START_DATE
#     end_date = cfg.DATA_END_DATE
#
#     mapper = SectorMapper()
#     sector_symbol = mapper.get_benchmark_symbol(ticker)
#     logger.info(f"🌍 Identified Sector Benchmark: {sector_symbol}")
#
#     with ThreadPoolExecutor(max_workers=3) as executor:
#         # Task 1: Main Stock (IBKR)
#         future_stock = executor.submit(fetch_ticker_data, dm, ticker, start_date, end_date)
#
#         # Task 2: Context (YF)
#         future_qqq = executor.submit(fetch_yfinance_data, "QQQ", start_date, end_date)
#         future_sec = executor.submit(fetch_yfinance_data, sector_symbol, start_date, end_date)
#
#         # Wait for results
#         stock_df = future_stock.result()
#         _, qqq_df = future_qqq.result()
#         _, sec_df = future_sec.result()
#
#     context = {'qqq': qqq_df, 'sector': sec_df}
#     logger.info(f"✅ Data Loaded. NVDA: {len(stock_df)} | QQQ: {len(qqq_df)} | {sector_symbol}: {len(sec_df)}")
#     return stock_df, context


def load_data_sequential(dm, ticker):
    """
    Fetches Target, QQQ, and Sector sequentially to avoid ThreadPool errors.
    """
    logger.info("🐢 Starting Sequential Data Download...")

    start_date = cfg.DATA_START_DATE
    end_date = cfg.DATA_END_DATE

    mapper = SectorMapper()
    sector_symbol = mapper.get_benchmark_symbol(ticker)
    logger.info(f"🌍 Identified Sector Benchmark: {sector_symbol}")

    # 1. Main Stock
    logger.info(f"📥 Fetching {ticker}...")
    stock_df = clean_raw_data(dm.get_stock_data(ticker, start_date=start_date, end_date=end_date))
    # logger.info(f"DEBUG_CLEAN[{ticker}] range: {stock_df.index.min()} -> {stock_df.index.max()}")
    # logger.info(f"DEBUG_CLEAN[{ticker}] head:\n{stock_df.head(3)}")
    # logger.info(f"DEBUG_CLEAN[{ticker}] tail:\n{stock_df.tail(3)}")

    # 2. Context (QQQ)
    logger.info(f"📥 Fetching QQQ...")
    qqq_df = clean_raw_data(dm.get_stock_data("QQQ", start_date=start_date, end_date=end_date))
    # logger.info(f"DEBUG#10 qqq_df Data:\n{qqq_df.head(3)} ...\n{qqq_df.tail(3)}")

    # 3. Context (Sector)
    logger.info(f"📥 Fetching {sector_symbol}...")
    sec_df = clean_raw_data(dm.get_stock_data(sector_symbol, start_date=start_date, end_date=end_date))
    # logger.info(f"DEBUG#11 sec_df Data:\n{sec_df.head(3)} ...\n{sec_df.tail(3)}")

    context = {'qqq': qqq_df, 'sector': sec_df}
    logger.info(f"✅ Data Loaded. {cfg.TARGET_TICKER}: {len(stock_df)} | QQQ: {len(qqq_df)} | {sector_symbol}: {len(sec_df)}")
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

    wins = df[(df['Prediction'] == 'UP') & (df['Is_Correct'] == True)]
    losses = df[(df['Prediction'] == 'UP') & (df['Is_Correct'] == False)]

    fig.add_trace(go.Scatter(x=wins['Date'], y=wins['High'] * 1.02, mode='markers',
                             marker=dict(color='green', size=12, symbol='triangle-down'), name='Win'), row=1, col=1)
    fig.add_trace(go.Scatter(x=losses['Date'], y=losses['High'] * 1.02, mode='markers',
                             marker=dict(color='red', size=12, symbol='x'), name='Loss'), row=1, col=1)

    regime_map = {'BEARISH': 0, 'RANGING_BULL': 1, 'TRENDING_UP': 2}
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

    df['Rolling_Win'] = df['Is_Correct'].rolling(20).mean() * 100
    fig.add_trace(go.Scatter(x=df['Date'], y=df['System_Score'], name='Score', line=dict(color='cyan')), row=4, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Rolling_Win'], name='Win Rate', line=dict(color='white', dash='dot'),
                             yaxis='y2'), row=4, col=1)
    fig.add_hline(y=50, line_dash="solid", line_color="red", row=4, col=1)

    start_d = df['Date'].iloc[0].strftime('%Y-%m-%d')
    end_d = df['Date'].iloc[-1].strftime('%Y-%m-%d')
    title_text = f"Gen-5 Orchestra: {ticker} ({start_d} to {end_d}) | Acc: {global_acc:.1f}% | Win: {win_rate:.1f}%"

    fig.update_layout(title=title_text, template="plotly_dark", height=1200, xaxis_rangeslider_visible=False)
    fig.write_html(filename)
    logger.info(f"📄 Chart Saved: {filename}")


def run_simulation():
    logger.info(f"🚀 STARTING GEN-5 PRO SIMULATION FOR {cfg.TARGET_TICKER}")

    # 1. Connect
    dm = DataSourceManager(use_ibkr=True, allow_fallback=True, port=cfg.IBKR_PORT)
    try:
        dm.connect_to_ibkr()
    except:
        logger.info("⚠️ IBKR Connection Failed. Using Fallback Data Source.")
        pass

    try:
        # 2. Parallel Download (Single Source of Truth)
        df, context_data = load_data_sequential(dm, cfg.TARGET_TICKER)

        if df.empty:
            logger.error("❌ Main Stock Data is Empty.")
            return

        # 3. Load Optimized Params
        params = load_optimized_params(cfg.TARGET_TICKER)

        # # 2. Load Smart Context
        # context_data = load_smart_context(TICKER)

        advisor = ProfessionalStockAdvisor(model_dir=cfg.MODELS_DIR, data_source_manager=dm)
        advisor.calculator = RobustFeatureCalculator(dm, context_data, False, params)
        advisor.log.info(f"🛠️ Using Robust Feature Calculator with Sector Data")

        # end_date = datetime.now().date()
        # start_date = end_date - timedelta(days=1200)
        # logger.info("📥 Downloading NVDA Data...")
        # df = clean_raw_data(dm.get_stock_data(TICKER, start_date=start_date, end_date=end_date))
        #
        # if df.empty: return

        # 5. Run Loop
        simulation_log = []

        # --- CALCULATE START INDEX DYNAMICALLY ---
        chart_start_date = cfg.DATA_END_DATE - timedelta(days=cfg.CHART_YEARS * 365)
        chart_start_ts = pd.Timestamp(chart_start_date)

        # Smart Slicing based on Date
        valid_indices = [i for i in range(len(df) - LOOK_AHEAD_DAYS) if df.index[i] >= chart_start_ts]

        if not valid_indices:
            logger.error("❌ No data found in the requested chart range.")
            return

        logger.info(f"📊 Simulating range: {df.index[valid_indices[0]].date()} -> {df.index[valid_indices[-1]].date()}")

        logger.info("🏃 Conductor Starting...")
        for i in tqdm(valid_indices):
            current_date = df.index[i]
            data_slice = df.iloc[:i + 1].copy()

            features_df = advisor.calculator.calculate_all_features(data_slice)
            if features_df.empty: continue
            features = features_df.iloc[-1].to_dict()

            regime = MarketRegimeDetector.detect_regime(features)
            score = StrategyOrchestra.get_score(features, regime, params)
            pred = "UP" if score >= 50 else "DOWN"

            future_close = df.iloc[i + LOOK_AHEAD_DAYS]['close']
            actual = "UP" if future_close > df.iloc[i]['close'] * 1.005 else "DOWN"
            is_correct = (pred == actual)

            simulation_log.append({
                'Date': current_date,
                'Open': df.iloc[i]['open'], 'High': df.iloc[i]['high'], 'Low': df.iloc[i]['low'],
                'Close': df.iloc[i]['close'],
                'Regime': regime, 'System_Score': score, 'Prediction': pred,
                'Actual_Direction': actual, 'Is_Correct': is_correct
            })

        res = pd.DataFrame(simulation_log)
        if res.empty:
            logger.error("❌ Simulation Log is Empty.")
            return

        # 1) Keep only the user-selected window (CHART_YEARS)
        visible_start = pd.Timestamp(cfg.DATA_END_DATE) - pd.Timedelta(days=int(cfg.CHART_YEARS * 365))
        res = res[res['Date'] >= visible_start]

        # 2) Enforce strict chronological order for the chart
        res = res.sort_values('Date').reset_index(drop=True)

        # logger.info(f"DEBUG#5 sorted res range: {res['Date'].min()} -> {res['Date'].max()}")
        # logger.info(f"DEBUG#5 sorted res count: {len(res)}")
        # logger.info(f"DEBUG#7 res head:\n{res[['Date', 'Open', 'High', 'Low', 'Close']].head(5)}")
        # logger.info(f"DEBUG#7 res tail:\n{res[['Date', 'Open', 'High', 'Low', 'Close']].tail(5)}")

        acc = res['Is_Correct'].mean() * 100
        up_trades = res[res['Prediction'] == 'UP']
        win_rate = up_trades['Is_Correct'].mean() * 100 if not up_trades.empty else 0.0

        logger.info(f"\n📊 GEN-5 RESULTS")
        logger.info(f"🎯 Global Accuracy: {acc:.2f}%")
        logger.info(f"🚀 System Win Rate: {win_rate:.2f}%")

        generate_chart(res, cfg.TARGET_TICKER, acc, win_rate)

    finally:
        if dm.isConnected():
            logger.info("🔌 Disconnecting...")
            dm.disconnect()


if __name__ == "__main__":
    run_simulation()