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

# --- Imports ---
from data_source_manager import DataSourceManager
from stockwise_simulation import ProfessionalStockAdvisor, FeatureCalculator, clean_raw_data

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", stream=sys.stdout)
logger = logging.getLogger("OrchestraSim")

TICKER = "NVDA"
MODEL_DIR = "models/NASDAQ-gen3-dynamic"
LOOK_AHEAD_DAYS = 5


# --- CLASS 1: THE REGIME DETECTOR ---
class MarketRegimeDetector:
    @staticmethod
    def detect_regime(latest_row):
        close = latest_row['close']
        sma_50 = latest_row.get('sma_50', close)
        sma_200 = latest_row.get('sma_200', close)
        adx = latest_row.get('ADX_14', 0)

        if close < sma_200: return "BEARISH"
        if close > sma_200:
            if adx > 25 and close > sma_50:
                return "TRENDING_UP"
            else:
                return "RANGING_BULL"
        return "RANGING_BULL"


# --- CLASS 2: THE STRATEGY ORCHESTRA ---
class StrategyOrchestra:
    @staticmethod
    def get_score(features, regime):
        if regime == "TRENDING_UP":
            return StrategyOrchestra._agent_breakout(features)
        elif regime == "RANGING_BULL":
            # Check for Early Breakout override
            if features['close'] > features.get('recent_high', 99999):
                return StrategyOrchestra._agent_breakout(features)
            return StrategyOrchestra._agent_dip_buyer(features)
        elif regime == "BEARISH":
            return StrategyOrchestra._agent_bear_defense(features)
        return 0

    @staticmethod
    def _agent_breakout(f):
        score = 0
        if f['close'] > f.get('recent_high', 99999): score += 55

        # Filters
        vol = f.get('volume', 0)
        vol_ma = f.get('vol_ma', 1)
        if vol > vol_ma * 1.1:
            score += 15
        else:
            score -= 10

        if f.get('rel_strength_qqq', 0) > 0: score += 10
        if f.get('rsi_14', 50) > 85: score -= 20
        return score

    @staticmethod
    def _agent_dip_buyer(f):
        score = 0
        rsi = f.get('rsi_14', 50)
        if rsi < 40:
            score += 40
        elif rsi < 50:
            score += 20

        sma_50 = f.get('sma_50', 0)
        close = f['close']
        if sma_50 > 0:
            dist = (close - sma_50) / sma_50
            if -0.03 < dist < 0.03: score += 20

        if f.get('ibs', 0.5) < 0.2: score += 15
        return score

    @staticmethod
    def _agent_bear_defense(f):
        if f.get('rsi_14', 50) < 20: return 55
        return 0

    # --- CLASS 3: ROBUST CALCULATOR ---


class RobustFeatureCalculator:
    def __init__(self, data_manager, contextual_data, is_cloud):
        self.qqq_data = contextual_data.get('qqq', pd.DataFrame())

    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if df.empty: return pd.DataFrame()

        try:
            df.columns = [col.lower() for col in df.columns]
            df['sma_50'] = df['close'].rolling(50).mean()
            df['sma_200'] = df['close'].rolling(200).mean()

            try:
                adx = df.ta.adx(length=14)
                if adx is not None: df = pd.concat([df, adx], axis=1)
            except:
                pass

            df['recent_high'] = df['high'].rolling(50).max().shift(1)
            df.ta.atr(length=14, append=True, col_names='atr_14')
            df.ta.rsi(length=14, append=True, col_names='rsi_14')
            df['vol_ma'] = df['volume'].rolling(20).mean()
            df['daily_return'] = df['close'].pct_change()
            df['ibs'] = (df['close'] - df['low']) / ((df['high'] - df['low']) + 1e-9)

            if not self.qqq_data.empty:
                qqq_ret = self.qqq_data['close'].pct_change()
                aligned_qqq = qqq_ret.reindex(df.index).fillna(0)
                df['rel_strength_qqq'] = df['daily_return'] - aligned_qqq
            else:
                df['rel_strength_qqq'] = 0.0

            df.fillna(0, inplace=True)
            return df

        except Exception as e:
            logger.error(f"Calc Error: {e}")
            return pd.DataFrame()


# --- UTILS ---
def load_robust_context_yf():
    import yfinance as yf
    try:
        qqq = yf.download("QQQ", period="2y", progress=False, auto_adjust=True)
        if isinstance(qqq.columns, pd.MultiIndex): qqq.columns = qqq.columns.droplevel(1)
        qqq.columns = [c.lower() for c in qqq.columns]
        return {'qqq': qqq}
    except:
        return {'qqq': pd.DataFrame()}


def generate_regime_chart(df, ticker, global_acc, win_rate):
    output_dir = "debug_charts"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    filename = os.path.join(output_dir, f"{ticker}_Orchestra_Report.html")

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_heights=[0.4, 0.15, 0.15, 0.3],
                        subplot_titles=(
                            f'{ticker} Price & Trades (Candlesticks)',
                            'Active Regime (State)',
                            'System Decision (-1=Down, 0=Wait, 1=Up)',
                            'Orchestra Score & Win Rate'
                        ))

    # --- PANEL 1: CANDLESTICKS (NEW) ---
    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ), row=1, col=1)

    # Add Trades Markers
    wins = df[(df['Prediction'] == 'UP') & (df['Is_Correct'] == True)]
    losses = df[(df['Prediction'] == 'UP') & (df['Is_Correct'] == False)]

    # For markers to appear ABOVE candles, we offset the y-position slightly or use High
    fig.add_trace(go.Scatter(x=wins['Date'], y=wins['High'] * 1.02, mode='markers',
                             marker=dict(color='blue', size=12, symbol='triangle-down'), name='Win'), row=1, col=1)
    fig.add_trace(go.Scatter(x=losses['Date'], y=losses['High'] * 1.02, mode='markers',
                             marker=dict(color='red', size=12, symbol='x'), name='Loss'), row=1, col=1)

    # --- PANEL 2: REGIME STATE ---
    regime_map = {'BEARISH': 0, 'RANGING_BULL': 1, 'TRENDING_UP': 2}
    df['Regime_Val'] = df['Regime'].map(regime_map)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Regime_Val'], mode='lines', name='Regime',
                             line=dict(shape='hv', color='purple', width=2)), row=2, col=1)
    fig.update_yaxes(tickvals=[0, 1, 2], ticktext=["BEAR", "RANGE", "TREND"], row=2, col=1)

    # --- PANEL 3: TRI-STATE PREDICTION ---
    def get_signal_val(score):
        if score >= 50: return 1
        if score <= 30: return -1
        return 0

    df['Signal_Val'] = df['System_Score'].apply(get_signal_val)

    # No Shift - Sync with today
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Signal_Val'],
        mode='lines',
        name='Decision',
        line=dict(shape='hv', color='#00CC96', width=2),
        fill='tozeroy'
    ), row=3, col=1)
    fig.update_yaxes(tickvals=[-1, 0, 1], ticktext=["DOWN", "WAIT", "UP"], row=3, col=1)

    # --- PANEL 4: SCORE & WIN RATE ---
    df['Rolling_Win'] = df['Is_Correct'].rolling(20).mean() * 100
    fig.add_trace(go.Scatter(x=df['Date'], y=df['System_Score'], name='Score', line=dict(color='cyan')), row=4, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Rolling_Win'], name='Win Rate %', line=dict(color='white', dash='dot'),
                             yaxis='y2'), row=4, col=1)
    fig.add_hline(y=50, line_dash="solid", line_color="red", row=4, col=1)

    # Remove Range Slider (It messes up subplots)
    fig.update_layout(xaxis_rangeslider_visible=False)

    title_text = f"Gen-4 Orchestra: {ticker} | Acc: {global_acc:.1f}% | Win Rate: {win_rate:.1f}%"
    fig.update_layout(title=title_text, template="plotly_dark", height=1200)
    fig.write_html(filename)
    logger.info(f"📄 Chart Saved: {filename}")


def run_simulation():
    logger.info(f"🚀 STARTING ORCHESTRA SIMULATION FOR {TICKER}")

    dm = DataSourceManager(use_ibkr=True, allow_fallback=True, port=7497)
    try:
        dm.connect_to_ibkr()
    except:
        pass

    context_data = load_robust_context_yf()

    advisor = ProfessionalStockAdvisor(model_dir=MODEL_DIR, data_source_manager=dm)
    advisor.calculator = RobustFeatureCalculator(dm, context_data, False)
    advisor.log.setLevel(logging.ERROR)

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=1200)
    logger.info("📥 Downloading NVDA Data...")
    df = clean_raw_data(dm.get_stock_data(TICKER, start_date=start_date, end_date=end_date))

    if df.empty: return

    simulation_log = []
    test_idx_start = 250

    logger.info("🏃 Conductor Starting...")
    for i in tqdm(range(test_idx_start, len(df) - LOOK_AHEAD_DAYS)):
        current_date = df.index[i]
        data_slice = df.iloc[:i + 1].copy()

        features_df = advisor.calculator.calculate_all_features(data_slice)
        if features_df.empty: continue
        features = features_df.iloc[-1].to_dict()

        regime = MarketRegimeDetector.detect_regime(features)
        score = StrategyOrchestra.get_score(features, regime)
        pred = "UP" if score >= 50 else "DOWN"

        future_close = df.iloc[i + LOOK_AHEAD_DAYS]['close']
        actual = "UP" if future_close > df.iloc[i]['close'] * 1.005 else "DOWN"
        is_correct = (pred == actual)

        # --- CAPTURE OHLC FOR CANDLESTICKS ---
        simulation_log.append({
            'Date': current_date,
            'Open': df.iloc[i]['open'],
            'High': df.iloc[i]['high'],
            'Low': df.iloc[i]['low'],
            'Close': df.iloc[i]['close'],
            'Regime': regime,
            'System_Score': score,
            'Prediction': pred,
            'Actual_Direction': actual,
            'Is_Correct': is_correct
        })

    res = pd.DataFrame(simulation_log)
    if res.empty: return

    acc = res['Is_Correct'].mean() * 100
    up_trades = res[res['Prediction'] == 'UP']
    win_rate = up_trades['Is_Correct'].mean() * 100 if not up_trades.empty else 0.0

    logger.info(f"\n📊 ORCHESTRA RESULTS")
    logger.info(f"🎯 Global Accuracy: {acc:.2f}%")
    logger.info(f"🚀 System Win Rate: {win_rate:.2f}%")

    generate_regime_chart(res, TICKER, acc, win_rate)


if __name__ == "__main__":
    run_simulation()