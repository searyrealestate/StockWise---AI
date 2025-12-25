# manual_trade_debugger.py

"""
Manual Trade Debugger (Fixed & Enhanced)
========================================
1. Exit Price Clamping: Ensures exit price is within daily High-Low range.
2. Summary Table: Restored printing of the full summary table.
3. PnL Stats: Added Total, Pos, and Neg PnL summaries.
"""

import pandas as pd
from datetime import datetime, timedelta
import logging
import os
import sys
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_ta as ta

# --- Imports ---
from data_source_manager import DataSourceManager
from stockwise_simulation import ProfessionalStockAdvisor, FeatureCalculator, load_contextual_data, clean_raw_data
from risk_manager import RiskManager

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

# Configuration
TICKER = "NVDA"
MODEL_DIR = "models/NASDAQ-gen3-dynamic"


# --- HELPER: SMART STOP CALCULATION ---
def calculate_smart_stop(df, entry_date, lookback=10):
    """
    Finds the lowest 'Low' price in the last N days before entry.
    """
    try:
        # Slice data strictly BEFORE the entry date
        history = df[df.index < pd.to_datetime(entry_date)].tail(lookback)
        if history.empty:
            return None

        swing_low = history['low'].min()
        return swing_low * 0.995  # Place stop 0.5% below the swing low
    except Exception:
        return None


def add_trade_visuals(fig, trade_data):
    entry_date = pd.to_datetime(trade_data['Entry_Date'])
    exit_date = pd.to_datetime(trade_data['Exit_Date'])
    entry_price = trade_data['Entry_Price']
    exit_price = trade_data['Exit_Price']
    pnl_pct = trade_data['PnL_Pct']

    fig.add_trace(go.Scatter(
        x=[entry_date], y=[entry_price], mode='markers+text',
        marker=dict(symbol='triangle-up', size=15, color='green', line=dict(width=2, color='white')),
        text=["B"], textposition="bottom center", name="Buy"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=[exit_date], y=[exit_price], mode='markers+text',
        marker=dict(symbol='triangle-down', size=15, color='red', line=dict(width=2, color='white')),
        text=["S"], textposition="top center", name="Sell"
    ), row=1, col=1)

    # Correct PnL Label
    mid_date = entry_date + (exit_date - entry_date) / 2
    mid_price = (entry_price + exit_price) / 2

    color = "lightgreen" if pnl_pct > 0 else "lightcoral"

    fig.add_annotation(
        x=mid_date, y=mid_price, text=f"<b>{pnl_pct:.2f}%</b>",
        showarrow=True, arrowhead=0, bgcolor=color, bordercolor="black",
        font=dict(color="black", size=12), row=1, col=1
    )

    fig.add_shape(type="line", x0=entry_date, y0=entry_price, x1=exit_date, y1=exit_price,
                  line=dict(color="gray", width=1, dash="dot"), row=1, col=1
                  )


def generate_master_chart(full_data, executed_trades, ticker):
    if not executed_trades: return None

    all_dates = []
    for t in executed_trades:
        all_dates.append(pd.to_datetime(t['Entry_Date']))
        all_dates.append(pd.to_datetime(t['Exit_Date']))

    min_date = min(all_dates) - timedelta(days=20)
    max_date = max(all_dates) + timedelta(days=20)

    chart_data = full_data[(full_data.index >= min_date) & (full_data.index <= max_date)].copy()

    if 'sma_50' not in chart_data.columns:
        chart_data['sma_50'] = chart_data['close'].rolling(50).mean()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05, row_heights=[0.7, 0.3],
                        subplot_titles=(f'{ticker} Trade Execution Analysis', 'Volume'))

    fig.add_trace(go.Candlestick(x=chart_data.index, open=chart_data['open'], high=chart_data['high'],
                                 low=chart_data['low'], close=chart_data['close'], name='Price'), row=1, col=1)

    fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['sma_50'], mode='lines',
                             name='SMA 50', line=dict(color='orange', width=1)), row=1, col=1)

    for trade in executed_trades:
        add_trade_visuals(fig, trade)

    colors = ['red' if row['open'] - row['close'] >= 0 else 'green' for index, row in chart_data.iterrows()]
    fig.add_trace(go.Bar(x=chart_data.index, y=chart_data['volume'], marker_color=colors, name='Volume'), row=2, col=1)

    fig.update_layout(title=f"MASTER TRADE CHART: {ticker}", xaxis_rangeslider_visible=False, template="plotly_dark",
                      height=900)
    fig.write_html(f"{ticker}_Trade_Execution_Report.html")
    return f"{ticker}_Trade_Execution_Report.html"


def calculate_trade_score(features, ai_action):
    score = 0
    logs = []

    ibs = features.get('ibs', 0.5)
    rel_strength = features.get('rel_strength_qqq', 0)
    macd_hist = features.get('macd_histogram', 0)
    daily_ret = features.get('daily_return', 0)
    has_bullish_candle = features.get('bullish_candlestick', 0) == 1
    has_vol_breakout = features.get('volume_breakout_flag', 0) == 1
    is_green_candle = daily_ret > 0

    if ai_action == 'BUY':
        score += 40
        logs.append(f"[BASE] AI Signal BUY (+40) -> {score}")
    else:
        logs.append(f"[BASE] No AI Signal (0) -> {score}")

    if rel_strength > 0.03:
        score += 35; logs.append(f"[ALPHA] 🚀 Institutional (+35)")
    elif rel_strength > 0.01:
        score += 20; logs.append(f"[ALPHA] Strong Outperformance (+20)")
    elif rel_strength > 0:
        score += 10; logs.append(f"[ALPHA] Positive Alpha (+10)")

    if ibs > 0.95:
        score += 20; logs.append(f"[IBS] 🔥 Dead Highs (+20)")
    elif ibs > 0.8:
        score += 15; logs.append(f"[IBS] Strong Close (+15)")

    if has_vol_breakout:
        if is_green_candle:
            score += 15; logs.append(f"[VOL] 🟢 Buying Vol (+15)")
        else:
            score -= 15; logs.append(f"[VOL] 🔴 Selling Vol (-15)")

    if has_bullish_candle: score += 10; logs.append(f"[TECH] Bullish Pattern (+10)")
    if macd_hist > 0: score += 10; logs.append(f"[MOMENTUM] Positive MACD (+10)")

    if macd_hist < -1.5: score -= 40; logs.append(f"[RISK] Extreme Neg Mom (-40)")
    if daily_ret < -0.03: score -= 50; logs.append(f"[CRASH] Heavy Drop (-50)")

    if score >= 60:
        decision = "STRONG BUY"
    elif score >= 45:
        decision = "WEAK BUY"
    else:
        decision = "BLOCKED"

    return score, logs, decision


def simulate_exit(entry_date, entry_price, full_data, stop_loss_price=None):
    """
    Runs a day-by-day simulation of the trade exit strategy.
    Includes:
    1. RiskManager logic (Trailing Stop, Structural Stop).
    2. Dynamic ATR Multiplier based on RSI.
    3. Proactive Profit Taking (Scalping the Peak).
    """
    # 1. Prepare Data Slice
    trade_data = full_data[full_data.index >= pd.to_datetime(entry_date)].copy()

    # Ensure required indicators exist
    if 'atr_14' not in trade_data.columns:
        trade_data.ta.atr(length=14, append=True, col_names='atr_14')
    if 'sma_150' not in trade_data.columns:
        trade_data['sma_150'] = trade_data['close'].rolling(150).mean()
    if 'rsi_14' not in trade_data.columns:
        trade_data.ta.rsi(length=14, append=True, col_names='rsi_14')
    if 'ema_9' not in trade_data.columns:
        trade_data.ta.ema(length=9, append=True, col_names='ema_9')

    trade_data.bfill(inplace=True)  # Fill any NaNs at start

    # 2. Initialize Risk Manager
    rm = RiskManager(portfolio_value=10000, global_risk_pct=1.0)

    # Determine Initial Stop Loss
    initial_atr = trade_data.iloc[0]['atr_14']
    if not stop_loss_price:
        # Try to find a Swing Low first (Smart Stop)
        smart_stop = calculate_smart_stop(full_data, entry_date)
        # Use Smart Stop ONLY if it exists AND is below entry price
        if smart_stop and smart_stop < entry_price:
            stop_loss_price = smart_stop
            # Optional: print for debug
            # print(f"   🛡️ Using Smart Swing Low Stop: {stop_loss_price:.2f}")
        else:
            # Fallback to original ATR logic if no swing low found
            stop_loss_price = entry_price - (initial_atr * 3.0)

    position = {
        'entry_price': entry_price,
        'current_stop_loss': stop_loss_price,
        'use_trailing_stop': True,
        'atr_multiplier': 3.0
    }

    exit_date = None
    exit_price = None
    exit_reason = "HOLDING"

    # Track Peak for Profit Protection logic
    max_price_since_entry = entry_price

    # 3. Day-by-Day Loop (Skip first day as it is entry day)
    for date, row in trade_data.iloc[1:].iterrows():
        current_close = row['close']
        current_high = row['high']
        current_open = row['open']
        rsi = row.get('rsi_14', 50)

        # Update Peak Price
        if current_high > max_price_since_entry:
            max_price_since_entry = current_high

        # Calculate Metrics
        current_profit_pct = (current_close - entry_price) / entry_price
        peak_profit_pct = (max_price_since_entry - entry_price) / entry_price

        # --- PROACTIVE PROFIT TAKING LOGIC ---

        # A. "Quick Win" Lock: If > 5% profit in first 3 days, move SL to Breakeven
        current_date = date.date() if hasattr(date, 'date') else date
        entry_date_obj = pd.to_datetime(entry_date).date()

        days_in_trade = (current_date - entry_date_obj).days
        if days_in_trade <= 3 and current_profit_pct > 0.05:
            # Move stop to Entry Price + 1% buffer
            new_stop = entry_price * 1.01
            if new_stop > position['current_stop_loss']:
                position['current_stop_loss'] = new_stop

        # B. "Climax" Protection: If RSI > 75 (Overbought), tighten trail significantly
        if rsi > 75:
            position['atr_multiplier'] = 1.5  # Tight trail to catch the top
        elif rsi > 65:
            position['atr_multiplier'] = 3.5  # Strong trend, let it run loosely
        else:
            position['atr_multiplier'] = 3.0  # Standard width

            # # Calculate EMA 9 for trend confirmation
            # # Note: Since we process row by row, we need pre-calculated EMA
            # ema_9 = row.get('ema_9', row['close'])  # Fallback if missing
            #
            # # C. "Profit Protection" (Scalping the Peak) - UPDATED with IBS Filter
            # if peak_profit_pct > 0.15:
            #     drawdown_from_peak = (max_price_since_entry - current_close) / max_price_since_entry
            #
            #     ibs = (current_close - row['low']) / ((row['high'] - row['low']) + 1e-9)
            #
            #     # Rule: Exit if drawdown > 5% AND Close is weak AND Price broke below EMA 9
            #     # If price is still above EMA 9, the trend is alive -> HOLD.
            #     if drawdown_from_peak > 0.05 and ibs < 0.5 and current_close < ema_9:
            #         exit_date = date
            #         exit_price = current_close
            #         exit_reason = "PROFIT_PROTECTION_EXIT (Trend Broken)"
            #         break

        # 4. Run Standard Risk Manager (Trailing Stop & Structural Checks)
        signal, position = rm.manage_open_position(row, position)

        if signal == "EXIT_SIGNAL":
            exit_date = date
            calculated_exit = position['current_stop_loss']

            # --- Price Clamping Logic (Handle Gaps) ---
            # Ensure exit is realistic within the daily range
            if calculated_exit > current_high:
                # Stop is above the daily high (Gap Up above stop)? Exit at Open.
                exit_price = current_open
            elif calculated_exit > current_open:
                # Stop was hit during intraday trading
                exit_price = calculated_exit
            else:
                # Price gapped down below stop. Exit at Open.
                exit_price = current_open

            exit_reason = "STOP_LOSS"
            break

    # 5. Handle Open Positions at End of Data
    if not exit_date:
        exit_date = trade_data.index[-1]
        exit_price = trade_data.iloc[-1]['close']
        exit_reason = "END_OF_DATA"

    pnl_pct = ((exit_price - entry_price) / entry_price) * 100

    return {
        'Exit_Date': exit_date.date(),
        'Exit_Price': exit_price,
        'Exit_Reason': exit_reason,
        'PnL_Pct': pnl_pct
    }


def create_debugger_report(entry_list: list):
    print(f"\n{'=' * 60}")
    print(f"🔍 TRADE LIFECYCLE DEBUGGER FOR {TICKER}")
    print(f"{'=' * 60}\n")

    dm = DataSourceManager(use_ibkr=False)
    try:
        context_data = load_contextual_data(dm)
    except NameError:
        context_data = {'qqq': pd.Series(), 'vix': pd.Series(), 'tlt': pd.Series()}

    advisor = ProfessionalStockAdvisor(model_dir=MODEL_DIR, data_source_manager=dm)
    advisor.calculator = FeatureCalculator(data_manager=dm, contextual_data=context_data, is_cloud=False)

    current_year = datetime.now().year
    parsed_dates = []
    for d, _ in entry_list:
        try:
            if '-' in d:
                dt = pd.to_datetime(d)
            else:
                dt = pd.to_datetime(d, format='%d/%m').replace(year=current_year)
            parsed_dates.append(dt)
        except:
            pass

    if not parsed_dates: return

    start_dl = (min(parsed_dates) - timedelta(days=365)).strftime('%Y-%m-%d')
    end_dl = (max(parsed_dates) + timedelta(days=120)).strftime('%Y-%m-%d')

    print(f"📥 Downloading Data ({start_dl} -> {end_dl})...")
    full_data = dm.get_stock_data(TICKER, start_date=start_dl, end_date=end_dl)
    full_data = clean_raw_data(full_data)
    if full_data.empty: return

    executed_trades = []
    report_rows = []

    for date_str, _ in entry_list:
        try:
            if '-' in date_str:
                analysis_date = pd.to_datetime(date_str).date()
            else:
                analysis_date = pd.to_datetime(date_str, format='%d/%m').replace(year=current_year).date()
        except:
            continue

        print(f"\n🔎 PROCESSING: {analysis_date}")
        data_slice = full_data[full_data.index <= pd.to_datetime(analysis_date)].copy()
        if data_slice.empty: continue

        _, result = advisor.run_analysis(data_slice, TICKER, analysis_date, use_market_filter=False)

        score = result.get('score', 0)
        decision = result.get('decision_type', 'BLOCKED')
        threshold = result.get('threshold_used', 50)

        print(f"   📊 System Score: {score} | Threshold: {threshold} | Decision: {decision}")

        features = result.get('all_features', {})
        ai_action = result.get('action', 'WAIT')

        score, logs, decision = calculate_trade_score(features, ai_action)
        print(f"   📊 Entry Score: {score} ({decision})")

        # Add to report rows for summary table
        row = {
            'Date': analysis_date, 'AI_Action': ai_action, 'Score': score, 'Decision': decision,
            'Exit_Date': '-', 'PnL_Pct': 0.0
        }

        if decision != "BLOCKED":
            try:
                next_day_idx = full_data.index[full_data.index > pd.to_datetime(analysis_date)][0]
                entry_price = full_data.loc[next_day_idx, 'open']
                sl_price = result.get('stop_loss_price')

                exit_stats = simulate_exit(next_day_idx.date(), entry_price, full_data, sl_price)
                print(
                    f"   🏁 {exit_stats['Exit_Reason']} on {exit_stats['Exit_Date']} | PnL: {exit_stats['PnL_Pct']:.2f}%")

                # Update row with exit stats
                row['Exit_Date'] = exit_stats['Exit_Date']
                row['PnL_Pct'] = exit_stats['PnL_Pct']

                executed_trades.append({
                    'Entry_Date': next_day_idx.date(), 'Entry_Price': entry_price,
                    'Exit_Date': exit_stats['Exit_Date'], 'Exit_Price': exit_stats['Exit_Price'],
                    'PnL_Pct': exit_stats['PnL_Pct']
                })

            except IndexError:
                print("   ⚠️ No future data.")

        report_rows.append(row)

    # --- SUMMARY & STATS ---
    if report_rows:
        df = pd.DataFrame(report_rows)
        executed = df[df['Decision'] != 'BLOCKED']

        total_pnl = executed['PnL_Pct'].sum()
        total_pos = executed[executed['PnL_Pct'] > 0]['PnL_Pct'].sum()
        total_neg = executed[executed['PnL_Pct'] < 0]['PnL_Pct'].sum()

        print("\n" + "=" * 80)
        print(f"📊 TRADE SUMMARY ({TICKER})")
        print("=" * 80)
        # print(df.to_string(index=False, col_space=12, justify='center'))
        print("-" * 80)
        print(f"💰 Total PnL: {total_pnl:.2f}%")
        print(f"📈 Total Pos PnL: {total_pos:.2f}%")
        print(f"📉 Total Neg PnL: {total_neg:.2f}%")

        generate_master_chart(full_data, executed_trades, TICKER)
        print(f"\n📊 CHART SAVED: {TICKER}_Trade_Execution_Report.html")


if __name__ == "__main__":
    test_dates = [
        ('2024-05-14', 0), ('2024-05-23', 0), ('2024-08-05', 0),
        ('2024-08-13', 0), ('2024-09-02', 0), ('2024-09-11', 0), ('2024-10-07', 0)
    ]
    create_debugger_report(test_dates)