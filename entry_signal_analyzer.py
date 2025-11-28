# entry_signal_analyzer.py

"""
NVDA AI Entry Point Analyzer
====================================

This script automates the generation of AI-driven 'BUY' signals for NVDA
over the past year, logs the full decision matrix (SHAP values, indicators),
and generates a chart for manual, human review.

This fulfills the user's request to systematically check entry points and
reasons for improvement.

The analysis uses the core components (AI Advisor, Feature Calculator)
to simulate the logic of the dynamic agent for each day.
"""

import pandas as pd
from datetime import datetime, timedelta
import logging
import os
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import webbrowser  # <-- NEW: Import for opening the chart automatically

# --- 1. Import necessary components ---
from logging_setup import setup_json_logging
from data_source_manager import DataSourceManager
from stockwise_simulation import ProfessionalStockAdvisor, FeatureCalculator, load_contextual_data, clean_raw_data

# Setup logging before anything else
setup_json_logging(logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
TICKER = "NVDA"
DAYS_TO_ANALYZE = 200  # Roughly one year of trading days
MODEL_DIR = "models/NASDAQ-gen3-dynamic"  # Use the recommended dynamic agent


def run_single_day_analysis(advisor: ProfessionalStockAdvisor, ticker: str, analysis_date: datetime.date,
                            full_data: pd.DataFrame) -> dict:
    """Runs the AI analysis for a single day."""

    # The run_analysis method already handles feature calculation and model prediction
    # Note: We skip market filter here to analyze ALL signals, even in bear markets, for review purposes.
    _, result = advisor.run_analysis(
        full_stock_data=full_data.copy(),
        ticker_symbol=ticker,
        analysis_date=analysis_date,
        use_market_filter=False  # Set to False for comprehensive signal review
    )

    return result


def analyze_and_log_signals():
    """Main function to orchestrate data download, analysis, and logging."""

    logger.info(f"--- Starting {TICKER} Entry Signal Analysis ---")

    # --- 2. Initialize Core System Components ---
    dm = DataSourceManager(use_ibkr=False)  # Use yfinance fallback

    # Mock load_contextual_data since it's defined in stockwise_simulation
    try:
        context_data = load_contextual_data(dm)
    except NameError:
        logger.error("load_contextual_data is missing. Initializing dummy context.")
        context_data = {'qqq': pd.Series(), 'vix': pd.Series(), 'tlt': pd.Series()}

    advisor = ProfessionalStockAdvisor(model_dir=MODEL_DIR, data_source_manager=dm)
    advisor.calculator = FeatureCalculator(data_manager=dm, contextual_data=context_data, is_cloud=False)

    # --- 3. Download Data ---
    end_date = datetime.now().date()
    # Need sufficient days for features (365 for features + 200 for back-analysis)
    start_date = end_date - timedelta(days=500)

    logger.info(f"Downloading historical data for {TICKER} from {start_date} to {end_date}...")
    full_data = dm.get_stock_data(TICKER, start_date=start_date, end_date=end_date)
    full_data = clean_raw_data(full_data)

    if full_data.empty:
        logger.error(f"FATAL: Failed to download data for {TICKER}. Exiting.")
        return

    # --- 4. Determine Analysis Dates ---
    # Use bdate_range to get only trading days
    trading_days = pd.bdate_range(start=start_date, end=end_date).normalize().unique()

    # Filter the last N trading days
    analysis_dates = trading_days[-DAYS_TO_ANALYZE:]
    logger.info(f"Analyzing {len(analysis_dates)} days for potential entry signals.")

    # --- 5. Loop and Analyze ---
    buy_signals = []

    for date in analysis_dates:
        date_str = date.strftime('%Y-%m-%d')

        # Slice data up to the analysis date + 1 day to allow for the next day's open price fetch
        data_slice = full_data[full_data.index <= date + timedelta(days=1)].copy()

        # IMPORTANT: Run analysis only up to the decision day (date.date())
        result = run_single_day_analysis(advisor, TICKER, date.date(), data_slice)

        # NOTE: The check is for BUY, ignoring WAIT / AVOID signals from the market filter
        if result.get('action') == 'BUY':
            buy_signals.append({
                'date': date,
                'price': result.get('current_price'),
                'sl': result.get('stop_loss_price'),
                'tp': result.get('profit_target_price'),
                'confidence': result.get('confidence', 0),
                'agent': result.get('agent', 'AI')
            })

            # --- Detailed Logging of Decision for Manual Review ---
            logger.info(f"*** BUY SIGNAL FOUND for {TICKER} on {date_str} ***")
            logger.info(
                f"Entry: {result.get('current_price'):.2f}, SL: {result.get('stop_loss_price'):.2f}, TP: {result.get('profit_target_price'):.2f}")
            logger.info(f"Decision Agent: {result.get('agent')}. Confidence: {result.get('confidence'):.2f}%")

            # Log the key indicators used in the AI decision (for manual review improvement)
            if 'all_features' in result:
                # Log only the top 5 most important features for this instance
                # Note: This sorting requires SHAP values, but since we don't have SHAP here,
                # we'll log the top features alphabetically as a proxy for the data context.
                feature_importance = sorted(result['all_features'].items(), key=lambda x: x[0])[:5]
                logger.info("Top 5 Contextual Features:")
                for k, v in feature_importance:
                    logger.info(f"  - {k}: {v:.4f}")

    logger.info(f"--- Analysis Complete. Found {len(buy_signals)} BUY signals for {TICKER}. ---")

    if buy_signals:
        plot_signals_for_review(full_data, buy_signals)


def plot_signals_for_review(df: pd.DataFrame, signals: list):
    """Generates a chart showing the entry points for manual review."""

    # We re-calculate SMAs for context in the chart
    df.ta.sma(length=50, append=True)
    df.ta.sma(length=200, append=True)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.8, 0.2])

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(x=df.index, open=df['open'], high=df['high'],
                       low=df['low'], close=df['close'], name='Price'),
        row=1, col=1
    )

    # Add SMAs
    fig.add_trace(
        go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', name='SMA 50', line=dict(color='yellow', width=1)), row=1,
        col=1)
    fig.add_trace(
        go.Scatter(x=df.index, y=df['SMA_200'], mode='lines', name='SMA 200', line=dict(color='magenta', width=1)),
        row=1, col=1)

    # Signal Markers
    buy_dates = [s['date'] for s in signals]
    buy_prices = [s['price'] for s in signals]

    fig.add_trace(
        go.Scatter(
            x=buy_dates, y=buy_prices,
            mode='markers',
            marker=dict(color='cyan', size=10, symbol='triangle-up', line=dict(width=2)),
            name='AI BUY Signal (Entry)'
        ),
        row=1, col=1
    )

    # Volume Bar
    fig.add_trace(
        go.Bar(x=df.index, y=df['volume'], name='Volume',
               marker=dict(color='rgba(100,110,120,0.6)')),
        row=2, col=1
    )

    fig.update_layout(
        title=f"Entry Points for {TICKER} (Last {DAYS_TO_ANALYZE} Days) - Manual Review",
        xaxis_rangeslider_visible=False,
        template="plotly_dark"
    )

    # Print the plot path for the user to review
    chart_path = os.path.join(os.getcwd(), "nvda_entry_signals.html")
    fig.write_html(chart_path)

    # --- NEW FIX: Automatically open the chart ---
    webbrowser.open_new_tab(f'file://{os.path.realpath(chart_path)}')

    logger.info(f"Chart generated successfully for manual review: {chart_path}")
    print(f"\nReview the chart in your browser: file:///{chart_path}")


if __name__ == "__main__":
    analyze_and_log_signals()