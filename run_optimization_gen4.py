import logging
import sys
import time
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

# --- Import your existing modules ---
from data_source_manager import DataSourceManager
from mico_optimizer_gen4 import run_full_optimization
from mico_system import MichaAdvisor
from trading_models import (
    MeanReversionAdvisor,
    BreakoutAdvisor,
    SuperTrendAdvisor,
    MovingAverageCrossoverAdvisor,
    VolumeMomentumAdvisor
)
from utils import clean_raw_data

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", stream=sys.stdout)
logger = logging.getLogger("OptimizationRunner")


def download_data_with_progress(dm, symbol, start_date, end_date):
    logger.info(f"📥 Starting chunked download for {symbol} ({start_date} -> {end_date})...")
    current_date = start_date
    chunks = []
    diff = relativedelta(end_date, start_date)
    total_months = diff.years * 12 + diff.months + 1

    with tqdm(total=total_months, desc=f"Downloading {symbol}", unit="month") as pbar:
        while current_date < end_date:
            next_date = current_date + relativedelta(months=1)
            if next_date > end_date: next_date = end_date

            # Robust fetch with fallback is handled inside dm.get_stock_data
            df_chunk = dm.get_stock_data(symbol, start_date=current_date, end_date=next_date)

            if not df_chunk.empty: chunks.append(df_chunk)
            current_date = next_date
            pbar.update(1)
            time.sleep(0.1)

    if not chunks: return pd.DataFrame()

    logger.info("🔄 Merging and cleaning data...")
    full_df = pd.concat(chunks)
    full_df = full_df[~full_df.index.duplicated(keep='first')]
    full_df.sort_index(inplace=True)

    # Ensure columns are lowercase for technical analysis lib
    full_df.columns = [c.lower() for c in full_df.columns]

    return clean_raw_data(full_df)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 STOCKWISE OPTIMIZATION SUITE: NVDA (Gen-4)")
    print("=" * 60 + "\n")

    # 1. Initialize Data Manager (Explicit Port 7497)
    dm = DataSourceManager(use_ibkr=True, allow_fallback=True, host='127.0.0.1', port=7497)

    print("🔌 Connecting to Data Source...")
    if not dm.isConnected():
        if dm.connect_to_ibkr():
            print("✅ Connected to IBKR TWS")
        else:
            print("⚠️ IBKR Failed. Using Fallback (Alpaca/YF)")

    # 2. Initialize Advisors
    try:
        mico_advisor = MichaAdvisor(data_manager=dm)
        mean_rev_advisor = MeanReversionAdvisor(data_manager=dm)
        breakout_advisor = BreakoutAdvisor(data_manager=dm)
        supertrend_advisor = SuperTrendAdvisor(data_manager=dm)
        ma_cross_advisor = MovingAverageCrossoverAdvisor(data_manager=dm)
        vol_mom_advisor = VolumeMomentumAdvisor(data_manager=dm)
    except Exception as e:
        logger.error(f"Failed to initialize advisors: {e}")
        sys.exit(1)

    optimizer_advisors = {
        "MichaAdvisor": mico_advisor,
        "SuperTrend": supertrend_advisor,
        "MeanReversion": mean_rev_advisor,
        "Breakout": breakout_advisor,
        "MACrossover": ma_cross_advisor
    }

    # 3. Parameter Grid (Refined for NVDA)
    parameter_grids = {
        "MichaAdvisor": {
            'sma_short': [20, 50], 'sma_long': [100, 150, 200],
            'rsi_threshold': [65, 70, 75], 'atr_mult_stop': [2.5, 3.0],
            'min_conditions_to_buy': [3, 4]
        },
        "SuperTrend": {'length': [10, 14], 'multiplier': [2.5, 3.0, 4.0]},
        "MeanReversion": {'bb_length': [20], 'rsi_oversold': [25, 30, 35]},
        "Breakout": {'breakout_window': [20, 50]},
        "MACrossover": {'short_window': [10, 20], 'long_window': [50, 100]}
    }

    # 4. Download Data with WARMUP PERIOD
    target_symbol = "NVDA"
    end_date = datetime.now().date()

    # Optimization Period: Last 2 Years
    optimization_start_date = end_date - timedelta(days=730)

    # Data Fetch Period: Last 3.5 Years (Provides 1.5 years warmup for SMA200)
    fetch_start_date = end_date - timedelta(days=1200)

    logger.info(f"📅 Optimization Period: {optimization_start_date} -> {end_date}")
    logger.info(f"📥 Fetching data from {fetch_start_date} (to allow indicator warmup)...")

    stock_data = download_data_with_progress(dm, target_symbol, fetch_start_date, end_date)

    if stock_data.empty:
        logger.error("❌ Critical: No data fetched. Aborting.")
        sys.exit(1)

    print(f"✅ Data Ready: {len(stock_data)} rows loaded.")

    # 5. Run Optimization
    print("\n⚙️ Starting Optimization Engine...")

    try:
        best_params_df = run_full_optimization(
            optimizer_advisors=optimizer_advisors,
            parameter_grids=parameter_grids,
            symbol=target_symbol,
            start_date=optimization_start_date,  # Testing starts here
            end_date=end_date,
            pre_fetched_data=stock_data  # Data includes warmup before start_date
        )

        # --- SAVE TO JSON ---
        output_dir = os.path.join("models", "Gen-4")
        os.makedirs(output_dir, exist_ok=True)

        json_filename = f"optimization_results_{target_symbol}.json"
        json_path = os.path.join(output_dir, json_filename)

        if best_params_df is not None and not best_params_df.empty:
            # Save as JSON
            best_params_df.to_json(json_path, orient='records', indent=4)

            # Also save as CSV for easy reading
            csv_path = json_path.replace('.json', '.csv')
            best_params_df.to_csv(csv_path, index=False)

            print(f"\n🏆 Optimization Complete!")
            print(f"📂 Results saved to: {os.path.abspath(output_dir)}")
            print(f"📄 JSON: {json_filename}")
            print(f"📄 CSV:  {os.path.basename(csv_path)}")
        else:
            print("\n⚠️ Optimization finished but returned no results.")
            print("   (This likely means no parameter combination generated a positive profit during the test period.)")

    except KeyboardInterrupt:
        print("\n⚠️ Optimization interrupted by user.")
    except Exception as e:
        logger.error(f"Optimization run failed: {e}", exc_info=True)

    # Cleanup
    if dm.isConnected(): dm.disconnect()
    time.sleep(1)