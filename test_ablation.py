import pandas as pd
import pandas_ta as ta
try:
    ta.core.settings["verbose"] = False
except Exception:
    pass
import matplotlib.pyplot as plt
import system_config as cfg
from continuous_learning_analyzer import run_simulation
import logging

# Setup simple logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AblationTest")

# Import necessary modules for data loading
from continuous_learning_analyzer import load_data_sequential
from data_source_manager import DataSourceManager

# --- CONFIGURATION ---
# 1. Set the 3-Month Window
START_DATE = "2024-01-01"  # Change to your desired start date
END_DATE = "2025-04-01"  # Change to your desired end date

# 2. Define Features to Test
# Keys are the "Switch Names" used in your _agent_breakout code
ALL_FEATURES = [
    "ema_crossover",
    "macd",
    "adx",
    "kc_breakout",
    "volume",
    "rsi_safety",
    "kalman",    # Gen-7 Feature
    "wavelet",   # Gen-7 Feature
    "lyapunov"   # Gen-7 Feature (if used)
]


def run_ablation_suite():
    results = []

    print(f"--- STARTING ABLATION TEST ({START_DATE} to {END_DATE}) ---")

    # --- 0. PRE-LOAD DATA (Optimization) ---
    print("\n[INFO] Pre-loading data once for all tests...")
    dm = DataSourceManager(use_ibkr=cfg.EN_IBKR, allow_fallback=True) 
    # Use the same load function as the analyzer to ensure consistency
    stock_df, context_data = load_data_sequential(dm, cfg.TARGET_TICKER)
    
    if stock_df.empty:
        print("❌ Error: Could not load data. Aborting.")
        return

    preloaded_bundle = (stock_df, context_data)
    print(f"Data loaded. Stock: {len(stock_df)} rows.")

    # --- TEST 1: BASELINE (All Features On) ---
    print("\nRunning BASELINE (All On)...")
    cfg.DISABLED_FEATURES = []
    # Pass preloaded_data to avoid re-fetching
    stats = run_simulation(start_date=START_DATE, end_date=END_DATE, return_stats=True, preloaded_data=preloaded_bundle)
    stats['Test Type'] = 'Baseline'
    stats['Feature Config'] = 'ALL_ON'
    results.append(stats)

    # --- TEST 2: LEAVE-ONE-OUT (Find the broken feature) ---
    # We disable ONE feature at a time. If PnL goes UP, that feature is bad.
    print("\nRunning LEAVE-ONE-OUT Analysis...")
    for feature in ALL_FEATURES:
        print(f"   Testing without: {feature}")
        cfg.DISABLED_FEATURES = [feature]
        stats = run_simulation(start_date=START_DATE, end_date=END_DATE, return_stats=True, preloaded_data=preloaded_bundle)
        stats['Test Type'] = 'Drop-One'
        stats['Feature Config'] = f"NO_{feature.upper()}"
        results.append(stats)

    # --- TEST 3: STANDALONE (Solo Performance) ---
    # We disable EVERYTHING except one feature.
    print("\nRunning STANDALONE Analysis (Solo Power)...")
    for feature in ALL_FEATURES:
        print(f"   Testing ONLY: {feature}")
        # Create list of all features EXCEPT the current one
        disable_list = [f for f in ALL_FEATURES if f != feature]
        cfg.DISABLED_FEATURES = disable_list

        stats = run_simulation(start_date=START_DATE, end_date=END_DATE, return_stats=True, preloaded_data=preloaded_bundle)
        stats['Test Type'] = 'Standalone'
        stats['Feature Config'] = f"ONLY_{feature.upper()}"
        results.append(stats)

    # --- PROCESS & DISPLAY RESULTS ---
    df_results = pd.DataFrame(results)

    # Sort by PnL to see the best configuration
    df_results = df_results.sort_values(by="Total PnL", ascending=False)

    print("\n" + "=" * 50)
    print("FINAL RESULTS SUMMARY")
    print("=" * 50)
    print(df_results[['Test Type', 'Feature Config', 'Total PnL', 'Win Rate', 'Total Trades']].to_string(index=False))

    # --- PLOT CHART ---
    generate_comparison_chart(df_results)


def generate_comparison_chart(df):
    """Generates a bar chart comparing PnL across tests"""
    plt.figure(figsize=(14, 8))

    # Color coding
    colors = []
    for type_ in df['Test Type']:
        if type_ == 'Baseline':
            colors.append('blue')
        elif type_ == 'Drop-One':
            colors.append('red')  # Removing features
        else:
            colors.append('green')  # Standalone

    bars = plt.bar(df['Feature Config'], df['Total PnL'], color=colors)

    plt.axhline(0, color='black', linewidth=1)
    plt.title(f'System Performance by Feature Configuration ({START_DATE} - {END_DATE})')
    plt.ylabel('Total PnL ($)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}',
                 ha='center', va='bottom' if height > 0 else 'top')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_ablation_suite()