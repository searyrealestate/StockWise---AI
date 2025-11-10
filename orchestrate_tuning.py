"""
Comprehensive HTML Report Generator
===================================

This script serves as the final step in the version testing pipeline. It
gathers all the compiled results and visual artifacts for a specific version
test and consolidates them into a single, professional, self-contained HTML report.

The resulting report provides a complete overview of a version's performance,
from data generation metrics to final backtest results, making it easy to
analyze, share, and archive.

How it Works:
-------------
1.  **Generates Visuals**: It first runs the `visualize_results.py` script to
    ensure all comparison charts (Sharpe Ratio, Max Drawdown, Equity Curves)
    are created and up-to-date.
2.  **Loads Master Data**: It reads all the data sheets from the master Excel
    workbook (`Master_Test_Results.xlsx`).
3.  **Filters by Version**: It filters the data from each sheet to isolate the
    results corresponding to the `--version-id` provided by the user.
4.  **Creates HTML Tables**: The filtered data for each stage (Data Generation,
    Tuning, Evaluation, Backtesting) is converted into formatted HTML tables.
5.  **Embeds Charts**: It reads the generated PNG chart images, encodes them
    into base64 strings, and embeds them directly into the HTML file. This
    makes the final report fully self-contained.
6.  **Populates Template**: It injects the version information, change description,
    HTML tables, and embedded charts into a master HTML template.

Output:
-------
A single HTML file named `reports/StockWise_Report_<version_id>.html` that
can be viewed in any web browser without needing any external files.

Usage:
------
The script is run from the command line after the `results_compiler.py` has
been run for all stages of a version test.

    python generate_report.py --version-id <version_id>

Example:
    python generate_report.py --version-id "v3.2.0"
"""


import os
import json
import glob
import joblib
import logging
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm
import itertools
import concurrent.futures
from Create_parquet_file_NASDAQ import apply_triple_barrier
import numpy as np
from datetime import datetime
import sys


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s")
logger = logging.getLogger("Gen3TunerOrchestrator")

# --- Add File Logging ---
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_file_path = os.path.join(LOG_DIR, f"tuner_run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] (%(name)s) %(message)s'))
logger.addHandler(file_handler)

# --- Configuration ---
AGENT_DATA_DIRS = {
    '1pct': "models/NASDAQ-training set/features/1per_profit",
    '2pct': "models/NASDAQ-training set/features/2per_profit",
    '3pct': "models/NASDAQ-training set/features/3per_profit",
    '4pct': "models/NASDAQ-training set/features/4per_profit",
    'dynamic': "models/NASDAQ-training set/features/dynamic_profit"
}
# Define base options for menus
ALL_AGENTS = list(AGENT_DATA_DIRS.keys())
ALL_MODEL_TYPES = ['entry', 'profit_take', 'cut_loss']
ALL_CLUSTERS = ['low', 'mid', 'high']


def run_single_study(agent_name: str, model_type: str, cluster: str, n_trials=100):
    """
    Runs an Optuna hyperparameter tuning study for one specialist model,
    now including the Triple Barrier labeling parameters.
    """

    # --- Data loading logic is now INSIDE the worker function ---
    logger.info(f"Worker for [{agent_name}/{model_type}/{cluster}] loading its own data...")
    data_dir = AGENT_DATA_DIRS[agent_name]
    file_pattern = os.path.join(data_dir, '*_daily_context.parquet')
    all_files = glob.glob(file_pattern)

    if not all_files:
        logger.error(f"Worker [{agent_name}] found no data files. Terminating.")
        return

    df_list = [pd.read_parquet(f) for f in all_files]
    df = pd.concat(df_list, ignore_index=True)

    target_map = {
        'entry': 'target_entry',
        'profit_take': 'target_profit_take',
        'cut_loss': 'target_cut_loss'
    }
    target_col = target_map[model_type]

    logger.info(f"\n--- Starting Optuna study for [{agent_name} / {model_type} / {cluster}] ---")
    logger.info(f"Tuning both model and labeling parameters for: '{target_col}'")

    cluster_df = df[df['volatility_cluster'] == cluster].copy()

    # Base feature columns (excluding all potential labels)
    base_feature_cols = [col for col in df.columns if 'target' not in col and col != 'volatility_cluster']

    if cluster_df.empty:
        logger.warning(f"No data for cluster {cluster}. Skipping study.")
        return

    def objective(trial):
        # --- 1. Suggest Labeling Hyperparameters ---
        profit_take_mult = trial.suggest_float('profit_take_mult', 1.0, 4.0)
        stop_loss_mult = trial.suggest_float('stop_loss_mult', 1.5, 4.0)
        time_limit_bars = trial.suggest_int('time_limit_bars', 5, 25)  # In days for daily data
        rsi_overbought_threshold = trial.suggest_int('rsi_overbought_threshold', 65, 80)

        # --- 2. Re-generate Labels On-The-Fly ---
        temp_df = cluster_df.copy()
        tb_labels = apply_triple_barrier(
            close_prices=temp_df['close'],
            high_prices=temp_df['high'],
            low_prices=temp_df['low'],
            atr=temp_df['atr_14'],
            profit_take_mult=profit_take_mult,
            stop_loss_mult=stop_loss_mult,
            time_limit_bars=time_limit_bars,
            profit_mode='dynamic'
        )

        # Re-create all three target labels based on the new tb_labels
        temp_df['target_entry'] = np.where(tb_labels == 1, 1, 0)
        temp_df['target_cut_loss'] = np.where(tb_labels == -1, 1, 0)

        # Check if rsi_14 exists, which it should from base_feature_cols
        if 'rsi_14' in temp_df.columns:
            overbought_condition = temp_df['rsi_14'] > rsi_overbought_threshold
            temp_df['target_profit_take'] = np.where((tb_labels == 1) & (overbought_condition), 1, 0)
        else:
            # Fallback if rsi_14 is missing (should not happen)
            temp_df['target_profit_take'] = np.where(tb_labels == 1, 1, 0)

        # --- 3. Prepare Data with New Labels ---
        X = temp_df[base_feature_cols]
        y = temp_df[target_col]  # 'target_col' is correctly set by the outer function

        if y.nunique() < 2 or y.sum() < 10:
            return 0.0  # Return a score of 0 if the new labels are unusable

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # --- 4. Suggest Model Hyperparameters (as before) ---
        model_params = {
            'objective': 'binary', 'metric': 'binary_logloss', 'n_estimators': 1000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'verbose': -1, 'n_jobs': 1, 'seed': 42, 'class_weight': 'balanced'
        }

        model = lgb.LGBMClassifier(**model_params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        preds = model.predict(X_val)
        return f1_score(y_val, preds, zero_division=0)

    # Create a unique name for the study, e.g., "[4pct/entry/low]"
    study_name = f"[{agent_name}/{model_type}/{cluster}]"
    study = optuna.create_study(direction='maximize', study_name=study_name)
    study.optimize(objective, n_trials=n_trials)

    logger.info(f"Best F1-Score: {study.best_value:.4f}")
    logger.info("Best parameters found (including labeling strategy):")
    for key, value in study.best_params.items():
        logger.info(f"  - {key}: {value}")

    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)
    param_filename = f"best_params_{agent_name}_{model_type}_{cluster}.json"
    output_path = os.path.join(output_dir, param_filename)
    with open(output_path, 'w') as f:
        json.dump(study.best_params, f, indent=4)
    logger.info(f"✅ Best parameters saved to: {output_path}")


def get_user_selection(prompt: str, options: list) -> list:
    """Helper function for interactive menus."""
    print(f"\n{prompt}")
    for i, option in enumerate(options):
        print(f"{i + 1}. {option.title()}")
    print(f"{len(options) + 1}. All")

    choice = input(f"Please enter your selection (1-{len(options) + 1}): ")
    try:
        choice_int = int(choice)
        if 1 <= choice_int <= len(options):
            return [options[choice_int - 1]]
        elif choice_int == len(options) + 1:
            return options
    except ValueError:
        pass

    print("❌ Invalid selection. Please try again.")
    return get_user_selection(prompt, options)


def _get_single_choice_from_list(prompt: str, options: list) -> str:
    """
    NEW: A helper function for the 'Custom Run' menu.
    Prompts the user to select exactly ONE item from a list.
    """
    print(f"\n{prompt}")
    for i, option in enumerate(options):
        print(f"{i + 1}. {option.title()}")

    while True:
        choice = input(f"Please enter your selection (1-{len(options)}): ")
        try:
            choice_int = int(choice)
            if 1 <= choice_int <= len(options):
                return options[choice_int - 1]
            else:
                print(f"❌ Invalid selection. Must be between 1 and {len(options)}.")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")


def get_custom_jobs() -> list:
    """
    NEW: An interactive function to build a custom list of tuning jobs.
    Returns a list of tuples, e.g., [('4pct', 'entry', 'low'), ...].
    """
    custom_jobs_list = []
    print("\n--- Custom Job Builder ---")

    while True:
        print(f"\nBuilding Job #{len(custom_jobs_list) + 1}:")

        # 1. Select Agent
        agent = _get_single_choice_from_list("Select Agent:", ALL_AGENTS)

        # 2. Select Model Type
        model_type = _get_single_choice_from_list("Select Model Type:", ALL_MODEL_TYPES)

        # 3. Select Cluster
        cluster = _get_single_choice_from_list("Select Volatility Cluster:", ALL_CLUSTERS)

        job_tuple = (agent, model_type, cluster)
        custom_jobs_list.append(job_tuple)
        print(f"✅ Job added: {job_tuple}")

        while True:
            add_another = input("Add another job? (y/n): ").lower().strip()
            if add_another in ['y', 'n']:
                break

        if add_another == 'n':
            break

    return custom_jobs_list


if __name__ == "__main__":
    all_jobs = []

    # --- NEW: Top-Level Menu (Standard vs. Custom) ---
    print("\n" + "=" * 50)
    print("      StockWise Tuner Orchestrator")
    print("=" * 50)
    print("\nHow do you want to run the tuner?")
    print("1. Standard Run (Select categories, runs all combinations)")
    print("2. Custom Run (Define a specific list of jobs)")

    while True:
        run_mode = input("Please enter your selection (1-2): ").strip()
        if run_mode in ['1', '2']:
            break
        print("❌ Invalid selection. Please enter 1 or 2.")

        # --- ROUTE TO THE CORRECT LOGIC ---

    if run_mode == '1':
        logger.info("Starting Standard Run...")
        agents_to_tune = get_user_selection("Which agent do you want to optimize?", ALL_AGENTS)
        model_types_to_tune = get_user_selection("Which model type do you want to optimize?", ALL_MODEL_TYPES)
        clusters_to_tune = get_user_selection("Which volatility cluster do you want to optimize?", ALL_CLUSTERS)
        all_jobs = list(itertools.product(agents_to_tune, model_types_to_tune, clusters_to_tune))

    elif run_mode == '2':
        logger.info("Starting Custom Run...")
        all_jobs = get_custom_jobs()

        # --- Safety check and summary before running ---
    if not all_jobs:
        logger.warning("No jobs were defined. Exiting.")
        sys.exit(0)

    logger.info("\n" + "─" * 50)
    logger.info(f"Tuning run will consist of the following {len(all_jobs)} job(s):")
    for job in all_jobs:
        logger.info(f"  - {job}")
    logger.info("─" * 50 + "\n")

    # --- Run tuning jobs in parallel ---
    # MAX_WORKERS = max(1, 1 if not os.cpu_count() else os.cpu_count() // 2)
    MAX_WORKERS = 2
    logger.info(f"Initializing ProcessPoolExecutor with {MAX_WORKERS} workers.")

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # --- Submit agent_name, m_type, and cluster STRINGS, NOT the DataFrame ---
        futures = [executor.submit(run_single_study, agent, m_type, cluster, n_trials=100) for
                   agent, m_type, cluster in all_jobs]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures),
                           desc="Overall Tuning Progress"):
            try:
                future.result()
            except Exception as e:
                logger.error(f"A tuning job failed with an error: {e}")

    logger.info("\n🎉 All selected tuning studies have finished.")
    logger.info("\n🎉 All selected tuning studies have finished.")