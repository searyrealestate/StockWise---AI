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
from data_manager import DataManager
from tqdm import tqdm
import itertools
import concurrent.futures
from Create_parquet_file_NASDAQ import apply_triple_barrier
import numpy as np
from datetime import datetime

# # --- Setup logging ---
# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s")
# logger = logging.getLogger("Gen3TunerOrchestrator")
#
# # --- Configuration ---
# AGENT_DATA_DIRS = {
#     'dynamic': "models/NASDAQ-training set/features/dynamic_profit",
#     '4pct': "models/NASDAQ-training set/features/4per_profit", # Changed from '4%'
#     '3pct': "models/NASDAQ-training set/features/3per_profit", # Changed from '3%'
#     '2pct': "models/NASDAQ-training set/features/2per_profit", # Changed from '2%'
#     '1pct': "models/NASDAQ-training set/features/1per_profit", # Changed from '1%'
# }
#
#
# # def run_single_study(df: pd.DataFrame, agent_name: str, model_type: str, cluster: str, n_trials=100):
# #     """
# #     Runs an Optuna hyperparameter tuning study for one single specialist model.
# #     """
# #     target_map = {
# #         'entry': 'target_entry',
# #         'profit_take': 'target_profit_take',
# #         'cut_loss': 'target_cut_loss'
# #     }
# #     target_col = target_map[model_type]
# #
# #     logger.info(f"\n--- Starting Optuna study for [{agent_name} / {model_type} / {cluster}] ---")
# #     logger.info(f"Targeting label: '{target_col}'")
# #
# #     cluster_df = df[df['volatility_cluster'] == cluster].copy()
# #
# #     # Base feature columns (excluding all potential labels)
# #     base_feature_cols = [col for col in df.columns if 'target' not in col and col != 'volatility_cluster']
# #
# #     if cluster_df.empty:
# #         logger.warning(f"Not enough data or classes for {cluster}/{target_col}. Skipping study.")
# #         return
# #
# #     feature_cols = [col for col in df.columns if
# #                     col not in ['volatility_cluster', 'target_entry', 'target_profit_take', 'target_cut_loss',
# #                                 'target_trailing_stop']]
# #
# #     X = cluster_df[feature_cols]
# #     y = cluster_df[target_col]
# #     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# #
# #     def objective(trial):
# #         params = {
# #             'objective': 'binary',
# #             'metric': 'binary_logloss',
# #             'n_estimators': 1000,
# #             'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
# #             'num_leaves': trial.suggest_int('num_leaves', 20, 100),
# #             'max_depth': trial.suggest_int('max_depth', 5, 20),
# #             'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
# #             'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
# #             'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
# #             'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
# #             'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
# #             'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
# #             'verbose': -1,
# #             'n_jobs': 1,
# #             'seed': 42
# #         }
# #         model = lgb.LGBMClassifier(**params)
# #         model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
# #         preds = model.predict(X_val)
# #         return f1_score(y_val, preds, zero_division=0)
# #
# #     study = optuna.create_study(direction='maximize')
# #     study.optimize(objective, n_trials=n_trials)
# #
# #     logger.info(f"Best F1-Score: {study.best_value:.4f}")
# #     logger.info("Best parameters found:")
# #     for key, value in study.best_params.items():
# #         logger.info(f"  - {key}: {value}")
# #
# #     # Save the best parameters to the specific file the trainer expects
# #     output_dir = "models"
# #     os.makedirs(output_dir, exist_ok=True)
# #     param_filename = f"best_params_{agent_name}_{model_type}_{cluster}.json"
# #     output_path = os.path.join(output_dir, param_filename)
# #     with open(output_path, 'w') as f:
# #         json.dump(study.best_params, f, indent=4)
# #     logger.info(f"✅ Best parameters saved to: {output_path}")
#
# def run_single_study(df: pd.DataFrame, agent_name: str, model_type: str, cluster: str, n_trials=100):
#     """
#     Runs an Optuna hyperparameter tuning study for one specialist model,
#     now including the Triple Barrier labeling parameters.
#     """
#     target_map = {
#         'entry': 'target_entry',
#         'profit_take': 'target_profit_take',
#         'cut_loss': 'target_cut_loss'
#     }
#     target_col = target_map[model_type]
#
#     logger.info(f"\n--- Starting Optuna study for [{agent_name} / {model_type} / {cluster}] ---")
#     logger.info(f"Tuning both model and labeling parameters for: '{target_col}'")
#
#     cluster_df = df[df['volatility_cluster'] == cluster].copy()
#
#     # Base feature columns (excluding all potential labels)
#     base_feature_cols = [col for col in df.columns if 'target' not in col and col != 'volatility_cluster']
#
#     if cluster_df.empty:
#         logger.warning(f"No data for cluster {cluster}. Skipping study.")
#         return
#
#     def objective(trial):
#         # --- 1. Suggest Labeling Hyperparameters ---
#         profit_take_mult = trial.suggest_float('profit_take_mult', 1.0, 4.0)
#         stop_loss_mult = trial.suggest_float('stop_loss_mult', 1.5, 4.0)
#         time_limit_bars = trial.suggest_int('time_limit_bars', 5, 25)  # In days for daily data
#
#         # --- 2. Re-generate Labels On-The-Fly ---
#         # This is the core of the new advanced method
#         temp_df = cluster_df.copy()
#         tb_labels = apply_triple_barrier(
#             close_prices=temp_df['close'],
#             high_prices=temp_df['high'],
#             low_prices=temp_df['low'],
#             atr=temp_df['atr_14'],
#             profit_take_mult=profit_take_mult,
#             stop_loss_mult=stop_loss_mult,
#             time_limit_bars=time_limit_bars,
#             profit_mode='dynamic'  # Assuming dynamic for this example
#         )
#         temp_df[target_col] = np.where(tb_labels == 1, 1, 0)  # We only care about the BUY signal for entry
#
#         # --- 3. Prepare Data with New Labels ---
#         X = temp_df[base_feature_cols]
#         y = temp_df[target_col]
#
#         if y.nunique() < 2 or y.sum() < 10:
#             return 0.0  # Return a score of 0 if the new labels are unusable
#
#         X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#
#         # --- 4. Suggest Model Hyperparameters (as before) ---
#         model_params = {
#             'objective': 'binary', 'metric': 'binary_logloss', 'n_estimators': 1000,
#             'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
#             'num_leaves': trial.suggest_int('num_leaves', 20, 100),
#             'max_depth': trial.suggest_int('max_depth', 5, 20),
#             'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
#             'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
#             'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
#             'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
#             'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
#             'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
#             'verbose': -1, 'n_jobs': 1, 'seed': 42, 'class_weight': 'balanced'
#         }
#
#         model = lgb.LGBMClassifier(**model_params)
#         model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
#         preds = model.predict(X_val)
#         return f1_score(y_val, preds, zero_division=0)
#
#     study = optuna.create_study(direction='maximize')
#     study.optimize(objective, n_trials=n_trials)  # n_trials might need to be increased for this complex search
#
#     logger.info(f"Best F1-Score: {study.best_value:.4f}")
#     logger.info("Best parameters found (including labeling strategy):")
#     for key, value in study.best_params.items():
#         logger.info(f"  - {key}: {value}")
#
#     # Save the best parameters to the specific file
#     output_dir = "models"
#     os.makedirs(output_dir, exist_ok=True)
#     param_filename = f"best_params_{agent_name}_{model_type}_{cluster}.json"
#     output_path = os.path.join(output_dir, param_filename)
#     with open(output_path, 'w') as f:
#         json.dump(study.best_params, f, indent=4)
#     logger.info(f"✅ Best parameters saved to: {output_path}")
#
#
# def get_user_selection(prompt: str, options: list) -> list:
#     """Helper function for interactive menus."""
#     print(f"\n{prompt}")
#     for i, option in enumerate(options):
#         print(f"{i + 1}. {option.title()}")
#     print(f"{len(options) + 1}. All")
#
#     choice = input(f"Please enter your selection (1-{len(options) + 1}): ")
#     try:
#         choice_int = int(choice)
#         if 1 <= choice_int <= len(options):
#             return [options[choice_int - 1]]
#         elif choice_int == len(options) + 1:
#             return options
#     except ValueError:
#         pass
#
#     print("❌ Invalid selection. Please try again.")
#     return get_user_selection(prompt, options)  # Recursive call on invalid input
#
#
# if __name__ == "__main__":
#
#     # --- Interactive Menu (same as before) ---
#     agents = get_user_selection("Which agent do you want to optimize?", list(AGENT_DATA_DIRS.keys()))
#     model_types = get_user_selection("Which model type do you want to optimize?", ['entry', 'profit_take', 'cut_loss'])
#     clusters = get_user_selection("Which volatility cluster do you want to optimize?", ['low', 'mid', 'high'])
#
#     # --- Create a list of all jobs to run for the progress bar ---
#     jobs_to_run = list(itertools.product(agents, model_types, clusters))
#
#     # Pre-load data for each selected agent to avoid reloading in the loop
#     agent_data_cache = {}
#     for agent in agents:
#         logger.info(f"\nPre-loading data for Agent: {agent.upper()}...")
#         data_dir = AGENT_DATA_DIRS[agent]
#
#         # Manually find all files ending with '_daily_context.parquet'
#         file_pattern = os.path.join(data_dir, '*_daily_context.parquet')
#         all_files = glob.glob(file_pattern)
#
#         if not all_files:
#             logger.error(
#                 f"No data files matching '*_daily_context.parquet' found for agent '{agent}' in {data_dir}. It will be skipped.")
#             continue
#
#         # data_manager = DataManager(data_dir)
#         # full_df = data_manager.combine_feature_files(data_manager.get_available_symbols())
#         # if full_df.empty:
#         #     logger.error(f"No data found for agent '{agent}' in {data_dir}. It will be skipped.")
#         # else:
#         #     agent_data_cache[agent] = full_df# Load and combine the files into a single DataFrame
#
#         df_list = []
#         for f in tqdm(all_files, desc=f"Loading data for {agent}"):
#             try:
#                 df_list.append(pd.read_parquet(f))
#             except Exception as e:
#                 logger.warning(f"Could not load file {f}: {e}")
#
#         if not df_list:
#             logger.error(f"Failed to load any valid data for agent '{agent}'. It will be skipped.")
#             continue
#
#         full_df = pd.concat(df_list, ignore_index=True)
#         agent_data_cache[agent] = full_df
#         logger.info(f"✅ Successfully loaded and combined {len(all_files)} files for agent '{agent}'.")
#
#     # --- Parallel Execution using ProcessPoolExecutor for CPU-bound tasks ---
#     # Leave one or two cores free for the OS to prevent system slowdown
#     MAX_WORKERS = max(1, os.cpu_count() - 2 if os.cpu_count() else 1)
#     logger.info(f"Initializing ProcessPoolExecutor with {MAX_WORKERS} workers.")
#
#     with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
#         futures = []
#         for agent, m_type, cluster in jobs_to_run:
#             if agent in agent_data_cache:
#                 future = executor.submit(run_single_study, agent_data_cache[agent], agent, m_type, cluster,
#                                          n_trials=100)
#                 futures.append(future)
#
#         # Use tqdm to show progress as jobs are completed
#         for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures),
#                            desc="Overall Tuning Progress"):
#             try:
#                 future.result()
#             except Exception as e:
#                 logger.error(f"A tuning job failed with an error: {e}")
#
#     logger.info("\n🎉 All selected tuning studies have finished.")


# --- Setup logging ---
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
    'dynamic': "models/NASDAQ-training set/features/dynamic_profit",
    '4pct': "models/NASDAQ-training set/features/4per_profit",
    '3pct': "models/NASDAQ-training set/features/3per_profit",
    '2pct': "models/NASDAQ-training set/features/2per_profit",
    '1pct': "models/NASDAQ-training set/features/1per_profit",
}


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
        time_limit_bars = trial.suggest_int('time_limit_bars', 5, 25)

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
        temp_df[target_col] = np.where(tb_labels == 1, 1, 0)

        # --- 3. Prepare Data with New Labels ---
        X = temp_df[base_feature_cols]
        y = temp_df[target_col]

        if y.nunique() < 2 or y.sum() < 10:
            return 0.0

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

    study = optuna.create_study(direction='maximize')
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


if __name__ == "__main__":
    # --- Interactive Menus (Unchanged) ---
    agents_to_tune = get_user_selection("Which agent do you want to optimize?", list(AGENT_DATA_DIRS.keys()))
    model_types_to_tune = get_user_selection("Which model type do you want to optimize?",
                                             ['entry', 'profit_take', 'cut_loss'])
    clusters_to_tune = get_user_selection("Which volatility cluster do you want to optimize?", ['low', 'mid', 'high'])

    # --- Create the full list of jobs to run ---
    all_jobs = list(itertools.product(agents_to_tune, model_types_to_tune, clusters_to_tune))

    # --- Run tuning jobs in parallel ---
    # MAX_WORKERS = max(1, 1 if not os.cpu_count() else os.cpu_count() // 2)
    MAX_WORKERS = 3
    logger.info(f"Initializing ProcessPoolExecutor with {MAX_WORKERS} workers.")

    # --- Main loop to process each agent sequentially ---
    # for agent_name in agents_to_tune:
    #     logger.info(f"\n{'=' * 80}\n🚀 Starting full tuning pipeline for AGENT: {agent_name.upper()}\n{'=' * 80}")
    #
    #     # --- STEP 1: Load data ONLY for the current agent ---
    #     data_dir = AGENT_DATA_DIRS[agent_name]
    #     file_pattern = os.path.join(data_dir, '*_daily_context.parquet')
    #     all_files = glob.glob(file_pattern)
    #
    #     if not all_files:
    #         logger.error(f"No data files found for agent '{agent_name}' in {data_dir}. Skipping.")
    #         continue
    #
    #     df_list = [pd.read_parquet(f) for f in tqdm(all_files, desc=f"Loading data for {agent_name}")]
    #
    #     if not df_list:
    #         logger.error(f"Failed to load any valid data for agent '{agent_name}'. Skipping.")
    #         continue
    #
    #     full_df = pd.concat(df_list, ignore_index=True)
    #     logger.info(f"✅ Successfully loaded and combined {len(all_files)} files for agent '{agent_name}'.")
    #
    #     # --- STEP 2: Create the job list for the current agent's models ---
    #     jobs_to_run = list(itertools.product([agent_name], model_types_to_tune, clusters_to_tune))
    #
    #     # --- STEP 3: Run tuning jobs in parallel for the current agent ---
    #     MAX_WORKERS = max(1, os.cpu_count() - 2 if os.cpu_count() else 1)
    #     logger.info(f"Initializing ProcessPoolExecutor with {MAX_WORKERS} workers for '{agent_name}'.")
    #
    #     with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
    #         futures = [executor.submit(run_single_study, full_df, agent, m_type, cluster, n_trials=100) for
    #                    agent, m_type, cluster in jobs_to_run]
    #
    #         for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures),
    #                            desc=f"Tuning {agent_name}"):
    #             try:
    #                 future.result()
    #             except Exception as e:
    #                 logger.error(f"A tuning job for {agent_name} failed with an error: {e}")
    #
    #     # --- STEP 4: Memory is automatically released as the loop proceeds to the next agent ---
    #     logger.info(f"✅ Completed tuning pipeline for agent: {agent_name.upper()}")

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