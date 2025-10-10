"""
Gen-3 Hyperparameter Tuning Orchestrator
========================================

This script is a powerful orchestrator for running hyperparameter tuning studies
using Optuna. It is designed to systematically find the optimal set of parameters
for each of the individual specialist models within the Gen-3 multi-agent
architecture.

The script is highly flexible, featuring an interactive menu system that allows the
user to run tuning for a single model, a specific agent's suite of models, or all
models for all agents in one go.

Key Functionality:
------------------
-   **Interactive Menus**: On launch, the script presents a series of menus for the
    user to select which agent(s), model type(s), and volatility cluster(s) they
    wish to optimize.
-   **Granular Tuning**: For each combination selected, it runs a dedicated Optuna
    study (`run_single_study`) to find the best hyperparameters that maximize the
    F1-score for that specific model's task.
-   **Efficient Data Handling**: The script pre-loads the required datasets for the
    selected agents into a cache before starting the tuning loops. This avoids
    redundant file I/O and speeds up the overall process.
-   **Automated Saving**: The best parameters found for each model are automatically
    saved to a unique JSON file (e.g., `best_params_dynamic_entry_low.json`).
    This file is named to be discoverable by the `model_trainer.py` script,
    creating a seamless workflow for training with optimized settings.
-   **Master Progress Bar**: A `tqdm` progress bar provides a high-level overview
    of the entire tuning session, showing how many jobs have been completed out of
    the total selected.

Usage:
------
    python <name_of_this_script>.py

    The script will then guide you through the interactive selection menus.
"""

import os
import json
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

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s")
logger = logging.getLogger("Gen3TunerOrchestrator")

# --- Configuration ---
AGENT_DATA_DIRS = {
    'dynamic': "models/NASDAQ-training set/features/dynamic_profit",
    '4%': "models/NASDAQ-training set/features/4per_profit",
    '3%': "models/NASDAQ-training set/features/3per_profit",
    '2%': "models/NASDAQ-training set/features/2per_profit",
    '1%': "models/NASDAQ-training set/features/1per_profit",
}


def run_single_study(df: pd.DataFrame, agent_name: str, model_type: str, cluster: str, n_trials=100):
    """
    Runs an Optuna hyperparameter tuning study for one single specialist model.
    """
    target_map = {
        'entry': 'target_entry',
        'profit_take': 'target_profit_take',
        'cut_loss': 'target_cut_loss'
    }
    target_col = target_map[model_type]

    logger.info(f"\n--- Starting Optuna study for [{agent_name} / {model_type} / {cluster}] ---")
    logger.info(f"Targeting label: '{target_col}'")

    cluster_df = df[df['volatility_cluster'] == cluster].copy()
    if cluster_df.empty or cluster_df[target_col].nunique() < 2:
        logger.warning(f"Not enough data or classes for {cluster}/{target_col}. Skipping study.")
        return

    feature_cols = [col for col in df.columns if
                    col not in ['volatility_cluster', 'target_entry', 'target_profit_take', 'target_cut_loss',
                                'target_trailing_stop']]

    X = cluster_df[feature_cols]
    y = cluster_df[target_col]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'n_estimators': 1000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'verbose': -1,
            'n_jobs': -1,
            'seed': 42
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        preds = model.predict(X_val)
        return f1_score(y_val, preds, zero_division=0)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    logger.info(f"Best F1-Score: {study.best_value:.4f}")
    logger.info("Best parameters found:")
    for key, value in study.best_params.items():
        logger.info(f"  - {key}: {value}")

    # Save the best parameters to the specific file the trainer expects
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
    return get_user_selection(prompt, options)  # Recursive call on invalid input


if __name__ == "__main__":

    # --- Interactive Menu (same as before) ---
    agents = get_user_selection("Which agent do you want to optimize?", list(AGENT_DATA_DIRS.keys()))
    model_types = get_user_selection("Which model type do you want to optimize?", ['entry', 'profit_take', 'cut_loss'])
    clusters = get_user_selection("Which volatility cluster do you want to optimize?", ['low', 'mid', 'high'])

    # --- Create a list of all jobs to run for the progress bar ---
    jobs_to_run = list(itertools.product(agents, model_types, clusters))

    # Pre-load data for each selected agent to avoid reloading in the loop
    agent_data_cache = {}
    for agent in agents:
        logger.info(f"\nPre-loading data for Agent: {agent.upper()}...")
        data_dir = AGENT_DATA_DIRS[agent]
        data_manager = DataManager(data_dir)
        full_df = data_manager.combine_feature_files(data_manager.get_available_symbols())
        if full_df.empty:
            logger.error(f"No data found for agent '{agent}' in {data_dir}. It will be skipped.")
        else:
            agent_data_cache[agent] = full_df

    # --- Parallel Execution using ProcessPoolExecutor ---
    # Use a number of workers suitable for your machine's CPU cores (e.g., os.cpu_count() - 2)
    MAX_WORKERS = 6

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for agent, m_type, cluster in jobs_to_run:
            if agent in agent_data_cache:
                # Submit each study as a separate job to the process pool
                future = executor.submit(run_single_study, agent_data_cache[agent], agent, m_type, cluster,
                                         n_trials=100)
                futures.append(future)

        # Use tqdm to show progress as jobs are completed
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Overall Tuning Progress"):
            try:
                future.result() # Retrieve result to raise any exceptions that occurred in the process
            except Exception as e:
                logger.error(f"A tuning job failed with an error: {e}")

    logger.info("\n🎉 All selected tuning studies have finished.")