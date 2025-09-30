"""
hyperparameter_tuner.py (V2 - Flexible Targeting)

how to run the hyperparameter_tuner.py?
select and run one of the following, no need to run all of them:
python hyperparameter_tuner.py --agent dynamic
python hyperparameter_tuner.py --agent 2pct
python hyperparameter_tuner.py --agent 3pct
python hyperparameter_tuner.py --agent 4pct

to run for a specific model and cluster:
--agent is mandatory ; --model-type and --cluster are optional
python hyperparameter_tuner.py --agent dynamic --model-type <[entry, profit_take, cut_loss]> --cluster <[low, mid, high]>

python hyperparameter_tuner.py --agent dynamic --model-type cut_loss --cluster mid
python hyperparameter_tuner.py --agent dynamic --model-type cut_loss   # will run for all clusters [low, mid, high]
python hyperparameter_tuner.py --agent 4pct --model-type entry  # will run for all models [entry, profit_take, cut_loss]
"""

import os
import json
import logging
import pandas as pd
import lightgbm as lgb
import optuna
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from data_manager import DataManager

# --- Setup logging for the tuner ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (OptunaTuner) %(message)s")
logger = logging.getLogger("OptunaTuner")


def objective(trial, X_train, y_train, X_val, y_val):
    """
    Defines the objective function for Optuna.
    Trains a LightGBM model and returns the F1-score for maximization.
    """
    # Define a search space for hyperparameters
    param = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbosity': -1, 'n_jobs': -1,
        'random_state': 42, 'class_weight': 'balanced',
        'n_estimators': trial.suggest_int('n_estimators', 400, 4000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }

    model = lgb.LGBMClassifier(**param)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='f1',
              callbacks=[lgb.early_stopping(100, verbose=False)])

    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    return f1


def run_tuning_for_model(df: pd.DataFrame, feature_cols: list, agent_name: str,
                         model_type: str, cluster: str, n_trials: int = 100):
    """
    Orchestrates the tuning process for a single representative model from an agent's dataset.
    We tune the 'mid-volatility entry model' as it's often the most balanced.
    """
    target_model_name = f"{model_type.replace('_', ' ').title()} Model ({cluster.upper()} Vol) for {agent_name.upper()} Agent"
    logger.info(f"✨ Starting hyperparameter tuning for: {target_model_name}...")

    cluster_df = df[df['volatility_cluster'] == cluster].copy()
    label_col = f'target_{model_type}'  # e.g., 'target_cut_loss'

    if label_col not in cluster_df.columns:
        logger.error(f"FATAL: Label column '{label_col}' not found in the dataset.")
        return None

    X = cluster_df[feature_cols]
    y = cluster_df[label_col]

    if y.nunique() < 2:
        logger.warning(f"Target '{label_col}' has fewer than 2 unique classes. Skipping tuning.")
        return None

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=n_trials)

    logger.info(f"🎉 Tuning finished for {target_model_name}. Best F1-Score: {study.best_value:.4f}")
    logger.info(f"Optimal parameters found: {study.best_params}")
    return study.best_params


if __name__ == "__main__":
    AGENT_CONFIGS = {
        'dynamic': {'data_dir': "models/NASDAQ-training set/features/dynamic_profit"},
        '1pct': {'data_dir': "models/NASDAQ-training set/features/1per_profit"},
        '2pct': {'data_dir': "models/NASDAQ-training set/features/2per_profit"},
        '3pct': {'data_dir': "models/NASDAQ-training set/features/3per_profit"},
        '4pct': {'data_dir': "models/NASDAQ-training set/features/4per_profit"}
    }
    parser = argparse.ArgumentParser(description="Hyperparameter Tuner for StockWise Gen-3 Agents")
    parser.add_argument('--agent', required=True, type=str, choices=list(AGENT_CONFIGS.keys()),
                        help='Select which agent dataset to use.')

    # --- model-type optional with a default of 'all' ---
    parser.add_argument('--model-type', default='all', type=str, choices=['entry', 'profit_take', 'cut_loss', 'all'],
                        help="Select which specialist model type to tune. Default is 'all'.")
    # -- cluster optional, with a default of 'all' ---
    parser.add_argument('--cluster', default='all', type=str, choices=['low', 'mid', 'high', 'all'],
                        help="Select volatility cluster to tune. Default is 'all'.")

    args = parser.parse_args()

    config = AGENT_CONFIGS[args.agent]
    train_feature_dir = config['data_dir']

    logger.info(f"Loading data for '{args.agent}' agent from: {train_feature_dir}")
    train_data_manager = DataManager(train_feature_dir, label="Train")
    combined_df = train_data_manager.combine_feature_files(train_data_manager.get_available_symbols())

    if not combined_df.empty:
        gen3_feature_cols = [
            'volume_ma_20', 'rsi_14', 'momentum_5', 'macd', 'macd_signal', 'macd_histogram',
            'bb_position', 'volatility_20d', 'atr_14', 'adx', 'adx_pos', 'adx_neg', 'obv',
            'rsi_28', 'z_score_20', 'bb_width', 'correlation_50d_qqq', 'vix_close', 'bb_upper', 'bb_lower',
            'bb_middle', 'daily_return', 'kama_10', 'stoch_k', 'stoch_d', 'dominant_cycle'
        ]
        # --- HANDLE 'all' FOR BOTH ARGUMENTS ---
        model_types_to_run = ['entry', 'profit_take', 'cut_loss'] if args.model_type == 'all' else [args.model_type]
        clusters_to_run = ['low', 'mid', 'high'] if args.cluster == 'all' else [args.cluster]

        for model_type in model_types_to_run:
            for cluster in clusters_to_run:
                best_params = run_tuning_for_model(
                    df=combined_df, feature_cols=gen3_feature_cols, agent_name=args.agent,
                    model_type=model_type, cluster=cluster
                )
                if best_params:
                    output_path = f"models/best_params_{args.agent}_{model_type}_{cluster}.json"
                    logger.info(f"Saving best parameters for '{cluster}' cluster to '{output_path}'.")
                    with open(output_path, 'w') as f:
                        json.dump(best_params, f, indent=4)
                    logger.info("✅ Parameters saved.")