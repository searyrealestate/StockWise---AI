# how to run the hyperparameter_tuner.py?
# select and run one of the following, no need to run all of them:
# python hyperparameter_tuner.py --agent dynamic
# python hyperparameter_tuner.py --agent 2pct
# python hyperparameter_tuner.py --agent 3pct
# python hyperparameter_tuner.py --agent 4pct

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
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'n_jobs': -1,
        'random_state': 42,
        'class_weight': 'balanced',
        'n_estimators': trial.suggest_int('n_estimators', 400, 4000),
        # MODIFIED: Use suggest_float with log=True as recommended
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        # MODIFIED: Use suggest_float
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        # MODIFIED: Use suggest_float with log=True
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


def run_tuning_for_agent(df: pd.DataFrame, feature_cols: list, agent_name: str, n_trials: int = 100):
    """
    Orchestrates the tuning process for a single representative model from an agent's dataset.
    We tune the 'mid-volatility entry model' as it's often the most balanced.
    """
    target_model_name = f"Mid-Volatility Entry Model for {agent_name.upper()} Agent"
    logger.info(f"✨ Starting hyperparameter tuning for: {target_model_name}...")

    # Filter for the specific data we want to tune on
    cluster_df = df[df['Volatility_Cluster'] == 'mid'].copy()
    label_col = 'Target_Entry'

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
    # NEW: Agent configurations to map agent names to their data directories
    AGENT_CONFIGS = {
        'dynamic': {'data_dir': "models/NASDAQ-training set/features/dynamic_profit"},
        '2pct': {'data_dir': "models/NASDAQ-training set/features/2per_profit"},
        '3pct': {'data_dir': "models/NASDAQ-training set/features/3per_profit"},
        '4pct': {'data_dir': "models/NASDAQ-training set/features/4per_profit"}
    }

    # NEW: Use argparse to select the agent to tune
    parser = argparse.ArgumentParser(description="Hyperparameter Tuner for StockWise Gen-3 Agents")
    parser.add_argument('--agent', required=True, type=str, choices=list(AGENT_CONFIGS.keys()),
                        help='Select which agent to tune.')
    args = parser.parse_args()

    config = AGENT_CONFIGS[args.agent]
    train_feature_dir = config['data_dir']

    logger.info(f"Loading data for '{args.agent}' agent from: {train_feature_dir}")
    train_data_manager = DataManager(train_feature_dir, label="Train")
    symbols = train_data_manager.get_available_symbols()
    combined_df = train_data_manager.combine_feature_files(symbols)

    if combined_df.empty:
        logger.error("❌ Combined training DataFrame is empty. Cannot start tuning.")
    else:
        # MODIFIED: Corrected feature list
        gen3_feature_cols = [
            'Volume_MA_20', 'RSI_14', 'Momentum_5', 'MACD', 'MACD_Signal',
            'MACD_Histogram', 'BB_Position', 'Volatility_20D', 'ATR_14',
            'ADX', 'ADX_pos', 'ADX_neg', 'OBV', 'RSI_28',
            'Z_Score_20', 'BB_Width', 'Correlation_50D_QQQ',
            'BB_Upper', 'BB_Lower', 'BB_Middle', 'Daily_Return'
        ]

        best_params = run_tuning_for_agent(
            df=combined_df,
            feature_cols=gen3_feature_cols,
            agent_name=args.agent
        )

        if best_params:
            # NEW: Save results to the central parameters file
            output_path = "models/best_lgbm_params.json"
            logger.info(f"The best parameters will be saved to '{output_path}'.")

            # Add a confirmation step to prevent accidental overwrites
            confirm = input("Do you want to overwrite this file? (y/n): ").lower()
            if confirm == 'y':
                with open(output_path, 'w') as f:
                    json.dump(best_params, f, indent=4)
                logger.info(f"✅ Best parameters saved. You can now retrain your '{args.agent}' agent.")
            else:
                logger.info("Save operation cancelled by user.")