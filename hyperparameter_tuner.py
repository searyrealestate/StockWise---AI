import os
import json
import logging
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from data_manager import DataManager

# --- Setup logging for the tuner ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (OptunaTuner) %(message)s")
logger = logging.getLogger("OptunaTuner")


def objective(trial, X_train, y_train, X_val, y_val):
    """
    Defines the objective function for Optuna to minimize.
    This function trains a LightGBM model with a set of hyperparameters
    suggested by Optuna and returns the validation F1-score.
    """
    # Define a search space for hyperparameters
    param = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'n_estimators': trial.suggest_int('n_estimators', 500, 10000),
        # FIX: Use suggest_float with log=True as recommended by the warning
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'max_depth': trial.suggest_int('max_depth', 2, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 500),
        # FIX: Use suggest_float as recommended by the warning
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        # FIX: Use suggest_float as recommended by the warning
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        # FIX: Use suggest_float with log=True as recommended by the warning
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        # FIX: Use suggest_float with log=True as recommended by the warning
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1,
    }

    model = lgb.LGBMClassifier(**param)

    # Use early stopping to prevent overfitting
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='binary_logloss',
              callbacks=[lgb.early_stopping(100, verbose=False)])

    # Predict and return the F1-score
    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred, zero_division=0)

    # Optuna aims to minimize, so we return the negative F1-score
    return -f1


def run_tuning_for_model(df: pd.DataFrame, feature_cols: list, label_col: str, model_name: str):
    """
    Orchestrates the tuning process for a single specialist model.
    """
    logger.info(f"✨ Starting hyperparameter tuning for {model_name}...")

    X = df[feature_cols]
    y = df[label_col]

    # Filter out rows where the label is not 0 or 1, as this is a binary task
    binary_mask = y.isin([0, 1])
    X = X[binary_mask]
    y = y[binary_mask]

    if y.nunique() < 2:
        logger.warning(f"Target '{label_col}' has fewer than 2 unique classes. Skipping tuning.")
        return

    # Use a fixed train/val split for reproducible tuning
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Create an Optuna study to find the optimal hyperparameters
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=50)  # Run 50 trials

    # Get the best parameters and save them
    best_params = study.best_params
    best_f1 = -study.best_value

    logger.info(f"🎉 Tuning finished for {model_name}. Best F1-Score: {best_f1:.4f}")
    logger.info(f"Optimal parameters: {best_params}")

    # Save the best parameters to a JSON file
    output_path = f"models/best_lgbm_params_{model_name}.json"
    with open(output_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    logger.info(f"✅ Best parameters saved to {output_path}")


if __name__ == "__main__":
    # Define directories
    TRAIN_FEATURE_DIR = "models/NASDAQ-training set/features"

    # Define the model you want to tune.
    TARGET_CLUSTER = 'low'
    TARGET_MODEL_TYPE = 'Entry'

    # Load and combine all training data
    train_data_manager = DataManager(TRAIN_FEATURE_DIR, label="Train")
    symbols = train_data_manager.get_available_symbols()
    combined_df = train_data_manager.combine_feature_files(symbols)

    if combined_df.empty:
        logger.error("❌ Combined training DataFrame is empty. Cannot train models.")
    else:
        # Filter for the target cluster
        cluster_df = combined_df[combined_df['Volatility_Cluster'] == TARGET_CLUSTER].copy()

        # Get the feature columns
        gen3_feature_cols = [
            'Volume_MA_20', 'RSI_14', 'Momentum_5', 'MACD', 'MACD_Signal',
            'MACD_Histogram', 'BB_Upper', 'BB_Lower', 'BB_Middle',
            'BB_Position', 'Daily_Return', 'Volatility_20D', 'ATR_14',
            'ADX', 'ADX_pos', 'ADX_neg', 'OBV', 'RSI_28', 'Dominant_Cycle_126D',
            "Smoothed_Close_5D", "RSI_14_Smoothed",
            'Z_Score_20', 'BB_Width', 'Correlation_50D_QQQ'
        ]

        # Run the tuner
        run_tuning_for_model(
            df=cluster_df,
            feature_cols=gen3_feature_cols,
            label_col=f'Target_{TARGET_MODEL_TYPE}',
            model_name=f"{TARGET_MODEL_TYPE}_model_{TARGET_CLUSTER}_vol.pkl"
        )