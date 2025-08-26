import optuna
import lightgbm as lgb
import pandas as pd
import joblib
import json
import logging
import time
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from data_manager import DataManager

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("HyperparameterTuner")

# --- Global Variables ---
TRAIN_FEATURE_DIR = "models/NASDAQ-training set/features"
LABEL_COL = "Target"
N_TRIALS = 100  # Define the number of trials as a constant

logger.info(f"Loading and combining data from {TRAIN_FEATURE_DIR}...")
data_manager = DataManager(TRAIN_FEATURE_DIR, label="Train")
symbols = data_manager.get_available_symbols()
DF_TRAIN = data_manager.combine_feature_files(symbols)
logger.info(f"Data loaded. Shape: {DF_TRAIN.shape}")

study_start_time = time.time()


def objective(trial: optuna.Trial) -> float:
    # (The contents of this function are unchanged)
    if DF_TRAIN.empty:
        raise ValueError("Training DataFrame is empty. Cannot run trial.")

    params = {
        'objective': 'binary', 'metric': 'logloss', 'n_estimators': 10000,
        'verbosity': -1, 'n_jobs': -1, 'seed': 42, 'class_weight': 'balanced',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }

    expected_features = [
        'Volume_MA_20', 'RSI_14', 'Momentum_5', 'MACD', 'MACD_Signal',
        'MACD_Histogram', 'BB_Upper', 'BB_Lower', 'BB_Middle',
        'BB_Position', 'Daily_Return', 'Volatility_20D',
        'ATR_14', 'ADX', 'ADX_pos', 'ADX_neg', 'OBV', 'RSI_28',
        'Dominant_Cycle_126D'
    ]
    feature_cols = [col for col in expected_features if col in DF_TRAIN.columns]

    X = DF_TRAIN[feature_cols]
    y = DF_TRAIN[LABEL_COL]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='logloss',
              callbacks=[lgb.early_stopping(100, verbose=False)])

    preds = model.predict(X_val)
    f1 = f1_score(y_val, preds)

    return f1


# --- FIX #1: The callback function now accepts n_trials as an argument ---
def progress_callback(study, trial, n_trials):
    """
    This function is called after each trial. It prints a progress summary.
    """
    completed_trials = len(study.trials)

    if completed_trials > 0:
        elapsed_seconds = time.time() - study_start_time
        avg_time_per_trial = elapsed_seconds / completed_trials
        remaining_trials = n_trials - completed_trials
        eta_seconds = remaining_trials * avg_time_per_trial
        eta_str = str(timedelta(seconds=int(eta_seconds)))

        logger.info(
            f"Trial {completed_trials}/{n_trials} finished. "
            f"Last F1: {trial.value:.4f}, Best F1: {study.best_value:.4f}. "
            f"ETA: {eta_str}"
        )


if __name__ == "__main__":
    if DF_TRAIN.empty:
        logger.error("❌ No data loaded. Exiting hyperparameter tuning.")
    else:
        study = optuna.create_study(direction="maximize")

        # --- FIX #2: We use a lambda function to pass the N_TRIALS constant into our callback ---
        study.optimize(
            objective,
            n_trials=N_TRIALS,
            callbacks=[lambda study, trial: progress_callback(study, trial, N_TRIALS)]
        )

        logger.info("=" * 50)
        logger.info("🎉 Hyperparameter Tuning Finished! 🎉")
        logger.info(f"Number of finished trials: {len(study.trials)}")
        logger.info(f"Best trial's F1-score: {study.best_value:.4f}")

        logger.info("Found best hyperparameters:")
        best_params = study.best_trial.params
        print(json.dumps(best_params, indent=4))

        config_path = "models/best_lgbm_params.json"
        with open(config_path, "w") as f:
            json.dump(best_params, f, indent=4)
        logger.info(f"✅ Best parameters saved to {config_path}")