import os
import json
import joblib
import logging
import pandas as pd
import lightgbm as lgb  # Import the full lightgbm library for advanced features
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from data_manager import DataManager
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ModelTrainer")


class ModelTrainer:
    def __init__(self, model_dir: str, label_col: str = "Target", custom_params: dict = None):
        self.model_dir = model_dir
        self.label_col = label_col
        self.params = custom_params
        os.makedirs(self.model_dir, exist_ok=True)

    def train_model(self, df: pd.DataFrame, model_name: str) -> dict:
        logger.info("Preparing training data for Gen-2 model...")

        # --- CHANGE #1: Update the feature list to include all new features ---
        expected_features = [
            'Volume_MA_20', 'RSI_14', 'Momentum_5', 'MACD', 'MACD_Signal',
            'MACD_Histogram', 'BB_Upper', 'BB_Lower', 'BB_Middle',
            'BB_Position', 'Daily_Return', 'Volatility_20D',
            'ATR_14', 'ADX', 'ADX_pos', 'ADX_neg', 'OBV', 'RSI_28',
            'Dominant_Cycle_126D'
        ]

        final_feature_cols = [col for col in expected_features if col in df.columns]

        missing_features = set(expected_features) - set(final_feature_cols)
        if missing_features:
            logger.warning(f"⚠️ Missing expected features in training data: {sorted(list(missing_features))}.")

        if self.label_col not in df.columns:
            raise ValueError(f"Target column '{self.label_col}' missing.")

        logger.info(f"Using {len(final_feature_cols)} features for training.")

        X = df[final_feature_cols]
        y = df[self.label_col]

        # Save the list of feature columns used for this model
        feature_cols_path = os.path.join(self.model_dir, model_name.replace(".pkl", "_features.json"))
        with open(feature_cols_path, "w") as f:
            json.dump(final_feature_cols, f, indent=4)
        logger.info(f"Saved feature columns to: {feature_cols_path}")

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        logger.info("Training Advanced LightGBM model...")

        # --- CHANGE #2: Use more advanced model parameters ---
        if self.params:
            model_params = self.params.copy()
            model_params.update({
                'objective': 'binary',
                'n_estimators': 50000,  # Still use a high number for early stopping
                'seed': 42,
                'n_jobs': -1,
                'verbose': -1,
                'class_weight': 'balanced'
            })
        else:
            # Fallback to default parameters if none are provided
            model_params = {'objective': 'binary', 'n_estimators': 2000, 'learning_rate': 0.02}

        model = lgb.LGBMClassifier(**model_params)

        # --- CHANGE #3: Use early stopping to prevent overfitting ---
        # The model will stop training if performance on the validation set doesn't improve for 100 rounds.
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='logloss',
                  callbacks=[lgb.early_stopping(100, verbose=False)])

        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]  # Get probabilities for the positive class

        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "f1": f1_score(y_val, y_pred, zero_division=0),
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred, zero_division=0),
            "best_iteration": model.best_iteration_
        }

        logger.info(f"Metrics: {metrics}")

        model_path = os.path.join(self.model_dir, model_name)
        joblib.dump(model, model_path)
        logger.info(f"Model saved to: {model_path}")

        return metrics


if __name__ == "__main__":
    TRAIN_FEATURE_DIR = "models/NASDAQ-training set/features"
    MODEL_DIR = "models/NASDAQ-training set"
    MODEL_NAME = f"nasdaq_gen2_optimized_model_{datetime.now().strftime('%Y%m%d')}.pkl"
    PARAMS_PATH = "models/best_lgbm_params.json"

    # --- NEW: Load the best parameters from the tuning process ---
    try:
        with open(PARAMS_PATH, 'r') as f:
            best_params = json.load(f)
        logger.info(f"✅ Successfully loaded best parameters from {PARAMS_PATH}")
    except FileNotFoundError:
        logger.error(f"❌ Best parameter file not found at {PARAMS_PATH}. Cannot proceed.")
        best_params = None  # Set to None if file not found

    if best_params:
        # --- End of new code ---

        train_data_manager = DataManager(TRAIN_FEATURE_DIR, label="Train")
        symbols = train_data_manager.get_available_symbols()
        df = train_data_manager.combine_feature_files(symbols)

        if df.empty:
            logger.error("❌ Combined training DataFrame is empty. Cannot train model.")
        else:
            # Pass the loaded parameters into the ModelTrainer
            trainer = ModelTrainer(model_dir=MODEL_DIR, custom_params=best_params)  # Pass params here
            metrics = trainer.train_model(df, model_name=MODEL_NAME)

            print("\n📊 Gen-2 OPTIMIZED Model Validation Metrics:")
            for metric, value in metrics.items():
                print(f"{metric.capitalize():<15}: {value:.4f}" if isinstance(value,
                                    float) else f"{metric.capitalize():<15}: {value}")