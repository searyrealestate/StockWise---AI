import os
import json
import joblib
import logging
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
from data_manager import DataManager
from datetime import datetime

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s")
logger = logging.getLogger("Gen3ModelTrainer")


class Gen3ModelTrainer:
    """
    Trains the full suite of Gen-3 specialist "orchestra" models.
    This trainer iterates through each volatility cluster and trains three
    distinct binary models for each: Entry, Profit-Taking, and Risk Management.
    """

    def __init__(self, model_dir: str, custom_params: dict):
        self.model_dir = model_dir
        self.params = custom_params
        # Ensure the output directory for Gen-3 models exists
        os.makedirs(self.model_dir, exist_ok=True)
        logger.info(f"Gen-3 Model Trainer initialized. Models will be saved to: {self.model_dir}")

    def _train_single_model(self, df: pd.DataFrame, feature_cols: list, label_col: str, model_name: str) -> dict:
        """
        Internal helper function to train one specialist binary model.
        """
        if label_col not in df.columns:
            logger.error(f"Target column '{label_col}' not found in the DataFrame. Skipping training for {model_name}.")
            return {}

        X = df[feature_cols]
        y = df[label_col]

        # Filter out rows where the label is not 0 or 1, as this is a binary task
        binary_mask = y.isin([0, 1])
        X = X[binary_mask]
        y = y[binary_mask]

        if y.nunique() < 2:
            logger.warning(f"⚠️ Target '{label_col}' has fewer than 2 unique classes. Cannot train {model_name}.")
            return {}

        if len(y[y == 1]) < 10:
            logger.warning(
                f"⚠️ Insufficient positive samples ({len(y[y == 1])}) for '{label_col}'. Skipping {model_name}.")
            return {}

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        logger.info(f"Training {model_name} with {len(X_train)} samples. Target: '{label_col}'.")

        # --- Use more advanced model parameters ---
        model_params = self.params.copy()
        model_params.update({
            'objective': 'binary',  # Critical change: each model is a simple binary classifier
            'n_estimators': 10000,
            'seed': 42,
            'n_jobs': -1,
            'verbose': -1,
            'class_weight': 'balanced'
        })

        model = lgb.LGBMClassifier(**model_params)

        # Use early stopping to prevent overfitting
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='logloss',
                  callbacks=[lgb.early_stopping(100, verbose=False)])

        y_pred = model.predict(X_val)

        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "f1": f1_score(y_val, y_pred, zero_division=0),
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred, zero_division=0),
            "best_iteration": model.best_iteration_
        }

        logger.info(
            f"Metrics for {model_name}: F1={metrics['f1']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")

        # --- Save the model and its feature list ---
        model_path = os.path.join(self.model_dir, model_name)
        joblib.dump(model, model_path)

        feature_cols_path = model_path.replace(".pkl", "_features.json")
        with open(feature_cols_path, "w") as f:
            json.dump(feature_cols, f, indent=4)

        logger.info(f"✅ Model saved to: {model_path}")
        logger.info(f"✅ Features saved to: {feature_cols_path}")

        return metrics

    def train_specialist_models(self, df: pd.DataFrame):
        """
        Main orchestration method to train the entire suite of Gen-3 models.
        """
        logger.info("🚀 Starting Gen-3 'Orchestra' Model Training Pipeline...")

        # --- Define the full feature set for Gen-3 ---
        # Note: 'Volatility_Cluster' is used for filtering, NOT as a feature itself.
        gen3_feature_cols = [
            'Volume_MA_20', 'RSI_14', 'Momentum_5', 'MACD', 'MACD_Signal',
            'MACD_Histogram', 'BB_Upper', 'BB_Lower', 'BB_Middle',
            'BB_Position', 'Daily_Return', 'Volatility_20D', 'ATR_14',
            'ADX', 'ADX_pos', 'ADX_neg', 'OBV', 'RSI_28', 'Dominant_Cycle_126D',
            "Smoothed_Close_5D", "RSI_14_Smoothed",
            'Z_Score_20', 'BB_Width', 'Correlation_50D_QQQ'
        ]

        # Define the clusters and the models to be trained for each
        clusters = ['low', 'mid', 'high']
        model_specs = {
            'entry': 'Target_Entry',
            'profit_take': 'Target_Profit_Take',
            'cut_loss': 'Target_Cut_Loss'
        }

        for cluster in clusters:
            logger.info(f"\n{'─' * 20} Training models for VOLATILITY CLUSTER: '{cluster.upper()}' {'─' * 20}")

            cluster_df = df[df['Volatility_Cluster'] == cluster].copy()

            if cluster_df.empty:
                logger.warning(f"No data found for cluster '{cluster}'. Skipping training for this cluster.")
                continue

            for model_type, target_col in model_specs.items():
                model_filename = f"{model_type}_model_{cluster}_vol.pkl"
                self._train_single_model(
                    df=cluster_df,
                    feature_cols=gen3_feature_cols,
                    label_col=target_col,
                    model_name=model_filename
                )

        logger.info("\n🎉 Gen-3 Model Training Pipeline Finished Successfully!")


if __name__ == "__main__":
    # Define directories for training data and where models will be saved
    TRAIN_FEATURE_DIR = "models/NASDAQ-training set/features"
    GEN3_MODEL_DIR = "models/NASDAQ-gen3"  # New dedicated directory for Gen-3 models

    # Path to the optimized hyperparameters from a tuning process
    PARAMS_PATH = "models/best_lgbm_params.json"

    # Load the best hyperparameters
    try:
        with open(PARAMS_PATH, 'r') as f:
            best_params = json.load(f)
        logger.info(f"✅ Successfully loaded best parameters from {PARAMS_PATH}")
    except FileNotFoundError:
        logger.error(f"❌ Best parameter file not found at {PARAMS_PATH}. Cannot proceed.")
        best_params = None

    if best_params:
        # Load and combine all training data
        train_data_manager = DataManager(TRAIN_FEATURE_DIR, label="Train")
        symbols = train_data_manager.get_available_symbols()
        combined_df = train_data_manager.combine_feature_files(symbols)

        if combined_df.empty:
            logger.error("❌ Combined training DataFrame is empty. Cannot train models.")
        else:
            # Initialize and run the Gen-3 trainer
            trainer = Gen3ModelTrainer(model_dir=GEN3_MODEL_DIR, custom_params=best_params)
            trainer.train_specialist_models(combined_df)
