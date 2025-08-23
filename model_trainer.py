import os
import json
import joblib
import logging
import pandas as pd
from tqdm import tqdm  # Ensure tqdm is explicitly imported here
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from data_manager import DataManager

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ModelTrainer")


class ModelTrainer:
    def __init__(self, model_dir: str, label_col: str = "Target"):
        self.model_dir = model_dir
        self.label_col = label_col
        os.makedirs(self.model_dir, exist_ok=True)

    def train_model(self, df: pd.DataFrame, model_name: str = "nasdaq_general_model_lgbm_tech.pkl") -> dict:
        logger.info("Preparing training data...")

        # Explicitly define the 12 features expected by ProfessionalStockAdvisor
        # These names must exactly match how they are generated in data_manager.py
        # This list comes from the KeyError encountered during model evaluation,
        # indicating these are the features the end system expects.
        expected_features = [
            'Volume_MA_20',  # Volume Moving Average
            'RSI_14',  # Relative Strength Index
            'Momentum_5',  # 5-day Momentum / Rate of Change
            'MACD',  # Moving Average Convergence Divergence
            'MACD_Signal',  # MACD Signal Line
            'MACD_Histogram',  # MACD Histogram (MACD - MACD_Signal)
            'BB_Upper',  # Bollinger Band Upper
            'BB_Lower',  # Bollinger Band Lower
            'BB_Middle',  # Bollinger Band Middle
            'BB_Position',  # Bollinger Band %B
            'Daily_Return',  # Daily Percentage Change in Close Price
            'Volatility_20D'  # 20-day Volatility (Standard Deviation of Daily Returns)
        ]

        # Filter the DataFrame to include only the expected features and the target
        # Also ensure 'Symbol' and 'Datetime' are not treated as features
        final_feature_cols = [col for col in expected_features if col in df.columns]

        # Check if all expected features are present
        missing_features = [f for f in expected_features if f not in df.columns]
        if missing_features:
            logger.warning(f"⚠️ Missing expected features in training data: {missing_features}. "
                           "This might impact model performance if these features are critical.")
            # Adjust final_feature_cols to only include those present
            final_feature_cols = [f for f in expected_features if f in df.columns]

        # Ensure target column is present
        if self.label_col not in df.columns:
            logger.error(f"❌ Target column '{self.label_col}' not found in the training data.")
            raise ValueError(f"Target column '{self.label_col}' missing.")

        # Ensure there are actually features to train on
        if not final_feature_cols:
            logger.error("❌ No valid features found for training after filtering. Cannot train model.")
            return {}

        logger.info(
            f"Using {len(final_feature_cols)} features for training: {final_feature_cols[:5]}{'...' if len(final_feature_cols) > 5 else ''}")

        X = df[final_feature_cols]
        y = df[self.label_col]

        # Save the list of feature columns used for training to a JSON file
        feature_cols_path = model_name.replace(".pkl", ".json").replace("nasdaq_general_model",
                                                                        "feature_cols_nasdaq_general_model")
        feature_cols_full_path = os.path.join(self.model_dir, feature_cols_path)
        with open(feature_cols_full_path, "w") as f:
            json.dump(final_feature_cols, f)
        logger.info(f"Saved feature columns to: {feature_cols_full_path}")

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        logger.info("Training LightGBM model...")
        model = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            class_weight="balanced",  # Helps with imbalanced datasets
            random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)

        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "f1": f1_score(y_val, y_pred, zero_division=0),
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred, zero_division=0),
        }

        logger.info(f"Metrics: {metrics}")

        model_path = os.path.join(self.model_dir, model_name)
        joblib.dump(model, model_path)
        logger.info(f"Model saved to: {model_path}")

        return metrics


if __name__ == "__main__":
    TRAIN_FEATURE_DIR = "models/NASDAQ-training set"
    train_data_manager = DataManager(TRAIN_FEATURE_DIR, label="Train")
    model_dir = "models/NASDAQ-training set/features"

    symbols = train_data_manager.get_available_symbols()
    # The tqdm call needs to be outside or within the method being called by DataManager
    # For now, let's pass symbols directly and let combine_feature_files handle its own tqdm if needed.
    df = train_data_manager.combine_feature_files(symbols)

    # Check if the combined DataFrame is empty before proceeding
    if df.empty:
        logger.error("❌ Combined training DataFrame is empty. Cannot train model.")
    else:
        trainer = ModelTrainer(model_dir=model_dir)
        try:
            metrics = trainer.train_model(df, model_name="nasdaq_general_model_lgbm_tech-400stocks.pkl")
            print("\n📊 Model Evaluation Metrics:")
            for metric, value in metrics.items():
                print(f"{metric.capitalize():<10}: {value:.4f}")
        except Exception as e:
            logger.error(f"An error occurred during model training: {e}")
            import traceback
            traceback.print_exc()

