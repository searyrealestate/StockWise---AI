import os
import json
import joblib
import logging
import pandas as pd
from tqdm import tqdm
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from data_manager import DataManager  # Make sure this is updated with your latest version

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

        # Exclude non-feature columns
        exclude_cols = ["symbol", "Symbol", "Target", "Datetime"]
        feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]

        logger.info(f"Using {len(feature_cols)} features: {feature_cols[:5]}{'...' if len(feature_cols) > 5 else ''}")

        # Save feature list
        feature_path = os.path.join(self.model_dir, f"feature_cols_{model_name.replace('.pkl', '')}.json")
        with open(feature_path, "w") as f:
            json.dump(feature_cols, f)
        logger.info(f"Saved feature columns to: {feature_path}")

        X = df[feature_cols]
        y = df[self.label_col]

        logger.info("Splitting data into train/validation sets...")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        logger.info("Training LightGBM model...")
        model = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            class_weight="balanced",
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

    symbols = train_data_manager.get_available_symbols()
    df = train_data_manager.combine_feature_files(tqdm(symbols, desc="Loading training data"))

    trainer = ModelTrainer(model_dir="models/400_train_set")
    metrics = trainer.train_model(df, model_name="nasdaq_general_model_lgbm_tech-400stocks.pkl")
    print("\n📊 Model Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k.capitalize():<10}: {v:.4f}")

