import os
import joblib
import pandas as pd
from xgboost import XGBClassifier

"""
🧠 StockWise Model Trainer from Feature Files
─────────────────────────────────────────────

This script automates the training of XGBoost models using preprocessed feature datasets.
It scans a directory for feature files, trains models for each stock, and saves them
as `.pkl` files for later use in prediction or evaluation dashboards.

🔧 Key Features:
────────────────
- Loads feature datasets from `.pkl` files
- Trains XGBoost classifiers using volume-based features
- Saves trained models to disk
- Skips models that already exist

📁 Inputs:
──────────
- Feature files named like: `SYMBOL_features.pkl`

📤 Outputs:
───────────
- Trained models saved as: `SYMBOL_model.pkl`
"""


model_dir = "C:/Users/user/PycharmProjects/StockWise/models"


def train_model_from_file(feature_path):
    """
    📚 Trains an XGBoost model from a single feature file.
    📥 Parameters:
    feature_path: str — path to the .pkl feature file

    📤 Returns: None (saves model to disk)

    What it does:
    Loads the feature DataFrame
    Extracts features and target
    Trains an XGBoost classifier
    Saves the model as SYMBOL_model.pkl in model_dir
    """

    symbol = os.path.basename(feature_path).split("_features.pkl")[0]
    try:
        df = pd.read_pickle(feature_path).dropna()
        X = df[["Volume_Relative", "Volume_Delta", "Turnover", "Volume_Spike"]]
        y = df["Target"]
        model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            use_label_encoder=False,
            eval_metric="logloss"
        )
        model.fit(X, y)
        model_path = os.path.join(model_dir, f"{symbol}_model.pkl")
        joblib.dump(model, model_path)
        print(f"✅ Trained & saved model for {symbol}")
    except Exception as e:
        print(f"❌ Failed to train {symbol}: {e}")


def train_all_models_from_features(folder):
    """
    📚 Trains models for all feature files in a folder (if not already trained).

    📥 Parameters:
    folder: str — directory containing feature files

    📤 Returns: None (trains and saves models)

    What it does:
    Scans the folder for files ending in _features.pkl
    For each file, checks if a corresponding model already exists
    If not, calls train_model_from_file() to train and save the model

    """
    feature_files = [f for f in os.listdir(folder) if f.endswith("_features.pkl")]
    for f in feature_files:
        model_name = f.replace("_features.pkl", "_model.pkl")
        if not os.path.exists(os.path.join(folder, model_name)):
            train_model_from_file(os.path.join(folder, f))


# Entry point
if __name__ == "__main__":
    """
    📚 Entry point for script execution.
    
    📥 Parameters: None
    📤 Returns: None
    
    What it does:
    Calls train_all_models_from_features() on the model_dir
    """
    train_all_models_from_features(model_dir)
