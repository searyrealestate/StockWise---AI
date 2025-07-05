import os
import joblib
import logging
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ModelTrainer")


class ModelTrainer:
    """
    Trains and saves a general stock prediction model using combined NASDAQ data.
    """

    def __init__(self, model_dir: str, feature_cols: list[str], label_col: str = "Target"):
        """
        Initializes the trainer.

        Parameters:
        - model_dir (str): Directory to save trained models
        - feature_cols (list of str): List of feature column names
        - label_col (str): Name of the target column
        """
        self.model_dir = model_dir
        self.feature_cols = feature_cols
        self.label_col = label_col

        os.makedirs(self.model_dir, exist_ok=True)

    def train_model(self, df: pd.DataFrame, model_name: str = "nasdaq_general_model.pkl") -> dict:
        """
        Trains a RandomForest model and saves it.

        Parameters:
        - df (pd.DataFrame): Combined training data
        - model_name (str): Filename to save the model

        Returns:
        - dict of training metrics (accuracy, F1, etc.)
        """
        logger.info("Preparing training data...")

        # Filter numeric features only
        exclude_cols = ["symbol", "Symbol", "Target", "Datetime"]
        feature_cols = [col for col in self.feature_cols if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]

        logger.info(f"Using {len(feature_cols)} features: {feature_cols[:5]}{'...' if len(feature_cols) > 5 else ''}")

        X = df[feature_cols]
        y = df[self.label_col]

        logger.info("Splitting data into train/validation sets...")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        logger.info("Training RandomForest model...")
        # model = RandomForestClassifier(
        #     n_estimators=100,
        #     random_state=42,
        #     class_weight="balanced",
        #     verbose=0
        # )
        model = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=42
        )

        # model.fit(X_train, y_train)
        # logger.info("Evaluating model...")

        # # Define the parameter grid
        # param_grid = {
        #     "n_estimators": [100, 200, 300],
        #     "max_depth": [4, 6, 8, 10, None],
        #     "min_samples_split": [2, 5, 10],
        #     "min_samples_leaf": [1, 2, 4]
        # }
        #
        # # Set up the randomized search
        # search = RandomizedSearchCV(
        #     estimator=RandomForestClassifier(class_weight="balanced", random_state=42),
        #     param_distributions=param_grid,
        #     n_iter=10,  # You can increase this for more thorough search
        #     scoring="f1",
        #     cv=3,
        #     verbose=1,
        #     n_jobs=-1
        # )

        # # Fit the search
        # search.fit(X_train, y_train)
        #
        # # Use the best model found
        # model = search.best_estimator_
        # logger.info(f"Best parameters: {search.best_params_}")

        # model = LGBMClassifier(
        #     n_estimators=100,
        #     learning_rate=0.05,
        #     class_weight="balanced",
        #     random_state=42
        # )

        # model = XGBClassifier(
        #     n_estimators=100,
        #     learning_rate=0.05,
        #     scale_pos_weight=6.5,  # Adjust based on class imbalance ratio
        #     use_label_encoder=False,
        #     eval_metric="logloss",
        #     random_state=42
        # )
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

    def load_model(self, model_name: str = "nasdaq_general_model.pkl"):
        """
        Loads a trained model from disk.

        Parameters:
        - model_name (str): Filename of the model to load

        Returns:
        - Trained model object
        """
        model_path = os.path.join(self.model_dir, model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        logger.info(f"Loading model from: {model_path}")
        return joblib.load(model_path)


if __name__ == "__main__":
    from data_manager import DataManager

    TRAIN_FEATURE_DIR = "models/NASDAQ-training set"
    train_data_manager = DataManager(TRAIN_FEATURE_DIR, label="Train")

    symbols = train_data_manager.get_available_symbols()
    df = train_data_manager.combine_feature_files(tqdm(symbols[:100], desc="Loading features"))

    feature_cols = df.columns.tolist()
    feature_cols = [col for col in feature_cols if col != "Volume"]
    trainer = ModelTrainer(model_dir="models", feature_cols=feature_cols)
    # trainer.train_model(df, model_name="nasdaq_general_model_v3.pkl")
    # trainer.train_model(df, model_name="nasdaq_general_model_rf_baseline.pkl")
    trainer.train_model(df, model_name="nasdaq_general_model_lgbm_tech.pkl")
