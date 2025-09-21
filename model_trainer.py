import os
import json
import joblib
import logging
import pandas as pd
import lightgbm as lgb
import argparse
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
            'volume_ma_20', 'rsi_14', 'momentum_5', 'macd', 'macd_signal',
            'macd_histogram', 'bb_position', 'volatility_20d', 'atr_14',
            'adx', 'adx_pos', 'adx_neg', 'obv', 'rsi_28',
            'z_score_20', 'bb_width', 'correlation_50d_qqq',
            'bb_upper', 'bb_lower', 'bb_middle', 'daily_return'
        ]

        # Define the clusters and the models to be trained for each
        clusters = ['low', 'mid', 'high']
        model_specs = {
            'entry': 'target_entry',
            'profit_take': 'target_profit_take',
            'cut_loss': 'target_cut_loss'
        }

        for cluster in clusters:
            logger.info(f"\n{'─' * 20} Training models for VOLATILITY CLUSTER: '{cluster.upper()}' {'─' * 20}")

            cluster_df = df[df['volatility_cluster'] == cluster].copy()

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


# In model_trainer.py
# (The Gen3ModelTrainer class and other functions at the top of the file remain the same)

def run_training_job(data_dir: str, model_dir: str, params: dict):
    """
    Helper function to run a single, complete training job for one agent.
    """
    logger.info(f"\n{'=' * 80}\n🚀 STARTING TRAINING JOB FOR: {model_dir}\n{'=' * 80}")

    # Load and combine training data from the specified directory
    train_data_manager = DataManager(data_dir, label="Train")
    symbols = train_data_manager.get_available_symbols()
    combined_df = train_data_manager.combine_feature_files(symbols)

    if combined_df.empty:
        logger.error(f"❌ Combined training DataFrame is empty from {data_dir}. Cannot train models.")
        return

    # Initialize and run the Gen-3 trainer, saving models to the specified directory
    trainer = Gen3ModelTrainer(model_dir=model_dir, custom_params=params)
    trainer.train_specialist_models(combined_df)


if __name__ == "__main__":
    # Define the configurations for each agent
    AGENT_CONFIGS = {
        '2pct': {
            'data_dir': "models/NASDAQ-training set/features/2per_profit",
            'model_dir': "models/NASDAQ-gen3-2pct"
        },
        '3pct': {
            'data_dir': "models/NASDAQ-training set/features/3per_profit",
            'model_dir': "models/NASDAQ-gen3-3pct"
        },
        '4pct': {
            'data_dir': "models/NASDAQ-training set/features/4per_profit",
            'model_dir': "models/NASDAQ-gen3-4pct"
        },
        'dynamic': {
            'data_dir': "models/NASDAQ-training set/features/dynamic_profit",
            'model_dir': "models/NASDAQ-gen3-dynamic"
        }
    }

    # --- NEW: Interactive Menu ---
    print("How do you prefer to run the script?")
    print("1. 2% profit model")
    print("2. 3% profit model")
    print("3. 4% profit model")
    print("4. dynamic profit model")
    print("5. All models")

    choice = input("Please enter your selection (1-5): ")

    agents_to_train = []
    if choice == '1':
        agents_to_train.append('2pct')
    elif choice == '2':
        agents_to_train.append('3pct')
    elif choice == '3':
        agents_to_train.append('4pct')
    elif choice == '4':
        agents_to_train.append('dynamic')
    elif choice == '5':
        agents_to_train = list(AGENT_CONFIGS.keys())
    else:
        print("❌ Invalid selection. Please run the script again and choose a number between 1 and 5.")
        exit()
    # --- END NEW ---

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

    if best_params and agents_to_train:
        # Loop through the selected agents and run the training job for each
        for agent_name in agents_to_train:
            config = AGENT_CONFIGS[agent_name]
            run_training_job(
                data_dir=config['data_dir'],
                model_dir=config['model_dir'],
                params=best_params
            )

    logger.info("\n🎉 All selected training pipelines have finished.")
