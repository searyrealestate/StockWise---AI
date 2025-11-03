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
import numpy as np
from Create_parquet_file_NASDAQ import apply_triple_barrier

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s")
logger = logging.getLogger("Gen3ModelTrainer")


class Gen3ModelTrainer:
    """
    Trains the full suite of Gen-3 specialist "orchestra" models.
    This trainer iterates through each volatility cluster and trains three
    distinct binary models for each: Entry, Profit-Taking, and Risk Management.
    """

    def __init__(self, model_dir: str, agent_name: str):
        self.model_dir = model_dir
        self.agent_name = agent_name
        os.makedirs(self.model_dir, exist_ok=True)
        logger.info(f"Gen3-Model Trainer initialized. Models will be saved to: {self.model_dir}")

    def _extract_params(self, custom_params: dict) -> (dict, dict):
        """
        Splits the parameters dictionary from the tuner into
        labeling parameters and model parameters.
        """
        label_param_keys = ['profit_take_mult', 'stop_loss_mult', 'time_limit_bars', 'rsi_overbought_threshold']

        labeling_params = {k: v for k, v in custom_params.items() if k in label_param_keys}
        model_params = {k: v for k, v in custom_params.items() if k not in label_param_keys}

        # Check if we have the essential labeling params
        if 'time_limit_bars' not in labeling_params:
            logger.error(f"CRITICAL: Tuner params file is missing 'time_limit_bars'.")
            return None, model_params

        return labeling_params, model_params

    def _train_single_model(self, df: pd.DataFrame, feature_cols: list, label_col: str, model_name: str,
                            custom_params: dict) -> dict:
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
        model_params = custom_params.copy()
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
        # Feature list now includes the new Gen-3 features
        gen3_feature_cols = [
            'volume_ma_20', 'rsi_14', 'momentum_5', 'macd', 'macd_signal',
            'macd_histogram', 'bb_position', 'volatility_20d', 'atr_14',
            'adx', 'adx_pos', 'adx_neg', 'obv', 'rsi_28',
            'z_score_20', 'bb_width', 'correlation_50d_qqq', 'vix_close', 'corr_tlt', 'cmf',
            'bb_upper', 'bb_lower', 'bb_middle', 'daily_return',
            'kama_10', 'stoch_k', 'st_och_d', 'dominant_cycle'
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
                params_path = f"models/best_params_{self.agent_name}_{model_type}_{cluster}.json"
                try:
                    with open(params_path, 'r') as f:
                        custom_params = json.load(f)
                    logger.info(f"Loaded specific params for {model_filename} from {params_path}")
                except FileNotFoundError:
                    logger.critical(f"❌ FATAL: No parameter file found at {params_path}. Cannot train.")
                    logger.critical("Please run the 'orchestrate_tuning.py' script first!")
                    continue

                # 1. Split params into two groups
                labeling_params, model_params = self._extract_params(custom_params)

                if not labeling_params:
                    logger.critical(f"Skipping {model_filename} due to missing labeling parameters in JSON file.")
                    continue

                # 2. Re-label the DataFrame in memory
                logger.info(f"Re-labeling data for {model_filename} using optimized params...")
                re_labeled_df = cluster_df.copy()
                tb_labels = apply_triple_barrier(
                    close_prices=re_labeled_df['close'],
                    high_prices=re_labeled_df['high'],
                    low_prices=re_labeled_df['low'],
                    atr=re_labeled_df['atr_14'],
                    profit_take_mult=labeling_params.get('profit_take_mult', 2.0),
                    stop_loss_mult=labeling_params.get('stop_loss_mult', 2.5),
                    time_limit_bars=labeling_params.get('time_limit_bars', 15),
                    profit_mode='dynamic'
                )

                # 3. Re-create all target columns from the new labels
                re_labeled_df['target_entry'] = np.where(tb_labels == 1, 1, 0)
                re_labeled_df['target_cut_loss'] = np.where(tb_labels == -1, 1, 0)
                overbought_threshold = labeling_params.get('rsi_overbought_threshold', 75)
                overbought_condition = re_labeled_df['rsi_14'] > overbought_threshold
                re_labeled_df['target_profit_take'] = np.where((tb_labels == 1) & (overbought_condition), 1, 0)

                # 4. Train the model using the re-labeled data and model-specific params
                self._train_single_model(
                    df=re_labeled_df,  # Pass the re-labeled DataFrame
                    feature_cols=gen3_feature_cols,
                    label_col=target_col,
                    model_name=model_filename,
                    custom_params=model_params  # Pass only the model params
                )

        logger.info("\n🎉 Gen-3 Model Training Pipeline Finished Successfully!")


def run_training_job(agent_name: str, data_dir: str, model_dir: str): # <-- Updated signature
    """
    Helper function to run a single, complete training job for one agent.
    """
    logger.info(f"\n{'=' * 80}\n🚀 STARTING TRAINING JOB FOR: {model_dir}\n{'=' * 80}")
    train_data_manager = DataManager(data_dir, label="Train")
    symbols = train_data_manager.get_available_symbols()
    combined_df = train_data_manager.combine_feature_files(symbols)
    if combined_df.empty:
        logger.error(f"❌ Combined training DataFrame is empty from {data_dir}. Cannot train models.")
        return
    # The trainer now handles its own parameter loading
    trainer = Gen3ModelTrainer(model_dir=model_dir, agent_name=agent_name)
    trainer.train_specialist_models(combined_df)


if __name__ == "__main__":
    # Define the configurations for each agent
    AGENT_CONFIGS = {
        '1pct': {
            'data_dir': "models/NASDAQ-training set/features/1per_profit",
            'model_dir': "models/NASDAQ-gen3-1pct"
        },
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
    print("1. 1% profit model")
    print("2. 2% profit model")
    print("3. 3% profit model")
    print("4. 4% profit model")
    print("5. dynamic profit model")
    print("6. All models")

    choice = input("Please enter your selection (1-6): ")

    agents_to_train = []
    if choice == '1':
        agents_to_train.append('1pct')
    if choice == '2':
        agents_to_train.append('2pct')
    elif choice == '3':
        agents_to_train.append('3pct')
    elif choice == '4':
        agents_to_train.append('4pct')
    elif choice == '5':
        agents_to_train.append('dynamic')
    elif choice == '6':
        agents_to_train = list(AGENT_CONFIGS.keys())
    else:
        print("❌ Invalid selection. Please run the script again and choose a number between 1 and 6.")
        exit()
    # --- END NEW ---

    # # Path to the optimized hyperparameters from a tuning process
    # PARAMS_PATH = "models/best_lgbm_params.json"
    #
    # # Load the best hyperparameters
    # try:
    #     with open(PARAMS_PATH, 'r') as f:
    #         best_params = json.load(f)
    #     logger.info(f"✅ Successfully loaded best parameters from {PARAMS_PATH}")
    # except FileNotFoundError:
    #     logger.error(f"❌ Best parameter file not found at {PARAMS_PATH}. Cannot proceed.")
    #     best_params = None

    for agent_name in agents_to_train:
        config = AGENT_CONFIGS[agent_name]
        run_training_job(
            agent_name=agent_name,
            data_dir=config['data_dir'],
            model_dir=config['model_dir']
        )

    logger.info("\n🎉 All selected training pipelines have finished.")
