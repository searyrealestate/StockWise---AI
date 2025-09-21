import os
import json
import joblib
import pandas as pd
import logging
from sklearn.metrics import classification_report
from data_manager import DataManager
import glob

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s")
logger = logging.getLogger("Gen3ModelEvaluator")


class Gen3ModelEvaluator:
    """
    Evaluates the performance of the full suite of Gen-3 specialist models.
    It loads all nine models, segments the test data by volatility cluster,
    and reports the performance of each specialist model on its specific task.
    """

    def __init__(self, model_dir: str, test_data_manager: DataManager):
        self.model_dir = model_dir
        self.test_data_manager = test_data_manager
        self.models = {}
        self.feature_cols = {}
        self._load_all_models()

    def _load_all_models(self):
        """
        Scans the model directory to find and load all nine specialist models
        and their corresponding feature lists.
        """
        logger.info(f"🔍 Scanning for Gen-3 models in: {self.model_dir}")
        model_files = glob.glob(os.path.join(self.model_dir, "*.pkl"))

        if len(model_files) < 9:
            logger.warning(f"⚠️ Found only {len(model_files)} model files. Expected 9. Evaluation may be incomplete.")

        for model_path in model_files:
            model_name = os.path.basename(model_path).replace(".pkl", "")
            features_path = model_path.replace(".pkl", "_features.json")
            try:
                self.models[model_name] = joblib.load(model_path)
                with open(features_path, 'r') as f:
                    self.feature_cols[model_name] = json.load(f)
                logger.info(f"  ✅ Loaded model: {model_name}")
            except Exception as e:
                logger.error(f"❌ Failed to load model or features for {model_name}: {e}")

    def evaluate_all_models(self) -> pd.DataFrame:
        """
        Orchestrates the evaluation process for all loaded models.
        """
        logger.info("\n🚀 Starting Gen-3 Model Evaluation Pipeline...")
        test_df = self.test_data_manager.combine_feature_files(self.test_data_manager.get_available_symbols())

        if test_df.empty:
            logger.error("❌ Combined test DataFrame is empty. Cannot evaluate models.")
            return pd.DataFrame()

        all_results = []
        clusters = ['low', 'mid', 'high']
        model_specs = {
            'entry': 'target_entry',
            'profit_take': 'target_profit_take',
            'cut_loss': 'target_cut_loss'
        }

        for cluster in clusters:
            logger.info(f"\n{'─' * 20} Evaluating models for VOLATILITY CLUSTER: '{cluster.upper()}' {'─' * 20}")
            cluster_df = test_df[test_df['volatility_cluster'] == cluster].copy()

            if cluster_df.empty:
                logger.warning(f"No test data found for cluster '{cluster}'. Skipping evaluation.")
                continue

            for model_type, target_col in model_specs.items():
                model_name = f"{model_type}_model_{cluster}_vol"

                if model_name not in self.models:
                    logger.warning(f"Model '{model_name}' not loaded. Skipping its evaluation.")
                    continue

                report = self._evaluate_single_model(
                    df=cluster_df,
                    model_name=model_name,
                    label_col=target_col
                )
                if report:
                    # We are interested in the metrics for class '1' (the positive class)
                    positive_class_metrics = report.get('1', {})
                    if positive_class_metrics:
                        # Add model identifiers to the dictionary
                        positive_class_metrics['model_name'] = model_name
                        positive_class_metrics['cluster'] = cluster
                        positive_class_metrics['model_type'] = model_type
                        all_results.append(positive_class_metrics)

        if not all_results:
            logger.error("❌ No evaluation results were generated.")
            return pd.DataFrame()

        summary_df = pd.DataFrame(all_results)

        # Reorder columns for clarity
        cols_order = ['model_name', 'cluster', 'model_type', 'precision', 'recall', 'f1-score', 'support']
        summary_df = summary_df[cols_order]

        # Save detailed report to CSV
        output_path = os.path.join(self.model_dir, "gen3_model_performance_summary.csv")
        summary_df.to_csv(output_path, index=False)
        logger.info(f"\n🎉 Detailed evaluation report for positive class (1) saved to: {output_path}")

        return summary_df

    def _evaluate_single_model(self, df: pd.DataFrame, model_name: str, label_col: str) -> dict:
        """
        Evaluates a single specialist model on the provided DataFrame.
        """
        model = self.models[model_name]
        feature_cols = self.feature_cols[model_name]

        # Verify all necessary columns are present
        required_cols = feature_cols + [label_col]
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            logger.error(f"Missing required columns for {model_name} in the test data: {missing}.")
            return None

        X_test = df[feature_cols]
        y_test = df[label_col]

        if y_test.nunique() < 2:
            logger.warning(f"Target '{label_col}' for {model_name} has fewer than 2 classes in the test set. Skipping.")
            return None

        logger.info(f"Evaluating {model_name} on {len(X_test)} samples...")
        y_pred = model.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        # --- Print a concise summary to the console ---
        # We focus on the performance for the positive class '1' as it's the action signal
        f1_score = report.get('1', {}).get('f1-score', 0)
        precision = report.get('1', {}).get('precision', 0)
        recall = report.get('1', {}).get('recall', 0)

        print(f"  - Model: {model_name}")
        print(f"    - Metrics for class '1' (Action Signal):")
        print(f"      - F1-Score: {f1_score:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

        return report


def run_evaluation_job(model_dir: str, test_data_dir: str):
    """
    Helper function to run a single, complete evaluation job for one agent.
    """
    logger.info(f"\n{'=' * 80}\n🚀 STARTING EVALUATION FOR: {model_dir}\n{'=' * 80}")

    if not os.path.exists(model_dir):
        logger.error(f"FATAL: Model directory not found at '{model_dir}'. Please run the trainer for this agent first.")
        return

    test_data_manager = DataManager(test_data_dir, label="Test")
    if not test_data_manager.get_available_symbols():
        logger.error(f"FATAL: No test data found in '{test_data_dir}'. Please run the data generation script.")
        return

    evaluator = Gen3ModelEvaluator(model_dir=model_dir, test_data_manager=test_data_manager)
    evaluator.evaluate_all_models()


if __name__ == "__main__":
    # Define the configurations for each agent, mapping them to their models and test data
    AGENT_CONFIGS = {
        'dynamic': {
            'test_data_dir': "models/NASDAQ-testing set/features/dynamic_profit",
            'model_dir': "models/NASDAQ-gen3-dynamic"
        },
        '2pct': {
            'test_data_dir': "models/NASDAQ-testing set/features/2per_profit",
            'model_dir': "models/NASDAQ-gen3-2pct"
        },
        '3pct': {
            'test_data_dir': "models/NASDAQ-testing set/features/3per_profit",
            'model_dir': "models/NASDAQ-gen3-3pct"
        },
        '4pct': {
            'test_data_dir': "models/NASDAQ-testing set/features/4per_profit",
            'model_dir': "models/NASDAQ-gen3-4pct"
        }
    }

    # --- Interactive Menu ---
    print("Which agent's models would you like to evaluate?")
    print("1. Dynamic Profit Agent")
    print("2. 2% Net Profit Agent")
    print("3. 3% Net Profit Agent")
    print("4. 4% Net Profit Agent")
    print("5. All Agents")

    choice = input("Please enter your selection (1-5): ")

    agents_to_evaluate = []
    if choice == '1':
        agents_to_evaluate.append('dynamic')
    elif choice == '2':
        agents_to_evaluate.append('2pct')
    elif choice == '3':
        agents_to_evaluate.append('3pct')
    elif choice == '4':
        agents_to_evaluate.append('4pct')
    elif choice == '5':
        agents_to_evaluate = list(AGENT_CONFIGS.keys())
    else:
        print("❌ Invalid selection. Please run the script again and choose a number between 1 and 5.")
        exit()

    # Loop through the selected agents and run the evaluation job for each
    for agent_name in agents_to_evaluate:
        config = AGENT_CONFIGS[agent_name]
        run_evaluation_job(
            model_dir=config['model_dir'],
            test_data_dir=config['test_data_dir']
        )

    logger.info("\n🎉 All selected model evaluations have finished.")