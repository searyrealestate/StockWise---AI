import os
import json
import joblib
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_manager import DataManager


class ModelEvaluator:
    def __init__(self, model_path: str, test_data_manager: DataManager, label_col: str = "Target"):
        self.model_path = model_path
        self.test_data_manager = test_data_manager
        self.label_col = label_col
        self.model = self._load_model()
        self.feature_cols = self._load_feature_columns()  # This will now load the correct features

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        return joblib.load(self.model_path)

    def _load_feature_columns(self):
        # Construct the path to the feature columns JSON file
        # This assumes the JSON is saved next to the model PKL with a consistent naming convention
        json_path = self.model_path.replace(".pkl", ".json").replace("nasdaq_general_model",
                                                                     "feature_cols_nasdaq_general_model")

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Feature column file not found: {json_path}. "
                                    "Please ensure model_trainer.py saved the feature columns.")
        with open(json_path, "r") as f:
            feature_cols = json.load(f)
        print(f"[Evaluator] Loaded {len(feature_cols)} feature columns from: {json_path}")
        return feature_cols

    def evaluate_model_performance(self, output_csv: str = "summary_model_performance.csv"):
        symbols = self.test_data_manager.get_available_symbols()
        results = []

        print(f"🔍 Found {len(symbols)} test stocks. Evaluating model performance...")
        for symbol in tqdm(symbols, desc="Evaluating stocks"):
            try:
                df = self.test_data_manager.load_feature_file(symbol)

                if df is None or df.empty:
                    print(f"[Evaluator] Skipping {symbol}: No data or empty DataFrame after processing.")
                    continue

                # Ensure all required feature columns are present in the DataFrame for prediction
                # Filter out features that might be missing for this specific DataFrame (e.g., due to short data)
                features_for_this_df = [col for col in self.feature_cols if col in df.columns]

                if not features_for_this_df:
                    print(f"[Evaluator] Skipping {symbol}: No valid features found in DataFrame to make predictions.")
                    continue

                missing_cols_for_pred = [col for col in self.feature_cols if col not in features_for_this_df]
                if missing_cols_for_pred:
                    print(
                        f"[Evaluator] Warning for {symbol}: Missing features for prediction: {missing_cols_for_pred}. "
                        "Using available features only.")

                X = df[features_for_this_df]
                y_true = df[self.label_col]

                if X.empty or y_true.empty:
                    print(f"[Evaluator] Skipping {symbol}: Feature or target data is empty after filtering.")
                    continue

                # Ensure X has the same number of features as the model was trained with
                # This is crucial for consistent prediction.
                # If the loaded model's expected features (from self.feature_cols) differ from
                # what's available in X for the current DataFrame, you need to handle it.
                # The ideal scenario is that self.feature_cols already matches what the model expects.
                if len(features_for_this_df) != len(self.feature_cols):
                    print(f"[Evaluator] Warning: Number of features for {symbol} ({len(features_for_this_df)}) "
                          f"does not match trained model's features ({len(self.feature_cols)}). This may cause issues.")
                    # A more robust solution might involve re-aligning columns or padding,
                    # but for now, we proceed with available features hoping for the best or skipping.
                    # Given the fix to `data_manager.py`, this warning should appear less often.

                y_pred = self.model.predict(X)

                metrics = {
                    "symbol": symbol,
                    "accuracy": accuracy_score(y_true, y_pred),
                    "precision": precision_score(y_true, y_pred, zero_division=0),
                    "recall": recall_score(y_true, y_pred, zero_division=0),
                    "f1": f1_score(y_true, y_pred, zero_division=0),
                    "n_samples": len(df)
                }
                results.append(metrics)
            except Exception as e:
                print(f"[Evaluator] Failed on {symbol}: {e}")
                import traceback
                traceback.print_exc()  # Print full traceback for debugging

        summary_df = pd.DataFrame(results)
        summary_df.to_csv(output_csv, index=False)
        print(f"[Evaluator] Saved summary to {output_csv}")

        if not summary_df.empty and "accuracy" in summary_df.columns:
            avg_accuracy = summary_df["accuracy"].mean()
            print(f"[Evaluator] Average accuracy across {len(summary_df)} stocks: {avg_accuracy:.4f}")
        else:
            print("[Evaluator] No valid results to summarize.")

        return summary_df


if __name__ == "__main__":
    # This should point to the 'features' sub-directory of the testing set
    TEST_FEATURE_DIR = "models/NASDAQ-testing set/features"

    # This should point to the exact model file you just created
    MODEL_PATH = "models/NASDAQ-training set/features/nasdaq_general_model_lgbm_tech-400stocks.pkl"

    test_data_manager = DataManager(TEST_FEATURE_DIR, label="Test")
    evaluator = ModelEvaluator(MODEL_PATH, test_data_manager)

    summary_df = evaluator.evaluate_model_performance(output_csv="summary_model_performance.csv")
