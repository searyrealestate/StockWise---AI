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
        self.feature_cols = self._load_feature_columns()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        return joblib.load(self.model_path)

    def _load_feature_columns(self):
        json_path = self.model_path.replace(".pkl", ".json").replace("nasdaq_general_model", "feature_cols_nasdaq_general_model")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Feature column file not found: {json_path}")
        with open(json_path, "r") as f:
            feature_cols = json.load(f)
        print(f"[Evaluator] Loaded {len(feature_cols)} feature columns from: {json_path}")
        return feature_cols

    def evaluate_on_stocks(self, symbols: list[str], output_csv: str = "model_performance_summary.csv") -> pd.DataFrame:
        results = []

        for symbol in tqdm(symbols, desc="Evaluating stocks"):
            df = self.test_data_manager.load_feature_file(symbol)
            if df is None or self.label_col not in df.columns:
                continue

            try:
                X = df[self.feature_cols]
                y_true = df[self.label_col]
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
    TEST_FEATURE_DIR = "models/NASDAQ-testing set"
    test_data_manager = DataManager(TEST_FEATURE_DIR, label="Test")

    symbols = test_data_manager.get_available_symbols()
    print(f"🔍 Found {len(symbols)} test stocks")

    model_path = "models/400_train_set/nasdaq_general_model_lgbm_tech-400stocks.pkl"

    evaluator = ModelEvaluator(
        model_path=model_path,
        test_data_manager=test_data_manager
    )

    summary = evaluator.evaluate_on_stocks(symbols, output_csv="summary_lgbm_tech-dynamic-days-window-400stocks.csv")
