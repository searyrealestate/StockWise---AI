# how to run the analyze_model.py?
# python analyze_model.py --model-dir "models/NASDAQ-gen3-2pct"
# python analyze_model.py --model-dir "models/NASDAQ-gen3-3pct"
# python analyze_model.py --model-dir "models/NASDAQ-gen3-4pct"
# python analyze_model.py --model-dir "models/NASDAQ-gen3-dynamic"

import pandas as pd
import joblib
import json
import os
import glob
import argparse  # NEW: Import argparse
from tqdm import tqdm

# --- Configuration for Gen-3 ---
# MODIFIED: OUTPUT_FILE is now defined here, MODEL_DIR is removed
OUTPUT_FILE = "gen3_feature_importance_summary.csv"


def analyze_all_feature_importances(model_dir: str):
    """
    Loads all Gen-3 specialist models from a specified directory,
    analyzes their feature importances, and consolidates the results.
    """
    all_importances = []

    # 1. Find all model files in the specified directory
    model_files = glob.glob(os.path.join(model_dir, "*.pkl"))
    if not model_files:
        print(f"❌ Error: No model files found in '{model_dir}'. Please run the model trainer first.")
        return

    print(f"🔍 Found {len(model_files)} specialist models. Analyzing feature importances...")

    # 2. Loop through each model and extract importance
    for model_path in tqdm(model_files, desc="Processing models"):
        model_name = os.path.basename(model_path).replace(".pkl", "")
        features_path = model_path.replace(".pkl", "_features.json")

        try:
            model = joblib.load(model_path)
            with open(features_path, 'r') as f:
                feature_names = json.load(f)

            importances = model.feature_importances_

            df_temp = pd.DataFrame({
                'model_name': model_name,
                'feature_name': feature_names,
                'importance': importances
            })
            all_importances.append(df_temp)

        except FileNotFoundError:
            print(f"⚠️ Warning: Missing features file for {model_name}. Skipping.")
        except Exception as e:
            print(f"❌ Error analyzing model {model_name}: {e}. Skipping.")

    if not all_importances:
        print("❌ No feature importance data could be extracted.")
        return

    # 3. Consolidate and rank the data
    combined_df = pd.concat(all_importances, ignore_index=True)
    average_importance = combined_df.groupby('feature_name')['importance'].mean().reset_index()
    average_importance.rename(columns={'importance': 'average_importance'}, inplace=True)
    average_importance.sort_values(by='average_importance', ascending=False, inplace=True)
    average_importance['global_rank'] = range(1, len(average_importance) + 1)

    print("\n## 💡 Consolidated Feature Importance Analysis")
    print(f"\nThis table shows the average importance of each feature across all models in '{model_dir}'.")

    # 4. Save the report to a CSV file inside the specified model directory
    # MODIFIED: Use the dynamic 'model_dir' for the output path
    output_path = os.path.join(model_dir, OUTPUT_FILE)
    average_importance.to_csv(output_path, index=False)
    print(f"\n✅ Detailed feature importance report saved to: {output_path}")

    # 5. Print a summary to the console
    print("\n### Top 10 Most Important Features (Across All Models)")
    print(average_importance.head(10).to_markdown(index=False))


if __name__ == "__main__":
    # NEW: Use argparse to get the model directory from the user
    parser = argparse.ArgumentParser(description="Analyze feature importance for a set of Gen-3 models.")
    parser.add_argument(
        '--model-dir',
        required=True,
        type=str,
        help='Directory path where the trained models are saved (e.g., "models/NASDAQ-gen3-2pct").'
    )
    args = parser.parse_args()

    # Call the main function with the user-provided directory
    analyze_all_feature_importances(args.model_dir)