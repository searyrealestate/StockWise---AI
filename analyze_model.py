import pandas as pd
import joblib
import json
import os
import glob
from tqdm import tqdm

# --- Configuration for Gen-3 ---
MODEL_DIR = "models/NASDAQ-gen3"
OUTPUT_FILE = "gen3_feature_importance_summary.csv"


def analyze_all_feature_importances(model_dir: str):
    """
    Loads all Gen-3 specialist models, analyzes their feature importances,
    and consolidates the results into a single DataFrame and CSV file.
    """
    all_importances = []

    # 1. Find all model files in the dedicated Gen-3 directory
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

            # Create a temporary DataFrame for this model's data
            df_temp = pd.DataFrame({
                'model_name': model_name,
                'feature_name': feature_names,
                'importance': importances
            })
            all_importances.append(df_temp)

        except FileNotFoundError:
            print(f"⚠️ Warning: Missing features file for {model_name}. Skipping.")
            continue
        except Exception as e:
            print(f"❌ Error analyzing model {model_name}: {e}. Skipping.")
            continue

    if not all_importances:
        print("❌ No feature importance data could be extracted.")
        return

    # 3. Consolidate and rank the data
    combined_df = pd.concat(all_importances, ignore_index=True)

    # Calculate the average importance for each feature across all models
    average_importance = combined_df.groupby('feature_name')['importance'].mean().reset_index()
    average_importance.rename(columns={'importance': 'average_importance'}, inplace=True)
    average_importance.sort_values(by='average_importance', ascending=False, inplace=True)

    # Add a global rank
    average_importance['global_rank'] = range(1, len(average_importance) + 1)

    print("\n## 💡 Consolidated Feature Importance Analysis")
    print("\nThis table shows the average importance of each feature across all 9 specialist models.")

    # 4. Save the report to a CSV file
    output_path = os.path.join(MODEL_DIR, OUTPUT_FILE)
    average_importance.to_csv(output_path, index=False)
    print(f"\n✅ Detailed feature importance report saved to: {output_path}")

    # 5. Print a summary to the console
    print("\n### Top 10 Most Important Features (Across All Models)")
    print(average_importance.head(10).to_markdown(index=False))


if __name__ == "__main__":
    analyze_all_feature_importances(MODEL_DIR)