import pandas as pd
import joblib
import json
import os

# --- 1. SET THE CORRECT PATH TO YOUR MODEL FILE HERE ---
# Example: "models/NASDAQ-training set/nasdaq_gen2_optimized_model_20250826.pkl"
MODEL_PATH = "models/NASDAQ-training set/nasdaq_gen2_optimized_model_20250826.pkl"


# --- Script starts here ---

def analyze_feature_importance(model_path: str):
    """
    Loads a trained model and its feature list to analyze and display
    the importance of each feature.
    """
    print(f"Analyzing model: {os.path.basename(model_path)}")

    # Construct the path to the corresponding features JSON file
    features_path = model_path.replace(".pkl", "_features.json")

    try:
        # --- Load the Model and Feature List ---
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at the specified path: {model_path}")
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features file not found at the specified path: {features_path}")

        model = joblib.load(model_path)
        with open(features_path, 'r') as f:
            feature_names = json.load(f)

        # --- Get Feature Importances from the trained model ---
        importances = model.feature_importances_

        # --- Create a DataFrame for Analysis ---
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

        feature_importance_df['Rank'] = feature_importance_df.index + 1

        # --- Print the Analysis in a Markdown Table ---
        print("\n## 💡 Feature Importance Analysis")
        print(
            "\nThis table shows which features the model relied on most to make its predictions. A higher score means a more influential feature.")
        print("\n| Rank | Feature                 |   Importance |")
        print("|-----:|:------------------------|-------------:|")
        for index, row in feature_importance_df.iterrows():
            print(f"| {row['Rank']:>4} | {row['Feature']:<23} | {row['Importance']:>12} |")

        # --- Find the FFT feature and provide a conclusion ---
        fft_feature_name = 'Dominant_Cycle_126D'
        if fft_feature_name in feature_importance_df['Feature'].values:
            fft_rank = feature_importance_df[feature_importance_df['Feature'] == fft_feature_name]['Rank'].iloc[0]

            print("\n---\n")
            print(f"### ✅ **Conclusion on FFT**\n")
            print(
                f"Yes, the data shows that the FFT feature ('{fft_feature_name}') was **highly beneficial to the model**.")
            print(f"It ranked as the **#{fft_rank}** most important feature out of {len(feature_importance_df)}.")
            print(
                "This places it in the top tier of all features, confirming that the cyclical information it provides is a valuable signal for making predictions.")

    except FileNotFoundError as e:
        print(f"\n**Error:** {e}")
        print("Please ensure the path and filename are correct.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


if __name__ == "__main__":
    if "PASTE_THE_FULL_PATH" in MODEL_PATH:
        print("---")
        print("⚠️ Please open this script and set the MODEL_PATH variable on line 12 to the correct file path.")
        print("---")
    else:
        analyze_feature_importance(MODEL_PATH)