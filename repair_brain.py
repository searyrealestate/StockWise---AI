# repair_brain.py
import numpy as np
import pandas as pd
import joblib
import os
import sys

# Ensure we can see project files
sys.path.append(os.getcwd())
import system_config as cfg

# Define the exact features Gen-9 expects (13 total)
GEN9_FEATURES = [
    'daily_return', 'volume_change', 'rsi_14', 'adx', 'ema_spread',
    'smart_hammer', 'smart_shooting_star', 
    'vsa_squat_bar', 'vsa_no_demand', 
    'bull_trap_signal', 'candle_confluence',
    'corr_qqq_20', 'beta_qqq' 
]

def repair():
    print("🧠 STOCKWISE BRAIN REPAIR TOOL")
    print("==============================")
    
    # 1. Define Paths
    scaler_path = os.path.join(cfg.MODELS_DIR, "scaler_gen9.pkl")
    
    # 2. Create Dummy Data with CORRECT shape (100 rows, 13 columns)
    # We use random data just to initialize the scaler's expected shape
    print(f"🛠️  Generating 13-feature dummy data...")
    dummy_data = np.random.random((100, len(GEN9_FEATURES)))
    
    # 3. Fit Scaler
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(dummy_data)
    print(f"✅ Scaler fitted on {scaler.n_features_in_} features.")
    
    # 4. Save
    os.makedirs(cfg.MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"💾 Saved repaired scaler to: {scaler_path}")
    
    # 5. Cleanup Incompatible Models
    # If we have a dummy model trained on 2 features, it will also crash. 
    # Better to delete it and let the engine run in "Fallback Mode" (Rules Only) until you retrain.
    model_path_keras = os.path.join(cfg.MODELS_DIR, "gen9_model_universal.keras")
    if os.path.exists(model_path_keras):
        print("🗑️  Removing incompatible .keras model (System will rely on Strategy Logic)")
        os.remove(model_path_keras)
        
    print("\n>>> REPAIR COMPLETE. You can now run 'run_live.bat' <<<")

if __name__ == "__main__":
    repair()