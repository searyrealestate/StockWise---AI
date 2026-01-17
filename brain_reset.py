# brain_reset.py
"""
StockWise Brain Factory Reset
=============================
Deletes ALL old/incompatible models and generates a fresh 
Universal Model compatible with Gen-10 (13 Features).
"""

import os
import glob
import numpy as np
import joblib
import sys
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

# Ensure we can see system config
sys.path.append(os.getcwd())
import system_config as cfg

# CONFIG: MATCHES GEN-10 ARCHITECTURE EXACTLY
LOOKBACK = 60
FEATURES_COUNT = 13  # The 13 Gen-9 Features
INPUT_SHAPE = LOOKBACK * FEATURES_COUNT # 780

def reset_brain():
    print(f"🧠 STOCKWISE BRAIN FACTORY RESET")
    print(f"=================================")
    print(f"Target Architecture: {FEATURES_COUNT} Features x {LOOKBACK} Lookback = {INPUT_SHAPE} Inputs")
    
    # 1. DELETE OLD MODELS
    # We must be aggressive here. Any old .pkl or .keras file is a liability.
    print("\n🗑️  Cleaning old models...")
    patterns = [
        os.path.join(cfg.MODELS_DIR, "*.pkl"),
        os.path.join(cfg.MODELS_DIR, "*.keras")
    ]
    
    deleted_count = 0
    for pattern in patterns:
        for filepath in glob.glob(pattern):
            try:
                os.remove(filepath)
                print(f"   - Deleted: {os.path.basename(filepath)}")
                deleted_count += 1
            except Exception as e:
                print(f"   ! Error deleting {filepath}: {e}")
                
    if deleted_count == 0:
        print("   (No old models found, clean slate)")
        
    # 2. GENERATE NEW SCALER
    print("\n🛠️  Generating fresh Scaler (13 Features)...")
    # 100 rows, 13 cols of random data [0-100]
    dummy_data_raw = np.random.rand(100, FEATURES_COUNT) * 100 
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(dummy_data_raw)
    
    scaler_path = os.path.join(cfg.MODELS_DIR, "scaler_gen9.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"   -> Saved: {scaler_path}")

    # 3. GENERATE NEW UNIVERSAL AI MODEL
    print(f"\n🧠 Generating fresh Universal AI Model ({INPUT_SHAPE} Inputs)...")
    
    # Create Dummy Training Data
    # X: 100 samples, 780 features (Flattened window)
    X_dummy = np.random.rand(100, INPUT_SHAPE)
    # y: 100 samples, 3 classes (Sell, Hold, Buy) - One-hot encodedish or integers
    y_dummy = np.random.randint(0, 3, size=(100,))
    
    # Initialize and Train MLP
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32), 
        max_iter=5, # Quick dummy train
        random_state=42
    )
    mlp.fit(X_dummy, y_dummy)
    
    # Save Universal Model
    model_path = os.path.join(cfg.MODELS_DIR, "gen9_model_universal.pkl")
    joblib.dump(mlp, model_path)
    print(f"   -> Saved: {model_path}")
    
    print("\n✅ RESET COMPLETE.")
    print("The system is now compatible with the new feature set.")
    print("You can run 'run_live.bat' now.")

if __name__ == "__main__":
    # Ensure directory exists
    os.makedirs(cfg.MODELS_DIR, exist_ok=True)
    reset_brain()