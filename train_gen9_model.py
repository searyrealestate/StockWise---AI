
"""
Train Gen-9 "Sniper" LSTM (Universal Tech Model)
================================================
Trains a Generalized Deep Learning Core on a basket of Tech Stocks.
Target: 15-Day Directional Prediction.
"""

import pandas as pd
import numpy as np
import logging
import joblib 
from stockwise_ai_core import DataPreprocessor, StockWiseAI
import yfinance as yf
from sklearn.utils import class_weight
import system_config as cfg

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Trainer")

def train_model(cutoff_date=None):
    from data_source_manager import DataSourceManager
    from continuous_learning_analyzer import calculate_future_outcomes
    from feature_engine import RobustFeatureCalculator
    from sklearn.utils import shuffle
    
    symbols = cfg.TRAINING_SYMBOLS

    dsm = DataSourceManager()
    
    all_X = []
    all_y = []
    
    # Define Features
    from stockwise_ai_core import GEN9_FEATURES as features 
    
    # --- 1. COLLECT DATA ---
    for symbol in symbols:
        logger.info(f"Processing Training Data for {symbol}...")
        
        # Fetch
        # data_map = dsm.fetch_data_sequential([symbol])
        # --- MODIFIED: Force 7 Years of History ---
        start_date = "2018-01-01"  # 7 Years Back
        logger.info(f"Downloading {symbol} Daily data from {start_date}...")
        # Force Daily Data (interval="1d") which has NO limit
        df = yf.download(symbol, start=start_date, interval="1d", progress=False)

        # df = data_map.get(symbol)
        
        if df is None or df.empty:
            logger.warning(f"Skipping {symbol} (No Data)")
            continue
            
        # Cleanup Columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df.columns.values]
        df.columns = [c.lower() for c in df.columns]
        
        new_cols = {}
        for c in df.columns:
            if 'close' in c: new_cols[c] = 'close'
            elif 'open' in c: new_cols[c] = 'open'
            elif 'high' in c: new_cols[c] = 'high'
            elif 'low' in c: new_cols[c] = 'low'
            elif 'volume' in c: new_cols[c] = 'volume'
        df.rename(columns=new_cols, inplace=True)

        # Calculate the daily return and volume change - percentage change
        df['daily_return'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()

        # --- APPLY DATE CUTOFF ---
        if cutoff_date:
            original_len = len(df)
            df = df[df.index < cutoff_date]
            logger.info(f"   ✂️ Cutoff applied: {original_len} -> {len(df)} rows (Stopped at {cutoff_date})")
        
        # Calc Features
        calc = RobustFeatureCalculator(params={})
        df = calc.calculate_features(df)
        df.dropna(inplace=True)
        
        # Ensure Feature Columns Exist
        valid_symbol = True
        for f in features:
            if f not in df.columns:
                # Try simple mapping
                if f.upper() in df.columns: df[f] = df[f.upper()]
                elif 'ADX' in df.columns and f == 'adx_14': df[f] = df['ADX']
                elif 'RSI' in df.columns and f == 'rsi_14': df[f] = df['RSI']
                else:
                    logger.warning(f"Missing feature {f} for {symbol}")
                    valid_symbol = False
                    
        if not valid_symbol: continue

        # Label Generation
        labels = calculate_future_outcomes(df, lookahead=15)
        
        # Align lengths
        valid_len = len(labels) - 15
        labels = labels[:valid_len]
        df_train = df.iloc[:valid_len]
        
        if len(df_train) < 100:
            logger.warning(f"Not enough data for {symbol}")
            continue
            
        y_raw = np.array(labels)
        y_cat = np.eye(3)[y_raw]
        
        # Preprocessing (Per-Symbol Scaling is CRITICAL for Universal Model)
        preprocessor = DataPreprocessor(lookback=60, feature_cols=features)
        
        # Determine flat based on rudimentary check (will double check later)
        # For now, generate the 3D sequence, reshape later if needed
        X_sym = preprocessor.fit_transform(df_train, flat=False) 
        
        # Align y to X (X loses 'lookback' frames at start)
        y_sym = y_cat[60:]
        
        if len(X_sym) != len(y_sym):
            # Fallback trim
            min_len = min(len(X_sym), len(y_sym))
            X_sym = X_sym[:min_len]
            y_sym = y_sym[:min_len]
            
        all_X.append(X_sym)
        all_y.append(y_sym)
        logger.info(f"Added {len(X_sym)} samples from {symbol}")

    # --- 2. COMBINE & TRAIN ---
    if not all_X:
        logger.error("No training data collected!")
        return

    X_final = np.concatenate(all_X)
    y_final = np.concatenate(all_y)
    
    # Shuffle to break symbol correlations
    X_final, y_final = shuffle(X_final, y_final, random_state=42)
    
    logger.info(f"UNIVERSAL DATASET: {X_final.shape} Samples")
    
    # Initialize Core
    ai = StockWiseAI(lookback=60)
    
    # Check Backend (If Sklearn, flatten. If TF, keep 3D)
    if ai.backend == "sklearn":
        # Flatten: (N, 60, 4) -> (N, 240)
        X_final = X_final.reshape(X_final.shape[0], -1)
    
    logger.info(f"Training Universal Model (Backend: {ai.backend})...")
    
    ai.build_model(input_shape=(60, len(features)))
    logger.info(f"Training Universal Model (Backend: {ai.backend})...")
    
    ai.build_model(input_shape=(60, len(features)))

    # --- NEW CODE START: Class Weights ---
    # This prevents the "Lazy Student" problem where AI just guesses WAIT every time.
    from sklearn.utils import class_weight
    
    # Convert one-hot (e.g., [0, 0, 1]) back to integers (0, 1, 2)
    y_integers = np.argmax(y_final, axis=1)
    
    # Calculate mathematically balanced weights
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_integers),
        y=y_integers
    )
    class_weight_dict = dict(enumerate(class_weights))

    # Manually boost the importance of BUY (2) and SELL (0) signals
    # We tell the AI: "A missed Buy is 2x worse than a missed Wait"
    if 0 in class_weight_dict: class_weight_dict[0] *= 0.5  # SELL
    if 2 in class_weight_dict: class_weight_dict[2] *= 6.0  # BUY
    if 1 in class_weight_dict: class_weight_dict[1] *= 0.5  # WAIT (Less important)

    logger.info(f"⚖️ Applied Aggressive Class Weights: {class_weight_dict}")
    
    # Pass the weights to the training function
    ai.train(X_final, y_final, class_weight=class_weight_dict)
    
    # --- SAVE ---
    # 1. Save Primary (based on backend)
    model_path_tf = "models/gen9_model_universal.keras"
    model_path_sklearn = "models/gen9_model_universal.pkl"
    
    if ai.backend == "tensorflow":
        ai.model.save(model_path_tf)
        logger.info(f"Primary TF Model Saved to {model_path_tf}")
        
        # 2. Force Train & Save SKLearn Fallback (Critical for robust deployment)
        logger.info("Training Backup SKLearn Model...")
        from sklearn.neural_network import MLPClassifier
        # Flatten for SKLearn
        X_flat = X_final.reshape(X_final.shape[0], -1)
        y_flat = np.argmax(y_final, axis=1)
        
        backup_model = MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=200, random_state=42)
        backup_model.fit(X_flat, y_flat)
        joblib.dump(backup_model, model_path_sklearn)
        logger.info(f"Backup SKLearn Model Saved to {model_path_sklearn}")
        
    else:
        # Backend is already SKLearn
        joblib.dump(ai.model, model_path_sklearn)
        logger.info(f"Primary SKLearn Model Saved to {model_path_sklearn}")

if __name__ == "__main__":
    train_model()
