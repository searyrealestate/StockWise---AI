# # train_gen9_model.py

# """
# Train Gen-9 "Sniper" LSTM (Universal Tech Model)
# ================================================
# Trains a Generalized Deep Learning Core on a basket of Tech Stocks.
# Target: 15-Day Directional Prediction.
# """

# import pandas as pd
# import numpy as np
# import logging
# import joblib 
# from stockwise_ai_core import DataPreprocessor, StockWiseAI
# import yfinance as yf
# from sklearn.utils import class_weight
# import system_config as cfg

# # Configure Logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("Trainer")

# def train_model(cutoff_date=None):
#     from data_source_manager import DataSourceManager
#     from continuous_learning_analyzer import calculate_future_outcomes
#     from feature_engine import RobustFeatureCalculator
#     from sklearn.utils import shuffle
    
#     symbols = cfg.TRAINING_SYMBOLS

#     dsm = DataSourceManager()
    
#     all_X = []
#     all_y = []
    
#     # Define Features
#     from stockwise_ai_core import GEN9_FEATURES as features 
    
#     # --- 1. COLLECT DATA ---
#     for symbol in symbols:
#         logger.info(f"Processing Training Data for {symbol}...")
        
#         # Fetch
#         # data_map = dsm.fetch_data_sequential([symbol])
#         # --- MODIFIED: Force 7 Years of History ---
#         start_date = "2018-01-01"  # 7 Years Back
#         logger.info(f"Downloading {symbol} Daily data from {start_date}...")
#         # Force Daily Data (interval="1d") which has NO limit
#         df = yf.download(symbol, start=start_date, interval="1d", progress=False)

#         # df = data_map.get(symbol)
        
#         if df is None or df.empty:
#             logger.warning(f"Skipping {symbol} (No Data)")
#             continue
            
#         # Cleanup Columns
#         if isinstance(df.columns, pd.MultiIndex):
#             df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df.columns.values]
#         df.columns = [c.lower() for c in df.columns]
        
#         new_cols = {}
#         for c in df.columns:
#             if 'close' in c: new_cols[c] = 'close'
#             elif 'open' in c: new_cols[c] = 'open'
#             elif 'high' in c: new_cols[c] = 'high'
#             elif 'low' in c: new_cols[c] = 'low'
#             elif 'volume' in c: new_cols[c] = 'volume'
#         df.rename(columns=new_cols, inplace=True)

#         # Calculate the daily return and volume change - percentage change
#         df['daily_return'] = df['close'].pct_change()
#         df['volume_change'] = df['volume'].pct_change()

#         # --- APPLY DATE CUTOFF ---
#         if cutoff_date:
#             original_len = len(df)
#             df = df[df.index < cutoff_date]
#             logger.info(f"   ✂️ Cutoff applied: {original_len} -> {len(df)} rows (Stopped at {cutoff_date})")
        
#         # Calc Features
#         calc = RobustFeatureCalculator(params={})
#         df = calc.calculate_features(df)
#         df.dropna(inplace=True)
        
#         # Ensure Feature Columns Exist
#         valid_symbol = True
#         for f in features:
#             if f not in df.columns:
#                 # Try simple mapping
#                 if f.upper() in df.columns: df[f] = df[f.upper()]
#                 elif 'ADX' in df.columns and f == 'adx_14': df[f] = df['ADX']
#                 elif 'RSI' in df.columns and f == 'rsi_14': df[f] = df['RSI']
#                 else:
#                     logger.warning(f"Missing feature {f} for {symbol}")
#                     valid_symbol = False
                    
#         if not valid_symbol: continue

#         # Label Generation
#         labels = calculate_future_outcomes(df, lookahead=15)
        
#         # Align lengths
#         valid_len = len(labels) - 15
#         labels = labels[:valid_len]
#         df_train = df.iloc[:valid_len]
        
#         if len(df_train) < 100:
#             logger.warning(f"Not enough data for {symbol}")
#             continue
            
#         y_raw = np.array(labels)
#         y_cat = np.eye(3)[y_raw]
        
#         # Preprocessing (Per-Symbol Scaling is CRITICAL for Universal Model)
#         preprocessor = DataPreprocessor(lookback=60, feature_cols=features)
        
#         # Determine flat based on rudimentary check (will double check later)
#         # For now, generate the 3D sequence, reshape later if needed
#         X_sym = preprocessor.fit_transform(df_train, flat=False) 
        
#         # Align y to X (X loses 'lookback' frames at start)
#         y_sym = y_cat[60:]
        
#         if len(X_sym) != len(y_sym):
#             # Fallback trim
#             min_len = min(len(X_sym), len(y_sym))
#             X_sym = X_sym[:min_len]
#             y_sym = y_sym[:min_len]
            
#         all_X.append(X_sym)
#         all_y.append(y_sym)
#         logger.info(f"Added {len(X_sym)} samples from {symbol}")

#     # --- 2. COMBINE & TRAIN ---
#     if not all_X:
#         logger.error("No training data collected!")
#         return

#     X_final = np.concatenate(all_X)
#     y_final = np.concatenate(all_y)
    
#     # Shuffle to break symbol correlations
#     X_final, y_final = shuffle(X_final, y_final, random_state=42)
    
#     logger.info(f"UNIVERSAL DATASET: {X_final.shape} Samples")
    
#     # Initialize Core
#     ai = StockWiseAI(lookback=60)
    
#     # Check Backend (If Sklearn, flatten. If TF, keep 3D)
#     if ai.backend == "sklearn":
#         # Flatten: (N, 60, 4) -> (N, 240)
#         X_final = X_final.reshape(X_final.shape[0], -1)
    
#     logger.info(f"Training Universal Model (Backend: {ai.backend})...")
    
#     ai.build_model(input_shape=(60, len(features)))
#     logger.info(f"Training Universal Model (Backend: {ai.backend})...")
    
#     ai.build_model(input_shape=(60, len(features)))

#     # --- NEW CODE START: Class Weights ---
#     # This prevents the "Lazy Student" problem where AI just guesses WAIT every time.
#     from sklearn.utils import class_weight
    
#     # Convert one-hot (e.g., [0, 0, 1]) back to integers (0, 1, 2)
#     y_integers = np.argmax(y_final, axis=1)
    
#     # Calculate mathematically balanced weights
#     class_weights = class_weight.compute_class_weight(
#         class_weight='balanced',
#         classes=np.unique(y_integers),
#         y=y_integers
#     )
#     class_weight_dict = dict(enumerate(class_weights))

#     # Manually boost the importance of BUY (2) and SELL (0) signals
#     # We tell the AI: "A missed Buy is 2x worse than a missed Wait"
#     if 0 in class_weight_dict: class_weight_dict[0] *= 0.5  # SELL
#     if 2 in class_weight_dict: class_weight_dict[2] *= 6.0  # BUY
#     if 1 in class_weight_dict: class_weight_dict[1] *= 0.5  # WAIT (Less important)

#     logger.info(f"⚖️ Applied Aggressive Class Weights: {class_weight_dict}")
    
#     # Pass the weights to the training function
#     ai.train(X_final, y_final, class_weight=class_weight_dict)
    
#     # --- SAVE ---
#     # 1. Save Primary (based on backend)
#     model_path_tf = "models/gen9_model_universal.keras"
#     model_path_sklearn = "models/gen9_model_universal.pkl"
    
#     if ai.backend == "tensorflow":
#         ai.model.save(model_path_tf)
#         logger.info(f"Primary TF Model Saved to {model_path_tf}")
        
#         # 2. Force Train & Save SKLearn Fallback (Critical for robust deployment)
#         logger.info("Training Backup SKLearn Model...")
#         from sklearn.neural_network import MLPClassifier
#         # Flatten for SKLearn
#         X_flat = X_final.reshape(X_final.shape[0], -1)
#         y_flat = np.argmax(y_final, axis=1)
        
#         backup_model = MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=200, random_state=42)
#         backup_model.fit(X_flat, y_flat)
#         joblib.dump(backup_model, model_path_sklearn)
#         logger.info(f"Backup SKLearn Model Saved to {model_path_sklearn}")
        
#     else:
#         # Backend is already SKLearn
#         joblib.dump(ai.model, model_path_sklearn)
#         logger.info(f"Primary SKLearn Model Saved to {model_path_sklearn}")

# if __name__ == "__main__":
#     train_model()



"""
Train Gen-9 "Specialist" LSTM Models
====================================
Trains a DEDICATED Deep Learning Core for EACH ticker.
Fixes: Forces generation of BOTH .keras (TF) and .pkl (Sklearn) models to prevent fallback failures.
"""

# Import necessary libraries for data handling, math, logging, and system ops
import pandas as pd  # Dataframes
import numpy as np   # Arrays and math
import logging       # Logging events
import system_config as cfg # Configuration constants
import os            # File system operations
from sklearn.utils import class_weight # For balancing Buy/Sell/Wait classes
from sklearn.neural_network import MLPClassifier
import joblib

# Configure the logging system to show INFO level messages
logging.basicConfig(level=logging.INFO)
# Create a specific logger for the Trainer module
logger = logging.getLogger("Trainer")

def train_model(cutoff_date=None):
    # Lazy imports to avoid circular dependency issues at the top level
    from data_source_manager import DataSourceManager
    from feature_engine import RobustFeatureCalculator
    from stockwise_ai_core import StockWiseAI, GEN9_FEATURES
    
    # 1. SETUP
    # Load the list of symbols to train from config
    symbols = cfg.TRAINING_SYMBOLS 
    # Initialize the Data Source Manager to fetch prices
    dsm = DataSourceManager()
    # Initialize the Feature Calculator to compute indicators
    calc = RobustFeatureCalculator()
    
    # 2. FETCH MARKET CONTEXT
    # We need QQQ data to calculate market correlation and beta
    logger.info("Fetching Market Context (QQQ)...")
    # Download 3000 days of QQQ data
    qqq_df = dsm.get_stock_data("QQQ", days_back=3000, interval='1d')
    
    # Apply cutoff to QQQ as well if needed
    if cutoff_date and not qqq_df.empty:
        qqq_df = qqq_df[qqq_df.index < cutoff_date]
        
    # Store it in a dictionary to pass to the feature engine
    context_data = {'qqq': qqq_df}
    
    # --- 3. TRAINING LOOP ---
    # Iterate through each symbol (e.g., NVDA, AAPL)
    for symbol in symbols:
        # Log which symbol we are currently processing
        logger.info(f"\n{'='*40}")
        logger.info(f"TRAINING SPECIALIST MODEL FOR: {symbol}")
        logger.info(f"{'='*40}")
        
        try:
            # A. Fetch Data
            # Get data for the specific symbol
            df = dsm.get_stock_data(symbol, days_back=3000, interval='1d')
            
            # --- APPLY CUTOFF DATE ---
            # If we are running a strict test, we must hide the future from the AI
            if cutoff_date:
                original_len = len(df)
                df = df[df.index < cutoff_date]
                logger.info(f"Applied Cutoff {cutoff_date}: {original_len} -> {len(df)} rows")
            
            # Check if data is sufficient (need at least 500 bars for valid training)
            if df is None or len(df) < 500:
                logger.warning(f"Skipping {symbol}: Not enough data (Need 500+, Got {len(df)}).")
                continue
                
            # B. Engineering
            # Calculate technical indicators, adding QQQ context
            logger.info(f"Engineering Features...")
            df = calc.calculate_features(df, context_data=context_data)
            
            # C. Labeling (Local Logic to avoid circular import)
            # Define how far into the future we try to predict (15 days)
            lookahead = 15
            # Get the closing price 15 days in the future
            future_close = df['close'].shift(-lookahead)
            # Calculate the percentage return 15 days later
            future_ret = (future_close - df['close']) / df['close']
            
            # Define classification conditions
            conditions = [
                (future_ret > 0.05), # BUY - Class 2: BUY if return > 5%
                (future_ret < -0.05) # SELL - Class 0: SELL if return < -5%
            ]
            # Apply conditions to create the 'outcome_class' column (Default 1: WAIT)
            df['outcome_class'] = np.select(conditions, [2, 0], default=1) 
            # Drop rows with NaN values (mostly the last 15 days with no future data)
            df.dropna(inplace=True)
            
            # D. Initialize Specialist AI
            # Create an AI instance specifically for this symbol
            ai = StockWiseAI(symbol=symbol, lookback=60) 
            
            # Ensure all required features exist in the dataframe
            available_features = [f for f in GEN9_FEATURES if f in df.columns]
            # If some features are missing, verify and fill them
            if len(available_features) < len(GEN9_FEATURES):
                missing = set(GEN9_FEATURES) - set(available_features)
                logger.warning(f"Missing features: {missing}. Filling 0.")
                for m in missing: df[m] = 0.0
            
            # --- CRITICAL FIX: SCALE DATA & SAVE SCALER ---
            # We must use the AI's preprocessor to fit the data.
            # This handles scaling (0-1) and reshaping into windows.
            logger.info(f"Fitting Scaler & Creating Windows for {symbol}...")
            
            # Create tensor: (Samples, 60, 13)
            # fit_transform saves the scaler to models/{symbol}_scaler.pkl
            # We pass flat=False initially to keep structure, but handle flattening later if needed
            X_final = ai.preprocessor.fit_transform(df, flat=False)
            
            # Align Labels (The preprocessor eats the first 'lookback' rows)
            y_raw = df['outcome_class'].values[60:]
            
            # Safety trim if lengths mismatch slightly due to windowing
            min_len = min(len(X_final), len(y_raw))
            X_final = X_final[:min_len]
            y_raw = y_raw[:min_len]
            
            # One-Hot Encode Labels for Neural Network
            y_cat = pd.get_dummies(y_raw)
            # Ensure all columns [0, 1, 2] exist (even if one class is missing in data)
            for col in [0, 1, 2]:
                if col not in y_cat.columns: y_cat[col] = 0
            # Convert to numpy array
            y_final = y_cat[[0, 1, 2]].values
            
            # Check if we have any data left after processing
            if len(X_final) == 0:
                logger.warning(f"No valid data left for {symbol}.")
                continue

            # E. Train
            logger.info(f"Training on {len(X_final)} samples...")
            
            # Calculate Class Weights to handle imbalance (e.g., fewer BUYs than WAITs)
            y_integers = np.argmax(y_final, axis=1)
            class_weights = class_weight.compute_class_weight(
                class_weight='balanced',
                classes=np.unique(y_integers),
                y=y_integers
            )
            weight_dict = dict(zip(np.unique(y_integers), class_weights))
            
            # Aggressive Boost: Tell AI that BUY signals are extra important
            if 2 in weight_dict: weight_dict[2] *= 1.3
            
            # --- F. DUAL MODEL SAVING (THE FIX) ---
            # 1. Train and Save TensorFlow Model (if available)
            if ai.backend == "tensorflow":
                logger.info("Training Primary TensorFlow Model...")
                # Define input shape for LSTM
                input_shape = (60, len(GEN9_FEATURES))
                # ai.build_model(input_shape=input_shape)
                ai.build_model(input_shape=(60, len(GEN9_FEATURES)))
                ai.train(X_final, y_final, class_weight=weight_dict)
                # Ensure it is saved
                ai.model.save(ai.model_path)
                logger.info(f"Saved TF Model: {ai.model_path}")
            
            # 2. ALWAYS Train and Save Scikit-Learn Fallback Model
            # This ensures verify_sniper_logic.py works even if TF fails
            logger.info("Training Fallback Scikit-Learn Model...")
                                                          
            
            # Flatten X for MLP (Samples, 60*13)
            X_flat = X_final.reshape(X_final.shape[0], -1)
            # Convert Y to 1D array for Sklearn
            y_flat = np.argmax(y_final, axis=1)
            
            # Create MLP Classifier
            mlp = MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=200, random_state=42)
            mlp.fit(X_flat, y_flat)
            
            # Save to the .pkl path defined in the AI core
            
            joblib.dump(mlp, ai.model_path_mlp)
            logger.info(f"Saved Sklearn Fallback: {ai.model_path_mlp}")
            
        except Exception as e:
            # Catch any errors for this symbol so we don't crash the whole loop
            logger.error(f"Failed to train {symbol}: {e}", exc_info=True)

    logger.info("All Specialist Models Trained.")

if __name__ == "__main__":
    train_model()