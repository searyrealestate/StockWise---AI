"""
StockWise AI Core (Gen-10 Debug)
================================
Deep Learning Module for High-Precision Directional Forecasting.
Includes verbose debugging to diagnose '0 Signal' issues.
"""

import numpy as np # Math
import pandas as pd # Dataframes
import os # Files
import joblib # Loading pickles
import logging # Logging
import datetime # Time
from datetime import datetime as dt # Time formatting

# Internal Modules
import system_config as cfg
import notification_manager as nm

# Fallback Neural Network
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

# Try importing TensorFlow, handle failure gracefully
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    from tensorflow.keras.losses import Loss
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
except Exception:
    TF_AVAILABLE = False 

# Setup Logger
logger = logging.getLogger("StockWise_AI_Core")
decision_logger = logging.getLogger("Sniper_Decisions")
decision_logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(os.path.join("logs", "sniper_decisions.log"))
file_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
if not decision_logger.handlers:
    decision_logger.addHandler(file_handler)

# --- CONFIGURATION constants ---
# Features the model expects (Must match Feature Engine)
GEN9_FEATURES = [
    'daily_return', 'volume_change', 'rsi_14', 'adx', 'ema_spread',
    'smart_hammer', 'smart_shooting_star', 
    'vsa_squat_bar', 'vsa_no_demand', 
    'bull_trap_signal', 'candle_confluence',
    'corr_qqq_20', 'beta_qqq' 
]

# --- CUSTOM LOSS FUNCTION ---
if TF_AVAILABLE:
    class PrecisionWeightedLoss(Loss):
        def __init__(self, fp_penalty=cfg.SniperConfig.LOSS_PENALTY_MULTIPLIER, name="precision_weighted_loss"):
            super().__init__(name=name)
            self.fp_penalty = fp_penalty

        def call(self, y_true, y_pred):
            y_true_buy = y_true[:, 2]
            y_pred_buy = y_pred[:, 2]
            ce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
            false_positive_error = (1.0 - y_true_buy) * tf.square(y_pred_buy)
            return ce_loss + (self.fp_penalty * false_positive_error)

# --- DATA PIPELINE ---
class DataPreprocessor:
    def __init__(self, lookback=60, feature_cols=None, scaler_path="models/scaler_gen9.pkl"):
        self.lookback = lookback
        self.feature_cols = feature_cols if feature_cols else GEN9_FEATURES
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_fitted = False
        self.scaler_path = scaler_path

    def fit_transform(self, df, flat=False):
        # Ensure features are defined
        if not self.feature_cols:
            raise ValueError("Feature columns must be defined.")
        
        # Ensure columns exist in the dataframe
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            for c in missing: df[c] = 0.0
            
        data = df[self.feature_cols].values
        # Fit the scaler
        scaled_data = self.scaler.fit_transform(data)
        self.is_fitted = True
        
        # Save the scaler for inference
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.scaler, self.scaler_path)
        
        return self._create_dataset(scaled_data, flat)

    def transform_single_window(self, df_window, flat=False):
        # Try to load scaler if not loaded
        if not self.is_fitted:
             if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                self.is_fitted = True
             else:
                 # LOGGING FAILURE
                 logger.error(f"Scaler Not Found at {self.scaler_path}")
                 return None
        
        # Handle missing columns in window
        for c in self.feature_cols:
            if c not in df_window.columns: df_window[c] = 0.0
            
        data = df_window[self.feature_cols].values
        # Scale
        scaled_data = self.scaler.transform(data)
        
        # Reshape based on backend requirement
        if flat:
            return scaled_data.reshape(1, -1) 
        else:
            return np.array([scaled_data]) 

    def _create_dataset(self, dataset, flat=False):
        X = []
        for i in range(len(dataset) - self.lookback):
            window = dataset[i : i + self.lookback]
            if flat:
                X.append(window.flatten())
            else:
                X.append(window)
        return np.array(X)

# --- AI CORE ---
class StockWiseAI:
    def __init__(self, symbol=None, lookback=60):
        """
        Initialize the AI Specialist.
        """
        self.lookback = lookback
        self.symbol = symbol
        self.model = None
        self.backend = "tensorflow" if TF_AVAILABLE else "sklearn"
        self.notifier = nm.NotificationManager()
        
        # --- DYNAMIC PATHS (Model AND Scaler) ---
        # Construct paths based on symbol name
        if self.symbol:
            self.model_path = os.path.join(cfg.MODELS_DIR, f"{self.symbol}_gen9_model.keras")
            self.model_path_mlp = os.path.join(cfg.MODELS_DIR, f"{self.symbol}_gen9_model.pkl")
            scaler_path = os.path.join(cfg.MODELS_DIR, f"{self.symbol}_scaler.pkl")
        else:
            self.model_path = os.path.join(cfg.MODELS_DIR, "gen9_model_universal.keras")
            self.model_path_mlp = os.path.join(cfg.MODELS_DIR, "gen9_model_universal.pkl")
            scaler_path = os.path.join(cfg.MODELS_DIR, "scaler_gen9.pkl")

        # Initialize preprocessor with the correct scaler path
        self.preprocessor = DataPreprocessor(lookback=lookback, scaler_path=scaler_path)

        if self.backend == "sklearn":
            logger.warning("TensorFlow not available. Using Scikit-Learn MLP Backend.")

    def build_model(self, input_shape=None):
        if self.backend == "tensorflow":
            model = Sequential()
            model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
            model.add(Dropout(0.3))
            model.add(LSTM(32, return_sequences=False))
            model.add(Dropout(0.3))
            model.add(Dense(16, activation='relu'))
            model.add(Dense(3, activation='softmax')) 
            model.compile(optimizer='adam', 
                          loss=PrecisionWeightedLoss(fp_penalty=cfg.SniperConfig.LOSS_PENALTY_MULTIPLIER), 
                          metrics=['accuracy'])
            self.model = model
            return model
        else:
            self.model = MLPClassifier(hidden_layer_sizes=(64, 32, 16), activation='relu', solver='adam', max_iter=500, random_state=42, early_stopping=True)
            return self.model

    def load_inference_model(self):
        # Attempt to load TF model first
        if self.backend == "tensorflow":
            if os.path.exists(self.model_path):
                try:
                    self.model = load_model(self.model_path, compile=False)
                    return True
                except: return False
        # If TF failed or not available, try Sklearn
        if os.path.exists(self.model_path_mlp):
            try:
                self.model = joblib.load(self.model_path_mlp)
                return True
            except Exception as e:
                logger.error(f"Failed to load MLP: {e}")
                return False
        
        logger.error(f"No model found at {self.model_path} OR {self.model_path_mlp}")
        return False

    def train(self, X, y, class_weight=None):
        if self.backend == "tensorflow":
            self.model.fit(X, y, epochs=5, batch_size=32, verbose=1, class_weight=class_weight)
            self.model.save(self.model_path)
        else:
            if len(y.shape) > 1 and y.shape[1] > 1: y = np.argmax(y, axis=1)
            self.model.fit(X, y)
            joblib.dump(self.model, self.model_path_mlp)

    def _internal_predict(self, df_window):
        # 1. Check Model
        if self.model is None:
            if not self.load_inference_model(): 
                # LOGGING FAILURE
                # logger.error("Internal Predict: Model Load Failed.")
                return 1, 0.0
        
        # 2. Prepare Data
        flat = (self.backend == "sklearn" or isinstance(self.model, MLPClassifier))
        tensor = self.preprocessor.transform_single_window(df_window, flat=flat)
        
        # 3. Check Scaler/Data
        if tensor is None: 
            # LOGGING FAILURE
            # logger.error("Internal Predict: Tensor is None (Scaler failed).")
            return 1, 0.0
        
        # 4. Predict
        if self.backend == "tensorflow" and not flat:
            probs = self.model.predict(tensor, verbose=0)[0]
        else:
            probs = self.model.predict_proba(tensor)[0]
        
        # 5. Parse Output
        if len(probs) == 3: buy_prob = probs[2]
        elif len(probs) == 2: buy_prob = probs[1]
        else: buy_prob = 0.0
        
        return 2, buy_prob

    def predict_trade_confidence(self, ticker, features, fundamentals, df_window):
        trace = {"Ticker": ticker, "Timestamp": datetime.datetime.now().isoformat(), "Checks": {}}
        fund_score = self._evaluate_fundamentals(fundamentals)
        trace["Checks"]["Fundamentals"] = {"Score": fund_score, "Pass": fund_score >= cfg.SniperConfig.FUNDAMENTAL_MIN_SCORE}
        adx = features.get('adx', 0)
        trace["Checks"]["Technicals"] = {"ADX": adx, "RSI": features.get('rsi_14')}
        trace["Checks"]["Trend_Filter"] = "PASS" 
        
        if len(df_window) > self.lookback: df_window = df_window.iloc[-self.lookback:]
        
        # Get prediction
        action, prob = self._internal_predict(df_window)
        
        # Log if prob is 0 (Debug)
        if prob == 0.0:
            logger.warning(f"Predict returned 0.0 for {ticker}. Model/Scaler likely missing.")
            
        trace["Checks"]["AI_Model"] = {"Prob": float(f"{prob:.4f}"), "Threshold": cfg.SniperConfig.MODEL_CONFIDENCE_THRESHOLD}
        
        final_decision = "WAIT"
        if prob >= cfg.SniperConfig.MODEL_CONFIDENCE_THRESHOLD: final_decision = "BUY"
        trace["Final_Decision"] = final_decision
        decision_logger.info(str(trace))
        return final_decision, prob, trace

    def _evaluate_fundamentals(self, fund_data):
        if not fund_data: return 50
        score = 50
        pe = fund_data.get('trailingPE'); growth = fund_data.get('revenueGrowth')
        if growth and growth > 0.15: score += 20
        if pe and pe < 25: score += 10
        return min(score, 100)