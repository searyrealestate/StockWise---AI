
"""
StockWise AI Core (Gen-9)
=========================
Deep Learning Module for High-Precision Directional Forecasting.
Implements:
1. "Lookback-Aware" Data Pipeline (Adaptive Tensor/Flat Generation)
2. Precision-Weighted Logic
3. Dual-Backend: TensorFlow (Primary) -> Scikit-Learn MLP (Fallback)
4. "Fusion" Logic: Fundamentals + Technicals + AI
5. Deep Observability: "Thought Logger" & Telegram Alerts

Author: StockWise Gen-9 Agent
"""

import numpy as np
import pandas as pd
import os
import joblib
import logging
import datetime
from datetime import datetime as dt

# Internal Modules
import system_config as cfg
import notification_manager as nm

# Fallback Neural Network
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    from tensorflow.keras.losses import Loss
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
except Exception:
    TF_AVAILABLE = False # Catch DLL errors

# Setup Logger
logger = logging.getLogger("StockWise_AI_Core")
decision_logger = logging.getLogger("Sniper_Decisions")
decision_logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(os.path.join("logs", "sniper_decisions.log"))
file_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
if not decision_logger.handlers:
    decision_logger.addHandler(file_handler)

# --- CONFIGURATION constants ---
# --- CONFIGURATION constants ---
GEN9_FEATURES = [
    'daily_return', 'volume_change', 'rsi_14', 'adx', 'ema_spread',
    'smart_hammer', 'smart_shooting_star', 
    'vsa_squat_bar', 'vsa_no_demand', 
    'bull_trap_signal', 'candle_confluence'
]

# --- CUSTOM LOSS FUNCTION ---
if TF_AVAILABLE:
    class PrecisionWeightedLoss(Loss):
        """
        Custom Loss to maximize Precision.
        Penalizes False Positives (Bad Buys) 10x more than False Negatives.
        """
        def __init__(self, fp_penalty=cfg.SniperConfig.LOSS_PENALTY_MULTIPLIER, name="precision_weighted_loss"):
            super().__init__(name=name)
            self.fp_penalty = fp_penalty

        def call(self, y_true, y_pred):
            # One-Hot Encoding assumed: [SELL, WAIT, BUY]
            # Index 2 is BUY.
            
            # Extract BUY probability
            y_true_buy = y_true[:, 2]
            y_pred_buy = y_pred[:, 2]
            
            # Standard Cross Entropy
            ce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
            
            # False Positive Penalty:
            # If Truth says NOT BUY (y_true_buy == 0) but Pred says BUY (y_pred_buy high)
            # We add a penalty term.
            false_positive_error = (1.0 - y_true_buy) * tf.square(y_pred_buy)
            
            return ce_loss + (self.fp_penalty * false_positive_error)

# --- DATA PIPELINE ---
class DataPreprocessor:
    """
    Handles scaling and reshaping.
    """
    def __init__(self, lookback=60, feature_cols=None):
        self.lookback = lookback
        self.feature_cols = feature_cols if feature_cols else GEN9_FEATURES
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_fitted = False
        self.scaler_path = "models/scaler_gen9.pkl"

    def fit_transform(self, df, flat=False):
        """Fit scaler on Training Data and transform it."""
        if not self.feature_cols:
            raise ValueError("Feature columns must be defined.")
        
        data = df[self.feature_cols].values
        scaled_data = self.scaler.fit_transform(data)
        self.is_fitted = True
        
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.scaler, self.scaler_path)
        
        return self._create_dataset(scaled_data, flat)

    def transform(self, df, flat=False):
        """Transform Live/Test Data."""
        if not self.is_fitted:
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                self.is_fitted = True
            else:
                raise ValueError("Scaler must be fitted fit!")
                
        data = df[self.feature_cols].values
        scaled_data = self.scaler.transform(data)
        
        return self._create_dataset(scaled_data, flat)
    
    def transform_single_window(self, df_window, flat=False):
        """Transform single window for Inference."""
        if not self.is_fitted:
             if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                self.is_fitted = True
             else:
                 return None
        
        data = df_window[self.feature_cols].values
        scaled_data = self.scaler.transform(data)
        
        if flat:
            return scaled_data.reshape(1, -1) # (1, lookback * features)
        else:
            return np.array([scaled_data]) # (1, lookback, features)

    def _create_dataset(self, dataset, flat=False):
        """Convert flat array to Tensor or Flattened Window."""
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
    def __init__(self, lookback=60):
        self.lookback = lookback
        self.model = None
        self.preprocessor = DataPreprocessor(lookback=lookback)
        self.model_path = "models/gen9_model_universal.keras"
        self.model_path_mlp = "models/gen9_model_universal.pkl"
        self.backend = "tensorflow" if TF_AVAILABLE else "sklearn"
        self.notifier = nm.NotificationManager()
        
        if self.backend == "sklearn":
            logger.warning("TensorFlow not available. Using Scikit-Learn MLP Backend.")

    def build_model(self, input_shape=None):
        """Build Model (LSTM or MLP)."""
        if self.backend == "tensorflow":
            model = Sequential()
            model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
            model.add(Dropout(0.3))
            model.add(LSTM(32, return_sequences=False))
            model.add(Dropout(0.3))
            model.add(Dense(16, activation='relu'))
            model.add(Dense(3, activation='softmax')) # [SELL, WAIT, BUY]
            
            model.compile(optimizer='adam', 
                          loss=PrecisionWeightedLoss(fp_penalty=cfg.SniperConfig.LOSS_PENALTY_MULTIPLIER), 
                          metrics=['accuracy'])
            self.model = model
            return model
        else:
            # MLP mimicking the Deep Structure
            self.model = MLPClassifier(
                hidden_layer_sizes=(64, 32, 16),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=42,
                early_stopping=True
            )
            return self.model

    def load_inference_model(self):
        """Lazy Load."""
        if self.backend == "tensorflow":
            if os.path.exists(self.model_path):
                try:
                    self.model = load_model(self.model_path, compile=False)
                    return True
                except:
                    return False
        else:
            if os.path.exists(self.model_path_mlp):
                self.model = joblib.load(self.model_path_mlp)
                return True
        return False

    def train(self, X, y,class_weight=None):
        """Unified Train."""
        if self.backend == "tensorflow":
            self.model.fit(X, y, epochs=5, batch_size=32, verbose=1, class_weight=class_weight)
            self.model.save(self.model_path)
        else:
            # Scikit learn needs 1D labels
            if len(y.shape) > 1 and y.shape[1] > 1:
                y = np.argmax(y, axis=1) # One-hot to Class ID
            self.model.fit(X, y)
            joblib.dump(self.model, self.model_path_mlp)

    def predict(self, df_window):
        """Legacy Predict (used by strategy_engine)."""
        # Alias for simple prediction, logic moved to predict_trade_confidence for full fusion
        # But we keep this for raw probability if needed
        return self._internal_predict(df_window)

    def _internal_predict(self, df_window):
        if self.model is None:
            if not self.load_inference_model():
                return 1, 0.0
        
        flat = (self.backend == "sklearn")
        tensor = self.preprocessor.transform_single_window(df_window, flat=flat)
        if tensor is None: return 1, 0.0
        
        if self.backend == "tensorflow":
            probs = self.model.predict(tensor, verbose=0)[0]
        else:
            probs = self.model.predict_proba(tensor)[0]
        
        if len(probs) == 3:
            buy_prob = probs[2]
        elif len(probs) == 2:
            buy_prob = probs[1] # Assume binary: 0=Other, 1=Buy
        else:
            buy_prob = 0.0
        return 2, buy_prob

    def predict_trade_confidence(self, ticker, features, fundamentals, df_window):
        """
        THE FUSION ENGINE
        -----------------
        Orchestrates the full decision chain.
        Returns: (Action, Probability, DecisionTrace)
        """
        trace = {
            "Ticker": ticker,
            "Timestamp": datetime.datetime.now().isoformat(),
            "Checks": {}
        }
        
        # 1. Fundamental Check
        fund_score = self._evaluate_fundamentals(fundamentals)
        trace["Checks"]["Fundamentals"] = {"Score": fund_score, "Pass": fund_score >= cfg.SniperConfig.FUNDAMENTAL_MIN_SCORE}
        
        # 2. Technical Check (Regime)
        # Assumed passed via features or calculated externally, but we log key indicators
        adx = features.get('adx', 0)
        trace["Checks"]["Technicals"] = {"ADX": adx, "RSI": features.get('rsi_14')}
        
        # 3. Trends & Regimes (Hard Filters)
        # Falling Knife Protection: Price must not be >10% below 200 SMA
        sma_200 = features.get('sma_200') or features.get('sma_long')
        current_price = features.get('close')
        
        # is_falling_knife = False
        # if sma_200 and current_price:
        #      if current_price < (sma_200 * 0.90):
        #          is_falling_knife = True
        #          trace["Checks"]["Trend_Filter"] = "BLOCKED (Falling Knife)"
        #          decision_logger.info(f"BLOCKED: {ticker} is a Falling Knife ({current_price} < 0.9*{sma_200})")
        #          return "WAIT", 0.0, trace
        if sma_200 and current_price:
             # STRICT RULE: Price must be ABOVE the 200 EMA
             if current_price < sma_200: 
                 trace["Checks"]["Trend_Filter"] = "BLOCKED (Downtrend < 200 SMA)"
                 decision_logger.info(f"BLOCKED: {ticker} in Downtrend ({current_price} < {sma_200})")
                 return "WAIT", 0.0, trace
        
        trace["Checks"]["Trend_Filter"] = "PASS"

        # 4. AI Inference
        # Enforce Lookback Slicing
        if len(df_window) > self.lookback:
            df_window = df_window.iloc[-self.lookback:]
            
        action, prob = self._internal_predict(df_window)
        trace["Checks"]["AI_Model"] = {"Prob": float(f"{prob:.4f}"), "Threshold": cfg.SniperConfig.MODEL_CONFIDENCE_THRESHOLD}
        
        # DECISION LOGIC
        final_decision = "WAIT"
        
        if (trace["Checks"]["Fundamentals"]["Pass"] and 
            prob >= cfg.SniperConfig.MODEL_CONFIDENCE_THRESHOLD):
            final_decision = "BUY"
            
            # FIRE ALERT
            if cfg.SniperConfig.ENABLE_TELEGRAM_ALERTS:
                current_price = df_window.iloc[-1]['close'] if 'close' in df_window.columns else 0.0
                target_price = current_price * (1 + cfg.SniperConfig.TARGET_PROFIT)
                stop_loss = current_price * (1 + cfg.SniperConfig.MAX_DRAWDOWN)
                max_buy_price = current_price * 1.005 # 0.5% Slippage Tolerance
                
                # Calculate Profit Range
                profit_at_entry = cfg.SniperConfig.TARGET_PROFIT
                profit_at_max = (target_price - max_buy_price) / max_buy_price
                
                msg = (f"🎯 SNIPER SIGNAL: {ticker}\n"
                       f"Entry: ${current_price:.2f}\n"
                       f"Max Buy: ${max_buy_price:.2f}\n"
                       f"Est. Profit: +{profit_at_max:.1%} - +{profit_at_entry:.1%}\n"
                       f"Target: ${target_price:.2f}\n"
                       f"Stop: ${stop_loss:.2f}\n"
                       f"Conf: {prob:.2%} | Fund: {fund_score}")
                # Prepare context for Smart Alerts
                alert_params = {
                    "price": current_price,
                    "target": target_price,
                    "stop_loss": stop_loss,
                    "timestamp": dt.now().isoformat()
                }

                self.notifier.send_alert(msg, ticker=ticker, current_params=alert_params)
        
        trace["Final_Decision"] = final_decision
        
        # LOG TRACE
        decision_logger.info(str(trace))
        
        return final_decision, prob, trace

    def _evaluate_fundamentals(self, fund_data):
        """
        Gen-9 Scoring logic for fundamentals.
        Max Score: 100
        """
        if not fund_data: return 50 # Neutral if missing
        
        score = 50
        
        # Extract Metrics
        pe = fund_data.get('trailingPE')
        fwd_pe = fund_data.get('forwardPE')
        growth = fund_data.get('revenueGrowth')
        margins = fund_data.get('profitMargins')
        peg = fund_data.get('pegRatio')
        
        # 1. Growth Engine (The most important for Tech)
        if growth and growth > 0.15: 
            score += 20
        elif growth and growth > 0.05:
            score += 10
            
        # 2. Profitability
        if margins and margins > 0.20:
            score += 10
            
        # 3. Valuation (PE) - Context Aware
        # Standard Value
        if pe and pe < 25: 
            score += 10
        
        # 4. Future Outlook (Forward PE vs Trailing)
        if pe and fwd_pe and fwd_pe < pe:
            score += 10 # Earnings expected to grow
            
        # 5. PEG Ratio (Growth at a Reasonable Price)
        # If PEG is missing, try to calc: PE / (Growth * 100)
        final_peg = peg
        if final_peg is None and pe and growth and growth > 0:
            try:
                final_peg = pe / (growth * 100)
            except:
                final_peg = None
                
        if final_peg and final_peg < 2.0:
            score += 10
            
        return min(score, 100)
