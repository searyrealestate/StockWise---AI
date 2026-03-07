# train_model.py

"""
StockWise Gen-12 Regime-Segregated AI Trainer
=============================================
The Model Factory.
This module completely abolishes ticker-specific models. It aggregates data across the
entire VIP universe, slices it mathematically by DSP Market Regime (Trend vs. Chop),
applies Feature Masking (Orthogonality), and trains Universal Master Models.
"""

import json
import os
import numpy as np
import pandas as pd
import joblib
import logging
import xgboost as xgb
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score

import system_config as cfg
from feature_engine import FeatureEngine
from data_source_manager import DataSourceManager

# Initialize Logger (Rule 2 Compliance)
logger = logging.getLogger("ML_Engine")

class RegimeModelTrainer:
    def __init__(self):
        """Initializes the Training Factory."""
        self.dm = DataSourceManager()
        self.fe = FeatureEngine()
        self.models_dir = cfg.MODELS_DIR
        
        # We will train and save these two exact models
        self.trend_model_path = os.path.join(self.models_dir, "Trend_Master_Model.pkl")
        self.chop_model_path = os.path.join(self.models_dir, "Chop_Master_Model.pkl")

    def load_and_split_ledger(self):
        """
        [Shadow Ledger Split]
        Reads the trade journal and splits it into Prediction Vector (pure math) 
        and Execution Vector (real-world fill probability).
        """
        try:
            with open(getattr(cfg, 'TRADE_JOURNAL_PATH', 'data/trade_journal.json'), 'r', encoding='utf-8') as f:
                journal_data = json.load(f)
                
            prediction_vector = []
            execution_vector = []
            
            for ticker, trades in journal_data.items():
                for trade in trades:
                    # All system signals go to the Prediction Vector to test mathematical edge
                    prediction_vector.append(trade)
                    
                    # Only trades confirmed by the user via Telegram go to the Execution Vector
                    if trade.get('status') == getattr(cfg, 'TRADE_STATUS_EXECUTED', 'CONFIRMED'):
                        execution_vector.append(trade)
                        
            logger.info(f"Ledger Split Complete: {len(prediction_vector)} Predictions | {len(execution_vector)} Executions.")
            return prediction_vector, execution_vector
            
        except FileNotFoundError:
            logger.warning(f"Trade journal not found. Starting with empty vectors.")
            return [], []
        except Exception as e:
            logger.error(f"Critical error loading Shadow Ledger: {str(e)}")
            return [], []
    
    def calculate_dynamic_reward(self, trade_data):
        """
        [Melting Period Analysis & Normalized Reward]
        Replaces simple binary Win/Loss with a time-decaying annualized return model.
        Penalizes setups that lock up capital for too long.
        """
        profit_pct = float(trade_data.get('net_profit_pct', 0.0))
        days_held = float(trade_data.get('days_held', 1.0))
        
        # Prevent division by zero
        days_held = max(days_held, 1.0)
        
        max_melting = float(getattr(cfg, 'MAX_MELTING_PERIOD_DAYS', 7.0))
        
        # Calculate daily normalized return
        daily_return = profit_pct / days_held
        
        if profit_pct > 0:
            if days_held > max_melting:
                # Time Decay Penalty: The trade took too long to materialize
                penalty_factor = max_melting / days_held
                reward = daily_return * penalty_factor
                logger.debug(f"Applied Melting Period Penalty: Target reached but took {days_held} days.")
            else:
                # Fast breakout reward
                reward = daily_return
        else:
            # For losses, the penalty is the absolute loss percentage multiplied by time torture
            # A slow loss is worse than a fast loss (locks up capital)
            time_multiplier = 1.0 + (days_held / max_melting)
            reward = profit_pct * time_multiplier
            
        return reward
    
    def _extract_features_from_trade(self, trade):
        """
        [Feature Extraction]
        Safely extracts the continuous and categorical features from a trade record
        to build the X vector for the Random Forest / LSTM model.
        """
        try:
            # Core numerical features
            tech_score = float(trade.get('tech_score', 0.0))
            ai_score = float(trade.get('ai_score', 0.0))
            master_score = float(trade.get('master_score', 0.0))
            
            # Market Regime mapping (Translating text to continuous space)
            regime = trade.get('market_regime', 'NEUTRAL').upper()
            if regime == 'TREND':
                regime_val = 1.0
            elif regime == 'CHOP':
                regime_val = -1.0
            else:
                regime_val = 0.0
                
            return [tech_score, ai_score, master_score, regime_val]
            
        except Exception as e:
            logger.debug(f"Failed to extract features for model training: {str(e)}")
            return None

    def prepare_training_data(self):
        """
        [Data Packaging & Target Variable Generation]
        Builds the X (features), Y (dynamic reward targets), and Weights matrices.
        Integrates the Prediction and Execution vectors to train against real-world friction.
        """
        # 1. Load the separated vectors
        prediction_vector, execution_vector = self.load_and_split_ledger()
        
        X_train = []
        y_train = []
        sample_weights = []
        
        # 2. Process Prediction Vector (Theoretical Edge Learning)
        for trade in prediction_vector:
            features = self._extract_features_from_trade(trade)
            if features is not None:
                # Replace standard Win/Loss binary target with the Time-Decayed normalized return
                reward = self.calculate_dynamic_reward(trade)
                
                X_train.append(features)
                y_train.append(reward)
                # Base learning weight for theoretical signals
                sample_weights.append(1.0) 
                
        # 3. Process Execution Vector (Real-world Friction & Liquidity Learning)
        for trade in execution_vector:
            features = self._extract_features_from_trade(trade)
            if features is not None:
                reward = self.calculate_dynamic_reward(trade)
                
                X_train.append(features)
                y_train.append(reward)
                # 2x Multiplier: Forces the ML model to prioritize actual filled execution results
                sample_weights.append(2.0) 
                
        if not X_train:
            logger.error("No training data could be prepared. Shadow Ledger might be empty or corrupted.")
            return np.array([]), np.array([]), np.array([])
            
        X_matrix = np.array(X_train)
        Y_matrix = np.array(y_train)
        Weights_matrix = np.array(sample_weights)
        
        logger.info(f"ML Data Preparation Complete. X-Shape: {X_matrix.shape}, Y-Shape: {Y_matrix.shape}")
        
        return X_matrix, Y_matrix, Weights_matrix
    
    def _generate_labels(self, df, lookahead=5, profit_target=0.03):
        """
        The Ground Truth Generator.
        We teach the AI by looking into the future. 
        Did the stock go up by 3% within the next 5 days? If yes, Label = 1. Else = 0.
        """
        logger.debug(f"Generating labels: {lookahead}-day lookahead, {profit_target*100}% target.")
        # Find the maximum high over the next 'lookahead' days
        future_high = df['high'].shift(-1).rolling(window=lookahead, min_periods=1).max()
        
        # Calculate maximum possible percentage gain
        max_gain = (future_high - df['close']) / df['close']
        
        # Label 1 if it hit our profit target, 0 if it failed
        df['target'] = np.where(max_gain >= profit_target, 1, 0)
        
        # SURGICAL FIX: Targeted drop. 
        # Only drop rows where the target couldn't be calculated (the last 5 days of the dataset).
        # Never use a blanket dropna() here.
        return df.dropna(subset=['target', 'close', 'er_slow'])

    def build_universal_dataset(self, symbols, days_back=730):
        """
        Fetches data for multiple stocks, calculates features, and stacks them into one massive dataframe.
        This provides the AI with a massive sample size, preventing ticker-specific overfitting.
        """
        logger.info(f"Building Universal Dataset from {len(symbols)} stocks over {days_back} days.")
        master_data = []

        for sym in symbols:
            try:
                # 1. Fetch raw data
                df = self.dm.get_stock_data(sym, days_back=days_back)
                if df is None or df.empty or len(df) < 100:
                    continue
                
                # 2. Calculate all 85+ features and DSP arrays
                df = self.fe.calculate_features(df, strategy_config={"active_indicators": ["all"]})
                
                # 3. Apply labels (Ground Truth)
                df = self._generate_labels(df, lookahead=5, profit_target=0.02)
                
                master_data.append(df)
                logger.debug(f"[{sym}] Processed {len(df)} rows for Universal Dataset.")
            except Exception as e:
                logger.error(f"[{sym}] Failed to process for dataset: {e}")

        if not master_data:
            logger.error("Failed to build Universal Dataset. No valid data returned.")
            return None

        # Combine all individual stock dataframes into one massive matrix
        universal_df = pd.concat(master_data, ignore_index=True)
        logger.info(f"Universal Dataset built successfully. Total rows: {len(universal_df)}")
        return universal_df

    def segregate_by_regime(self, df):
        """
        Slices the universal dataset into two distinct universes based on the DSP Gatekeeper.
        """
        logger.info("Segregating data by DSP Regimes (Trend vs Chop).")
        
        trend_thr = cfg.DSP_CONFIG.get("threshold_coherent_trend", 0.60)
        chop_thr = cfg.DSP_CONFIG.get("threshold_stochastic_chop", 0.30)
        
        df_trend = df[df['er_slow'] >= trend_thr].copy()
        df_chop = df[df['er_slow'] <= chop_thr].copy()
        
        logger.info(f"Segregation Complete -> Trend Rows: {len(df_trend)} | Chop Rows: {len(df_chop)}")
        return df_trend, df_chop

    def apply_feature_masking(self, df, regime_type):
        """
        Feature Orthogonality (Masking).
        Prevents the AI from looking at indicators that mathematically fail in certain regimes.
        """
        logger.debug(f"Applying Feature Masking for [{regime_type}] regime.")
        
        # Drop columns we can't train on (prices, targets, strings)
        base_drop = ['open', 'high', 'low', 'close', 'target', 'er_slow', 'er_fast']
        available_cols = [c for c in df.columns if c not in base_drop and np.issubdtype(df[c].dtype, np.number)]
        X = df[available_cols].copy()
        y = df['target'].copy()

        if regime_type == "TREND":
            # Blindfold the AI to Oscillators. In a strong trend, RSI stays overbought.
            # If the AI sees RSI=80, it might learn to sell, which is fatal in a mega-trend.
            masked_features = [col for col in X.columns if not any(x in col for x in ['rsi', 'stoch', 'cci', 'bb_'])]
            X = X[masked_features]
            
        elif regime_type == "CHOP":
            # Blindfold the AI to Trend Followers. Moving averages generate endless false signals in chop.
            masked_features = [col for col in X.columns if not any(x in col for x in ['sma', 'ema', 'supertrend', 'psar'])]
            X = X[masked_features]

        return X, y

    def train_and_save(self, X, y, weights, model_path, regime_name):
        """
        [Bug 1.1 Fix] Trains XGBClassifier (binary: profit or not) and saves
        the ACTUAL feature list used during training.
        """
        if len(X) < 50:
            logger.warning(f"Not enough data to train {regime_name} model ({len(X)} rows). Skipping.")
            return

        logger.info(f"Training [{regime_name}] Master Model on {len(X)} samples, {X.shape[1]} features.")

        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X, y, weights, test_size=0.2, random_state=42
        )

        # XGBClassifier for binary target (profit >= 2% within 5 days)
        model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )

        model.fit(X_train, y_train, sample_weight=w_train)

        # Evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        logger.info(f"[{regime_name}] Model Trained. Accuracy: {acc:.2%}, Precision: {prec:.2%}")

        # Save model
        joblib.dump(model, model_path)

        # [CRITICAL FIX] Save the ACTUAL feature columns used during training
        feature_list = list(X.columns) if hasattr(X, 'columns') else [f"f{i}" for i in range(X.shape[1])]
        feature_list_path = model_path.replace(".pkl", "_features.json")
        with open(feature_list_path, 'w') as f:
            json.dump(feature_list, f, indent=2)

        logger.info(f"[{regime_name}] Saved model + {len(feature_list)} features to {model_path}")

    def execute_training_pipeline(self, symbols=None):
        """
        [Bug 1.1 Fix] The Orchestrator for the Trainer.
        Uses the full technical pipeline: fetch → features → labels → regime split → mask → train.
        """
        logger.info("=== STARTING GEN-13 AI TRAINING CYCLE ===")

        # 1. Get symbols from VIP scanner results or use provided list
        if symbols is None:
            scanner_path = getattr(cfg, 'SCANNER_OUTPUT_PATH', 'data/vip_scanner_results.json')
            try:
                with open(scanner_path, 'r') as f:
                    scanner_data = json.load(f)
                symbols = list(scanner_data.keys()) if isinstance(scanner_data, dict) else scanner_data
            except Exception:
                symbols = getattr(cfg, 'DEFAULT_TRAINING_SYMBOLS', ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AMD', 'NFLX', 'SPY'])
                logger.warning(f"Scanner results not found. Using default symbols: {symbols}")

        # 2. Build universal dataset with real technical features
        universal_df = self.build_universal_dataset(symbols)
        if universal_df is None or universal_df.empty:
            logger.error("Training aborted: No data from universal dataset.")
            return

        # 3. Segregate by DSP regime
        df_trend, df_chop = self.segregate_by_regime(universal_df)

        # 4. Train TREND model
        if len(df_trend) >= 50:
            X_trend, y_trend = self.apply_feature_masking(df_trend, "TREND")
            weights_trend = np.ones(len(y_trend))
            self.train_and_save(X_trend, y_trend, weights_trend, self.trend_model_path, "TREND")
        else:
            logger.warning(f"Not enough TREND data ({len(df_trend)} rows). Skipping.")

        # 5. Train CHOP model
        if len(df_chop) >= 50:
            X_chop, y_chop = self.apply_feature_masking(df_chop, "CHOP")
            weights_chop = np.ones(len(y_chop))
            self.train_and_save(X_chop, y_chop, weights_chop, self.chop_model_path, "CHOP")
        else:
            logger.warning(f"Not enough CHOP data ({len(df_chop)} rows). Skipping.")

        logger.info("=== AI TRAINING CYCLE COMPLETE ===")


if __name__ == "__main__":
    # Test execution block
    trainer = RegimeModelTrainer()
    trainer.execute_training_pipeline()