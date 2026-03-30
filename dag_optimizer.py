# dag_optimizer.py

"""
StockWise Gen-12 Information Theory Engine (The DAG Optimizer)
==============================================================
The Information Theory Expert.
This module does not trade. It operates asynchronously to analyze the predictive power 
of every technical indicator in the Feature Engine using XGBoost and SHAP values.
It outputs a mathematically perfect, non-redundant filter sequence (Directed Acyclic Graph) 
for Agent 2 (The Tactical Sniper) to use during live trading.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import system_config as cfg
from feature_engine import FeatureEngine
from data_source_manager import DataSourceManager

# Initialize Logger (Rule 2 Compliance)
logger = cfg.LoggerSetup.setup_logger("DAG_Optimizer")

# --- ADVANCED MATH IMPORTS ---
try:
    import xgboost as xgb
    import shap
    MATH_LIBS_AVAILABLE = True
except ImportError:
    logger.error("Critical ML libraries missing. Please run: pip install xgboost shap")
    MATH_LIBS_AVAILABLE = False


class InformationTheoryEngine:
    def __init__(self):
        """
        Initializes the Information Theory Expert.
        Sets up the paths to save the 'Brain' configurations (best_params.json).
        """
        self.dm = DataSourceManager()
        self.fe = FeatureEngine()
        self.params_file = os.path.join(cfg.DB_DIR, "best_params.json")
        self.dag_memory = self._load_json(self.params_file, default_type={})

    def _load_json(self, path, default_type):
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.debug(f"Failed to load JSON {path}: {e}")
        return default_type

    def _save_json(self, data, path):
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save DAG JSON: {e}")

    def generate_ground_truth(self, df, lookahead=3):
        """
        Calculates what ACTUALLY happened to the stock to teach the AI.
        We ask: "Did the stock go up more than 1% over the next 3 days?"
        """
        logger.debug(f"Generating forward-looking ground truth (Lookahead: {lookahead} days).")
        # Shift the close price backwards to align future prices with today's indicators
        df['future_close'] = df['close'].shift(-lookahead)
        
        # Target = 1 if the stock gained more than 1%, else 0
        df['target'] = np.where((df['future_close'] - df['close']) / df['close'] > 0.01, 1, 0)
        
        return df.dropna()

    def apply_mrmr_filter(self, shap_ranked_features, df, max_features=5):
        """
        Minimum Redundancy, Maximum Relevance (mRMR) Protocol.
        Prevents the Multicollinearity Trap. If SuperTrend and SMA50 are both highly ranked,
        they are redundant. We keep the best one and throw the other away.
        """
        logger.debug("Executing mRMR (Minimum Redundancy, Maximum Relevance) filtration.")
        
        final_dag_sequence = []
        correlation_matrix = df[shap_ranked_features].corr().abs()
        
        for feature in shap_ranked_features:
            if len(final_dag_sequence) == 0:
                # The #1 most powerful feature is automatically accepted
                final_dag_sequence.append(feature)
                continue
                
            # Check correlation against already accepted features
            is_redundant = False
            for accepted_feature in final_dag_sequence:
                corr_value = correlation_matrix.loc[feature, accepted_feature]
                if corr_value > 0.70: # If it is 70% correlated to something we already have, discard it.
                    is_redundant = True
                    logger.debug(f"mRMR Veto: Dropping '{feature}' (Highly correlated {corr_value:.2f} to '{accepted_feature}')")
                    break
                    
            if not is_redundant:
                final_dag_sequence.append(feature)
                
            if len(final_dag_sequence) >= max_features:
                break
                
        return final_dag_sequence

    def calculate_shap_dag(self, symbol, df, regime_name):
        """
        The core intelligence. Trains a micro-XGBoost tree to find non-linear 
        indicator combinations, then uses SHAP to extract the exact hierarchy.
        """
        logger.info(f"Extracting Information Gain for {symbol} in [{regime_name}] Regime.")
        
        # 1. Strip out non-predictive metadata
        ignore_cols = ['open', 'high', 'low', 'close', 'future_close', 'target']
        feature_cols = [col for col in df.columns if col not in ignore_cols and np.issubdtype(df[col].dtype, np.number)]
        
        if len(feature_cols) < 5 or len(df) < 50:
            logger.warning(f"Insufficient data to compute SHAP for {symbol}.")
            return []

        X = df[feature_cols]
        y = df['target']
        
        try:
            # 2. Train the XGBoost Tree
            # We use shallow trees (max_depth=3) to prevent overfitting the noise
            model = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
            model.fit(X, y)
            
            # 3. Dissect the AI's Brain using SHAP
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            
            # 4. Calculate Absolute Mean SHAP (Global Feature Importance)
            mean_shap_values = np.abs(shap_values).mean(axis=0)
            
            # Tie the scores back to the column names and sort descending
            shap_dict = dict(zip(feature_cols, mean_shap_values))
            ranked_features = sorted(shap_dict, key=shap_dict.get, reverse=True)
            
            # 5. Filter out redundancy (mRMR)
            optimal_dag = self.apply_mrmr_filter(ranked_features, df, max_features=5)
            
            logger.info(f"DAG Compiled for {symbol} ({regime_name}): {optimal_dag}")
            return optimal_dag
            
        except Exception as e:
            logger.error(f"SHAP Extraction failed: {e}", exc_info=True)
            return []

    def optimize_watchlist(self, watchlist_symbols):
        """
        The Main Orchestrator for the DAG Optimizer.
        Iterates through the VIP watchlist, splits data by DSP Regime, and builds the DAGs.
        """
        if not MATH_LIBS_AVAILABLE:
            return

        logger.info(f"Initiating DAG Optimization for {len(watchlist_symbols)} VIP stocks.")
        
        trend_threshold = cfg.DSP_CONFIG.get("threshold_coherent_trend", 0.60)
        chop_threshold = cfg.DSP_CONFIG.get("threshold_stochastic_chop", 0.30)
        
        for symbol in watchlist_symbols:
            try:
                # 1. Fetch 1 Year of Data (Needed for statistical significance)
                df = self.dm.get_stock_data(symbol, days_back=365)
                if df is None or df.empty:
                    continue
                
                # 2. Calculate ALL Features (This is computationally heavy, which is why we only do it on the VIP list)
                df = self.fe.calculate_features(df, strategy_config={"active_indicators": ["all"]})
                
                # 3. Generate Ground Truth
                df = self.generate_ground_truth(df, lookahead=3)
                
                # 4. Segregate the Data by DSP Regime
                # We extract the rows where the Efficiency Ratio proved the stock was trending
                df_trend = df[df['er_slow'] >= trend_threshold].copy()
                
                # We extract the rows where the Efficiency Ratio proved the stock was chopping
                df_chop = df[df['er_slow'] <= chop_threshold].copy()
                
                # 5. Initialize the memory structure for this stock
                if symbol not in self.dag_memory:
                    self.dag_memory[symbol] = {}
                    
                self.dag_memory[symbol]["last_optimized"] = datetime.now().isoformat()
                
                # 6. Extract the DAG for the Trend Regime
                if len(df_trend) > 50:
                    trend_dag = self.calculate_shap_dag(symbol, df_trend, "TREND")
                    self.dag_memory[symbol]["TREND_DAG"] = trend_dag
                else:
                    logger.debug(f"[{symbol}] Insufficient historical TREND data to build DAG.")
                    
                # 7. Extract the DAG for the Chop Regime
                if len(df_chop) > 50:
                    chop_dag = self.calculate_shap_dag(symbol, df_chop, "CHOP")
                    self.dag_memory[symbol]["CHOP_DAG"] = chop_dag
                else:
                    logger.debug(f"[{symbol}] Insufficient historical CHOP data to build DAG.")
                    
            except Exception as e:
                logger.error(f"DAG Optimization failed for {symbol}: {e}")
                
        # 8. Save the brains to disk
        self._save_json(self.dag_memory, self.params_file)
        logger.info("DAG Optimization Complete. Strategy Engine parameters updated.")


if __name__ == "__main__":
    # Test execution block
    logger.info("Manual Execution of DAG Optimizer requested.")
    
    # We load the VIP Watchlist from the Scout's ledger
    hunter_watchlist_file = os.path.join(cfg.DB_DIR, "daily_review_list.json")
    try:
        with open(hunter_watchlist_file, "r") as f:
            vip_dict = json.load(f)
            vip_symbols = list(vip_dict.keys())
    except:
        vip_symbols = cfg.WATCHLIST # Fallback
        
    expert = InformationTheoryEngine()
    expert.optimize_watchlist(vip_symbols)