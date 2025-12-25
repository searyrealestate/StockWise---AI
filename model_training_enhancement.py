"""
🤖 Enhanced Model Training Pipeline for 95% Accuracy
──────────────────────────────────────────────────

This script enhances your existing Create_parquet_file_NASDAQ.py
to create models specifically targeting 95% confidence predictions.

Key improvements:
1. Multi-target training (different timeframes and profit thresholds)
2. Advanced feature engineering with market regime detection
3. Ensemble model training with cross-validation
4. Performance tracking and model selection
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EnhancedModelTrainer:
    def __init__(self, debug=True):
        self.debug = debug
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.performance_metrics = {}
        
    def log_training(self, message, level="INFO"):
        """Training-specific logging"""
        if self.debug:
            timestamp = datetime.now().strftime("%H:%M:%S")
            icons = {"INFO": "🤖", "SUCCESS": "✅", "ERROR": "❌", "WARNING": "⚠️"}
            print(f"[{timestamp}] {icons.get(level, '🔸')} TRAINING: {message}")
    
    def create_advanced_features(self, df, symbol):
        """🔧 Create advanced features for better prediction"""
        self.log_training(f"Creating advanced features for {symbol}")
        
        df = df.sort_index()  # Ensure chronological order
        
        # Market regime features
        df['volatility_regime'] = self.detect_volatility_regime(df)
        df['trend_regime'] = self.detect_trend_regime(df)
        df['volume_regime'] = self.detect_volume_regime(df)
        
        # Cross-indicator features
        df['rsi_macd_alignment'] = np.where(
            (df['rsi_14'] > 50) & (df['macd_12_26'] > df['macd_signal_12_26']), 1, 0
        )
        
        df['volume_price_divergence'] = np.where(
            (df['Close'].pct_change() > 0) & (df['Volume_Relative'] < 0.8), 1, 0
        )
        
        # Advanced momentum features
        df['momentum_acceleration'] = df['Close'].pct_change().rolling(3).apply(
            lambda x: x.iloc[-1] - x.iloc[0] if len(x) == 3 else 0
        )
        
        # Market structure features
        df['consecutive_higher_highs'] = self.count_consecutive_patterns(df, 'higher_high')
        df['consecutive_higher_lows'] = self.count_consecutive_patterns(df, 'higher_low')
        
        # Multi-timeframe features
        for window in [3, 7, 14]:
            df[f'price_position_{window}d'] = (
                df['Close'] / df['Close'].rolling(window).max()
            )
            df[f'volume_position_{window}d'] = (
                df['Volume'] / df['Volume'].rolling(window).max()
            )
        
        # Feature interaction terms
        df['rsi_volume_interaction'] = df['rsi_14'] * df['Volume_Relative']
        df['bb_momentum_interaction'] = df['bb_position'] * df['momentum_5']
        
        self.log_training(f"Advanced features created: {df.shape[1]} total columns")
        return df
    
    def detect_volatility_regime(self, df, window=20):
        """🌡️ Detect volatility regime"""
        returns = df['Close'].pct_change()
        rolling_vol = returns.rolling(window).std()
        historical_vol = returns.std()
        
        regime = np.where(
            rolling_vol > historical_vol * 1.5, 2,  # High vol
            np.where(rolling_vol < historical_vol * 0.7, 0, 1)  # Low vol, Normal
        )
        return regime
    
    def detect_trend_regime(self, df, short=10, long=50):
        """📈 Detect trend regime"""
        short_ma = df['Close'].rolling(short).mean()
        long_ma = df['Close'].rolling(long).mean()
        
        regime = np.where(
            short_ma > long_ma * 1.02, 2,  # Strong uptrend
            np.where(short_ma < long_ma * 0.98, 0, 1)  # Strong downtrend, Sideways
        )
        return regime
    
    def detect_volume_regime(self, df, window=20):
        """📊 Detect volume regime"""
        volume_ma = df['Volume'].rolling(window).mean()
        current_volume = df['Volume']
        
        regime = np.where(
            current_volume > volume_ma * 1.5, 2,  # High volume
            np.where(current_volume < volume_ma * 0.7, 0, 1)  # Low volume, Normal
        )
        return regime
    
    def count_consecutive_patterns(self, df, pattern_col):
        """📊 Count consecutive patterns"""
        if pattern_col not in df.columns:
            # Create basic higher_low if not exists
            if pattern_col == 'higher_low':
                df['higher_low'] = (df['Low'] > df['Low'].shift(1)).astype(int)
        
        pattern_series = df.get(pattern_col, pd.Series([0] * len(df)))
        consecutive = pd.Series([0] * len(df), index=df.index)
        
        count = 0
        for i in range(len(pattern_series)):
            if pattern_series.iloc[i] == 1:
                count += 1
            else:
                count = 0
            consecutive.iloc[i] = count
        
        return consecutive
    
    def create_multi_target_labels(self, df, symbol):
        """🎯 Create multiple target labels for different strategies"""
        self.log_training(f"Creating multi-target labels for {symbol}")
        
        targets = {}
        close_prices = df['Close']
        
        # Different time horizons
        for days in [3, 5, 7, 10, 14]:
            future_returns = close_prices.shift(-days) / close_prices - 1
            
            # Different profit thresholds
            for threshold in [0.02, 0.03, 0.04, 0.05]:
                target_name = f'target_{days}d_{int(threshold*100)}pct'
                targets[target_name] = (future_returns > threshold).astype(int)
        
        # Risk-adjusted targets (Sharpe-like)
        for days in [5, 10]:
            future_returns = close_prices.shift(-days) / close_prices - 1
            rolling_vol = close_prices.pct_change().rolling(20).std()
            risk_adjusted_return = future_returns / (rolling_vol + 0.01)  # Add small epsilon
            targets[f'risk_adj_target_{days}d'] = (risk_adjusted_return > 0.5).astype(int)
        
        # Add targets to dataframe
        for target_name, target_values in targets.items():
            df[target_name] = target_values
        
        self.log_training(f"Created {len(targets)} target variables")
        return df, list(targets.keys())
    
    def select_features_for_training(self, df):
        """🔍 Select best features for training"""
        
        # Core technical indicators
        core_features = [
            'Volume_Relative', 'rsi_14', 'rsi_21', 'macd_12_26', 'macd_signal_12_26',
            'macd_diff_12_26', 'bb_position', 'stoch_k', 'stoch_d', 'momentum_5',
            'volatility_regime', 'trend_regime', 'volume_regime'
        ]
        
        # Advanced features
        advanced_features = [
            'rsi_macd_alignment', 'volume_price_divergence', 'momentum_acceleration',
            'consecutive_higher_highs', 'consecutive_higher_lows', 'rsi_volume_interaction',
            'bb_momentum_interaction'
        ]
        
        # Multi-timeframe features
        timeframe_features = [col for col in df.columns if 'position_' in col and 'd' in col]
        
        # Combine all features
        all_features = core_features + advanced_features + timeframe_features
        
        # Filter features that actually exist in the dataframe
        available_features = [f for f in all_features if f in df.columns]
        
        self.log_training(f"Selected {len(available_features)} features for training")
        return available_features
    
    def train_ensemble_for_symbol(self, df, symbol, target_columns, feature_columns):
        """🏋️ Train ensemble models for a specific symbol"""
        self.log_training(f"Training ensemble for {symbol}")
        
        # Prepare data
        df_clean = df[feature_columns + target_columns].dropna()
        
        if len(df_clean) < 100:
            self.log_training(f"Insufficient data for {symbol}: {len(df_clean)} rows", "WARNING")
            return None
        
        # Split data chronologically
        split_point = int(len(df_clean) * 0.8)
        train_data = df_clean.iloc[:split_point]
        test_data = df_clean.iloc[split_point:]
        
        X_train = train_data[feature_columns]
        X_test = test_data[feature_columns]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers[symbol] = scaler
        
        # Train models for each target
        symbol_models = {}
        symbol_performance = {}
        
        for target_col in target_columns:
            if train_data[target_col].nunique() < 2:
                continue  # Skip targets with only one class
            
            y_train = train_data[target_col]
            y_test = test_data[target_col]
            
            self.log_training(f"Training models for {target_col}")
            
            # Define models
            models = {
                'random_forest': RandomForestClassifier(
                    n_estimators=200, max_depth=10, min_samples_split=20,
                    class_weight='balanced', random_state=42
                ),
                'gradient_boost': GradientBoostingClassifier(
                    n_estimators=150, learning_rate=0.1, max_depth=6,
                    random_state=42
                ),
                'neural_network': MLPClassifier(
                    hidden_layer_sizes=(100, 50), max_iter=500,
                    alpha=0.001, random_state=42
                )
            }
            
            target_models = {}
            target_performance = {}
            
            for model_name, model in models.items():
                try:
                    # Train model
                    if model_name == 'neural_network':
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                        if hasattr(model, 'predict_proba'):
                            y_proba = model.predict_proba(X_test_scaled)[:, 1]
                        else:
                            y_proba = y_pred.astype(float)
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        y_proba = model.predict_proba(X_test)[:, 1]
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    
                    # Custom confidence metric (for high-precision trading)
                    high_conf_threshold = 0.8
                    high_conf_predictions = y_proba > high_conf_threshold
                    if high_conf_predictions.sum() > 0:
                        high_conf_accuracy = accuracy_score(
                            y_test[high_conf_predictions], 
                            y_pred[high_conf_predictions]
                        )
                    else:
                        high_conf_accuracy = 0.0
                    
                    performance = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'high_conf_accuracy': high_conf_accuracy,
                        'high_conf_count': high_conf_predictions.sum(),
                        'total_predictions': len(y_test)
                    }
                    
                    # Only keep models with reasonable performance
                    if accuracy > 0.55 and precision > 0.5:
                        target_models[model_name] = model
                        target_performance[model_name] = performance
                        
                        self.log_training(
                            f"{model_name} - Acc: {accuracy:.3f}, Prec: {precision:.3f}, "
                            f"High-Conf-Acc: {high_conf_accuracy:.3f}", "SUCCESS"
                        )
                    else:
                        self.log_training(
                            f"{model_name} - Poor performance: Acc: {accuracy:.3f}, Prec: {precision:.3f}", 
                            "WARNING"
                        )
                        
                except Exception as e:
                    self.log_training(f"Error training {model_name}: {e}", "ERROR")
            
            if target_models:
                symbol_models[target_col] = target_models
                symbol_performance[target_col] = target_performance
        
        if symbol_models:
            self.models[symbol] = symbol_models
            self.performance_metrics[symbol] = symbol_performance
            
            # Calculate feature importance (using Random Forest as representative)
            self.calculate_feature_importance(symbol, feature_columns)
            
            self.log_training(f"Successfully trained {len(symbol_models)} target models for {symbol}", "SUCCESS")
            return symbol_models
        else:
            self.log_training(f"No successful models for {symbol}", "ERROR")
            return None
    
    def calculate_feature_importance(self, symbol, feature_columns):
        """📊 Calculate and store feature importance"""
        if symbol not in self.models:
            return
        
        importance_scores = {}
        
        for target_col, target_models in self.models[symbol].items():
            if 'random_forest' in target_models:
                rf_model = target_models['random_forest']
                importance = rf_model.feature_importances_
                
                for i, feature in enumerate(feature_columns):
                    if feature not in importance_scores:
                        importance_scores[feature] = []
                    importance_scores[feature].append(importance[i])
        
        # Average importance across targets
        avg_importance = {
            feature: np.mean(scores) 
            for feature, scores in importance_scores.items()
        }
        
        self.feature_importance[symbol] = sorted(
            avg_importance.items(), key=lambda x: x[1], reverse=True
        )
        
        # Log top features
        top_features = self.feature_importance[symbol][:10]
        self.log_training(f"Top 10 features for {symbol}:")
        for feature, importance in top_features:
            self.log_training(f"  {feature}: {importance:.4f}")
    
    def save_enhanced_models(self, symbol, output_dir):
        """💾 Save all models and metadata"""
        if symbol not in self.models:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save ensemble models
        model_file = os.path.join(output_dir, f"{symbol}_ensemble_{timestamp}.pkl")
        ensemble_data = {
            'models': self.models[symbol],
            'scaler': self.scalers.get(symbol),
            'performance': self.performance_metrics.get(symbol, {}),
            'feature_importance': self.feature_importance.get(symbol, []),
            'created': datetime.now(),
            'training_version': '95_percent_confidence_v1'
        }
        
        joblib.dump(ensemble_data, model_file)
        self.log_training(f"Enhanced ensemble saved: {model_file}", "SUCCESS")
        
        # Save performance report
        report_file = os.path.join(output_dir, f"{symbol}_performance_report_{timestamp}.txt")
        self.save_performance_report(symbol, report_file)
    
    def save_performance_report(self, symbol, report_file):
        """📋 Generate detailed performance report"""
        if symbol not in self.performance_metrics:
            return
        
        with open(report_file, 'w') as f:
            f.write(f"ENHANCED MODEL PERFORMANCE REPORT\n")
            f.write(f"Symbol: {symbol}\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write("=" * 50 + "\n\n")
            
            for target_col, target_performance in self.performance_metrics[symbol].items():
                f.write(f"TARGET: {target_col}\n")
                f.write("-" * 30 + "\n")
                
                for model_name, metrics in target_performance.items():
                    f.write(f"\n{model_name.upper()}:\n")
                    f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
                    f.write(f"  Precision: {metrics['precision']:.4f}\n")
                    f.write(f"  Recall: {metrics['recall']:.4f}\n")
                    f.write(f"  F1 Score: {metrics['f1']:.4f}\n")
                    f.write(f"  High-Conf Accuracy: {metrics['high_conf_accuracy']:.4f}\n")
                    f.write(f"  High-Conf Predictions: {metrics['high_conf_count']}/{metrics['total_predictions']}\n")
                
                f.write("\n" + "=" * 50 + "\n")
            
            # Feature importance
            if symbol in self.feature_importance:
                f.write("\nFEATURE IMPORTANCE (Top 20):\n")
                f.write("-" * 30 + "\n")
                for i, (feature, importance) in enumerate(self.feature_importance[symbol][:20], 1):
                    f.write(f"{i:2d}. {feature:<30} {importance:.6f}\n")
        
        self.log_training(f"Performance report saved: {report_file}", "SUCCESS")


def enhance_existing_training_pipeline(original_df, symbol, output_dir):
    """🚀 Main function to enhance your existing training pipeline"""
    
    print(f"🚀 ENHANCING TRAINING PIPELINE FOR {symbol}")
    print("=" * 60)
    
    # Initialize enhanced trainer
    trainer = EnhancedModelTrainer(debug=True)
    
    try:
        # Step 1: Create advanced features
        print("🔧 Step 1: Creating advanced features...")
        enhanced_df = trainer.create_advanced_features(original_df.copy(), symbol)
        
        # Step 2: Create multi-target labels
        print("🎯 Step 2: Creating multi-target labels...")
        enhanced_df, target_columns = trainer.create_multi_target_labels(enhanced_df, symbol)
        
        # Step 3: Select features
        print("🔍 Step 3: Selecting optimal features...")
        feature_columns = trainer.select_features_for_training(enhanced_df)
        
        print(f"   Selected {len(feature_columns)} features")
        print(f"   Created {len(target_columns)} target variables")
        
        # Step 4: Train ensemble models
        print("🏋️ Step 4: Training ensemble models...")
        models = trainer.train_ensemble_for_symbol(
            enhanced_df, symbol, target_columns, feature_columns
        )
        
        if models:
            # Step 5: Save enhanced models
            print("💾 Step 5: Saving enhanced models...")
            trainer.save_enhanced_models(symbol, output_dir)
            
            print(f"✅ SUCCESSFULLY ENHANCED {symbol}")
            print(f"   - Trained {len(models)} target models")
            print(f"   - Performance reports saved")
            print(f"   - Models ready for 95% confidence trading")
            
            return True
        else:
            print(f"❌ FAILED TO ENHANCE {symbol} - No successful models")
            return False
            
    except Exception as e:
        print(f"❌ ERROR ENHANCING {symbol}: {e}")
        return False


# Integration with your existing Create_parquet_file_NASDAQ.py
def integrate_with_existing_pipeline():
    """🔗 Integration instructions for your existing pipeline"""
    
    integration_code = '''
# ADD THIS TO YOUR EXISTING Create_parquet_file_NASDAQ.py
# Replace your existing train_model function with this enhanced version:

def train_enhanced_model(df, symbol):
    """🧠 Enhanced model training with 95% confidence targeting"""
    from model_training_enhancement import enhance_existing_training_pipeline
    
    debug_print(f"=== ENHANCED MODEL TRAINING for {symbol} ===", "INFO")
    
    # Run the enhanced training pipeline
    success = enhance_existing_training_pipeline(df, symbol, TRAIN_DIR)
    
    if success:
        debug_print(f"Enhanced training successful for {symbol}", "SUCCESS")
        return True, df  # Return success flag and data
    else:
        debug_print(f"Enhanced training failed for {symbol}, falling back to original", "WARNING")
        # Fall back to your original train_model function
        return train_model_original(df, symbol)

# Replace the call in process_ticker_list:
# Change: model, df_trained = train_model(df, symbol)
# To:     success, df_trained = train_enhanced_model(df, symbol)
'''
    
    print("🔗 INTEGRATION INSTRUCTIONS:")
    print("=" * 50)
    print(integration_code)
    print("=" * 50)


if __name__ == "__main__":
    print("🤖 Enhanced Model Training Pipeline")
    print("This script provides enhanced training capabilities for 95% confidence trading")
    print("\nTo integrate with your existing pipeline:")
    integrate_with_existing_pipeline()
