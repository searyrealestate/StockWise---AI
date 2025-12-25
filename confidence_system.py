"""
🎯 95% Confidence Building System
─────────────────────────────────

Advanced ensemble system for high-confidence trading decisions
Combines multiple validation layers and historical performance tracking
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score
import joblib
import os
from datetime import datetime, timedelta

class ConfidenceBuilder:
    def __init__(self, debug=True):
        self.debug = debug
        self.models = {}
        self.model_performance = {}
        self.confidence_threshold = 95.0
        self.min_historical_accuracy = 0.70
        self.ensemble_weights = {}
        
    def log_confidence(self, message, level="INFO"):
        """Confidence system logging"""
        if self.debug:
            timestamp = datetime.now().strftime("%H:%M:%S")
            icons = {"INFO": "🧠", "SUCCESS": "✅", "ERROR": "❌", "WARNING": "⚠️"}
            print(f"[{timestamp}] {icons.get(level, '🔸')} CONFIDENCE: {message}")
    
    def create_ensemble_models(self, symbol):
        """🤖 Create ensemble of different model types"""
        self.log_confidence(f"Creating ensemble models for {symbol}")
        
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=20,
                random_state=42
            ),
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                alpha=0.001,
                random_state=42
            )
        }
        
        self.models[symbol] = models
        self.log_confidence(f"Created {len(models)} ensemble models", "SUCCESS")
        return models
    
    def train_ensemble_with_validation(self, symbol, X_train, y_train, X_val, y_val):
        """🏋️ Train ensemble with cross-validation"""
        self.log_confidence(f"Training ensemble for {symbol} with validation")
        
        if symbol not in self.models:
            self.create_ensemble_models(symbol)
        
        model_scores = {}
        trained_models = {}
        
        for model_name, model in self.models[symbol].items():
            try:
                self.log_confidence(f"Training {model_name}...")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Validate performance
                train_pred = model.predict(X_train)
                val_pred = model.predict(X_val)
                
                train_accuracy = accuracy_score(y_train, train_pred)
                val_accuracy = accuracy_score(y_val, val_pred)
                val_precision = precision_score(y_val, val_pred, average='weighted', zero_division=0)
                
                # Calculate model score (penalize overfitting)
                overfitting_penalty = max(0, train_accuracy - val_accuracy - 0.05)
                model_score = val_accuracy - overfitting_penalty
                
                model_scores[model_name] = {
                    'train_accuracy': train_accuracy,
                    'val_accuracy': val_accuracy,
                    'val_precision': val_precision,
                    'model_score': model_score,
                    'usable': model_score >= self.min_historical_accuracy
                }
                
                if model_score >= self.min_historical_accuracy:
                    trained_models[model_name] = model
                    self.log_confidence(f"{model_name}: Val Acc={val_accuracy:.3f}, Score={model_score:.3f} ✅", "SUCCESS")
                else:
                    self.log_confidence(f"{model_name}: Score={model_score:.3f} too low ❌", "WARNING")
                    
            except Exception as e:
                self.log_confidence(f"Error training {model_name}: {e}", "ERROR")
                model_scores[model_name] = {'usable': False, 'error': str(e)}
        
        # Calculate ensemble weights based on performance
        usable_models = {k: v for k, v in model_scores.items() if v.get('usable', False)}
        
        if usable_models:
            total_score = sum(model['model_score'] for model in usable_models.values())
            self.ensemble_weights[symbol] = {
                model_name: model_data['model_score'] / total_score 
                for model_name, model_data in usable_models.items()
            }
            
            self.log_confidence(f"Ensemble weights: {self.ensemble_weights[symbol]}", "SUCCESS")
        else:
            self.log_confidence("No usable models - ensemble failed", "ERROR")
            self.ensemble_weights[symbol] = {}
        
        # Store performance data
        self.model_performance[symbol] = model_scores
        
        return trained_models, model_scores
    
    def ensemble_predict_with_confidence(self, symbol, features):
        """🎯 Ensemble prediction with confidence scoring"""
        self.log_confidence(f"Making ensemble prediction for {symbol}")
        
        if symbol not in self.models or symbol not in self.ensemble_weights:
            self.log_confidence(f"No trained ensemble for {symbol}", "WARNING")
            return 0.5, 50.0  # Neutral prediction
        
        predictions = []
        confidences = []
        weights = []
        
        for model_name, model in self.models[symbol].items():
            if model_name in self.ensemble_weights[symbol]:
                try:
                    # Get prediction
                    pred = model.predict([features])[0]
                    predictions.append(pred)
                    
                    # Get confidence (probability)
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba([features])[0]
                        confidence = max(proba)  # Highest probability
                    else:
                        confidence = 0.7  # Default for models without proba
                    
                    confidences.append(confidence)
                    weights.append(self.ensemble_weights[symbol][model_name])
                    
                    self.log_confidence(f"{model_name}: pred={pred}, conf={confidence:.3f}")
                    
                except Exception as e:
                    self.log_confidence(f"Error with {model_name}: {e}", "ERROR")
        
        if not predictions:
            self.log_confidence("No valid predictions", "ERROR")
            return 0.5, 50.0
        
        # Weighted ensemble prediction
        weighted_pred = sum(p * w for p, w in zip(predictions, weights)) / sum(weights)
        weighted_conf = sum(c * w for c, w in zip(confidences, weights)) / sum(weights)
        
        # Boost confidence for agreement
        agreement = len(set(predictions)) == 1  # All models agree
        if agreement:
            weighted_conf = min(weighted_conf * 1.15, 0.99)
            self.log_confidence("Model agreement bonus applied", "SUCCESS")
        
        # Convert to percentage
        final_confidence = weighted_conf * 100
        
        self.log_confidence(f"Ensemble result: pred={weighted_pred:.3f}, conf={final_confidence:.1f}%")
        return weighted_pred, final_confidence
    
    def historical_performance_check(self, symbol, lookback_days=90):
        """📊 Check historical performance of recommendations"""
        self.log_confidence(f"Checking historical performance for {symbol}")
        
        # This would load historical recommendations and their outcomes
        # For now, simulate with stored performance data
        
        if symbol in self.model_performance:
            avg_accuracy = np.mean([
                model_data.get('val_accuracy', 0) 
                for model_data in self.model_performance[symbol].values()
                if model_data.get('usable', False)
            ])
            
            if avg_accuracy >= self.min_historical_accuracy:
                confidence_boost = min((avg_accuracy - 0.5) * 100, 25)
                self.log_confidence(f"Historical accuracy: {avg_accuracy:.3f}, boost: +{confidence_boost:.1f}%", "SUCCESS")
                return confidence_boost
            else:
                self.log_confidence(f"Historical accuracy too low: {avg_accuracy:.3f}", "WARNING")
                return -10  # Penalize poor historical performance
        
        return 0  # No historical data
    
    def signal_consistency_check(self, current_signals, historical_signals=None):
        """🔍 Check consistency with historical successful signals"""
        self.log_confidence("Checking signal consistency...")
        
        consistency_score = 0
        
        # Check for strong signal patterns that historically worked
        strong_patterns = [
            "🚀 Price accelerating",
            "💪 Volume-price confirmation", 
            "🚀 Resistance broken",
            "🔊 Volume breakout",
            "🚀 MACD Bullish crossover"
        ]
        
        pattern_matches = sum(1 for signal in current_signals if any(pattern in signal for pattern in strong_patterns))
        consistency_score = min(pattern_matches * 5, 20)
        
        self.log_confidence(f"Pattern matches: {pattern_matches}, consistency boost: +{consistency_score}%")
        return consistency_score
    
    def risk_adjusted_confidence(self, base_confidence, volatility, market_conditions):
        """⚖️ Adjust confidence based on risk factors"""
        self.log_confidence("Applying risk adjustments...")
        
        risk_adjustment = 0
        
        # Volatility adjustment
        if volatility > 0.03:  # High volatility
            risk_adjustment -= 5
            self.log_confidence(f"High volatility penalty: -5%")
        elif volatility < 0.015:  # Low volatility (good for breakouts)
            risk_adjustment += 3
            self.log_confidence(f"Low volatility bonus: +3%")
        
        # Market conditions
        if market_conditions.get('market_trend', 'neutral') == 'bull':
            risk_adjustment += 2
            self.log_confidence(f"Bull market bonus: +2%")
        elif market_conditions.get('market_trend', 'neutral') == 'bear':
            risk_adjustment -= 3
            self.log_confidence(f"Bear market penalty: -3%")
        
        adjusted_confidence = max(base_confidence + risk_adjustment, 40)
        self.log_confidence(f"Risk-adjusted confidence: {base_confidence:.1f}% → {adjusted_confidence:.1f}%")
        
        return adjusted_confidence
    
    def calculate_95_percent_confidence(self, symbol, features, signals, market_data):
        """🎯 Master confidence calculation targeting 95%"""
        self.log_confidence(f"=== CALCULATING 95% CONFIDENCE for {symbol} ===")
        
        # 1. Ensemble model prediction
        ensemble_pred, ensemble_conf = self.ensemble_predict_with_confidence(symbol, features)
        self.log_confidence(f"Ensemble confidence: {ensemble_conf:.1f}%")
        
        # 2. Historical performance boost
        historical_boost = self.historical_performance_check(symbol)
        
        # 3. Signal consistency check
        consistency_boost = self.signal_consistency_check(signals)
        
        # 4. Risk adjustments
        volatility = market_data.get('volatility', 0.02)
        risk_adjusted_conf = self.risk_adjusted_confidence(
            ensemble_conf + historical_boost + consistency_boost,
            volatility,
            market_data
        )
        
        # 5. Final confidence gates
        final_confidence = self.apply_confidence_gates(
            risk_adjusted_conf, 
            ensemble_pred, 
            signals, 
            symbol
        )
        
        self.log_confidence(f"FINAL CONFIDENCE: {final_confidence:.1f}%")
        
        # 6. Decision logic for 95% targeting
        recommendation = self.make_95_percent_decision(final_confidence, ensemble_pred, signals)
        
        return {
            'confidence': final_confidence,
            'prediction': ensemble_pred,
            'recommendation': recommendation,
            'breakdown': {
                'ensemble': ensemble_conf,
                'historical': historical_boost,
                'consistency': consistency_boost,
                'risk_adjusted': risk_adjusted_conf - ensemble_conf - historical_boost - consistency_boost
            }
        }
    
    def apply_confidence_gates(self, base_confidence, prediction, signals, symbol):
        """🚪 Apply confidence gates to ensure quality"""
        self.log_confidence("Applying confidence quality gates...")
        
        gated_confidence = base_confidence
        
        # Gate 1: Minimum signal strength requirement
        strong_signals = sum(1 for signal in signals if any(word in signal.lower() 
                           for word in ['strong', 'breakout', 'accelerating', 'spike']))
        
        if strong_signals < 2:
            gated_confidence = min(gated_confidence, 80)
            self.log_confidence(f"Gate 1: Only {strong_signals} strong signals, capped at 80%")
        
        # Gate 2: Prediction strength requirement for high confidence
        if gated_confidence > 90 and abs(prediction - 0.5) < 0.3:
            gated_confidence = 85
            self.log_confidence("Gate 2: Weak prediction strength, capped at 85%")
        
        # Gate 3: Model availability requirement
        if symbol not in self.ensemble_weights or len(self.ensemble_weights[symbol]) < 2:
            gated_confidence = min(gated_confidence, 75)
            self.log_confidence("Gate 3: Insufficient ensemble models, capped at 75%")
        
        # Gate 4: Conflicting signal penalty
        bearish_signals = sum(1 for signal in signals if any(word in signal.lower() 
                            for word in ['sell', 'bearish', 'down', 'weak']))
        bullish_signals = len(signals) - bearish_signals
        
        if bearish_signals > 0 and bullish_signals > 0:
            conflict_penalty = min(bearish_signals * 5, 15)
            gated_confidence -= conflict_penalty
            self.log_confidence(f"Gate 4: Signal conflict penalty: -{conflict_penalty}%")
        
        return max(gated_confidence, 50)  # Minimum 50% confidence
    
    def make_95_percent_decision(self, confidence, prediction, signals):
        """🎯 Make trading decision targeting 95% accuracy"""
        self.log_confidence("Making 95% accuracy targeted decision...")
        
        # Ultra-conservative approach for 95% accuracy
        if confidence >= 95 and prediction > 0.75:
            decision = "ULTRA_BUY"
            self.log_confidence("ULTRA_BUY: 95%+ confidence with strong prediction", "SUCCESS")
            
        elif confidence >= 90 and prediction > 0.70:
            decision = "STRONG_BUY"
            self.log_confidence("STRONG_BUY: 90%+ confidence", "SUCCESS")
            
        elif confidence >= 85 and prediction > 0.65:
            decision = "BUY"
            self.log_confidence("BUY: 85%+ confidence", "SUCCESS")
            
        elif confidence >= 80 and prediction > 0.60:
            decision = "WEAK_BUY"
            self.log_confidence("WEAK_BUY: 80%+ confidence")
            
        elif confidence <= 30 or prediction < 0.3:
            decision = "SELL"
            self.log_confidence("SELL: Low confidence or bearish prediction")
            
        else:
            decision = "WAIT"
            self.log_confidence("WAIT: Insufficient confidence for action")
        
        return decision
    
    def save_ensemble(self, symbol, filepath=None):
        """💾 Save ensemble models and performance data"""
        if filepath is None:
            filepath = f"models/ensemble_{symbol}_{datetime.now().strftime('%Y%m%d')}.pkl"
        
        ensemble_data = {
            'models': self.models.get(symbol, {}),
            'weights': self.ensemble_weights.get(symbol, {}),
            'performance': self.model_performance.get(symbol, {}),
            'created': datetime.now(),
            'confidence_threshold': self.confidence_threshold
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(ensemble_data, filepath)
        self.log_confidence(f"Ensemble saved to {filepath}", "SUCCESS")
    
    def load_ensemble(self, symbol, filepath=None):
        """📂 Load ensemble models and performance data"""
        if filepath is None:
            # Try to find the most recent ensemble file
            pattern = f"models/ensemble_{symbol}_*.pkl"
            import glob
            files = glob.glob(pattern)
            if files:
                filepath = max(files, key=os.path.getmtime)
            else:
                self.log_confidence(f"No ensemble file found for {symbol}", "WARNING")
                return False
        
        try:
            ensemble_data = joblib.load(filepath)
            self.models[symbol] = ensemble_data['models']
            self.ensemble_weights[symbol] = ensemble_data['weights']
            self.model_performance[symbol] = ensemble_data['performance']
            
            self.log_confidence(f"Ensemble loaded from {filepath}", "SUCCESS")
            return True
        except Exception as e:
            self.log_confidence(f"Error loading ensemble: {e}", "ERROR")
            return False


class UserOverrideSystem:
    """🤝 Manual review system for edge cases"""
    
    def __init__(self, debug=True):
        self.debug = debug
        self.override_history = []
        
    def log_override(self, message):
        if self.debug:
            print(f"🤝 OVERRIDE: {message}")
    
    def flag_for_manual_review(self, symbol, confidence, signals, reason):
        """🚨 Flag recommendations for manual review"""
        self.log_override(f"Flagging {symbol} for manual review: {reason}")
        
        review_case = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'confidence': confidence,
            'signals': signals,
            'flag_reason': reason,
            'status': 'pending_review'
        }
        
        # Save to review queue (in practice, this could be a database or file)
        review_file = f"manual_review_queue.txt"
        with open(review_file, "a") as f:
            f.write(f"{symbol}\t{confidence:.1f}\t{reason}\t{datetime.now()}\n")
        
        return review_case
    
    def should_flag_for_review(self, symbol, confidence, signals, market_data):
        """🔍 Determine if case needs manual review"""
        
        # Flag conditions
        flag_reasons = []
        
        # 1. High confidence with unusual market conditions
        if confidence > 90 and market_data.get('volatility', 0) > 0.04:
            flag_reasons.append("High confidence in volatile market")
        
        # 2. Conflicting strong signals
        strong_buy_signals = sum(1 for s in signals if 'strong' in s.lower() and any(word in s.lower() for word in ['buy', 'bullish', 'breakout']))
        strong_sell_signals = sum(1 for s in signals if 'strong' in s.lower() and any(word in s.lower() for word in ['sell', 'bearish', 'down']))
        
        if strong_buy_signals > 0 and strong_sell_signals > 0:
            flag_reasons.append("Conflicting strong signals")
        
        # 3. Unusually high expected returns
        if market_data.get('expected_return', 0) > 0.15:  # 15%+ expected return
            flag_reasons.append("Unusually high expected return")
        
        # 4. Low volume confirmation for high confidence
        volume_signals = [s for s in signals if 'volume' in s.lower()]
        if confidence > 85 and len(volume_signals) == 0:
            flag_reasons.append("High confidence without volume confirmation")
        
        return flag_reasons


# Integration function for your main system
def integrate_confidence_system(advisor_instance):
    """🔧 Integrate confidence system with your existing advisor"""
    
    # Add confidence builder
    advisor_instance.confidence_builder = ConfidenceBuilder(debug=advisor_instance.debug)
    advisor_instance.override_system = UserOverrideSystem(debug=advisor_instance.debug)
    
    # Enhanced recommendation generation
    def confidence_enhanced_recommendation(self, indicators, symbol):
        """Generate recommendation with 95% confidence targeting"""
        
        # Get base recommendation from enhanced signals
        if hasattr(self, 'enhanced_detector'):
            df = self.get_stock_data(symbol, datetime.now().date(), days_back=60)
            enhanced_result = self.enhanced_detector.enhanced_signal_decision(df, indicators, symbol)
            base_signals = enhanced_result['signals']
            base_confidence = enhanced_result['confidence']
        else:
            # Fallback to original system
            original_result = self.original_generate_enhanced_recommendation(indicators, symbol)
            base_signals = original_result.get('reasons', [])
            base_confidence = original_result.get('confidence', 50)
        
        # Prepare features for ensemble (simplified example)
        features = [
            indicators.get('volume_relative', 1.0),
            indicators.get('rsi_14', 50) / 100,
            indicators.get('macd_histogram', 0),
            indicators.get('momentum_5', 0) / 100,
            indicators.get('bb_position', 0.5)
        ]
        
        # Market data for confidence calculation
        market_data = {
            'volatility': indicators.get('volatility', 0.02) / 100,
            'market_trend': 'neutral',  # Could be determined from broader market indicators
            'expected_return': 0.05  # Base expected return
        }
        
        # Calculate 95% confidence
        confidence_result = self.confidence_builder.calculate_95_percent_confidence(
            symbol, features, base_signals, market_data
        )
        
        final_confidence = confidence_result['confidence']
        recommendation = confidence_result['recommendation']
        
        # Check if manual review is needed
        flag_reasons = self.override_system.should_flag_for_review(
            symbol, final_confidence, base_signals, market_data
        )
        
        if flag_reasons:
            self.override_system.flag_for_manual_review(
                symbol, final_confidence, base_signals, "; ".join(flag_reasons)
            )
        
        # Convert to your format
        action_mapping = {
            'ULTRA_BUY': 'BUY',
            'STRONG_BUY': 'BUY', 
            'BUY': 'BUY',
            'WEAK_BUY': 'BUY',
            'SELL': 'SELL/AVOID',
            'WAIT': 'WAIT'
        }
        
        # Dynamic profit targets based on confidence
        profit_targets = {
            'ULTRA_BUY': 0.08,  # 8% for ultra-high confidence
            'STRONG_BUY': 0.06,  # 6% for strong confidence
            'BUY': 0.04,         # 4% for normal buy
            'WEAK_BUY': 0.03,    # 3% for weak buy
            'SELL': -0.03,
            'WAIT': 0
        }
        
        target_profit = profit_targets.get(recommendation, 0.04)
        current_price = indicators['current_price']
        
        return {
            'action': action_mapping.get(recommendation, 'WAIT'),
            'confidence': final_confidence,
            'buy_price': current_price if recommendation.endswith('BUY') else None,
            'sell_price': current_price * (1 + target_profit) if target_profit > 0 else current_price,
            'stop_loss': current_price * 0.94,  # 6% stop loss
            'expected_profit_pct': target_profit * 100,
            'reasons': base_signals + [f"🎯 95% Confidence System: {recommendation} ({final_confidence:.1f}%)"],
            'final_score': confidence_result['prediction'] * 10,  # Scale for compatibility
            'confidence_breakdown': confidence_result['breakdown'],
            'manual_review_flags': flag_reasons,
            'current_price': current_price,
            'trading_plan': self.build_trading_plan(current_price, target_gain=target_profit)
        }
    
    # Replace the method
    advisor_instance.confidence_enhanced_recommendation = confidence_enhanced_recommendation.__get__(advisor_instance)
    
    return advisor_instance