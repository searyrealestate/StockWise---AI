"""
🎯 Enhanced Signal Detection Functions for 95% Confidence Trading
────────────────────────────────────────────────────────────────

Key improvements:
1. Pre-breakout scanning for early signal detection
2. Dynamic thresholds based on market conditions
3. Multi-timeframe confirmation system
4. Advanced momentum detection with acceleration
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class EnhancedSignalDetector:
    def __init__(self, debug=True):
        self.debug = debug
        self.market_regime = "normal"  # normal, volatile, trending
        self.confidence_threshold = 95.0
        
    def log_signal(self, message, level="INFO"):
        """Enhanced logging for signal detection"""
        if self.debug:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] 🎯 SIGNAL: {message}")
    
    def detect_market_regime(self, df, lookback=20):
        """🌡️ Adaptive market regime detection"""
        self.log_signal("Detecting market regime...")
        
        # Calculate volatility
        returns = df['Close'].pct_change().dropna()
        recent_vol = returns.tail(lookback).std() * 100
        historical_vol = returns.std() * 100
        
        # Calculate trend strength
        sma_20 = df['Close'].rolling(20).mean()
        trend_strength = abs(df['Close'].iloc[-1] - sma_20.iloc[-1]) / sma_20.iloc[-1]
        
        # Determine regime
        if recent_vol > historical_vol * 1.5:
            self.market_regime = "volatile"
            vol_multiplier = 0.7  # Lower thresholds in volatile markets
        elif trend_strength > 0.05:
            self.market_regime = "trending" 
            vol_multiplier = 1.2  # Higher thresholds in strong trends
        else:
            self.market_regime = "normal"
            vol_multiplier = 1.0
            
        self.log_signal(f"Market regime: {self.market_regime} (vol_mult: {vol_multiplier:.2f})")
        return vol_multiplier
    
    def pre_breakout_scan(self, indicators, current_price):
        """🔍 Early breakout detection before full confirmation"""
        self.log_signal("Scanning for pre-breakout conditions...")
        
        signals = []
        score = 0
        
        # 1. Coiling price action (low volatility before breakout)
        bb_width = (indicators.get('bb_upper', 0) - indicators.get('bb_lower', 0)) / current_price
        if bb_width < 0.04:  # Tight Bollinger Bands
            score += 2
            signals.append("🎯 Price coiling - Bollinger Bands tightening")
            
        # 2. Volume drying up before breakout
        vol_ratio = indicators.get('volume_relative', 1.0)
        if 0.6 < vol_ratio < 0.9:  # Below average but not dead
            score += 1.5
            signals.append("📊 Volume consolidating - preparing for move")
            
        # 3. RSI building base above 40
        rsi = indicators.get('rsi_14', 50)
        if 42 < rsi < 58:  # Not oversold, building strength
            score += 1
            signals.append("⚡ RSI building base - momentum gathering")
            
        # 4. Price near key moving average support
        sma_20 = indicators.get('sma_20', current_price)
        distance_to_sma = abs(current_price - sma_20) / current_price
        if distance_to_sma < 0.02:  # Within 2% of SMA20
            score += 1.5
            signals.append("🎯 Price near SMA20 support - setup forming")
            
        # 5. MACD histogram flattening (momentum shift)
        macd_hist = indicators.get('macd_histogram', 0)
        if -0.1 < macd_hist < 0.1:  # Near zero line
            score += 1
            signals.append("🔄 MACD histogram flattening - momentum shift")
            
        self.log_signal(f"Pre-breakout score: {score:.1f}/7.0")
        return score, signals
    
    def enhanced_momentum_detection(self, df, indicators):
        """🚀 Advanced momentum detection with acceleration"""
        self.log_signal("Analyzing enhanced momentum patterns...")
        
        current_price = indicators['current_price']
        signals = []
        score = 0
        
        # 1. Price acceleration (increasing rate of change)
        if len(df) >= 10:
            roc_5 = (current_price / df['Close'].iloc[-6] - 1) * 100
            roc_3 = (current_price / df['Close'].iloc[-4] - 1) * 100
            roc_1 = (current_price / df['Close'].iloc[-2] - 1) * 100
            
            # Check for acceleration
            if roc_1 > roc_3 > roc_5 and roc_1 > 1.0:
                score += 3
                signals.append(f"🚀 Price accelerating: 1d={roc_1:.1f}%, 3d={roc_3:.1f}%, 5d={roc_5:.1f}%")
                
        # 2. Volume-price confirmation
        vol_ratio = indicators.get('volume_relative', 1.0)
        momentum_5 = indicators.get('momentum_5', 0)
        
        if vol_ratio > 1.2 and momentum_5 > 2.0:
            score += 2.5
            signals.append(f"💪 Volume-price confirmation: Vol={vol_ratio:.1f}x, Mom={momentum_5:.1f}%")
            
        # 3. Multi-timeframe RSI alignment
        rsi_14 = indicators.get('rsi_14', 50)
        rsi_21 = indicators.get('rsi_21', 50)
        
        if rsi_14 > 45 and rsi_21 > 45 and rsi_14 > rsi_21:
            score += 1.5
            signals.append(f"📈 RSI momentum: 14d={rsi_14:.1f}, 21d={rsi_21:.1f}")
            
        # 4. MACD momentum building
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        macd_hist = indicators.get('macd_histogram', 0)
        
        if macd > macd_signal and macd_hist > 0 and macd_hist > -0.05:
            score += 2
            signals.append(f"🔄 MACD building: Hist={macd_hist:.3f}")
            
        self.log_signal(f"Enhanced momentum score: {score:.1f}/9.0")
        return score, signals
    
    def adaptive_support_resistance(self, df, indicators, vol_multiplier):
        """🎯 Dynamic support/resistance with adaptive thresholds"""
        self.log_signal("Analyzing adaptive support/resistance...")
        
        current_price = indicators['current_price']
        signals = []
        score = 0
        
        # Adaptive thresholds based on market regime
        base_threshold = 0.02 * vol_multiplier  # 2% base, adjusted for regime
        
        # 1. Dynamic resistance break
        resistance_20 = indicators.get('resistance_20', current_price)
        distance_to_resistance = (resistance_20 - current_price) / current_price
        
        if distance_to_resistance < base_threshold:  # Near resistance
            if distance_to_resistance > 0:
                score += 1.5
                signals.append(f"🎯 Approaching resistance: {distance_to_resistance*100:.1f}% away")
            else:  # Broke through
                score += 3
                signals.append(f"🚀 Resistance broken: {abs(distance_to_resistance)*100:.1f}% above")
                
        # 2. Support strength
        support_20 = indicators.get('support_20', current_price)
        distance_to_support = (current_price - support_20) / current_price
        
        if distance_to_support > base_threshold:  # Well above support
            score += 1
            signals.append(f"💪 Strong support base: {distance_to_support*100:.1f}% above")
            
        # 3. Bollinger Band positioning
        bb_position = indicators.get('bb_position', 0.5)
        
        if 0.3 <= bb_position <= 0.7:  # Healthy middle range
            score += 1
            signals.append(f"✅ Healthy BB position: {bb_position:.2f}")
        elif bb_position < 0.3:  # Oversold opportunity
            score += 2
            signals.append(f"🔥 Oversold BB position: {bb_position:.2f}")
            
        self.log_signal(f"Adaptive S/R score: {score:.1f}/6.0")
        return score, signals
    
    def stealth_volume_analysis(self, df, indicators):
        """🕵️ Advanced volume analysis for institutional activity"""
        self.log_signal("Analyzing stealth volume patterns...")
        
        signals = []
        score = 0
        
        # 1. Accumulation pattern (rising price, steady volume)
        vol_current = indicators.get('volume_current', 0)
        vol_avg_20 = indicators.get('volume_avg_20', 0)
        momentum_5 = indicators.get('momentum_5', 0)
        
        if vol_avg_20 > 0:
            vol_consistency = vol_current / vol_avg_20
            if 0.8 <= vol_consistency <= 1.3 and momentum_5 > 1.0:
                score += 2
                signals.append(f"🏗️ Accumulation pattern: Steady vol, rising price")
                
        # 2. Volume spike with price follow-through
        vol_relative = indicators.get('volume_relative', 1.0)
        if vol_relative > 1.5:
            if momentum_5 > 0:  # Price following volume
                score += 2.5
                signals.append(f"🔊 Volume breakout: {vol_relative:.1f}x with price follow-through")
            else:  # Volume without price action (warning)
                score -= 1
                signals.append(f"⚠️ Volume spike without price confirmation")
                
        # 3. Volume trend (increasing over time)
        vol_avg_10 = indicators.get('volume_avg_10', 0)
        if vol_avg_20 > 0 and vol_avg_10 > 0:
            vol_trend = vol_avg_10 / vol_avg_20
            if vol_trend > 1.1:  # 10-day avg > 20-day avg
                score += 1
                signals.append(f"📈 Rising volume trend: {vol_trend:.2f}")
                
        self.log_signal(f"Stealth volume score: {score:.1f}/5.5")
        return score, signals
    
    def calculate_dynamic_confidence(self, all_scores, market_conditions):
        """🎯 Dynamic confidence calculation"""
        self.log_signal("Calculating dynamic confidence...")
        
        # Base scores from different analyses
        pre_breakout_score, momentum_score, sr_score, volume_score = all_scores
        
        # Weight scores based on market regime
        if self.market_regime == "volatile":
            weights = [0.15, 0.35, 0.25, 0.25]  # Emphasize momentum in volatile markets
        elif self.market_regime == "trending":
            weights = [0.20, 0.30, 0.30, 0.20]  # Balanced approach in trending markets
        else:
            weights = [0.25, 0.25, 0.25, 0.25]  # Equal weights in normal markets
            
        # Calculate weighted score
        max_scores = [7.0, 9.0, 6.0, 5.5]  # Maximum possible scores for each category
        normalized_scores = [score/max_score for score, max_score in zip(all_scores, max_scores)]
        weighted_score = sum(w * s for w, s in zip(weights, normalized_scores))
        
        # Convert to confidence percentage
        base_confidence = 50 + (weighted_score * 45)  # 50-95% range
        
        # Boost confidence for signal alignment
        signal_alignment = sum(1 for score in all_scores if score > 0)
        alignment_bonus = min(signal_alignment * 2, 10)
        
        final_confidence = min(base_confidence + alignment_bonus, 99.9)
        
        self.log_signal(f"Confidence calculation:")
        self.log_signal(f"  - Weighted score: {weighted_score:.3f}")
        self.log_signal(f"  - Base confidence: {base_confidence:.1f}%")
        self.log_signal(f"  - Alignment bonus: +{alignment_bonus:.1f}%")
        self.log_signal(f"  - Final confidence: {final_confidence:.1f}%")
        
        return final_confidence
    
    def enhanced_signal_decision(self, df, indicators, symbol):
        """🧠 Master signal decision with enhanced logic"""
        self.log_signal(f"=== ENHANCED SIGNAL DECISION for {symbol} ===")
        
        current_price = indicators['current_price']
        
        # 1. Detect market regime and get multiplier
        vol_multiplier = self.detect_market_regime(df)
        
        # 2. Run all enhanced analyses
        pre_score, pre_signals = self.pre_breakout_scan(indicators, current_price)
        momentum_score, momentum_signals = self.enhanced_momentum_detection(df, indicators)
        sr_score, sr_signals = self.adaptive_support_resistance(df, indicators, vol_multiplier)
        volume_score, volume_signals = self.stealth_volume_analysis(df, indicators)
        
        # 3. Calculate dynamic confidence
        all_scores = [pre_score, momentum_score, sr_score, volume_score]
        confidence = self.calculate_dynamic_confidence(all_scores, self.market_regime)
        
        # 4. Make decision based on enhanced criteria
        total_score = sum(all_scores)
        max_possible = 27.5  # Sum of max scores
        score_percentage = (total_score / max_possible) * 100
        
        # Enhanced decision logic
        if confidence >= 85 and total_score >= 12:
            action = "STRONG_BUY"
            target_gain = 0.06 + (score_percentage - 40) * 0.001  # Dynamic target
        elif confidence >= 75 and total_score >= 8:
            action = "BUY"
            target_gain = 0.04 + (score_percentage - 30) * 0.001
        elif confidence >= 65 and total_score >= 5:
            action = "WEAK_BUY"
            target_gain = 0.03
        elif total_score <= -5:
            action = "SELL"
            target_gain = -0.03
        else:
            action = "WAIT"
            target_gain = 0
            
        # Combine all signals
        all_signals = pre_signals + momentum_signals + sr_signals + volume_signals
        
        result = {
            'action': action,
            'confidence': confidence,
            'total_score': total_score,
            'score_breakdown': {
                'pre_breakout': pre_score,
                'momentum': momentum_score,
                'support_resistance': sr_score,
                'volume': volume_score
            },
            'market_regime': self.market_regime,
            'target_gain_pct': target_gain * 100,
            'signals': all_signals
        }
        
        self.log_signal(f"DECISION: {action} (Confidence: {confidence:.1f}%)")
        self.log_signal(f"Total Score: {total_score:.1f}/{max_possible} ({score_percentage:.1f}%)")
        
        return result

# Example usage integration with your existing code
def integrate_enhanced_signals(advisor_instance):
    """🔧 Integration function for your existing StockAdvisor"""
    
    # Add enhanced detector to your advisor
    advisor_instance.enhanced_detector = EnhancedSignalDetector(debug=advisor_instance.debug)
    
    # Override the generate_enhanced_recommendation method
    def enhanced_generate_recommendation(self, indicators, symbol):
        """Enhanced version of your existing method"""
        
        # Get stock data for enhanced analysis
        df = self.get_stock_data(symbol, datetime.now().date(), days_back=60)
        if df is None:
            return self.original_generate_enhanced_recommendation(indicators, symbol)
            
        # Run enhanced signal detection
        enhanced_result = self.enhanced_detector.enhanced_signal_decision(df, indicators, symbol)
        
        # If enhanced system is highly confident, use its recommendation
        if enhanced_result['confidence'] >= 85:
            self.log(f"Using enhanced signal decision: {enhanced_result['action']}", "SUCCESS")
            
            # Convert to your format
            return {
                'action': enhanced_result['action'].replace('_', '/'),
                'confidence': enhanced_result['confidence'],
                'buy_price': indicators['current_price'],
                'sell_price': indicators['current_price'] * (1 + enhanced_result['target_gain_pct']/100),
                'stop_loss': indicators['current_price'] * 0.94,  # 6% stop loss
                'expected_profit_pct': enhanced_result['target_gain_pct'],
                'reasons': enhanced_result['signals'],
                'final_score': enhanced_result['total_score'],
                'signal_breakdown': enhanced_result['score_breakdown'],
                'current_price': indicators['current_price'],
                'trading_plan': self.build_trading_plan(indicators['current_price'])
            }
        else:
            # Fall back to original system
            return self.original_generate_enhanced_recommendation(indicators, symbol)
    
    # Store original method and replace
    advisor_instance.original_generate_enhanced_recommendation = advisor_instance.generate_enhanced_recommendation
    advisor_instance.generate_enhanced_recommendation = enhanced_generate_recommendation.__get__(advisor_instance)
    
    return advisor_instance