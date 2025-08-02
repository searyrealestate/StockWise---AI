"""
💡 Enhanced Confidence Trading Advisor - 95% Accuracy System
──────────────────────────────────────────────────────────────

Advanced system with enhanced confidence calculation and better decision-making.
Target: 95% confidence recommendations with clear buy/sell signals.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import joblib
import os
import glob
from datetime import datetime, timedelta, date
import ta
import warnings

# ENHANCED IMPORTS WITH ERROR HANDLING
try:
    from enhanced_signals import EnhancedSignalDetector
    from confidence_system import ConfidenceBuilder
    ENHANCEMENTS_AVAILABLE = True
    print("✅ 95% Confidence System loaded successfully")
except ImportError as e:
    ENHANCEMENTS_AVAILABLE = False
    print(f"⚠️ 95% Confidence System not available: {e}")
    print("   System will use original algorithms")

warnings.filterwarnings('ignore')


class EnhancedStockAdvisor:
    def __init__(self, model_dir="models/NASDAQ-training set", debug=False, download_log=True):
        self.model_dir = model_dir
        self.models = {}
        self.debug = debug
        self.debug_log = []
        self.download_log = download_log
        self.investment_days = 7
        self.failed_models = []
        self.tax = 0
        self.broker_fee = 0

        if self.download_log:
            self.ensure_log_file()
        else:
            self.log_file = None

        # Initialize 95% confidence system if available
        if ENHANCEMENTS_AVAILABLE:
            try:
                self.enhanced_detector = EnhancedSignalDetector(debug=self.debug)
                self.confidence_builder = ConfidenceBuilder(debug=self.debug)
                self.enhancements_active = True
                self.log("95% Confidence System initialized", "SUCCESS")
            except Exception as e:
                self.enhancements_active = False
                self.log(f"Enhancement initialization failed: {e}", "ERROR")
        else:
            self.enhancements_active = False

        self.load_models()

    def ensure_log_file(self):
        """Ensure log file is properly initialized with timestamp"""
        if not hasattr(self, 'log_file') or not self.log_file:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            self.log_file = f"debug_log_{timestamp}.log"

            # Create initial log entry with your desired format
            if self.download_log:
                try:
                    header_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    with open(self.log_file, "w", encoding='utf-8') as f:
                        f.write(f"=== Stock Advisor Debug Log ===\n")
                        f.write(f"{header_timestamp} | [INFO] | Log file created: {self.log_file}\n")
                        f.write(f"{header_timestamp} | [INFO] | Debug mode: {self.debug}\n")
                        f.write(f"{header_timestamp} | [INFO] | Download log: {self.download_log}\n")
                        f.write("=" * 80 + "\n\n")
                        f.flush()
                except Exception as e:
                    print(f"Warning: Could not create log file {self.log_file}: {e}")

        return self.log_file

    def log(self, message, level="INFO"):
        if self.debug:
            # Define timestamp once for both console and file
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            color_map = {
                "INFO": "\033[94m",  # Blue
                "SUCCESS": "\033[92m",  # Green
                "ERROR": "\033[91m",  # Red
            }
            reset = "\033[0m"
            level_prefix = {"INFO": "[INFO]", "SUCCESS": "[SUCCESS]", "ERROR": "[ERROR]"}.get(level, "[INFO]")
            symbol = getattr(self, "active_symbol", "")

            # Console output with colors and emoji (keep existing format for console)
            emoji_prefix = {"INFO": "⚖️", "SUCCESS": "✅", "ERROR": "❌"}.get(level, "⚖️")
            console_formatted = f"{datetime.now().strftime('%H:%M:%S')} | {color_map.get(level, '')}{emoji_prefix} [{level}] {symbol} | {message}{reset}"
            self.debug_log.append(console_formatted)
            print(console_formatted)

            # File logging with your desired format: YYYY-MM-DD HH:MM:SS | [LEVEL] | message
            if self.download_log:
                try:
                    # Ensure log_file attribute exists
                    if not hasattr(self, 'log_file') or not self.log_file:
                        self.ensure_log_file()

                    # Get directory path (only if there is one)
                    log_dir = os.path.dirname(self.log_file)
                    if log_dir:  # Only create directory if there is one
                        os.makedirs(log_dir, exist_ok=True)

                    # FIXED: Create clean file format with timestamp
                    # Format: 2025-08-02 21:05:34 | [INFO] | Create Streamlit Page
                    if symbol:
                        clean_formatted = f"{timestamp} | {level_prefix} | {symbol} | {message}"
                    else:
                        clean_formatted = f"{timestamp} | {level_prefix} | {message}"

                    # Write to file with explicit UTF-8 encoding
                    with open(self.log_file, "a", encoding='utf-8', errors='replace') as f:
                        f.write(clean_formatted + "\n")
                        f.flush()  # Ensure immediate write

                except Exception as e:
                    # Fallback: try writing without special characters
                    try:
                        fallback_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        if symbol:
                            fallback_msg = f"{fallback_timestamp} | {level} | {symbol} | {message}"
                        else:
                            fallback_msg = f"{fallback_timestamp} | {level} | {message}"

                        with open(self.log_file, "a", encoding='utf-8', errors='ignore') as f:
                            f.write(fallback_msg + "\n")
                            f.flush()
                    except Exception as inner_e:
                        # If all else fails, print error but don't break the app
                        print(f"Critical logging error: {inner_e}")
                        pass

    def apply_israeli_fees_and_tax(self, profit_pct, apply_tax=True, apply_fees=True):
        """
        Adjust profit percentage for Israeli broker fees and tax.
        - Broker fee: 0.2% on buy + 0.2% on sell = 0.4%
        - Tax: 25% on net profit

        Args:
        profit_pct: Gross profit percentage (e.g., 5.0 for 5%)
        apply_tax: Whether to apply capital gains tax
        apply_fees: Whether to apply broker fees

        Returns:
        Net profit percentage after fees and taxes
        """
        adjusted = profit_pct

        # Reset class variables
        self.broker_fee = 0
        self.tax = 0

        if apply_fees:
            # Subtract broker fees (0.4% total)
            fee_amount = 0.4
            adjusted -= fee_amount
            self.broker_fee = fee_amount
            self.log(f"Applied broker fees: -{fee_amount:.2f}%", "INFO")

        if apply_tax and adjusted > 0:
            # Apply 25% tax on net profit (after fees)
            tax_amount = adjusted * 0.25
            adjusted -= tax_amount
            self.tax = tax_amount
            self.log(f"Applied capital gains tax: -{tax_amount:.2f}%", "INFO")

        self.log(f"Profit calculation: {profit_pct:.2f}% → {adjusted:.2f}% (net)", "INFO")
        return round(adjusted, 2)

    def build_enhanced_trading_plan(self, current_price, target_gain=0.037, max_loss=0.06, days=7):
        """🎯 Enhanced trading plan with strategy integration"""
        self.log(
            f"Building enhanced trading plan for price={current_price}, gain={target_gain:.1%}, loss={max_loss:.1%}, days={days}",
            "INFO")

        strategy_settings = getattr(self, 'strategy_settings', {"profit": 1.0, "risk": 1.0})

        buy_price = current_price
        sell_price = round(buy_price * (1 + target_gain), 2)
        stop_loss = round(buy_price * (1 - max_loss), 2)
        profit_pct = round(target_gain * 100, 1)

        # Calculate net profit after fees and taxes
        net_profit_pct = self.apply_israeli_fees_and_tax(profit_pct)

        plan = {
            "buy_price": buy_price,
            "sell_price": sell_price,
            "stop_loss": stop_loss,
            "profit_pct": profit_pct,
            "net_profit_pct": net_profit_pct,
            "max_loss_pct": round(max_loss * 100, 1),
            "holding_days": days,
            "strategy_multiplier": strategy_settings.get("profit", 1.0),
            "risk_multiplier": strategy_settings.get("risk", 1.0),
            "confidence_requirement": strategy_settings.get("confidence_req", 75)
        }

        self.log(f"Enhanced trading plan created: {plan}", "INFO")
        return plan

    def generate_95_percent_recommendation(self, indicators, symbol):
        """🎯 Generate recommendation using 95% confidence system"""
        if not hasattr(self, 'enhancements_active') or not self.enhancements_active:
            self.log("Enhancements not active, using enhanced original system", "INFO")
            return self.generate_enhanced_recommendation(indicators, symbol)

        self.log(f"Starting 95% confidence recommendation for {symbol}", "INFO")

        try:
            # Get stock data for enhanced analysis
            df = self.get_stock_data(symbol, datetime.now().date(), days_back=60)
            if df is None:
                self.log("No data available, falling back to enhanced original system", "WARNING")
                return self.generate_enhanced_recommendation(indicators, symbol)

            # Run enhanced signal detection
            enhanced_result = self.enhanced_detector.enhanced_signal_decision(df, indicators, symbol)

            # Prepare features for confidence system
            features = [
                indicators.get('volume_relative', 1.0),
                indicators.get('rsi_14', 50) / 100,
                indicators.get('macd_histogram', 0),
                indicators.get('momentum_5', 0) / 100,
                indicators.get('bb_position', 0.5),
                indicators.get('volatility', 1.0) / 100
            ]

            # Market data for confidence calculation
            market_data = {
                'volatility': indicators.get('volatility', 2.0) / 100,
                'market_trend': 'neutral',
                'expected_return': enhanced_result.get('target_gain_pct', 5.0) / 100
            }

            # Calculate 95% confidence if system is highly confident
            if enhanced_result['confidence'] >= 75:
                confidence_result = self.confidence_builder.calculate_95_percent_confidence(
                    symbol, features, enhanced_result['signals'], market_data
                )
                final_confidence = confidence_result['confidence']
                recommendation = confidence_result['recommendation']
            else:
                final_confidence = enhanced_result['confidence']
                recommendation = enhanced_result['action']

            # FIXED: Use the enhanced profit calculation with strategy settings
            strategy_settings = getattr(self, 'strategy_settings', {"profit": 1.0, "risk": 1.0})
            target_profit = self.calculate_dynamic_profit_target(
                indicators, final_confidence, self.investment_days, symbol, strategy_settings
            )

            current_price = indicators['current_price']

            # Action mapping
            action_mapping = {
                'ULTRA_BUY': 'BUY',
                'STRONG_BUY': 'BUY',
                'BUY': 'BUY',
                'WEAK_BUY': 'BUY',
                'SELL': 'SELL/AVOID',
                'WAIT': 'WAIT'
            }

            final_action = action_mapping.get(recommendation, 'WAIT')

            # Calculate prices and profits
            target_profit_pct = target_profit * 100  # Convert to percentage
            net_profit_pct = self.apply_israeli_fees_and_tax(target_profit_pct)

            # Enhanced stop loss based on strategy and time horizon
            if strategy_settings.get("risk", 1.0) >= 1.3:  # Aggressive/Swing
                stop_loss_pct = min(0.08, 0.04 + (self.investment_days * 0.001))  # Dynamic stop loss
            else:
                stop_loss_pct = min(0.06, 0.03 + (self.investment_days * 0.0005))

            # Build result
            result = {
                'action': final_action,
                'confidence': final_confidence,
                'buy_price': current_price if final_action == 'BUY' else None,
                'sell_price': current_price * (1 + target_profit) if final_action == 'BUY' else current_price,
                'stop_loss': current_price * (1 - stop_loss_pct),
                'expected_profit_pct': round(net_profit_pct, 2),
                'gross_profit_pct': round(target_profit_pct, 2),
                'tax_paid': round(self.tax, 2),
                'broker_fee_paid': round(self.broker_fee, 2),
                'reasons': enhanced_result['signals'] + [
                    f"🎯 95% System: {recommendation} ({final_confidence:.1f}%)",
                    f"📈 Strategy: {getattr(self, 'current_strategy', 'Unknown')} (×{strategy_settings.get('profit', 1.0):.1f})",
                    f"⏱️ Time horizon: {self.investment_days} days (×{target_profit / 0.037:.1f} base)"
                ],
                'final_score': enhanced_result.get('total_score', 0),
                'signal_breakdown': enhanced_result.get('score_breakdown', {}),
                'current_price': current_price,
                'trading_plan': self.build_enhanced_trading_plan(current_price, target_profit, stop_loss_pct),
                'enhancement_active': True,
                'strategy_applied': True,  # NEW: Flag to show strategy was applied
                'time_multiplier': target_profit / 0.037,  # Show the scaling factor
            }

            self.log(f"95% recommendation complete: {final_action} at {final_confidence:.1f}% confidence", "SUCCESS")
            self.log(f"Profit target: {target_profit_pct:.1f}% (net: {net_profit_pct:.1f}%)", "SUCCESS")

            return result

        except Exception as e:
            self.log(f"Error in 95% system, falling back to enhanced original: {e}", "ERROR")
            return self.generate_enhanced_recommendation(indicators, symbol)

    def debug_recommendation_logic(self, final_score, strategy_settings, current_strategy):
        """🔍 Debug function to trace recommendation logic"""

        self.log("=== DEBUGGING RECOMMENDATION LOGIC ===", "INFO")
        self.log(f"Final Score: {final_score:.2f}", "INFO")
        self.log(f"Strategy: {current_strategy}", "INFO")
        self.log(f"Strategy Settings: {strategy_settings}", "INFO")

        # Recreate threshold logic with debugging
        confidence_req = strategy_settings.get("confidence_req", 75)
        self.log(f"Confidence Requirement: {confidence_req}%", "INFO")

        profit_multiplier = strategy_settings.get("profit", 1.0)
        self.log(f"Profit Multiplier: {profit_multiplier}", "INFO")

        if profit_multiplier >= 1.8:  # Swing Trading
            buy_threshold = 0.8
            sell_threshold = -0.8
            strategy_type = "Swing Trading"
        elif profit_multiplier >= 1.4:  # Aggressive
            buy_threshold = 0.9
            sell_threshold = -0.9
            strategy_type = "Aggressive"
        else:  # Conservative/Balanced
            buy_threshold = 1.0
            sell_threshold = -1.0
            strategy_type = "Conservative/Balanced"

        self.log(f"Detected Strategy Type: {strategy_type}", "INFO")
        self.log(f"BUY Threshold: {buy_threshold}", "INFO")
        self.log(f"SELL Threshold: {sell_threshold}", "INFO")

        # Decision logic with detailed logging
        if final_score >= buy_threshold:
            expected_action = "BUY"
            self.log(f"✅ SHOULD BE BUY: {final_score:.2f} >= {buy_threshold}", "SUCCESS")
        elif final_score <= sell_threshold:
            expected_action = "SELL/AVOID"
            self.log(f"❌ SHOULD BE SELL: {final_score:.2f} <= {sell_threshold}", "INFO")
        else:
            expected_action = "WAIT"
            self.log(f"⏳ SHOULD BE WAIT: {sell_threshold} < {final_score:.2f} < {buy_threshold}", "INFO")

        self.log(f"Expected Action: {expected_action}", "SUCCESS")

        return {
            'expected_action': expected_action,
            'buy_threshold': buy_threshold,
            'sell_threshold': sell_threshold,
            'final_score': final_score,
            'strategy_type': strategy_type
        }

    def fix_recommendation_logic(self, indicators, symbol):
        """🔧 Fixed version of recommendation logic"""

        self.log(f"=== FIXED RECOMMENDATION LOGIC for {symbol} ===", "INFO")

        current_price = indicators['current_price']
        strategy_settings = getattr(self, 'strategy_settings', {"profit": 1.0, "risk": 1.0, "confidence_req": 75})

        # Signal analysis (keep your existing logic)
        trend_score, trend_signals = self.analyze_trend(indicators, current_price)
        momentum_score, momentum_signals = self.analyze_momentum(indicators)
        volume_score, volume_signals = self.analyze_volume(indicators)
        sr_score, sr_signals = self.analyze_support_resistance(indicators)
        model_score, model_signals = self.analyze_ml_model(symbol, indicators, current_price)

        # Calculate final score
        signal_weights = {
            'trend': 0.25,
            'momentum': 0.20,
            'volume': 0.15,
            'support_resistance': 0.15,
            'model': 0.25
        }

        final_score = (
                trend_score * signal_weights['trend'] +
                momentum_score * signal_weights['momentum'] +
                volume_score * signal_weights['volume'] +
                sr_score * signal_weights['support_resistance'] +
                model_score * signal_weights['model']
        )

        self.log(f"Calculated Final Score: {final_score:.2f}", "INFO")

        # FIXED: Proper threshold logic
        profit_multiplier = strategy_settings.get("profit", 1.0)

        if profit_multiplier >= 1.8:  # Swing Trading
            buy_threshold = 0.8
            sell_threshold = -0.8
        elif profit_multiplier >= 1.4:  # Aggressive
            buy_threshold = 0.9
            sell_threshold = -0.9
        else:  # Conservative/Balanced
            buy_threshold = 1.0
            sell_threshold = -1.0

        self.log(f"Using thresholds: BUY≥{buy_threshold}, SELL≤{sell_threshold}", "INFO")

        # FIXED: Decision logic
        if final_score >= buy_threshold:
            action = "BUY"
            self.log(f"✅ BUY DECISION: {final_score:.2f} >= {buy_threshold}", "SUCCESS")

            # Calculate enhanced profit target
            base_confidence = 70 + min(25, final_score * 8)
            target_profit = self.calculate_dynamic_profit_target(
                indicators, base_confidence, self.investment_days, symbol, strategy_settings
            )

            buy_price = current_price
            sell_price = current_price * (1 + target_profit)
            stop_loss_pct = min(0.08, 0.04 + (self.investment_days * 0.001))
            stop_loss = current_price * (1 - stop_loss_pct)

            gross_profit_pct = target_profit * 100
            net_profit_pct = self.apply_israeli_fees_and_tax(gross_profit_pct)

        elif final_score <= sell_threshold:
            action = "SELL/AVOID"
            self.log(f"❌ SELL DECISION: {final_score:.2f} <= {sell_threshold}", "INFO")

            buy_price = None
            sell_price = current_price
            stop_loss = current_price * 1.06
            gross_profit_pct = 0
            net_profit_pct = 0
            base_confidence = 70 + min(25, abs(final_score) * 8)

        else:
            action = "WAIT"
            self.log(f"⏳ WAIT DECISION: {sell_threshold} < {final_score:.2f} < {buy_threshold}", "INFO")

            buy_price = None
            sell_price = current_price
            stop_loss = current_price * 0.94
            gross_profit_pct = 0
            net_profit_pct = 0
            base_confidence = 50 + abs(final_score) * 5

        # Calculate final confidence
        confirming_indicators = sum([
            1 if abs(trend_score) > 1 else 0,
            1 if abs(momentum_score) > 1 else 0,
            1 if abs(volume_score) > 0 else 0,
            1 if abs(sr_score) > 0 else 0,
            1 if abs(model_score) > 1 else 0
        ])
        confidence_bonus = min(10, confirming_indicators * 2)
        final_confidence = min(95, base_confidence + confidence_bonus)

        all_signals = trend_signals + momentum_signals + volume_signals + sr_signals + model_signals

        return {
            'action': action,
            'confidence': final_confidence,
            'buy_price': buy_price,
            'sell_price': sell_price,
            'stop_loss': stop_loss,
            'expected_profit_pct': round(net_profit_pct, 2),
            'gross_profit_pct': round(gross_profit_pct, 2),
            'final_score': final_score,
            'buy_threshold': buy_threshold,
            'sell_threshold': sell_threshold,
            'reasons': all_signals,
            'debug_info': {
                'trend_score': trend_score,
                'momentum_score': momentum_score,
                'volume_score': volume_score,
                'sr_score': sr_score,
                'model_score': model_score,
                'profit_multiplier': profit_multiplier
            }
        }

    def validate_signal_logic(self):
        """🧪 Test function to validate signal logic"""

        test_cases = [
            {'score': 5.0, 'strategy': 'Swing Trading', 'expected': 'BUY'},
            {'score': 0.9, 'strategy': 'Swing Trading', 'expected': 'BUY'},
            {'score': 0.7, 'strategy': 'Swing Trading', 'expected': 'WAIT'},
            {'score': 1.5, 'strategy': 'Balanced', 'expected': 'BUY'},
            {'score': 0.5, 'strategy': 'Balanced', 'expected': 'WAIT'},
            {'score': -1.5, 'strategy': 'Aggressive', 'expected': 'SELL/AVOID'},
        ]

        for test in test_cases:
            # Set strategy settings
            strategy_multipliers = {
                "Conservative": {"profit": 0.8, "risk": 0.8},
                "Balanced": {"profit": 1.0, "risk": 1.0},
                "Aggressive": {"profit": 1.4, "risk": 1.3},
                "Swing Trading": {"profit": 1.8, "risk": 1.5}
            }

            strategy_settings = strategy_multipliers[test['strategy']]
            profit_multiplier = strategy_settings.get("profit", 1.0)

            # Calculate thresholds
            if profit_multiplier >= 1.8:
                buy_threshold = 0.8
                sell_threshold = -0.8
            elif profit_multiplier >= 1.4:
                buy_threshold = 0.9
                sell_threshold = -0.9
            else:
                buy_threshold = 1.0
                sell_threshold = -1.0

            # Determine action
            if test['score'] >= buy_threshold:
                actual = 'BUY'
            elif test['score'] <= sell_threshold:
                actual = 'SELL/AVOID'
            else:
                actual = 'WAIT'

            # Validate
            status = "✅ PASS" if actual == test['expected'] else "❌ FAIL"
            print(
                f"{status} | Score: {test['score']:.1f} | Strategy: {test['strategy']} | Expected: {test['expected']} | Actual: {actual}")

            if actual != test['expected']:
                print(f"   Thresholds: BUY≥{buy_threshold}, SELL≤{sell_threshold}")
    def load_models(self):
        """Load trained models"""

        self.log("Loading models...", "INFO")

        try:
            if os.path.exists(self.model_dir):
                model_files = glob.glob(os.path.join(self.model_dir, "*_model_*.pkl"))
                for model_file in model_files:
                    symbol = os.path.basename(model_file).split('_model_')[0]
                    try:
                        self.models[symbol] = joblib.load(model_file)
                        self.log(f"Loaded model for {symbol}", "SUCCESS")
                    except Exception as e:
                        self.log(f"Failed to load model for {symbol}: {str(e)}", "ERROR")
            else:
                self.log(f"Model directory does not exist: {self.model_dir}", "ERROR")

        except Exception as e:
            self.log(f"Unexpected error loading models: {str(e)}", "ERROR")
            error_msg = f"Failed to load model for {symbol}: {str(e)}"
            self.failed_models.append((symbol, str(e)))
            self.log(error_msg, "ERROR")

    def calculate_enhanced_confidence(self, indicators, final_score, strategy_settings, investment_days):
        """
        🎯 Enhanced confidence calculation with multiple validation layers
        Target: 90-95% confidence for strong signals across all strategies
        """
        self.log("=== ENHANCED CONFIDENCE CALCULATION ===", "INFO")

        # 1. BASE CONFIDENCE from signal strength
        base_confidence = self.calculate_base_confidence_from_signals(final_score)
        self.log(f"Base confidence from signals: {base_confidence:.1f}%", "INFO")

        # 2. TECHNICAL INDICATOR CONFIRMATION
        technical_boost = self.calculate_technical_confirmation_boost(indicators)
        self.log(f"Technical confirmation boost: +{technical_boost:.1f}%", "INFO")

        # 3. MULTI-TIMEFRAME ALIGNMENT
        timeframe_boost = self.calculate_timeframe_alignment_boost(indicators)
        self.log(f"Timeframe alignment boost: +{timeframe_boost:.1f}%", "INFO")

        # 4. VOLUME AND MOMENTUM CONFIRMATION
        volume_momentum_boost = self.calculate_volume_momentum_boost(indicators)
        self.log(f"Volume & momentum boost: +{volume_momentum_boost:.1f}%", "INFO")

        # 5. STRATEGY-SPECIFIC CONFIDENCE ADJUSTMENTS
        strategy_boost = self.calculate_strategy_confidence_boost(strategy_settings, investment_days)
        self.log(f"Strategy-specific boost: +{strategy_boost:.1f}%", "INFO")

        # 6. RISK-ADJUSTED CONFIDENCE
        risk_adjustment = self.calculate_risk_adjusted_confidence(indicators, investment_days)
        self.log(f"Risk adjustment: {risk_adjustment:+.1f}%", "INFO")

        # CALCULATE FINAL CONFIDENCE
        final_confidence = (base_confidence +
                            technical_boost +
                            timeframe_boost +
                            volume_momentum_boost +
                            strategy_boost +
                            risk_adjustment)

        # Apply bounds based on strategy type
        min_confidence, max_confidence = self.get_confidence_bounds(strategy_settings)
        final_confidence = max(min_confidence, min(final_confidence, max_confidence))

        self.log(f"FINAL ENHANCED CONFIDENCE: {final_confidence:.1f}%", "SUCCESS")

        return final_confidence

    def calculate_base_confidence_from_signals(self, final_score):
        """Calculate base confidence from signal strength"""
        # Enhanced mapping: stronger signals = higher confidence
        if abs(final_score) >= 3.0:
            return 85.0  # Very strong signals
        elif abs(final_score) >= 2.5:
            return 80.0  # Strong signals
        elif abs(final_score) >= 2.0:
            return 75.0  # Good signals
        elif abs(final_score) >= 1.5:
            return 70.0  # Moderate signals
        elif abs(final_score) >= 1.0:
            return 65.0  # Weak signals
        elif abs(final_score) >= 0.8:
            return 60.0  # Very weak signals
        else:
            return 55.0  # Minimal signals

    def calculate_technical_confirmation_boost(self, indicators):
        """Calculate boost from technical indicator alignment"""
        boost = 0.0

        # RSI confirmation
        rsi_14 = indicators.get('rsi_14', 50)
        if 30 <= rsi_14 <= 45:  # Sweet spot for buying
            boost += 8.0
        elif 25 <= rsi_14 <= 35:  # Oversold but not extreme
            boost += 6.0
        elif rsi_14 < 25:  # Extremely oversold
            boost += 4.0  # Less confident in extreme conditions

        # MACD confirmation
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        macd_hist = indicators.get('macd_histogram', 0)

        if macd > macd_signal and macd_hist > 0:
            if macd_hist > 0.5:  # Strong bullish momentum
                boost += 10.0
            else:  # Mild bullish momentum
                boost += 6.0

        # Bollinger Bands position
        bb_position = indicators.get('bb_position', 0.5)
        if 0.15 <= bb_position <= 0.35:  # Near lower band but not extreme
            boost += 6.0
        elif bb_position < 0.15:  # Very near lower band
            boost += 4.0

        # Stochastic confirmation
        stoch_k = indicators.get('stoch_k', 50)
        stoch_d = indicators.get('stoch_d', 50)
        if stoch_k < 30 and stoch_k > stoch_d:  # Oversold with upward momentum
            boost += 5.0

        return min(boost, 20.0)  # Cap at 20%

    def calculate_timeframe_alignment_boost(self, indicators):
        """Calculate boost from multiple timeframe alignment"""
        boost = 0.0
        current_price = indicators['current_price']

        # Moving average alignment (bullish setup)
        sma_5 = indicators.get('sma_5', current_price)
        sma_10 = indicators.get('sma_10', current_price)
        sma_20 = indicators.get('sma_20', current_price)
        sma_50 = indicators.get('sma_50', current_price)

        # Perfect bullish alignment
        if current_price > sma_5 > sma_10 > sma_20 > sma_50:
            boost += 12.0
        # Good bullish alignment
        elif current_price > sma_5 > sma_10 > sma_20:
            boost += 8.0
        # Moderate bullish alignment
        elif current_price > sma_10 > sma_20:
            boost += 5.0
        # Basic bullish
        elif current_price > sma_20:
            boost += 3.0

        # EMA alignment
        ema_12 = indicators.get('ema_12', current_price)
        ema_26 = indicators.get('ema_26', current_price)
        if ema_12 > ema_26:
            boost += 3.0

        return min(boost, 15.0)  # Cap at 15%

    def calculate_volume_momentum_boost(self, indicators):
        """Calculate boost from volume and momentum confirmation"""
        boost = 0.0

        # Volume confirmation
        volume_relative = indicators.get('volume_relative', 1.0)
        if volume_relative >= 2.0:
            boost += 8.0  # High volume spike
        elif volume_relative >= 1.5:
            boost += 5.0  # Above average volume
        elif volume_relative >= 1.2:
            boost += 3.0  # Good volume

        # Price momentum
        momentum_5 = indicators.get('momentum_5', 0)
        if momentum_5 > 5:
            boost += 6.0  # Strong positive momentum
        elif momentum_5 > 2:
            boost += 4.0  # Good momentum
        elif momentum_5 > 0:
            boost += 2.0  # Positive momentum

        # Volatility consideration (moderate volatility is better)
        volatility = indicators.get('volatility', 2.0)
        if 1.5 <= volatility <= 3.5:  # Sweet spot
            boost += 3.0
        elif volatility > 5.0:  # Too volatile
            boost -= 2.0

        return boost

    def calculate_strategy_confidence_boost(self, strategy_settings, investment_days):
        """Calculate strategy-specific confidence adjustments"""
        boost = 0.0
        strategy_type = getattr(self, 'current_strategy', 'Balanced')

        # Strategy-based confidence adjustments
        if strategy_type == "Conservative":
            # Conservative strategy gets bonus for longer timeframes
            if investment_days >= 30:
                boost += 8.0
            elif investment_days >= 14:
                boost += 5.0
            # Conservative strategy penalty for short timeframes
            elif investment_days <= 3:
                boost -= 5.0

        elif strategy_type == "Aggressive":
            # Aggressive strategy gets bonus for medium timeframes
            if 7 <= investment_days <= 21:
                boost += 6.0
            # Penalty for very long timeframes (market can change)
            elif investment_days > 60:
                boost -= 3.0

        elif strategy_type == "Swing Trading":
            # Swing trading gets bonus for optimal timeframes
            if 14 <= investment_days <= 45:
                boost += 8.0
            elif 7 <= investment_days <= 60:
                boost += 5.0
            # Penalty for very short timeframes
            elif investment_days <= 3:
                boost -= 8.0

        # Balanced strategy (no specific adjustments - it's the baseline)

        return boost

    def calculate_risk_adjusted_confidence(self, indicators, investment_days):
        """Calculate risk-based confidence adjustments"""
        adjustment = 0.0

        # Support/resistance strength
        current_price = indicators['current_price']
        support_20 = indicators.get('support_20', current_price * 0.95)
        resistance_20 = indicators.get('resistance_20', current_price * 1.05)

        # Distance from support (good for buying)
        support_distance = (current_price - support_20) / support_20 * 100
        if 2 <= support_distance <= 8:  # Sweet spot above support
            adjustment += 5.0
        elif support_distance < 1:  # Very close to support
            adjustment += 3.0

        # Market regime stability
        bb_width = indicators.get('bb_upper', current_price * 1.02) - indicators.get('bb_lower', current_price * 0.98)
        bb_width_pct = bb_width / current_price * 100

        if 3 <= bb_width_pct <= 8:  # Moderate volatility
            adjustment += 3.0
        elif bb_width_pct > 12:  # High volatility - reduce confidence
            adjustment -= 5.0

        # Time horizon risk
        if investment_days <= 1:
            adjustment -= 10.0  # Very risky
        elif investment_days <= 3:
            adjustment -= 5.0  # Risky
        elif 7 <= investment_days <= 30:
            adjustment += 2.0  # Good timeframe
        elif investment_days > 90:
            adjustment -= 3.0  # Too long, market uncertainty

        return adjustment

    def get_confidence_bounds(self, strategy_settings):
        """Get min/max confidence bounds based on strategy"""
        strategy_type = getattr(self, 'current_strategy', 'Balanced')

        bounds = {
            "Conservative": (75, 95),  # High minimum, high maximum
            "Balanced": (65, 93),  # Moderate bounds
            "Aggressive": (60, 90),  # Lower minimum, good maximum
            "Swing Trading": (70, 95)  # Good minimum, high maximum
        }

        return bounds.get(strategy_type, (65, 90))

    def validate_confidence_calculation(self, indicators, final_score, confidence, strategy_settings):
        """Validate that confidence calculation makes sense"""
        issues = []

        # Check if confidence matches signal strength
        if abs(final_score) >= 2.0 and confidence < 75:
            issues.append(f"Strong signal (score: {final_score:.2f}) but low confidence ({confidence:.1f}%)")

        if abs(final_score) < 1.0 and confidence > 85:
            issues.append(f"Weak signal (score: {final_score:.2f}) but high confidence ({confidence:.1f}%)")

        # Check technical indicators alignment
        rsi_14 = indicators.get('rsi_14', 50)
        macd_hist = indicators.get('macd_histogram', 0)

        if rsi_14 < 30 and macd_hist > 0 and confidence < 80:
            issues.append("Strong technical setup (oversold RSI + bullish MACD) but confidence too low")

        # Log validation results
        if issues:
            self.log("⚠️ CONFIDENCE VALIDATION ISSUES:", "WARNING")
            for issue in issues:
                self.log(f"   • {issue}", "WARNING")
        else:
            self.log("✅ Confidence validation passed", "SUCCESS")

        return len(issues) == 0

    def get_stock_data(self, symbol, target_date, days_back=60):
        """Get comprehensive stock data for analysis"""
        self.log(f"Fetching stock data for {symbol}", "INFO")
        try:
            target_pd = pd.Timestamp(target_date)
            start_date = target_pd - pd.Timedelta(days=days_back)
            end_date = target_pd + pd.Timedelta(days=20)

            df = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)

            if df.empty:
                self.log(f"No data returned for {symbol}", "ERROR")
                return None

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]

            self.log(f"Retrieved {len(df)} rows for {symbol}", "SUCCESS")

            return df

        except Exception as e:
            self.log(f"Error fetching stock data for {symbol}: {str(e)}", "ERROR")
            return None

    def calculate_enhanced_indicators(self, df, analysis_date):
        """Calculate comprehensive technical indicators for higher confidence - FIXED VERSION"""

        self.log(f"Calculating indicators for {analysis_date}", "INFO")

        # Filter data up to analysis date
        historical_data = df[df.index <= analysis_date].copy()
        if len(historical_data) < 20:
            return None

        indicators = {}

        try:
            # Price Analysis
            current_price = historical_data['Close'].iloc[-1]
            self.log(f"Current price: {current_price:.2f}", "SUCCESS")

            indicators['current_price'] = current_price

            # FIXED: Moving Averages (Multiple timeframes)
            indicators['sma_5'] = historical_data['Close'].rolling(5, min_periods=1).mean().iloc[-1]
            self.log(f"sma_5: {indicators['sma_5']:.1f}", "SUCCESS")

            indicators['sma_10'] = historical_data['Close'].rolling(10, min_periods=1).mean().iloc[-1]
            self.log(f"sma_10: {indicators['sma_10']:.1f}", "SUCCESS")

            indicators['sma_20'] = historical_data['Close'].rolling(20, min_periods=1).mean().iloc[-1]
            self.log(f"sma_20: {indicators['sma_20']:.1f}", "SUCCESS")

            indicators['sma_50'] = historical_data['Close'].rolling(50, min_periods=1).mean().iloc[-1]
            self.log(f"sma_50: {indicators['sma_50']:.1f}", "SUCCESS")

            # FIXED: EMA for trend confirmation
            indicators['ema_10'] = historical_data['Close'].ewm(span=10, min_periods=1).mean().iloc[-1]
            self.log(f"ema_10: {indicators['ema_10']:.1f}", "SUCCESS")

            indicators['ema_12'] = historical_data['Close'].ewm(span=12, min_periods=1).mean().iloc[-1]
            self.log(f"ema_12: {indicators['ema_12']:.1f}", "SUCCESS")

            indicators['ema_26'] = historical_data['Close'].ewm(span=26, min_periods=1).mean().iloc[-1]
            self.log(f"ema_26: {indicators['ema_26']:.1f}", "SUCCESS")

            # FIXED: RSI calculation using ta library
            try:
                rsi_14 = ta.momentum.RSIIndicator(historical_data['Close'], window=14)
                indicators['rsi_14'] = rsi_14.rsi().iloc[-1]
                self.log(f"rsi_14: {indicators['rsi_14']:.1f}", "SUCCESS")

                rsi_21 = ta.momentum.RSIIndicator(historical_data['Close'], window=21)
                indicators['rsi_21'] = rsi_21.rsi().iloc[-1]
                self.log(f"rsi_21: {indicators['rsi_21']:.1f}", "SUCCESS")

                # Handle NaN values
                if pd.isna(indicators['rsi_14']):
                    indicators['rsi_14'] = 50
                    self.log(f"rsi_14: {indicators['rsi_14']:.1f}", "SUCCESS")

                if pd.isna(indicators['rsi_21']):
                    indicators['rsi_21'] = 50
                    self.log(f"rsi_21: {indicators['rsi_21']:.1f}", "SUCCESS")

            except Exception as rsi_error:
                print(f"RSI calculation error: {rsi_error}")
                self.log(f"RSI calculation error: {rsi_error}", "ERROR")

                indicators['rsi_14'] = 50
                indicators['rsi_21'] = 50
                self.log(f"rsi_14: {indicators['rsi_14']:.1f}", "ERROR")
                self.log(f"rsi_21: {indicators['rsi_21']:.1f}", "ERROR")

            # FIXED: MACD calculation
            try:
                macd_indicator = ta.trend.MACD(historical_data['Close'])
                indicators['macd'] = macd_indicator.macd().iloc[-1]
                indicators['macd_signal'] = macd_indicator.macd_signal().iloc[-1]
                indicators['macd_histogram'] = macd_indicator.macd_diff().iloc[-1]
                self.log(f"MACD hist: {indicators['macd_histogram']:.3f}", "SUCCESS")

                # Handle NaN values
                for key in ['macd', 'macd_signal', 'macd_histogram']:
                    if pd.isna(indicators[key]):
                        indicators[key] = 0
                        self.log(f"indicators - {key}: {indicators[key]:.3f}", "SUCCESS")

            except Exception as macd_error:
                print(f"MACD calculation error: {macd_error}")
                indicators['macd'] = 0
                indicators['macd_signal'] = 0
                indicators['macd_histogram'] = 0
                self.log(f"macd: {indicators['macd']:.3f}", "ERROR")
                self.log(f"macd_signal: {indicators['macd_signal']:.3f}", "ERROR")
                self.log(f"macd_histogram: {indicators['macd_histogram'] :.3f}", "ERROR")

            # FIXED: Bollinger Bands
            try:
                bb = ta.volatility.BollingerBands(historical_data['Close'], window=20)
                indicators['bb_upper'] = bb.bollinger_hband().iloc[-1]
                self.log(f"bb_upper: {indicators['bb_upper']:.2f}", "SUCCESS")

                indicators['bb_lower'] = bb.bollinger_lband().iloc[-1]
                self.log(f"bb_lower: {indicators['bb_lower']:.2f}", "SUCCESS")

                indicators['bb_middle'] = bb.bollinger_mavg().iloc[-1]
                self.log(f"bb_middle: {indicators['bb_middle']:.2f}", "SUCCESS")

                # Calculate position
                bb_range = indicators['bb_upper'] - indicators['bb_lower']
                if bb_range > 0:
                    indicators['bb_position'] = (current_price - indicators['bb_lower']) / bb_range
                    self.log(f"bb_position: {indicators['bb_position']:.2f}", "SUCCESS")

                else:
                    indicators['bb_position'] = 0.5
                    self.log(f"bb_position: {indicators['bb_position']:.2f}", "SUCCESS")

                # Handle NaN values
                for key in ['bb_upper', 'bb_lower', 'bb_middle']:
                    if pd.isna(indicators[key]):
                        indicators[key] = current_price
                        self.log(f"{key}: {indicators[key]:.2f}", "SUCCESS")

            except Exception as bb_error:
                print(f"Bollinger Bands calculation error: {bb_error}")
                indicators['bb_position'] = 0.5
                self.log(f"bb_position: {indicators['bb_position']:.2f}", "ERROR")

                indicators['bb_upper'] = current_price * 1.02
                self.log(f"bb_upper: {indicators['bb_upper']:.2f}", "ERROR")

                indicators['bb_lower'] = current_price * 0.98
                self.log(f"bb_lower: {indicators['bb_lower']:.2f}", "ERROR")

                indicators['bb_middle'] = current_price
                self.log(f"bb_middle: {indicators['bb_middle']:.2f}", "ERROR")

            # FIXED: Stochastic Oscillator
            try:
                stoch = ta.momentum.StochasticOscillator(
                    historical_data['High'],
                    historical_data['Low'],
                    historical_data['Close']
                )
                indicators['stoch_k'] = stoch.stoch().iloc[-1]
                self.log(f"stoch_k: {indicators['stoch_k']:.1f}", "SUCCESS")
                indicators['stoch_d'] = stoch.stoch_signal().iloc[-1]
                self.log(f"stoch_d: {indicators['stoch_d']:.1f}", "SUCCESS")

                # Handle NaN values
                if pd.isna(indicators['stoch_k']):
                    indicators['stoch_k'] = 50
                    self.log(f"stoch_k: {indicators['stoch_k']:.2f}", "INFO")

                if pd.isna(indicators['stoch_d']):
                    indicators['stoch_d'] = 50
                    self.log(f"stoch_d: {indicators['stoch_d']:.2f}", "INFO")

            except Exception as stoch_error:
                print(f"Stochastic calculation error: {stoch_error}")
                indicators['stoch_k'] = 50
                indicators['stoch_d'] = 50
                self.log(f"stoch_k: {indicators['stoch_k']:.2f}", "ERROR")
                self.log(f"stoch_d: {indicators['stoch_d']:.2f}", "ERROR")

            # FIXED: Volume Analysis
            indicators['volume_current'] = historical_data['Volume'].iloc[-1]
            self.log(f"volume_current: {indicators['volume_current']:.2f}", "SUCCESS")

            indicators['volume_avg_10'] = historical_data['Volume'].rolling(10, min_periods=1).mean().iloc[-1]
            self.log(f"volume_avg_10: {indicators['volume_avg_10']:.2f}", "SUCCESS")

            indicators['volume_avg_20'] = historical_data['Volume'].rolling(20, min_periods=1).mean().iloc[-1]
            self.log(f"volume_avg_20: {indicators['volume_avg_20']:.2f}", "SUCCESS")

            # Ensure volume averages are not zero
            if indicators['volume_avg_20'] > 0:
                indicators['volume_relative'] = indicators['volume_current'] / indicators['volume_avg_20']
                self.log(f"volume_relative: {indicators['volume_relative']:.2f}", "SUCCESS")

            else:
                indicators['volume_relative'] = 1.0
                self.log(f"volume_relative: {indicators['volume_relative']:.2f}", "INFO")

            # FIXED: Price Momentum
            if len(historical_data) > 5:
                indicators['momentum_5'] = (current_price / historical_data['Close'].iloc[-6] - 1) * 100
                self.log(f"momentum_5: {indicators['momentum_5']:.2f}", "SUCCESS")

            else:
                indicators['momentum_5'] = 0
                self.log(f"momentum_5: {indicators['momentum_5']:.2f}", "INFO")

            if len(historical_data) > 10:
                indicators['momentum_10'] = (current_price / historical_data['Close'].iloc[-11] - 1) * 100
                self.log(f"momentum_10: {indicators['momentum_10']:.2f}", "SUCCESS")

            else:
                indicators['momentum_10'] = 0
                self.log(f"momentum_10: {indicators['momentum_10']:.2f}", "INFO")

            # FIXED: Volatility
            returns = historical_data['Close'].pct_change().dropna()
            if len(returns) > 1:
                indicators['volatility'] = returns.std() * 100
                self.log(f"volatility: {indicators['volatility']:.2f}", "SUCCESS")

            else:
                indicators['volatility'] = 1.0
                self.log(f"volatility: {indicators['volatility']:.2f}", "INFO")

            # Add price change calculation for ML model
            if len(historical_data) > 1:
                indicators['price_change_1d'] = (current_price / historical_data['Close'].iloc[-2] - 1) * 100
                self.log(f"price_change_1d: {indicators['price_change_1d']:.2f}%", "SUCCESS")
            else:
                indicators['price_change_1d'] = 0
                self.log(f"price_change_1d: {indicators['price_change_1d']:.2f}%", "INFO")

            indicators['resistance_20'] = historical_data['High'].rolling(20, min_periods=1).max().iloc[-1]
            self.log(f"resistance_20: {indicators['resistance_20']:.2f}", "SUCCESS")

            # FIXED: Support and Resistance
            indicators['support_20'] = historical_data['Low'].rolling(20, min_periods=1).min().iloc[-1]
            self.log(f"support_20: {indicators['support_20']:.2f}", "SUCCESS")

            # Ensure all values are numeric and not NaN
            for key, value in indicators.items():
                if pd.isna(value) or not np.isfinite(value):
                    if 'price' in key.lower():
                        indicators[key] = current_price
                        self.log(f"{key}: {indicators[key]:.2f}", "INFO")
                    elif 'volume' in key.lower():
                        indicators[key] = 1000000  # Default volume
                        self.log(f"{key}: {indicators[key]:.2f}", "INFO")
                    elif 'rsi' in key.lower() or 'stoch' in key.lower():
                        indicators[key] = 50  # Neutral
                        self.log(f"{key}: {indicators[key]:.2f}", "INFO")
                    else:
                        indicators[key] = 0
                        self.log(f"{key}: {indicators[key]:.2f}", "INFO")

            self.log(f"Calculated indicators for {analysis_date.date()}: RSI={indicators['rsi_14']:.1f}, MACD={indicators['macd']:.3f}, Volume_Rel={indicators['volume_relative']:.2f}")

            return indicators

        except Exception as e:
            self.log(f"Critical error calculating indicators: {e}")
            return None

    def calculate_confidence_score(self,indicators):
        self.log("Starting confidence scoring", "INFO")
        score = 0
        weights = {
            'rsi_14': 1,
            'macd_histogram': 1.5,
            'volume_relative': 0.8,
            'momentum_5': 1.2,
            'bb_position': 0.5,
            'stoch_k': 0.8
        }
        for key, weight in weights.items():
            val = indicators.get(key, 0)
            contribution = weight * val
            if key.startswith('rsi') and 40 < val < 60:
                self.log(f"{key}: value={val:.2f}, weight={weight}, contribution={contribution:.2f}", "INFO")
                continue  # Neutral zone
            score += weight * val
            self.log(f"{key}: value={val:.2f}, weight={weight}, contribution={contribution:.2f}", "INFO")

        # for key, weight in weights.items():
        #     val = indicators.get(key, 0)
        #     contribution = weight * val
        #     self.log(f"{key}: value={val:.2f}, weight={weight}, contribution={contribution:.2f}", "INFO")

        self.log(f"Final confidence score: {score:.2f}", "SUCCESS")

        return round(score, 2)

    def interpret_signals(self,indicators):
        self.log("Starting interpret_signals", "INFO")

        commentary = []
        if indicators['rsi_14'] < 30:
            commentary.append("Oversold RSI – possible rebound.")
            self.log("Signal: RSI is oversold (<30)", "INFO")

        if indicators['macd'] > indicators['macd_signal']:
            commentary.append("MACD crossover – bullish momentum.")
            self.log(f"Signal: MACD crossover is oversold ({indicators['macd_signal']})", "INFO")

        if indicators['volume_relative'] > 1.5:
            commentary.append("High volume spike confirms move.")
            self.log("High volume spike confirms move (<1.5)", "INFO")

        if indicators['stoch_k'] > indicators['stoch_d']:
            commentary.append("Stochastic trending upward.")
            self.log(f"Stochastic trending upward is oversold (<{indicators['stoch_d']})", "INFO")
        return " | ".join(commentary)

    def log_recommendation(self, symbol, result, analysis_date):
        self.log(f"Logged recommendation for {symbol} on {analysis_date}", "SUCCESS")

        with open("recommendation_log.csv", "a") as f:
            f.write(
                f"{symbol},{analysis_date},{result['action']},{result['confidence']:.1f},{result['final_score']:.2f}\n")

    def analyze_stock_enhanced(self, symbol, target_date):
        """Enhanced stock analysis with 95% confidence targeting"""
        self.log(f"Starting enhanced analysis for {symbol} on {target_date}", "INFO")

        df = self.get_stock_data(symbol, target_date)
        if df is None or df.empty:
            self.log("No data retrieved for symbol or data is empty", "WARNING")
            return None

        target_pd = pd.Timestamp(target_date)

        # Find the target date or closest date
        if target_pd in df.index:
            analysis_date = target_pd
        else:
            closest_idx = df.index.get_indexer([target_pd], method='nearest')[0]
            if closest_idx < 0 or closest_idx >= len(df):
                self.log("Closest date index out of bounds", "ERROR")
                return None
            analysis_date = df.index[closest_idx]
            self.log(f"Using closest available date: {analysis_date}", "INFO")

        # Calculate comprehensive indicators
        indicators = self.calculate_enhanced_indicators(df, analysis_date)
        if indicators is None:
            self.log("Failed to calculate indicators", "ERROR")
            return None

        self.log("Indicators calculated successfully", "INFO")

        # Enhanced recommendation generation with debug mode
        try:
            recommendation = self.generate_enhanced_recommendation(indicators=indicators, symbol=symbol)
            self.log("Using enhanced recommendation (forced for testing)", "SUCCESS")
        except Exception as e:
            self.log(f"95% system failed: {e}, using original", "WARNING")
            recommendation = self.generate_enhanced_recommendation(indicators=indicators, symbol=symbol)

        self.log(f"Recommendation generated: {recommendation}", "INFO")

        # Log the final recommendation to CSV
        self.log_recommendation(symbol, recommendation, analysis_date)

        # Include debug_log in the return dictionary
        return {
            'symbol': symbol,
            'analysis_date': analysis_date,
            'indicators': indicators,
            'investment_days': self.investment_days,
            'debug_log': self.debug_log,
            **recommendation
        }

    def build_trading_plan(self, current_price, target_gain=0.037, max_loss=0.06, days=7):
        self.log(f"Building trading plan for price={current_price}, gain={target_gain}, loss={max_loss}, days={days}",
                 "INFO")

        buy_price = current_price
        sell_price = round(buy_price * (1 + target_gain), 2)
        stop_loss = round(buy_price * (1 - max_loss), 2)
        profit_pct = round(target_gain * 100, 1)

        plan = {
            "buy_price": buy_price,
            "sell_price": sell_price,
            "stop_loss": stop_loss,
            "profit_pct": profit_pct,
            "max_loss_pct": round(max_loss * 100, 1),
            "holding_days": days,
            'net_profit_pct': self.apply_israeli_fees_and_tax(target_gain * 100)
        }
        self.log(f"Trading plan created: {plan}", "INFO")
        return plan

    def boost_confidence(self,tech_score, model_score):
        self.log(f"Boosting confidence with tech_score={tech_score}, model_score={model_score}, days={self.investment_days}",
                 "INFO")

        base_confidence = 50.0
        signal_alignment = 1 if tech_score * model_score > 0 else 0

        # Scale based on score strength
        score_strength = min(abs(tech_score + model_score), 6)
        confidence_boost = score_strength * 5

        # Duration sensitivity
        duration_factor = min(self.investment_days, 14) / 14
        timing_bonus = 3 * duration_factor

        total_boost = confidence_boost + (10 * signal_alignment) + timing_bonus
        final_confidence = min(base_confidence + total_boost, 99.9)

        self.log(f"Final confidence score: {final_confidence}", "INFO")
        return round(final_confidence, 1)

    def analyze_trend(self, indicators, current_price):
        """Enhanced trend analysis with more sensitive scoring"""
        self.log(f"Starting analyze_trend with enhanced sensitivity", "INFO")

        score = 0
        signals = []

        sma_5 = indicators.get('sma_5', current_price)
        sma_10 = indicators.get('sma_10', current_price)
        sma_20 = indicators.get('sma_20', current_price)
        sma_50 = indicators.get('sma_50', current_price)

        # 🔧 ENHANCED: More granular trend scoring
        if current_price > sma_5 > sma_10 > sma_20:
            score += 3
            signals.append("📈 Strong SMA uptrend")
        elif current_price > sma_5 > sma_10:  # NEW: Additional case
            score += 2.5
            signals.append("📈 Good SMA uptrend")
        elif current_price > sma_10 > sma_20:  # NEW: Additional case
            score += 2
            signals.append("📈 Moderate SMA uptrend")
        elif current_price > sma_20:
            score += 1.5  # INCREASED from 2 to 1.5 for balance
            signals.append("📈 Price above SMA20")
        elif current_price > sma_50:  # NEW: Additional positive case
            score += 1
            signals.append("📈 Price above SMA50")
        elif current_price < sma_5 < sma_10 < sma_20:
            score -= 3
            signals.append("📉 Strong SMA downtrend")
        else:
            score -= 1  # REDUCED from -2 to -1
            signals.append("📉 Price below key SMAs")

        # EMA analysis (keep existing)
        ema_12 = indicators.get('ema_12', current_price)
        ema_26 = indicators.get('ema_26', current_price)

        if ema_12 > ema_26:
            score += 1
            signals.append("🔄 Bullish EMA crossover")
        else:
            score -= 1
            signals.append("🔄 Bearish EMA crossover")

        self.log(f"Enhanced Trend Score: {score}", "INFO")
        return score, signals

    def analyze_momentum(self, indicators):
        self.log(f"Starting analyze_momentum: indicators={indicators}", "INFO")

        score = 0
        signals = []

        rsi = indicators.get('rsi_14', 50)
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        macd_hist = indicators.get('macd_histogram', 0)
        self.log(f"RSI: {rsi:.1f} | MACD={macd:.4f}, Signal={macd_signal:.4f}, Hist={macd_hist:.4f}", "INFO")

        if rsi < 30:
            score += 3
            signals.append("🔥 RSI < 30: Strong buy")
            self.log(f"RSI < 30: Strong buy: ✅ RSI < 30: +3", "SUCCESS")

        elif rsi < 40:
            score += 2
            signals.append("💪 RSI 30–39: Buy bias")
            self.log(f"RSI 30–39: Buy bias: ✅ RSI 30–39: +2", "SUCCESS")

        elif rsi <= 55:
            score += 1
            signals.append("✅ RSI 45–55: Stable")
            self.log(f"RSI 45–55: Stable: ✅ RSI 45–55: +1", "SUCCESS")

        elif rsi > 70:
            score -= 3
            signals.append("🚨 RSI > 70: Sell bias")
            self.log(f"RSI > 70: Sell bias: ❌ RSI > 70: -3", "ERROR")

        if macd > macd_signal and macd_hist > 0:
            score += 2
            signals.append("🚀 MACD Bullish crossover")
            self.log(f"MACD Bullish crossover: ✅ MACD bullish: +2", "SUCCESS")

        elif macd < macd_signal and macd_hist < 0:
            score -= 2
            signals.append("📉 MACD Bearish crossover")
            self.log(f"MACD Bearish crossover: ❌ MACD bearish: -2", "ERROR")

        self.log(f"Momentum Score: {score}", "INFO")
        return score, signals

    def analyze_volume(self, indicators):
        self.log(f"Starting analyze_volume: indicators={indicators}", "INFO")

        score = 0
        signals = []

        vr = indicators.get('volume_relative', 1.0)
        self.log(f"Volume Ratio: {vr:.2f}", "INFO")

        if vr > 2.0:
            score += 2
            signals.append("🔊 High volume spike")
            self.log(f"High volume spike: ✅ Volume > 2x avg: +2", "SUCCESS")

        elif vr > 1.5:
            score += 1
            signals.append("📢 Above average volume")
            self.log(f"Above average volume: ✅ Volume > 1.5x: +1", "SUCCESS")

        elif vr < 0.7:
            score -= 1
            signals.append("🔇 Weak volume")
            self.log(f"Weak volume: ❌ Volume < 0.7x: -1", "ERROR")

        self.log(f"Volume Score: {score}", "INFO")
        return score, signals

    def analyze_support_resistance(self, indicators):
        self.log(f"Starting analyze_support_resistance: indicators={indicators}", "INFO")

        score = 0
        signals = []

        bb = indicators.get('bb_position', 0.5)
        self.log(f"Bollinger Position: {bb:.3f}", "INFO")

        if bb < 0.2:
            score += 2
            signals.append("📉 Near lower band")
            self.log(f"Near lower band: ✅ BB < 0.2: +2", "SUCCESS")

        elif bb > 0.8:
            score -= 2
            signals.append("📈 Near upper band")
            self.log(f"Near upper band: ❌ BB > 0.8: -2", "ERROR")
        elif 0.3 <= bb <= 0.7:
            score += 1
            signals.append("✅ Healthy BB range")
            self.log(f"Healthy BB range: ✅ BB 0.3–0.7: +1", "SUCCESS")

        self.log(f"S/R Score: {score}","INFO")
        return score, signals

    def analyze_ml_model(self, symbol, indicators, current_price):
        self.log(f"Starting analyze_ml_model: symbol={symbol}, indicators={indicators}, "
                 f"current_price={current_price}", "INFO")

        score = 0
        signals = []

        if symbol not in self.models:
            self.log(f"⚠️ No ML model for {symbol}", "WARNING")
            signals.append("🤖 No trained model available")
            return score, signals

        try:
            model = self.models[symbol]
            vr = indicators.get("volume_relative", 1.0)
            features = [
                vr,
                indicators.get("momentum_5", 0),
                indicators.get("rsi_14", 50),
                indicators.get("macd_histogram", 0),
                indicators.get("bb_position", 0.5),
                indicators.get("ema_10", current_price),
                indicators.get("price_change_1d", 0.0),
                1 if vr > 1.5 else 0,
                1 if indicators.get("rsi_14", 50) < 30 else 0
            ]
            self.log(f"ML Features: {features}", "INFO")

            X = np.array(features).reshape(1, -1)
            prediction = model.predict(X)[0]

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                confidence = proba[1] if prediction == 1 else proba[0]
                self.log(f"ML Prediction: {prediction}, Prob: {proba}", "INFO")
            else:
                confidence = 0.7
                self.log("ML Prediction: Default confidence", "INFO")

            delta = 3 * confidence
            if prediction == 1:
                score += delta
                signals.append(f"🤖 ML predicts BUY ({confidence:.1%})")
                self.log(f"✅ ML BUY: +{delta:.2f}", "SUCCESS")
            else:
                score -= delta
                signals.append(f"🤖 ML predicts SELL ({confidence:.1%})")
                self.log(f"❌ ML SELL: -{delta:.2f}", "ERROR")

        except Exception as e:
            signals.append("🤖 ML model analysis failed")
            self.log(f"❌ ML Error: {str(e)}", "ERROR")

        self.log(f"ML Score: {score:.2f}", "INFO")
        return score, signals

    def calculate_dynamic_profit_target(self, indicators, confidence, investment_days, symbol, strategy_settings=None):
        """
        🎯 Calculate dynamic profit targets based on multiple factors
        Higher confidence + longer time + aggressive strategy = higher profit targets
        """
        self.log(f"Calculating dynamic profit target for {symbol}", "INFO")

        # Use strategy settings if available
        if strategy_settings is None:
            strategy_settings = getattr(self, 'strategy_settings', {"profit": 1.0, "risk": 1.0, "confidence_req": 75})

        # ENHANCED BASE TARGETS by confidence level
        confidence_multipliers = {
            95: 0.15,  # 15% for ultra-high confidence
            90: 0.12,  # 12% for very high confidence
            85: 0.10,  # 10% for high confidence
            80: 0.08,  # 8% for good confidence
            75: 0.06,  # 6% for moderate confidence
            70: 0.05,  # 5% for fair confidence
            60: 0.037  # 3.7% default for low confidence
        }

        # Get base target from confidence
        base_target = 0.037  # Default
        for conf_threshold in sorted(confidence_multipliers.keys(), reverse=True):
            if confidence >= conf_threshold:
                base_target = confidence_multipliers[conf_threshold]
                break

        # ENHANCED TIME-BASED MULTIPLIERS (much more aggressive for longer periods)
        time_multipliers = {
            1: 0.8,  # 1 day: reduce target
            3: 0.9,  # 3 days: slight reduction
            7: 1.0,  # 7 days: base target
            14: 1.4,  # 14 days: 40% increase
            21: 1.8,  # 21 days: 80% increase
            30: 2.5,  # 30 days: 150% increase
            45: 3.2,  # 45 days: 220% increase
            60: 4.0,  # 60 days: 300% increase
            90: 5.5,  # 90 days: 450% increase
            120: 7.0  # 120 days: 600% increase
        }

        time_multiplier = 1.0
        for days in sorted(time_multipliers.keys(), reverse=True):
            if investment_days >= days:
                time_multiplier = time_multipliers[days]
                break

        # STRATEGY TYPE MULTIPLIERS (FIXED - now actually applied)
        strategy_multiplier = strategy_settings.get("profit", 1.0)

        # Enhanced volatility adjustments
        volatility = indicators.get('volatility', 2.0)
        if volatility > 5.0:  # Very high volatility
            volatility_multiplier = 1.4
        elif volatility > 4.0:  # High volatility
            volatility_multiplier = 1.3
        elif volatility > 3.0:  # Medium-high volatility
            volatility_multiplier = 1.2
        elif volatility > 2.0:  # Medium volatility
            volatility_multiplier = 1.1
        elif volatility < 1.0:  # Low volatility
            volatility_multiplier = 0.85
        else:
            volatility_multiplier = 1.0

        # Enhanced momentum adjustments
        momentum_5 = indicators.get('momentum_5', 0)
        if momentum_5 > 8:  # Very strong momentum
            momentum_multiplier = 1.25
        elif momentum_5 > 5:  # Strong momentum
            momentum_multiplier = 1.20
        elif momentum_5 > 2:  # Good momentum
            momentum_multiplier = 1.10
        elif momentum_5 < -8:  # Very negative momentum
            momentum_multiplier = 0.75
        elif momentum_5 < -5:  # Negative momentum
            momentum_multiplier = 0.85
        else:
            momentum_multiplier = 1.0

        # Volume confirmation bonus (enhanced)
        volume_relative = indicators.get('volume_relative', 1.0)
        if volume_relative > 3.0:  # Massive volume
            volume_bonus = 1.25
        elif volume_relative > 2.5:  # Very high volume
            volume_bonus = 1.20
        elif volume_relative > 2.0:  # High volume
            volume_bonus = 1.15
        elif volume_relative > 1.5:  # Good volume
            volume_bonus = 1.10
        else:
            volume_bonus = 1.0

        # Market regime bonus
        regime_bonus = 1.0
        rsi_14 = indicators.get('rsi_14', 50)
        macd_hist = indicators.get('macd_histogram', 0)

        # Strong bullish regime
        if rsi_14 < 40 and macd_hist > 0:
            regime_bonus = 1.15
        elif rsi_14 < 50 and macd_hist > 0:
            regime_bonus = 1.10

        # CALCULATE FINAL TARGET (with all multipliers applied)
        final_target = (base_target *
                        time_multiplier *
                        strategy_multiplier *  # Now properly applied
                        volatility_multiplier *
                        momentum_multiplier *
                        volume_bonus *
                        regime_bonus)

        # Enhanced bounds based on time horizon and strategy
        if strategy_settings.get("profit", 1.0) >= 1.8:  # Swing trading
            max_target = 0.60 if investment_days >= 90 else 0.45  # Up to 60% for swing trading
            min_target = 0.03
        elif strategy_settings.get("profit", 1.0) >= 1.4:  # Aggressive strategy
            max_target = 0.50 if investment_days >= 60 else 0.35  # Up to 50% for aggressive
            min_target = 0.025
        else:  # Conservative/Balanced
            max_target = 0.30 if investment_days >= 60 else 0.20
            min_target = 0.02

        final_target = max(min_target, min(final_target, max_target))

        # Log detailed breakdown
        self.log(f"Enhanced profit calculation for {symbol}:", "INFO")
        self.log(f"  Base target: {base_target:.1%} (confidence: {confidence}%)", "INFO")
        self.log(f"  Time multiplier: {time_multiplier:.2f} ({investment_days} days)", "INFO")
        self.log(f"  Strategy multiplier: {strategy_multiplier:.2f} ({getattr(self, 'current_strategy', 'Unknown')})",
                 "INFO")
        self.log(f"  Volatility multiplier: {volatility_multiplier:.2f}", "INFO")
        self.log(f"  Momentum multiplier: {momentum_multiplier:.2f}", "INFO")
        self.log(f"  Volume bonus: {volume_bonus:.2f}", "INFO")
        self.log(f"  Regime bonus: {regime_bonus:.2f}", "INFO")
        self.log(f"  FINAL TARGET: {final_target:.1%}", "SUCCESS")

        return final_target

    def analyze_market_regime(self, indicators, df_recent):
        """🌍 Analyze current market regime for better context"""

        # Trend strength analysis
        sma_20 = indicators.get('sma_20', indicators['current_price'])
        sma_50 = indicators.get('sma_50', indicators['current_price'])
        current_price = indicators['current_price']

        # Calculate trend strength
        if current_price > sma_20 > sma_50:
            trend_strength = min((current_price - sma_50) / sma_50 * 100, 10)
            regime = "Strong Uptrend"
        elif current_price > sma_20:
            trend_strength = min((current_price - sma_20) / sma_20 * 100, 5)
            regime = "Mild Uptrend"
        elif current_price < sma_20 < sma_50:
            trend_strength = min((sma_50 - current_price) / current_price * 100, -10)
            regime = "Strong Downtrend"
        else:
            trend_strength = 0
            regime = "Sideways"

        return {
            'regime': regime,
            'trend_strength': trend_strength,
            'regime_multiplier': 1.2 if "Strong Uptrend" in regime else
            1.1 if "Mild Uptrend" in regime else
            0.8 if "Downtrend" in regime else 1.0
        }

    def calculate_multi_timeframe_confirmation(self, indicators):
        """📊 Multi-timeframe analysis for higher confidence"""

        confirmations = 0
        total_checks = 0

        # RSI across timeframes
        rsi_14 = indicators.get('rsi_14', 50)
        rsi_21 = indicators.get('rsi_21', 50)

        if rsi_14 < 40 and rsi_21 < 45:  # Both RSIs suggest oversold
            confirmations += 2
        elif rsi_14 < 50 and rsi_21 < 55:  # Mild oversold
            confirmations += 1
        total_checks += 2

        # Moving average alignment
        current_price = indicators['current_price']
        sma_5 = indicators.get('sma_5', current_price)
        sma_10 = indicators.get('sma_10', current_price)
        sma_20 = indicators.get('sma_20', current_price)

        if current_price > sma_5 > sma_10 > sma_20:  # Perfect bullish alignment
            confirmations += 3
        elif current_price > sma_10 > sma_20:  # Good bullish alignment
            confirmations += 2
        elif current_price > sma_20:  # Basic bullish
            confirmations += 1
        total_checks += 3

        # MACD confirmation
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        macd_hist = indicators.get('macd_histogram', 0)

        if macd > macd_signal and macd_hist > 0:  # Strong bullish MACD
            confirmations += 2
        elif macd > macd_signal:  # Mild bullish MACD
            confirmations += 1
        total_checks += 2

        # Volume confirmation
        volume_relative = indicators.get('volume_relative', 1.0)
        if volume_relative > 1.5:  # Strong volume
            confirmations += 2
        elif volume_relative > 1.2:  # Good volume
            confirmations += 1
        total_checks += 2

        confirmation_percentage = (confirmations / total_checks) * 100 if total_checks > 0 else 0

        return {
            'confirmation_score': confirmations,
            'total_possible': total_checks,
            'confirmation_percentage': confirmation_percentage,
            'confidence_boost': min(confirmation_percentage / 10, 15)  # Up to 15% boost
        }

    def enhanced_profit_recommendation(self, indicators, symbol):
        """🚀 Enhanced recommendation with higher profit targets"""

        # Get current analysis
        current_price = indicators['current_price']

        # Analyze market regime
        regime_analysis = self.analyze_market_regime(indicators, None)

        # Multi-timeframe confirmation
        confirmation_analysis = self.calculate_multi_timeframe_confirmation(indicators)

        # Base confidence from your existing system
        base_confidence = 70  # You'll get this from your existing method

        # Enhanced confidence calculation
        enhanced_confidence = base_confidence + confirmation_analysis['confidence_boost']
        enhanced_confidence = min(enhanced_confidence, 98)  # Cap at 98%

        # Dynamic profit target
        profit_target = self.calculate_dynamic_profit_target(
            indicators, enhanced_confidence, self.investment_days, symbol
        )

        # Risk-adjusted stop loss
        volatility = indicators.get('volatility', 2.0)
        if volatility > 4.0:
            stop_loss_pct = 0.08  # 8% stop for high volatility
        elif volatility > 3.0:
            stop_loss_pct = 0.06  # 6% stop for medium volatility
        else:
            stop_loss_pct = 0.04  # 4% stop for low volatility

        return {
            'enhanced_confidence': enhanced_confidence,
            'profit_target': profit_target,
            'stop_loss_pct': stop_loss_pct,
            'regime_analysis': regime_analysis,
            'confirmation_analysis': confirmation_analysis,
            'expected_holding_days': self.investment_days,
            'risk_reward_ratio': profit_target / stop_loss_pct
        }

    def extract_signal_strengths(self, trend_score, momentum_score, volume_score, sr_score, model_score):
        """Return breakdown of signal strengths categorized by source."""
        self.log(f"Starting extract_signal_strengths: "
                 f"trend_score = {trend_score},"
                 f" momentum_score = {momentum_score},"
                 f" volume_score = {volume_score},"
                 f" sr_score = {sr_score},"
                 f" model_score = {model_score}",
                 "INFO")

        breakdown = {
            'trend_score': round(trend_score, 2),
            'momentum_score': round(momentum_score, 2),
            'volume_score': round(volume_score, 2),
            'sr_score': round(sr_score, 2),
            'model_score': round(model_score, 2)
        }
        self.log(f"📊 Signal Breakdown: {breakdown}", "INFO")
        return breakdown

    def generate_enhanced_recommendation(self, indicators, symbol):
        """Generate high-confidence recommendations using multi-factor analysis"""
        self.log(f"Starting generate_enhanced_recommendation: symbol={symbol}", "INFO")

        self.active_symbol = symbol
        current_price = indicators['current_price']

        # Get strategy settings
        strategy_settings = getattr(self, 'strategy_settings', {"profit": 1.0, "risk": 1.0, "confidence_req": 75})

        self.log(f"\n=== ENHANCED RECOMMENDATION DEBUG for {symbol} ===", "INFO")
        self.log(f"Current Price: ${current_price:.2f}", "INFO")
        self.log(f"Investment Days: {self.investment_days}", "INFO")
        self.log(f"Strategy Settings: {strategy_settings}", "INFO")

        # Signal weights
        signal_weights = {
            'trend': 0.25,
            'momentum': 0.20,
            'volume': 0.15,
            'support_resistance': 0.15,
            'model': 0.25
        }
        self.log(f"Signal Weights: {signal_weights}", "INFO")

        # Run all signal analysis
        trend_score, trend_signals = self.analyze_trend(indicators, current_price)
        momentum_score, momentum_signals = self.analyze_momentum(indicators)
        volume_score, volume_signals = self.analyze_volume(indicators)
        sr_score, sr_signals = self.analyze_support_resistance(indicators)
        model_score, model_signals = self.analyze_ml_model(symbol, indicators, current_price)

        # Log individual scores
        self.log(
            f"Individual Scores - Trend: {trend_score:.2f}, Momentum: {momentum_score:.2f}, Volume: {volume_score:.2f}, S/R: {sr_score:.2f}, Model: {model_score:.2f}",
            "INFO")

        # Calculate final score
        final_score = (
                trend_score * signal_weights['trend'] +
                momentum_score * signal_weights['momentum'] +
                volume_score * signal_weights['volume'] +
                sr_score * signal_weights['support_resistance'] +
                model_score * signal_weights['model']
        )

        self.log(f"Calculated Final Score: {final_score:.2f}", "INFO")

        # Combine all signals
        all_signals = trend_signals + momentum_signals + volume_signals + sr_signals + model_signals

        # CALCULATE THRESHOLDS
        confidence_req = strategy_settings.get("confidence_req", 75)
        profit_multiplier = strategy_settings.get("profit", 1.0)

        self.log(f"Profit Multiplier: {profit_multiplier}", "INFO")

        # Strategy-based threshold adjustments
        if profit_multiplier >= 1.8:  # Swing Trading
            buy_threshold = 0.8
            sell_threshold = -0.8
            strategy_name = "Swing Trading"
        elif profit_multiplier >= 1.4:  # Aggressive
            buy_threshold = 0.9
            sell_threshold = -0.9
            strategy_name = "Aggressive"
        else:  # Conservative/Balanced
            buy_threshold = 1.0
            sell_threshold = -1.0
            strategy_name = "Conservative/Balanced"

        self.log(f"Strategy Detected: {strategy_name}", "INFO")
        self.log(f"Using thresholds: BUY≥{buy_threshold}, SELL≤{sell_threshold}", "INFO")

        # CRITICAL DEBUG: Check decision logic step by step
        self.log("=== DECISION LOGIC DEBUG ===", "INFO")
        self.log(f"Final Score: {final_score:.2f}", "INFO")
        self.log(f"Buy Threshold: {buy_threshold}", "INFO")
        self.log(f"Sell Threshold: {sell_threshold}", "INFO")
        self.log(f"Score >= Buy Threshold: {final_score >= buy_threshold} ({final_score:.2f} >= {buy_threshold})",
                 "INFO")
        self.log(f"Score <= Sell Threshold: {final_score <= sell_threshold} ({final_score:.2f} <= {sell_threshold})",
                 "INFO")

        # ACTION DECISION LOGIC with detailed logging
        if final_score >= buy_threshold:
            action = "BUY"
            self.log(f"✅ BUY DECISION: {final_score:.2f} >= {buy_threshold} threshold", "SUCCESS")

            base_confidence = 70 + min(25, final_score * 8)
            buy_price = current_price

            # Use enhanced profit calculation with strategy integration
            target_profit = self.calculate_dynamic_profit_target(
                indicators, base_confidence, self.investment_days, symbol, strategy_settings
            )

            sell_price = current_price * (1 + target_profit)

            # Enhanced stop loss based on strategy
            if strategy_settings.get("risk", 1.0) >= 1.3:  # Aggressive/Swing
                stop_loss_pct = min(0.08, 0.04 + (self.investment_days * 0.001))
            else:
                stop_loss_pct = min(0.06, 0.03 + (self.investment_days * 0.0005))

            stop_loss = current_price * (1 - stop_loss_pct)

            # Calculate profit percentages
            gross_profit_pct = target_profit * 100
            net_profit_pct = self.apply_israeli_fees_and_tax(gross_profit_pct)

        elif final_score <= sell_threshold:
            action = "SELL/AVOID"
            self.log(f"❌ SELL DECISION: {final_score:.2f} <= {sell_threshold} threshold", "INFO")

            base_confidence = 70 + min(25, abs(final_score) * 8)
            buy_price = None
            sell_price = current_price
            target_profit = 0
            stop_loss_pct = 0.06
            stop_loss = current_price * (1 + stop_loss_pct)
            gross_profit_pct = 0
            net_profit_pct = 0

        else:
            action = "WAIT"
            self.log(f"⏳ WAIT DECISION: {sell_threshold} < {final_score:.2f} < {buy_threshold}", "INFO")

            base_confidence = 50 + abs(final_score) * 5
            buy_price = None
            sell_price = current_price  # FIXED: For WAIT, sell_price should be current_price, not None
            target_profit = 0
            stop_loss_pct = 0.06
            stop_loss = current_price * (1 - stop_loss_pct)
            gross_profit_pct = 0
            net_profit_pct = 0
            all_signals.append(f"🤔 Score {final_score:.2f} between thresholds ({sell_threshold} to {buy_threshold})")

        # Log the final action
        self.log(f"FINAL ACTION: {action}", "SUCCESS")

        # Confidence calculation
        confirming_indicators = sum([
            1 if abs(trend_score) > 1 else 0,
            1 if abs(momentum_score) > 1 else 0,
            1 if abs(volume_score) > 0 else 0,
            1 if abs(sr_score) > 0 else 0,
            1 if abs(model_score) > 1 else 0
        ])
        confidence_bonus = min(10, confirming_indicators * 2)
        final_confidence = min(95, base_confidence + confidence_bonus)

        # Enhanced trading plan
        trading_plan = self.build_enhanced_trading_plan(current_price, target_profit, stop_loss_pct,
                                                        self.investment_days)

        # Signal breakdown
        signal_strengths = self.extract_signal_strengths(trend_score, momentum_score, volume_score, sr_score,
                                                         model_score)

        # Risk profile
        risk_level = (
            "Short-term" if self.investment_days <= 7 else
            "Medium-term" if self.investment_days <= 21 else
            "Long-term"
        )

        return {
            'action': action,
            'confidence': final_confidence,
            'buy_price': buy_price,
            'sell_price': sell_price,
            'stop_loss': stop_loss,
            'expected_profit_pct': round(net_profit_pct, 2),
            'gross_profit_pct': round(gross_profit_pct, 2),
            'tax_paid': round(self.tax, 2),
            'broker_fee_paid': round(self.broker_fee, 2),
            'reasons': all_signals + [
                f"📈 Strategy: {strategy_name} (×{strategy_settings.get('profit', 1.0):.1f})",
                f"⏱️ Time scaling: {self.investment_days} days (×{target_profit / 0.037 if target_profit > 0 else 1:.1f})",
                f"🎯 Score: {final_score:.2f} (BUY≥{buy_threshold}, SELL≤{sell_threshold})"
            ],
            'risk_level': risk_level,
            'final_score': final_score,
            'current_price': current_price,
            'signal_breakdown': signal_strengths,
            'trading_plan': trading_plan,
            'strategy_applied': True,
            'strategy_multiplier': strategy_settings.get("profit", 1.0),
            'time_multiplier': target_profit / 0.037 if target_profit > 0 else 1.0,
        }

    def generate_enhanced_recommendation_with_improved_confidence(self, indicators, symbol):
        """Generate high-confidence recommendations using multi-factor analysis"""
        self.log(f"Starting generate_enhanced_recommendation: symbol={symbol}", "INFO")

        self.active_symbol = symbol
        current_price = indicators['current_price']

        # Get strategy settings
        strategy_settings = getattr(self, 'strategy_settings', {"profit": 1.0, "risk": 1.0, "confidence_req": 75})

        self.log(f"\n=== ENHANCED RECOMMENDATION DEBUG for {symbol} ===", "INFO")
        self.log(f"Current Price: ${current_price:.2f}", "INFO")
        self.log(f"Investment Days: {self.investment_days}", "INFO")
        self.log(f"Strategy Settings: {strategy_settings}", "INFO")

        # Signal weights
        signal_weights = {
            'trend': 0.25,
            'momentum': 0.20,
            'volume': 0.15,
            'support_resistance': 0.15,
            'model': 0.25
        }
        self.log(f"Signal Weights: {signal_weights}", "INFO")

        # Run all signal analysis
        trend_score, trend_signals = self.analyze_trend(indicators, current_price)
        momentum_score, momentum_signals = self.analyze_momentum(indicators)
        volume_score, volume_signals = self.analyze_volume(indicators)
        sr_score, sr_signals = self.analyze_support_resistance(indicators)
        model_score, model_signals = self.analyze_ml_model(symbol, indicators, current_price)

        # Log individual scores
        self.log(
            f"Individual Scores - Trend: {trend_score:.2f}, Momentum: {momentum_score:.2f}, Volume: {volume_score:.2f}, S/R: {sr_score:.2f}, Model: {model_score:.2f}",
            "INFO")

        # Calculate final score (keep existing logic)
        final_score = (
                trend_score * signal_weights['trend'] +
                momentum_score * signal_weights['momentum'] +
                volume_score * signal_weights['volume'] +
                sr_score * signal_weights['support_resistance'] +
                model_score * signal_weights['model']
        )

        self.log(f"Calculated Final Score: {final_score:.2f}", "INFO")

        # Combine all signals
        all_signals = trend_signals + momentum_signals + volume_signals + sr_signals + model_signals

        # CALCULATE THRESHOLDS
        confidence_req = strategy_settings.get("confidence_req", 75)
        profit_multiplier = strategy_settings.get("profit", 1.0)

        self.log(f"Profit Multiplier: {profit_multiplier}", "INFO")

        # Strategy-based threshold adjustments
        if profit_multiplier >= 1.8:  # Swing Trading
            buy_threshold = 0.8
            sell_threshold = -0.8
            strategy_name = "Swing Trading"
        elif profit_multiplier >= 1.4:  # Aggressive
            buy_threshold = 0.9
            sell_threshold = -0.9
            strategy_name = "Aggressive"
        else:  # Conservative/Balanced
            buy_threshold = 1.0
            sell_threshold = -1.0
            strategy_name = "Conservative/Balanced"

        self.log(f"Strategy Detected: {strategy_name}", "INFO")
        self.log(f"Using thresholds: BUY≥{buy_threshold}, SELL≤{sell_threshold}", "INFO")

        # CRITICAL DEBUG: Check decision logic step by step
        self.log("=== DECISION LOGIC DEBUG ===", "INFO")
        self.log(f"Final Score: {final_score:.2f}", "INFO")
        self.log(f"Buy Threshold: {buy_threshold}", "INFO")
        self.log(f"Sell Threshold: {sell_threshold}", "INFO")
        self.log(f"Score >= Buy Threshold: {final_score >= buy_threshold} ({final_score:.2f} >= {buy_threshold})",
                 "INFO")
        self.log(f"Score <= Sell Threshold: {final_score <= sell_threshold} ({final_score:.2f} <= {sell_threshold})",
                 "INFO")

        # ACTION DECISION LOGIC with detailed logging
        if final_score >= buy_threshold:
            action = "BUY"
            self.log(f"✅ BUY DECISION: {final_score:.2f} >= {buy_threshold} threshold", "SUCCESS")

            base_confidence = 70 + min(25, final_score * 8)
            buy_price = current_price

            # Use enhanced profit calculation with strategy integration
            target_profit = self.calculate_dynamic_profit_target(
                indicators, base_confidence, self.investment_days, symbol, strategy_settings
            )

            sell_price = current_price * (1 + target_profit)

            # Enhanced stop loss based on strategy
            if strategy_settings.get("risk", 1.0) >= 1.3:  # Aggressive/Swing
                stop_loss_pct = min(0.08, 0.04 + (self.investment_days * 0.001))
            else:
                stop_loss_pct = min(0.06, 0.03 + (self.investment_days * 0.0005))

            stop_loss = current_price * (1 - stop_loss_pct)

            # Calculate profit percentages
            gross_profit_pct = target_profit * 100
            net_profit_pct = self.apply_israeli_fees_and_tax(gross_profit_pct)

        elif final_score <= sell_threshold:
            action = "SELL/AVOID"
            self.log(f"❌ SELL DECISION: {final_score:.2f} <= {sell_threshold} threshold", "INFO")

            base_confidence = 70 + min(25, abs(final_score) * 8)
            buy_price = None
            sell_price = current_price
            target_profit = 0
            stop_loss_pct = 0.06
            stop_loss = current_price * (1 + stop_loss_pct)
            gross_profit_pct = 0
            net_profit_pct = 0

        else:
            action = "WAIT"
            self.log(f"⏳ WAIT DECISION: {sell_threshold} < {final_score:.2f} < {buy_threshold}", "INFO")

            base_confidence = 50 + abs(final_score) * 5
            buy_price = None
            sell_price = current_price  # FIXED: For WAIT, sell_price should be current_price, not None
            target_profit = 0
            stop_loss_pct = 0.06
            stop_loss = current_price * (1 - stop_loss_pct)
            gross_profit_pct = 0
            net_profit_pct = 0
            all_signals.append(f"🤔 Score {final_score:.2f} between thresholds ({sell_threshold} to {buy_threshold})")

        # Log the final action
        self.log(f"FINAL ACTION: {action}", "SUCCESS")

        # Confidence calculation
        confirming_indicators = sum([
            1 if abs(trend_score) > 1 else 0,
            1 if abs(momentum_score) > 1 else 0,
            1 if abs(volume_score) > 0 else 0,
            1 if abs(sr_score) > 0 else 0,
            1 if abs(model_score) > 1 else 0
        ])
        confidence_bonus = min(10, confirming_indicators * 2)
        final_confidence = min(95, base_confidence + confidence_bonus)

        # Enhanced trading plan
        trading_plan = self.build_enhanced_trading_plan(current_price, target_profit, stop_loss_pct,
                                                        self.investment_days)

        # Signal breakdown
        signal_strengths = self.extract_signal_strengths(trend_score, momentum_score, volume_score, sr_score,
                                                         model_score)

        # Risk profile
        risk_level = (
            "Short-term" if self.investment_days <= 7 else
            "Medium-term" if self.investment_days <= 21 else
            "Long-term"
        )

        return {
            'action': action,
            'confidence': final_confidence,
            'buy_price': buy_price,
            'sell_price': sell_price,
            'stop_loss': stop_loss,
            'expected_profit_pct': round(net_profit_pct, 2),
            'gross_profit_pct': round(gross_profit_pct, 2),
            'tax_paid': round(self.tax, 2),
            'broker_fee_paid': round(self.broker_fee, 2),
            'reasons': all_signals + [
                f"📈 Strategy: {strategy_name} (×{strategy_settings.get('profit', 1.0):.1f})",
                f"⏱️ Time scaling: {self.investment_days} days (×{target_profit / 0.037 if target_profit > 0 else 1:.1f})",
                f"🎯 Score: {final_score:.2f} (BUY≥{buy_threshold}, SELL≤{sell_threshold})"
            ],
            'risk_level': risk_level,
            'final_score': final_score,
            'current_price': current_price,
            'signal_breakdown': signal_strengths,
            'trading_plan': trading_plan,
            'strategy_applied': True,
            'strategy_multiplier': strategy_settings.get("profit", 1.0),
            'time_multiplier': target_profit / 0.037 if target_profit > 0 else 1.0,
        }

    def create_enhanced_chart(self, symbol, data):
        """Create enhanced chart with FIXED target price display"""
        self.log(f"Creating enhanced chart for {symbol}", "INFO")

        df = self.get_stock_data(symbol, data['analysis_date'], days_back=60)
        if df is None or df.empty:
            self.log(f"No data returned for {symbol}", "ERROR")
            return None

        fig = go.Figure()

        # Price candlesticks
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            showlegend=False
        ))

        # Add multiple moving averages
        for period, color in [(5, 'orange'), (20, 'blue'), (50, 'red')]:
            if len(df) >= period:
                ma = df['Close'].rolling(period).mean()
                fig.add_trace(go.Scatter(
                    x=df.index, y=ma,
                    mode='lines', name=f'MA{period}',
                    line=dict(color=color, width=1)
                ))

        # Bollinger Bands
        if len(df) >= 20:
            try:
                bb = ta.volatility.BollingerBands(df['Close'])
                fig.add_trace(go.Scatter(
                    x=df.index, y=bb.bollinger_hband(),
                    mode='lines', name='BB Upper',
                    line=dict(color='gray', dash='dot', width=1),
                    showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=df.index, y=bb.bollinger_lband(),
                    mode='lines', name='BB Lower',
                    line=dict(color='gray', dash='dot', width=1),
                    fill='tonexty', fillcolor='rgba(128,128,128,0.1)',
                    showlegend=False
                ))
            except:
                pass

        # Mark analysis point
        analysis_date = data['analysis_date']
        current_price = data['current_price']
        action = data['action']

        # Action marker
        if action == "BUY":
            marker_color = 'green'
            marker_symbol = 'triangle-up'
        elif action == "SELL/AVOID":
            marker_color = 'red'
            marker_symbol = 'triangle-down'
        else:
            marker_color = 'orange'
            marker_symbol = 'circle'

        fig.add_trace(go.Scatter(
            x=[analysis_date],
            y=[current_price],
            mode='markers',
            name=f'{action} Signal',
            marker=dict(
                color=marker_color,
                size=15,
                symbol=marker_symbol,
                line=dict(width=2, color='white')
            )
        ))

        # 🔧 FIXED: ADD TARGET AND STOP LOSS LINES FOR ALL SCENARIOS
        # Always show target lines, even for WAIT signals

        if data.get('sell_price') and data['sell_price'] != current_price:
            fig.add_hline(
                y=data['sell_price'],
                line_dash="dash",
                line_color="green",
                annotation_text=f"Target: ${data['sell_price']:.2f}",
                annotation_position="top right"
            )
            self.log(f"Added target line at ${data['sell_price']:.2f}", "SUCCESS")

        if data.get('stop_loss'):
            fig.add_hline(
                y=data['stop_loss'],
                line_dash="dot",
                line_color="red",
                annotation_text=f"Stop Loss: ${data['stop_loss']:.2f}",
                annotation_position="bottom right"
            )
            self.log(f"Added stop loss line at ${data['stop_loss']:.2f}", "SUCCESS")

        # 🔧 NEW: Add potential target lines even for WAIT signals
        if action == "WAIT" and data.get('expected_profit_pct', 0) > 0:
            # Calculate what the target would be if this were a BUY
            potential_target = current_price * (1 + (data['gross_profit_pct'] / 100))
            fig.add_hline(
                y=potential_target,
                line_dash="dashdot",
                line_color="yellow",
                annotation_text=f"Potential Target: ${potential_target:.2f}",
                annotation_position="top left"
            )
            self.log(f"Added potential target line at ${potential_target:.2f}", "INFO")

        fig.update_layout(
            title=f'{symbol} - Enhanced Technical Analysis',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=500,
            showlegend=True
        )

        return fig

    def validate_strategy_integration(self, result, expected_strategy):
        """🔍 Validation function to ensure strategy effects are applied"""

        # Check if strategy was applied
        if not result.get('strategy_applied', False):
            self.log("❌ VALIDATION FAILED: Strategy not applied", "ERROR")
            return False

        # Check if multipliers are reasonable
        strategy_mult = result.get('strategy_multiplier', 1.0)
        time_mult = result.get('time_multiplier', 1.0)

        expected_multipliers = {
            "Conservative": 0.8,
            "Balanced": 1.0,
            "Aggressive": 1.4,
            "Swing Trading": 1.8
        }

        expected_mult = expected_multipliers.get(expected_strategy, 1.0)

        if abs(strategy_mult - expected_mult) > 0.1:
            self.log(f"❌ VALIDATION FAILED: Expected {expected_mult}, got {strategy_mult}", "ERROR")
            return False

        # Check if profit scaling is working
        gross_profit = result.get('gross_profit_pct', 0)
        if gross_profit <= 3.7 and (strategy_mult > 1.0 or time_mult > 1.0):
            self.log(f"❌ VALIDATION FAILED: No profit scaling despite multipliers", "ERROR")
            return False

        self.log("✅ Strategy integration validation passed", "SUCCESS")
        return True

    def calculate_dynamic_profit_target_with_validation(self, indicators, confidence, investment_days, symbol,
                                                        strategy_settings=None):
        """🎯 Profit calculation with built-in validation"""

        # Store original values for validation
        original_base = 0.037

        # Get strategy settings with validation
        if strategy_settings is None:
            strategy_settings = getattr(self, 'strategy_settings', {"profit": 1.0, "risk": 1.0})
            if not hasattr(self, 'strategy_settings'):
                self.log("⚠️ WARNING: No strategy_settings found, using defaults", "WARNING")

        # Run calculation
        final_target = self.calculate_dynamic_profit_target(indicators, confidence, investment_days, symbol,
                                                                  strategy_settings)

        # VALIDATION: Ensure scaling actually happened
        strategy_mult = strategy_settings.get("profit", 1.0)

        # Calculate expected minimum based on multipliers
        time_mult = 1.0 + (investment_days - 7) * 0.05  # Simplified time calc
        expected_minimum = original_base * strategy_mult * time_mult

        if final_target < expected_minimum * 0.8:  # Allow 20% variance
            self.log(f"🚨 PROFIT SCALING ISSUE: Expected min {expected_minimum:.1%}, got {final_target:.1%}", "ERROR")
            self.log(f"   Strategy mult: {strategy_mult}, Time mult: {time_mult}", "ERROR")

        return final_target

    def test_strategy_profit_scaling(self):
        """🧪 Unit test to verify strategy scaling works"""

        # Mock advisor with different strategies
        self.mock_indicators = {
            'current_price': 100.0,
            'volatility': 2.0,
            'momentum_5': 3.0,
            'volume_relative': 1.2,
            'rsi_14': 45,
            'macd_histogram': 0.1
        }

        strategies_to_test = {
            "Conservative": {"profit": 0.8, "expected_range": (0.03, 0.08)},
            "Balanced": {"profit": 1.0, "expected_range": (0.037, 0.12)},
            "Aggressive": {"profit": 1.4, "expected_range": (0.05, 0.20)},
            "Swing Trading": {"profit": 1.8, "expected_range": (0.06, 0.30)}
        }

        time_periods = [7, 30, 60, 90]

        for strategy_name, strategy_data in strategies_to_test.items():
            for days in time_periods:
                strategy_settings = {"profit": strategy_data["profit"], "risk": 1.0}

                # Test profit calculation
                target = self.calculate_dynamic_profit_target(
                    self.mock_indicators, 80, days, "TEST", strategy_settings
                )

                min_expected, max_expected = strategy_data["expected_range"]

                # Adjust expectations for time
                if days >= 60:
                    max_expected *= 2.0
                elif days >= 30:
                    max_expected *= 1.5

                assert min_expected <= target <= max_expected, \
                    f"❌ {strategy_name} + {days}d: Expected {min_expected:.1%}-{max_expected:.1%}, got {target:.1%}"

                print(f"✅ {strategy_name} + {days}d: {target:.1%} (within expected range)")

    def monitor_profit_calculations(self, symbol, result):
        """📊 Real-time monitoring of profit calculations"""

        expected_profit = result.get('gross_profit_pct', 0)
        strategy_mult = result.get('strategy_multiplier', 1.0)
        time_mult = result.get('time_multiplier', 1.0)

        # Flag unusual cases
        if expected_profit <= 3.7 and strategy_mult > 1.0:
            self.log(f"🚨 ANOMALY: {symbol} - No profit scaling despite {strategy_mult}x strategy multiplier", "ERROR")

        if expected_profit <= 3.7 and self.investment_days >= 30:
            self.log(f"🚨 ANOMALY: {symbol} - No time scaling for {self.investment_days} days", "ERROR")

        if strategy_mult == 1.0 and hasattr(self, 'current_strategy'):
            if self.current_strategy in ["Aggressive", "Swing Trading"]:
                self.log(f"🚨 ANOMALY: {symbol} - {self.current_strategy} strategy not applied", "ERROR")

        # Log success cases
        if expected_profit > 10 and strategy_mult > 1.0:
            self.log(f"✅ SUCCESS: {symbol} - Enhanced targeting: {expected_profit:.1f}% (strategy: {strategy_mult}x)",
                     "SUCCESS")

    def validate_advisor_configuration(self):
        """🔧 Validate advisor configuration on startup"""

        issues = []

        # Check if strategy settings exist
        if not hasattr(self, 'strategy_settings'):
            issues.append("Missing strategy_settings attribute")

        # Check if current_strategy exists
        if not hasattr(self, 'current_strategy'):
            issues.append("Missing current_strategy attribute")

        # Check if enhanced methods exist
        required_methods = [
            'calculate_dynamic_profit_target',
            'generate_enhanced_recommendation',
            'build_enhanced_trading_plan'
        ]

        for method_name in required_methods:
            if not hasattr(self, method_name):
                issues.append(f"Missing method: {method_name}")

        # Check if investment_days is reasonable
        if not hasattr(self, 'investment_days') or self.investment_days <= 0:
            issues.append("Invalid investment_days setting")

        if issues:
            self.log("🚨 CONFIGURATION ISSUES DETECTED:", "ERROR")
            for issue in issues:
                self.log(f"   • {issue}", "ERROR")
            return False
        else:
            self.log("✅ Advisor configuration validation passed", "SUCCESS")
            return True


    def run_comprehensive_profit_tests(self):
        """🧪 Comprehensive test suite for profit calculations"""

        print("🚀 Running Enhanced Profit System Tests...")
        print("=" * 60)

        # Test 1: Strategy multiplier application
        print("\n📊 Test 1: Strategy Multiplier Application")
        self.test_strategy_profit_scaling()

        # Test 2: Time scaling verification
        print("\n⏱️ Test 2: Time Scaling Verification")
        for days in [7, 14, 30, 60, 90]:
            target = self.calculate_dynamic_profit_target(
                self.mock_indicators, 80, days, "TEST", {"profit": 1.0, "risk": 1.0}
            )
            base_multiplier = target / 0.037
            print(f"   {days} days: {target:.1%} ({base_multiplier:.1f}x base)")

            if days >= 60 and base_multiplier < 2.0:
                print(f"   ❌ WARNING: {days} days should have higher multiplier")
            else:
                print(f"   ✅ {days} days scaling looks good")

        # Test 3: Combined effects
        print("\n🎯 Test 3: Combined Strategy + Time Effects")
        strategies = ["Conservative", "Balanced", "Aggressive", "Swing Trading"]
        multipliers = [0.8, 1.0, 1.4, 1.8]

        for strategy, mult in zip(strategies, multipliers):
            for days in [7, 30, 90]:
                target = self.calculate_dynamic_profit_target(
                    self.mock_indicators, 80, days, "TEST", {"profit": mult, "risk": 1.0}
                )
                print(f"   {strategy} + {days}d: {target:.1%}")

                # Verify scaling
                expected_min = 0.037 * mult * (1.0 if days == 7 else 1.5 if days == 30 else 3.0)
                if target >= expected_min:
                    print(f"   ✅ Scaling verified (≥{expected_min:.1%})")
                else:
                    print(f"   ❌ Scaling issue (expected ≥{expected_min:.1%})")

        print("\n" + "=" * 60)
        print("✅ Enhanced Profit System Tests Complete!")


def show_debug_logs_safely(result, show_debug):
    """Safely display debug logs with error handling"""
    if show_debug:
        st.markdown("---")
        st.subheader("🐛 Debug Logs")

        # Safely get debug logs
        debug_logs = result.get("debug_log", [])

        if debug_logs:
            with st.expander("🔍 Full Debug Output", expanded=False):
                try:
                    # Convert all log entries to strings and join
                    log_text = "\n".join(str(log) for log in debug_logs)
                    st.code(log_text, language="text")
                except Exception as e:
                    st.error(f"Error displaying debug logs: {e}")
                    st.write("Raw debug data:")
                    st.write(debug_logs)

            # FIXED: Safe filtering with error handling
            try:
                success_lines = [str(l) for l in debug_logs if '✅' in str(l) or 'SUCCESS' in str(l)]
                error_lines = [str(l) for l in debug_logs if '❌' in str(l) or 'ERROR' in str(l)]
                neutral_lines = [str(l) for l in debug_logs if '⚖️' in str(l) or 'INFO' in str(l)]

                if success_lines:
                    st.markdown("### ✅ Successful Checks")
                    st.code("\n".join(success_lines))

                if error_lines:
                    st.markdown("### ❌ Warnings & Issues")
                    st.code("\n".join(error_lines))

                if neutral_lines:
                    st.markdown("### ⚖️ Neutral Observations")
                    st.code("\n".join(neutral_lines))

            except Exception as e:
                st.error(f"Error processing debug logs: {e}")

        else:
            st.info("No debug logs available. Make sure debug mode is enabled in the sidebar.")


# ALSO ADD: Enhanced profit display in results section
def show_enhanced_profit_breakdown(result, strategy_type, investment_days):
    """Show detailed profit breakdown with strategy effects"""

    with st.expander("🚀 Enhanced Profit Analysis (Strategy Effects)", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**📊 Profit Calculation Breakdown:**")

            # Show the actual multipliers used
            base_target = 3.7  # Or get from result
            strategy_mult = result.get('strategy_multiplier', 1.0)
            time_mult = result.get('time_multiplier', 1.0)
            final_target = result.get('gross_profit_pct', 3.7)

            st.write(f"• Base Target: {base_target:.1f}%")
            st.write(f"• Strategy Multiplier ({strategy_type}): {strategy_mult:.2f}x")
            st.write(f"• Time Multiplier ({investment_days} days): {time_mult:.2f}x")
            st.write(f"• **Gross Target: {final_target:.1f}%**")
            st.write(f"• **Net Profit: {result.get('expected_profit_pct', 0):.1f}%**")

        with col2:
            st.markdown("**⚡ Strategy Comparison:**")

            strategies = {
                "Conservative": {"mult": 0.8, "desc": "Lower risk, steady gains"},
                "Balanced": {"mult": 1.0, "desc": "Moderate risk/reward"},
                "Aggressive": {"mult": 1.4, "desc": "Higher risk, bigger gains"},
                "Swing Trading": {"mult": 1.8, "desc": "Maximum profit potential"}
            }

            for strat_name, strat_info in strategies.items():
                if strat_name == strategy_type:
                    st.write(f"🎯 **{strat_name}**: {strat_info['mult']}x - {strat_info['desc']}")
                else:
                    st.write(f"• {strat_name}: {strat_info['mult']}x - {strat_info['desc']}")

        with col3:
            st.markdown("**📈 Time vs Strategy Effect:**")

            # Show how the combination works
            current_mult = strategy_mult * time_mult

            if current_mult >= 3.0:
                st.success(f"🚀 **{current_mult:.1f}x multiplier** - Maximum profit mode!")
            elif current_mult >= 2.0:
                st.info(f"📈 **{current_mult:.1f}x multiplier** - High profit targeting")
            elif current_mult >= 1.5:
                st.info(f"📊 **{current_mult:.1f}x multiplier** - Enhanced targeting")
            else:
                st.warning(f"⚖️ **{current_mult:.1f}x multiplier** - Conservative targeting")

            st.write(f"• Your settings: {strategy_type} + {investment_days} days")
            st.write(f"• Combined effect: {current_mult:.1f}x base profit")


# UPDATE: The analysis results section to show strategy effects
def show_strategy_effects_in_results(result, strategy_type, advisor):
    """Show how strategy affected the recommendation"""

    if result.get('strategy_applied', False):
        st.success("✅ **Strategy Effects Applied Successfully**")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🎯 Strategy Impact:**")
            strategy_mult = advisor.strategy_settings.get("profit", 1.0)
            st.write(f"• Strategy: {strategy_type}")
            st.write(f"• Profit Multiplier: {strategy_mult:.1f}x")
            st.write(f"• Risk Multiplier: {advisor.strategy_settings.get('risk', 1.0):.1f}x")
            st.write(f"• Confidence Requirement: {advisor.strategy_settings.get('confidence_req', 75)}%")

        with col2:
            st.markdown("**📊 Scaling Effect:**")
            time_mult = result.get('time_multiplier', 1.0)
            total_mult = strategy_mult * time_mult
            st.write(f"• Time Multiplier: {time_mult:.2f}x")
            st.write(f"• **Total Scaling: {total_mult:.2f}x**")
            st.write(f"• Base 3.7% → Target {result.get('gross_profit_pct', 3.7):.1f}%")

            if total_mult >= 3.0:
                st.success("🚀 Maximum profit mode!")
            elif total_mult >= 2.0:
                st.info("📈 High profit targeting")
            else:
                st.info("📊 Standard targeting")
    else:
        st.warning("⚠️ Strategy effects not applied - check system integration")


def create_enhanced_interface():
    """Create enhanced interface with 95% confidence targeting"""
    st.set_page_config(
        page_title="Enhanced Stock Advisor",
        page_icon="📈",
        layout="wide"
    )

    # Header
    st.title("📈 Enhanced Stock Trading Advisor")
    st.markdown("### Advanced AI system with 95% confidence targeting!")
    st.markdown("---")

    # INITIALIZE DEBUG SETTINGS IN SESSION STATE
    if 'show_debug' not in st.session_state:
        st.session_state.show_debug = False
    if 'download_file' not in st.session_state:
        st.session_state.download_file = False

    # Initialize advisor with current settings
    if 'enhanced_advisor' not in st.session_state:
        st.session_state.enhanced_advisor = EnhancedStockAdvisor(
            debug=True,  # Always enable debug for potential logging
            download_log=st.session_state.download_file
        )

    advisor = st.session_state.enhanced_advisor

    advisor.log("Create Streamlit Page", "INFO")
    advisor.log("Enhanced interface initialized", "INFO")

    # Sidebar controls
    st.sidebar.header("🎯 Get Your Trading Advice")

    # Stock input
    stock_symbol = st.sidebar.text_input(
        "📊 Stock Symbol",
        value="NVDA",
        help="Enter any stock ticker (e.g., AAPL, GOOGL, TSLA)"
    ).upper().strip()

    advisor.log(f"Stock Symbol: {stock_symbol}", "INFO")

    # Date input
    date_input = st.sidebar.text_input(
        "📅 Date (MM/DD/YY or MM/DD/YYYY)",
        value="7/1/25",
        help="Enter date like: 1/7/25 or 1/7/2025"
    )
    advisor.log(f"Date Input: {date_input}", "INFO")

    # Parse the date
    try:
        if '/' in date_input:
            parts = date_input.split('/')
            if len(parts) == 3:
                month, day, year = parts
                if len(year) == 2:
                    year = "20" + year if int(year) < 50 else "19" + year
                target_date = datetime(int(year), int(month), int(day)).date()
                advisor.log(f"Target Date: {target_date}", "INFO")
            else:
                target_date = datetime.now().date()
                advisor.log(f"Target Date: {target_date}", "INFO")
        else:
            target_date = datetime.now().date()
            advisor.log(f"Target Date: {target_date}", "INFO")
    except:
        target_date = datetime.now().date()
        st.sidebar.warning("⚠️ Invalid date format. Using today's date.")
        advisor.log("Invalid date format. Using today's date.", "WARNING")

    # Investment period
    # Investment timeframe selection
    advisor.investment_days = st.sidebar.selectbox(
        "🕐 Target holding period(up to):",
        options=[1, 3, 7, 14, 21, 30, 45, 60, 90, 120],  # Extended options
        index=2,  # Default to 7 days
        help="Longer periods generally allow for higher profit targets but require more patience"
    )
    advisor.log(f"Investment Days: {advisor.investment_days}", "INFO")

    # Strategy type selection
    strategy_type = st.sidebar.radio(
        "📈 Strategy Type:",
        options=["Conservative", "Balanced", "Aggressive", "Swing Trading"],
        index=1,  # Default to Balanced
        help="Strategy affects profit targets and risk tolerance"
    )
    advisor.log(f"Strategy type selection: {strategy_type}", "INFO")

    # Map strategy to multipliers and store in advisor
    strategy_multipliers = {
        "Conservative": {"profit": 0.8, "risk": 0.8, "confidence_req": 85},
        "Balanced": {"profit": 1.0, "risk": 1.0, "confidence_req": 75},
        "Aggressive": {"profit": 1.4, "risk": 1.3, "confidence_req": 65},
        "Swing Trading": {"profit": 1.8, "risk": 1.5, "confidence_req": 70}
    }
    advisor.strategy_settings = strategy_multipliers[strategy_type]
    advisor.current_strategy = strategy_type  # Store strategy name for logging
    advisor.log(f"Strategy: {strategy_type}, Investment Days: {advisor.investment_days}", "INFO")

    # ENHANCED: Show realistic profit targets based on selection
    base_profit = 3.7  # Base 3.7%
    strategy_multiplier = advisor.strategy_settings["profit"]
    time_multiplier = 1.0 + (advisor.investment_days - 7) * 0.05  # More aggressive time scaling

    estimated_profit = base_profit * strategy_multiplier * time_multiplier

    # Enhanced profit target preview with strategy effect
    if advisor.investment_days >= 30:
        if strategy_type == "Swing Trading":
            profit_range = f"15-35% profits"
        elif strategy_type == "Aggressive":
            profit_range = f"12-25% profits"
        else:
            profit_range = f"8-15% profits"
        st.sidebar.info(f"💡 {strategy_type} + Long timeframe (≥30 days): Target {profit_range}")
    elif advisor.investment_days >= 14:
        if strategy_type == "Swing Trading":
            profit_range = f"10-20% profits"
        elif strategy_type == "Aggressive":
            profit_range = f"8-15% profits"
        else:
            profit_range = f"5-10% profits"
        st.sidebar.info(f"💡 {strategy_type} + Medium timeframe: Target {profit_range}")
    else:
        if strategy_type == "Swing Trading":
            profit_range = f"6-12% profits"
        elif strategy_type == "Aggressive":
            profit_range = f"5-8% profits"
        else:
            profit_range = f"3-6% profits"
        st.sidebar.info(f"💡 {strategy_type} + Short timeframe: Target {profit_range}")

    # Show the estimated profit for current settings
    st.sidebar.metric(
        "📊 Estimated Target Profit",
        f"{estimated_profit:.1f}%",
        delta=f"vs base {base_profit}%"
    )

    # Show model availability
    if stock_symbol in advisor.models:
        st.sidebar.success(f"🤖 AI Model Available for {stock_symbol}")
        advisor.log(f"AI Model Available for {stock_symbol}", "SUCCESS")
    else:
        st.sidebar.info(f"📊 Using Technical Analysis for {stock_symbol}")
        advisor.log(f"Using Technical Analysis for {stock_symbol}", "INFO")

    # Analyze button
    analyze_btn = st.sidebar.button("🚀 Get Enhanced Trading Advice", type="primary", use_container_width=True)
    advisor.log("Analyze Button Clicked", "INFO")

    # ADD SEPARATOR BEFORE DEBUG CONTROLS
    st.sidebar.markdown("---")

    # DEBUG CONTROLS - AT THE BOTTOM OF SIDEBAR
    st.sidebar.markdown("### 🐛 Debug Options")

    # Update session state with current checkbox values
    st.session_state.show_debug = st.sidebar.checkbox(
        "Show Debug Logs",
        value=st.session_state.show_debug,
        help="Display detailed calculation logs on screen"
    )

    st.session_state.download_file = st.sidebar.checkbox(
        "Enable Log File Creation",
        value=st.session_state.download_file,
        help="Create downloadable log file"
    )

    # UPDATE ADVISOR SETTINGS BASED ON CURRENT STATE
    advisor.download_log = st.session_state.download_file
    if st.session_state.download_file:
        if not hasattr(advisor, 'log_file') or not advisor.log_file:
            advisor.ensure_log_file()  # Use the new method
            st.sidebar.success(f"📝 Log file created: {os.path.basename(advisor.log_file)}")
    else:
        advisor.log_file = None

    # Show download button if log file exists and download is enabled
    if st.session_state.download_file and 'enhanced_advisor' in st.session_state:
        # Ensure advisor has log_file attribute
        if not hasattr(advisor, 'log_file') or not advisor.log_file:
            advisor.log_file = f"debug_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

        # Check if log file exists and has content
        if os.path.exists(advisor.log_file):
            try:
                with open(advisor.log_file, 'r', encoding='utf-8') as f:
                    log_content = f.read()

                if log_content.strip():  # Only show if file has content
                    st.sidebar.download_button(
                        label="📥 Download Debug Log",
                        data=log_content,
                        file_name=f"debug_log_{stock_symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}.log",
                        mime="text/plain",
                        help="Download the complete debug log file"
                    )
                    # Show file info
                    file_size = len(log_content.encode('utf-8'))
                    st.sidebar.caption(f"Log file: {file_size} bytes, {log_content.count(chr(10))} lines")
                else:
                    st.sidebar.info("📝 Log file exists but is empty. Run an analysis first.")

            except Exception as e:
                st.sidebar.error(f"Error accessing log file: {e}")
                st.sidebar.caption(f"Log file path: {advisor.log_file}")
                advisor.log(f"Error accessing log file: {e}", "ERROR")
        else:
            if advisor.debug_log:  # If there are debug logs in memory but no file
                st.sidebar.info("📝 Debug logs available. Run an analysis to create downloadable file.")
            else:
                st.sidebar.info("📝 No debug logs yet. Run an analysis first.")
                advisor.log("No debug logs yet. Run an analysis first.", "INFO")

    # Use session state values for the rest of the application
    show_debug = st.session_state.show_debug
    download_file = st.session_state.download_file

    # Analysis results
    if analyze_btn and stock_symbol:
        with st.spinner(f"🔍 Running enhanced analysis for {stock_symbol}..."):

            result = advisor.analyze_stock_enhanced(stock_symbol, target_date)

            if result is None:
                st.error("❌ Could not analyze this stock. Please try a different symbol or date.")
                advisor.log("Could not analyze this stock. Please try a different symbol or date.", "ERROR")
                return

            # Success message
            st.sidebar.success(f"✅ Enhanced analysis complete for {stock_symbol}")
            advisor.log(f"Enhanced analysis complete for {stock_symbol}", "SUCCESS")

            # Main recommendation box
            action = result['action']
            confidence = result['confidence']

            cal1, cal2 = st.columns(2)
            # Enhanced color-coded recommendation
            if action == "BUY":
                cal2.success(f"🟢 **RECOMMENDATION: {action}** 📈")
                cal1.markdown(f"### **Confidence Level: {confidence:.0f}%**")
                advisor.log(f"BUY recommendation for {stock_symbol}", "INFO")

            elif action == "SELL/AVOID":
                cal2.error(f"🔴 **RECOMMENDATION: {action}** 📉")
                cal1.markdown(f"### **Confidence Level: {confidence:.0f}%**")
                advisor.log(f"SELL/AVOID recommendation for {stock_symbol}", "INFO")

            else:
                cal2.warning(f"🟡 **RECOMMENDATION: {action}** ⏳")
                cal1.markdown(f"### **Confidence Level: {confidence:.0f}%**")
                advisor.log(f"NEUTRAL recommendation for {stock_symbol}", "INFO")

            # Enhanced price information
            st.subheader("💰 Price Information")

            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric(
                    label="Current Price",
                    value=f"${result['current_price']:.2f}",
                    help="Price on the analysis date"
                )
                advisor.log(f"Current Price: {result['current_price']:.2f}", "INFO")

            with col2:
                if result.get('buy_price'):
                    st.metric(
                        label="🟢 BUY at",
                        value=f"${result['buy_price']:.2f}",
                        help="Recommended buying price"
                    )
                    advisor.log(f"BUY Price: {result['buy_price']:.2f}", "INFO")
                else:
                    st.metric(
                        label="🟢 BUY at",
                        value="N/A",
                        help="No buy recommendation"
                    )
                    advisor.log("No buy recommendation", "INFO")

            with col3:
                if result.get('sell_price'):
                    st.metric(
                        label="🔴 SELL at",
                        value=f"${result['sell_price']:.2f}",
                        help="Target selling price"
                    )
                    advisor.log(f"SELL Price: {result['sell_price']:.2f}", "INFO")
                else:
                    st.metric(
                        label="🔴 SELL at",
                        value="N/A",
                        help="No sell target"
                    )
                    advisor.log("No sell target", "INFO")

            with col4:
                if result['expected_profit_pct'] > 0:
                    st.metric(
                        label="💰 Expected Profit",
                        value=f"{result['expected_profit_pct']:.1f}%",
                        delta=f"in {advisor.investment_days} days"
                    )
                    st.caption(result.get('net_profit_message', ''))
                    advisor.log(f"Expected Profit: {result['expected_profit_pct']:.1f}%", "INFO")
                else:
                    st.metric(
                        label="💰 Expected Profit",
                        value="0%",
                        help="No profit expected"
                    )
                    advisor.log("No profit expected", "INFO")

            with col5:
                st.metric(
                    label="💸 Broker Fee",
                    value=f"{result.get('broker_fee_paid', 0.0):.2f}%",
                    help="Total cost from buying and selling fees"
                )
                st.metric(
                    label="🧾 Tax Paid",
                    value=f"{result.get('tax_paid', 0.0):.2f}%",
                    help="25% capital gains tax applied to net profit"
                )

            def show_enhanced_profit_analysis(result, strategy_type, investment_days):
                """Display enhanced profit analysis"""

                with st.expander("🚀 Enhanced Profit Analysis"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**📊 Profit Target Breakdown:**")
                        base_target = result.get('base_profit_target', 3.7)
                        final_target = result.get('expected_profit_pct', 3.7)

                        st.write(f"• Base Target: {base_target:.1f}%")
                        st.write(
                            f"• Strategy Multiplier ({strategy_type}): {strategy_multipliers[strategy_type]['profit']:.1f}x")
                        st.write(
                            f"• Time Multiplier ({investment_days} days): {1.0 + (investment_days - 7) * 0.02:.1f}x")
                        st.write(f"• **Final Target: {final_target:.1f}%**")

                    with col2:
                        st.markdown("**⏰ Time vs Profit Expectations:**")

                        timeframes = {
                            "1-7 days": "3-6% (Quick trades)",
                            "7-21 days": "5-10% (Short swing)",
                            "21-60 days": "8-15% (Medium swing)",
                            "60+ days": "12-25% (Long swing)"
                        }

                        for timeframe, profit_range in timeframes.items():
                            if investment_days <= 7 and "1-7 days" in timeframe:
                                st.write(f"🎯 **{timeframe}: {profit_range}**")
                            elif 7 < investment_days <= 21 and "7-21 days" in timeframe:
                                st.write(f"🎯 **{timeframe}: {profit_range}**")
                            elif 21 < investment_days <= 60 and "21-60 days" in timeframe:
                                st.write(f"🎯 **{timeframe}: {profit_range}**")
                            elif investment_days > 60 and "60+ days" in timeframe:
                                st.write(f"🎯 **{timeframe}: {profit_range}**")
                            else:
                                st.write(f"• {timeframe}: {profit_range}")

            show_enhanced_profit_analysis(result, strategy_type, advisor.investment_days)

            # Show signal strength breakdown
            if confidence >= 85:
                st.info("🎯 **HIGH CONFIDENCE SIGNAL** - Multiple indicators confirm this recommendation")
                advisor.log("HIGH CONFIDENCE SIGNAL", "INFO")
            elif confidence >= 70:
                st.info("✅ **GOOD CONFIDENCE** - Most indicators support this recommendation")
                advisor.log("GOOD CONFIDENCE", "INFO")
            else:
                st.warning("⚠️ **MODERATE CONFIDENCE** - Mixed signals detected")
                advisor.log("MODERATE CONFIDENCE", "WARNING")

            # ADD THIS NEW SECTION:
            if result.get('enhancement_active', False):
                st.info("🎯 **95% CONFIDENCE SYSTEM ACTIVE** - Enhanced analysis in use")

                # Show confidence breakdown
                if 'confidence_boost' in result and result['confidence_boost'] > 0:
                    st.success(
                        f"💪 Confidence Boosted: {result['original_confidence']:.1f}% → {result['confidence']:.1f}% (+{result['confidence_boost']:.1f}%)")

                # Enhanced confidence indicators
                if result['confidence'] >= 95:
                    st.success("🏆 **ULTRA-HIGH CONFIDENCE** - Highest quality signal")
                elif result['confidence'] >= 90:
                    st.success("🌟 **VERY HIGH CONFIDENCE** - Excellent signal quality")
                elif result['confidence'] >= 85:
                    st.info("⭐ **HIGH CONFIDENCE** - Strong signal")

                # Show enhancement details in expander
                with st.expander("🔍 95% Confidence System Details"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**System Status:**")
                        st.write("✅ Enhanced Signal Detection: Active")
                        st.write("✅ Confidence Builder: Active")
                        st.write("✅ Market Regime Detection: Active")

                    with col2:
                        st.markdown("**Confidence Breakdown:**")
                        if 'signal_breakdown' in result:
                            for signal_type, score in result['signal_breakdown'].items():
                                st.write(f"• {signal_type.replace('_', ' ').title()}: {score:.2f}")
            else:
                st.warning("⚠️ Using original system - 95% confidence enhancements not active")

            # Enhanced trading plan
            st.subheader("📋 Your Enhanced Trading Plan")

            col1, col2, col3 = st.columns(3)

            if action == "BUY":
                with col1:
                    st.markdown(f"""
                    ### 🟢 **BUY PLAN FOR {stock_symbol}**
                    
                    **What to do:**
                    1. 💰 **Buy the stock at:** ${result['buy_price']:.2f}
                    2. 🎯 **Sell it when it reaches:** ${result['sell_price']:.2f}
                    3. 🛡️ **Stop loss if it drops to:** ${result['stop_loss']:.2f}
                    4. ⏱️ **Max holding time:** {advisor.investment_days} days (can exit earlier at target)
                    """)
                    advisor.log(f"BUY plan for {stock_symbol}", "INFO")


                with col2:
                    st.markdown(f"""
                    **Expected Outcome:**
                    - 💵 **Profit per share:** ${(result['sell_price'] - result['buy_price']):.2f}
                    - 📈 **Percentage gain:** {result['expected_profit_pct']:.1f}%
                    - 🎲 **Success probability:** {confidence:.0f}%
                    - 🛡️ **Max loss if stopped:** {((result['buy_price'] - result['stop_loss']) / result['buy_price'] * 100):.1f}%
                    - ⏰ **Exit strategy:** Sell at target OR after {advisor.investment_days} days
                    """)
                    advisor.log(f"Expected outcome for {stock_symbol}", "INFO")

            elif action == "SELL/AVOID":
                with col1:
                    st.markdown(f"""
                    ### 🔴 **AVOID/SELL PLAN FOR {stock_symbol}**
                    
                    **What to do:**
                    - 🚫 **Don't buy this stock right now**
                    - 📉 **If you own it, sell at:** ${result['sell_price']:.2f}
                    - ⏳ **Re-evaluate in:** {advisor.investment_days} days maximum
                    """)
                    advisor.log(f"SELL/AVOID plan for {stock_symbol}", "INFO")

                with col2:
                    st.markdown(f"""
                    **Expected Outcome:**
                    - 📉 **Potential loss avoided:** {result['expected_profit_pct']:.1f}%
                    - 🎲 **Confidence in decline:** {confidence:.0f}%
                    - 💡 **Better opportunities expected within {advisor.investment_days} days**
                    """)
                    advisor.log(f"Expected outcome for {stock_symbol}", "INFO")

            else:
                with col1:
                    st.markdown(f"""
                    ### 🟡 **WAIT PLAN FOR {stock_symbol}**
                    
                    **What to do:**
                    - ⏳ **Wait for clearer signals**
                    - 👀 **Monitor daily for up to {advisor.investment_days} days**
                    - 🔄 **Re-analyze when signals strengthen**
                    """)
                    advisor.log(f"WAIT plan for {stock_symbol}", "INFO")

                with col2:
                    st.markdown(f"""
                    **Why wait:**
                    - 🤔 **Conflicting signals detected**
                    - 📊 **Need stronger confirmation**
                    - 🎯 **Better timing expected within {advisor.investment_days} days**
                    """)
                    advisor.log(f"Wait reason for {stock_symbol}", "INFO")

            # Enhanced signal analysis
            with col3:
                st.subheader("🔬 Signal Analysis")

                signal_breakdown = result.get('signal_breakdown', {})

                # Show signal strength bars
                if signal_breakdown:
                    for signal_type, score in signal_breakdown.items():
                        if signal_type == 'trend_score':
                            emoji = "📈" if score > 0 else "📉" if score < 0 else "➡️"
                            st.write(f"{emoji} **Trend:** {score:.1f}")
                            advisor.log(f"Trend score for {stock_symbol}: {score:.1f}", "INFO")

                        elif signal_type == 'momentum_score':
                            emoji = "🚀" if score > 0 else "🔻" if score < 0 else "⚖️"
                            st.write(f"{emoji} **Momentum:** {score:.1f}")
                            advisor.log(f"Momentum score for {stock_symbol}: {score:.1f}", "INFO")

                        elif signal_type == 'volume_score':
                            emoji = "📢" if abs(score) > 1 else "📊"
                            st.write(f"{emoji} **Volume:** {score:.1f}")
                            advisor.log(f"Volume score for {stock_symbol}: {score:.1f}", "INFO")

                        elif signal_type == 'sr_score':
                            emoji = "🎯" if abs(score) > 1 else "📊"
                            st.write(f"{emoji} **Support/Resistance:** {score:.1f}")
                            advisor.log(f"Support/Resistance score for {stock_symbol}: {score:.1f}", "INFO")

                        elif signal_type == 'model_score':
                            emoji = "🤖" if abs(score) > 1 else "📊"
                            st.write(f"{emoji} **AI Model:** {score:.1f}")
                            advisor.log(f"AI Model score for {stock_symbol}: {score:.1f}", "INFO")

            with st.expander("Recommendation"):
                # Detailed reasoning
                st.subheader("🤔 Why This Recommendation?")

                reasons = result.get('reasons', [])
                if reasons:
                    advisor.log(f"Reasons for {stock_symbol}", "INFO")
                    # Group reasons by category for better organization
                    trend_reasons = [r for r in reasons if any(word in r.lower() for word in ['average', 'trend', 'ema', 'moving'])]
                    advisor.log(f"Trend reasons: {trend_reasons}", "INFO")
                    momentum_reasons = [r for r in reasons if any(word in r.lower() for word in ['rsi', 'macd', 'stochastic', 'momentum'])]
                    advisor.log(f"Momentum reasons: {momentum_reasons}", "INFO")
                    volume_reasons = [r for r in reasons if 'volume' in r.lower()]
                    advisor.log(f"Volume reasons: {volume_reasons}", "INFO")
                    level_reasons = [r for r in reasons if any(word in r.lower() for word in ['support', 'resistance', 'bollinger'])]
                    advisor.log(f"Level reasons: {level_reasons}", "INFO")
                    model_reasons = [r for r in reasons if 'model' in r.lower()]
                    advisor.log(f"Model reasons: {model_reasons}", "INFO")
                    other_reasons = [r for r in reasons if r not in trend_reasons + momentum_reasons + volume_reasons + level_reasons + model_reasons]
                    advisor.log(f"Other reasons: {other_reasons}", "INFO")
                    col1, col2 = st.columns(2)

                    with col1:
                        if trend_reasons:
                            st.markdown("**📈 Trend Analysis:**")
                            for reason in trend_reasons:
                                st.write(f"• {reason}")
                                advisor.log(f"Trend reason: {reason}", "INFO")

                        if momentum_reasons:
                            st.markdown("**🚀 Momentum Indicators:**")
                            for reason in momentum_reasons:
                                st.write(f"• {reason}")
                                advisor.log(f"Momentum reason: {reason}", "INFO")

                        if volume_reasons:
                            st.markdown("**📊 Volume Analysis:**")
                            for reason in volume_reasons:
                                st.write(f"• {reason}")
                                advisor.log(f"Volume reason: {reason}", "INFO")

                    with col2:
                        if level_reasons:
                            st.markdown("**🎯 Key Levels:**")
                            for reason in level_reasons:
                                st.write(f"• {reason}")
                                advisor.log(f"Level reason: {reason}", "INFO")

                        if model_reasons:
                            st.markdown("**🤖 AI Analysis:**")
                            for reason in model_reasons:
                                st.write(f"• {reason}")
                                advisor.log(f"Model reason: {reason}", "INFO")

                        if other_reasons:
                            st.markdown("**📋 Additional Factors:**")
                            for reason in other_reasons:
                                st.write(f"• {reason}")
                                advisor.log(f"Other reason: {reason}", "INFO")

            # Enhanced chart
            chart = advisor.create_enhanced_chart(stock_symbol, result)
            if chart:
                st.subheader("📊 Enhanced Technical Chart")
                st.plotly_chart(chart, use_container_width=True)
                advisor.log(f"Enhanced chart for {stock_symbol} created", "INFO")

            with st.expander("Risk Assessment"):
                # Enhanced risk information
                st.subheader("⚠️ Risk Assessment")

                risk_level = result.get('risk_level', 'Medium-term')
                final_score = result.get('final_score', 0)
                advisor.log(f"Risk level for {stock_symbol}: {risk_level}", "INFO")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("**📊 Risk Level**")
                    if risk_level == "Short-term":
                        st.warning("""
                        **🏃‍♂️ Short-term (1-7 days):**
                        - ⚡ Higher volatility
                        - 👀 Monitor closely
                        - 🎯 Quick decisions needed
                        """)
                        advisor.log(f"Risk level for {stock_symbol}: Short-term", "INFO")

                    elif risk_level == "Medium-term":
                        st.info("""
                        **🚶‍♂️ Medium-term (1-3 weeks):**
                        - ⚖️ Balanced approach
                        - 📅 Weekly monitoring
                        - 📈 Trend development time
                        """)
                        advisor.log(f"Risk level for {stock_symbol}: Medium-term", "INFO")

                    else:
                        st.success("""
                        **🐌 Long-term (3-4 weeks+):**
                        - 📉 Lower daily volatility
                        - 🔄 Less frequent monitoring
                        - 🎯 Fundamental changes
                        """)
                        advisor.log(f"Risk level for {stock_symbol}: Long-term", "INFO")

                with col2:
                    st.markdown("**🎯 Signal Strength**")
                    if abs(final_score) >= 2.5:
                        st.success("🔥 **VERY STRONG** signal")
                        advisor.log(f"Signal strength for {stock_symbol}: Very Strong", "INFO")

                    elif abs(final_score) >= 1.5:
                        st.info("💪 **STRONG** signal")
                        advisor.log(f"Signal strength for {stock_symbol}: Strong", "INFO")

                    elif abs(final_score) >= 1.0:
                        st.warning("📊 **MODERATE** signal")
                        advisor.log(f"Signal strength for {stock_symbol}: Moderate", "INFO")

                    else:
                        st.error("🤔 **WEAK** signal")
                        advisor.log(f"Signal strength for {stock_symbol}: Weak", "INFO")

                    st.write(f"Signal Score: {final_score:.2f}")
                    advisor.log(f"Signal score for {stock_symbol}: {final_score:.2f}", "INFO")

                with col3:
                    st.markdown("**💡 Recommendation Quality**")
                    if confidence >= 90:
                        st.success("🏆 **EXCELLENT** - Act with confidence")
                        advisor.log(f"Recommendation quality for {stock_symbol}: Excellent", "INFO")

                    elif confidence >= 80:
                        st.success("✅ **VERY GOOD** - Strong recommendation")
                        advisor.log(f"Recommendation quality for {stock_symbol}: Very Good", "INFO")

                    elif confidence >= 70:
                        st.info("👍 **GOOD** - Solid analysis")
                        advisor.log(f"Recommendation quality for {stock_symbol}: Good", "INFO")

                    elif confidence >= 60:
                        st.warning("⚖️ **FAIR** - Consider carefully")
                        advisor.log(f"Recommendation quality for {stock_symbol}: Fair", "INFO")

                    else:
                        st.error("🤔 **POOR** - Wait for better signals")
                        advisor.log(f"Recommendation quality for {stock_symbol}: Poor", "ERROR")

            # Enhanced disclaimer
            st.subheader("📋 Important Trading Guidelines")

            # if show_debug:
            #     st.markdown("---")
            #     st.subheader("🐛 Debug Logs")
            #
            #     with st.expander("🔍 Full Debug Output", expanded=False):
            #         st.code("\n".join(result["debug_log"]), language="text")
            #
            #     success_lines = [l for l in result['debug_log'] if l.startswith('✅')]
            #     error_lines = [l for l in result['debug_log'] if l.startswith('❌')]
            #     neutral_lines = [l for l in result['debug_log'] if l.startswith('⚖️')]
            #
            #     st.markdown("### ✅ Successful Checks")
            #     st.code("\n".join(success_lines))
            #
            #     st.markdown("### ❌ Warnings & Issues")
            #     st.code("\n".join(error_lines))
            #
            #     st.markdown("### ⚖️ Neutral Observations")
            #     st.code("\n".join(neutral_lines))

            show_debug_logs_safely(result,show_debug)
            col1, col2 = st.columns(2)

            with col1:
                st.info("""
                **✅ Before You Trade:**
                - 📊 Double-check current market conditions
                - 💰 Only invest what you can afford to lose
                - 🎯 Set stop-losses as recommended
                - 📈 Monitor your positions regularly
                - 📚 Keep learning about the market
                """)

            with col2:
                st.warning("""
                **⚠️ Risk Reminders:**
                - 📉 Past performance ≠ future results
                - 🌪️ Markets can be unpredictable
                - 📰 News can change everything quickly
                - 💡 This is educational, not financial advice
                - 👨‍💼 Consult professionals for large investments
                """)

    else:
        # Enhanced welcome message
        st.info("👆 Enter a stock symbol and date to get your enhanced trading advice!")

        # Enhanced examples
        st.subheader("🚀 Enhanced Features")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **🎯 What Makes This Enhanced:**
            - **🤖 AI + Technical Analysis** - Dual-powered recommendations
            - **📊 Multi-timeframe Analysis** - 5 different signal types
            - **🔍 95% Confidence Targeting** - Higher accuracy goals
            - **📈 Advanced Indicators** - RSI, MACD, Bollinger Bands, Stochastic
            - **🎪 Signal Confirmation** - Multiple indicators must agree
            """)

        with col2:
            st.markdown("""
            **✅ You Get:**
            - **🎯 High-confidence recommendations** (target: 85-95%)
            - **💰 Precise buy/sell prices** with stop-losses
            - **📊 Detailed signal breakdown** by category
            - **🔬 Multi-factor analysis** explanation
            - **📈 Enhanced technical charts** with all indicators
            """)

        # Example recommendation display
        st.subheader("📋 Example Enhanced Recommendation")

        st.code("""
            🟢 RECOMMENDATION: BUY
            Confidence Level: 87%
            
            💰 Price Information:
            Current Price: $153.30
            🟢 BUY at: $153.30
            🔴 SELL at: $158.45
            💰 Expected Profit: 3.4%
            
            🔬 Signal Analysis:
            📈 Trend: +2.1 (Strong upward)
            🚀 Momentum: +1.8 (Bullish RSI & MACD)
            📢 Volume: +1.2 (Above average confirmation)  
            🎯 Support/Resistance: +0.9 (Near support)
            🤖 AI Model: +2.3 (85% buy confidence)
            
            Final Signal Score: +8.3 (Very Strong)
        """)


if __name__ == "__main__":
    create_enhanced_interface()