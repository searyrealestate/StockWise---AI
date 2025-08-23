"""
Enhanced Stock Trading Algorithm Debug and Fix
============================================

Issue: Algorithm generating WAIT signals instead of actionable BUY/SELL recommendations
Solution: Multiple debugging and enhancement approaches
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import ta


class TradingAlgorithmDebugger:
    """
    Debug and fix trading algorithm issues
    """

    def __init__(self, advisor):
        self.advisor = advisor
        self.debug_results = []

    def diagnose_wait_signals_issue(self, symbol="NVDA", start_date="2025-04-21"):
        """
        Comprehensive diagnosis of why algorithm generates WAIT signals
        """
        print("🔍 DIAGNOSING WAIT SIGNALS ISSUE")
        print("=" * 60)

        # 1. Check Signal Strength Distribution
        self.check_signal_strength_distribution(symbol, start_date)

        # 2. Analyze Strategy Thresholds
        self.analyze_strategy_thresholds()

        # 3. Check Data Quality Issues
        self.check_data_quality_issues(symbol, start_date)

        # 4. Test Different Time Periods
        self.test_different_time_periods(symbol)

        # 5. Check Model Availability
        self.check_model_availability(symbol)

        return self.debug_results

    def check_signal_strength_distribution(self, symbol, start_date):
        """
        Check if signals are consistently weak
        """
        print("\n1. 📊 SIGNAL STRENGTH ANALYSIS")
        print("-" * 40)

        # Simulate signal analysis for recent dates
        test_dates = pd.date_range(start=start_date, end=datetime.now(), freq='7D')

        signal_history = []

        for test_date in test_dates[-10:]:  # Last 10 periods
            try:
                # Get data for this date
                df = yf.download(symbol, start=test_date - timedelta(days=60),
                                 end=test_date + timedelta(days=1), progress=False)

                if df.empty:
                    continue

                indicators = self.calculate_quick_indicators(df, test_date)
                if indicators:
                    signals = self.analyze_signals_quickly(indicators)
                    signal_history.append({
                        'date': test_date,
                        'final_score': signals['final_score'],
                        'trend': signals['trend_score'],
                        'momentum': signals['momentum_score'],
                        'volume': signals['volume_score']
                    })

            except Exception as e:
                print(f"   ⚠️ Error for {test_date.date()}: {e}")

        if signal_history:
            df_signals = pd.DataFrame(signal_history)

            print(f"📈 Signal Analysis for {symbol} (Last 10 periods):")
            print(f"   Average Final Score: {df_signals['final_score'].mean():.2f}")
            print(f"   Score Range: {df_signals['final_score'].min():.2f} to {df_signals['final_score'].max():.2f}")
            print(f"   Strong Signals (>1.0): {len(df_signals[df_signals['final_score'] > 1.0])}/10")
            print(f"   Weak Signals (<0.5): {len(df_signals[abs(df_signals['final_score']) < 0.5])}/10")

            # Identify the issue
            if df_signals['final_score'].max() < 1.0:
                print("   🚨 ISSUE FOUND: All signals are weak (<1.0)")
                self.debug_results.append("WEAK_SIGNALS: All recent signals below action thresholds")

                # Check individual components
                if df_signals['trend'].mean() < 0.5:
                    print("   📉 Trend signals are consistently weak")
                if df_signals['momentum'].mean() < 0.5:
                    print("   🔻 Momentum signals are consistently weak")
                if df_signals['volume'].mean() < 0.5:
                    print("   📊 Volume confirmation is weak")

        else:
            print("   ❌ No signal data available for analysis")
            self.debug_results.append("NO_DATA: Unable to retrieve signal data")

    def analyze_strategy_thresholds(self):
        """
        Check if strategy thresholds are too restrictive
        """
        print("\n2. 🎯 STRATEGY THRESHOLD ANALYSIS")
        print("-" * 40)

        strategy_settings = getattr(self.advisor, 'strategy_settings', {})
        current_strategy = getattr(self.advisor, 'current_strategy', 'Unknown')

        print(f"Current Strategy: {current_strategy}")
        print(f"Strategy Settings: {strategy_settings}")

        # Define thresholds based on current strategy
        if current_strategy == "Conservative":
            expected_buy_threshold = 1.8
            expected_confidence_req = 85
        elif current_strategy == "Aggressive":
            expected_buy_threshold = 0.6
            expected_confidence_req = 60
        elif current_strategy == "Swing Trading":
            expected_buy_threshold = 0.9
            expected_confidence_req = 70
        else:  # Balanced
            expected_buy_threshold = 1.2
            expected_confidence_req = 75

        print(f"Expected BUY threshold: {expected_buy_threshold}")
        print(f"Expected confidence requirement: {expected_confidence_req}%")

        # Check if thresholds are too restrictive
        if expected_buy_threshold >= 1.5:
            print("   ⚠️ POTENTIAL ISSUE: BUY threshold might be too high")
            print("   💡 SUGGESTION: Try 'Aggressive' or 'Swing Trading' strategy")
            self.debug_results.append("HIGH_THRESHOLD: BUY threshold may be too restrictive")

        if expected_confidence_req >= 80:
            print("   ⚠️ POTENTIAL ISSUE: Confidence requirement might be too high")
            print("   💡 SUGGESTION: Lower confidence requirements or use different strategy")
            self.debug_results.append("HIGH_CONFIDENCE_REQ: Confidence requirement may be too high")

    def check_data_quality_issues(self, symbol, start_date):
        """
        Check for data quality issues that might affect signals
        """
        print("\n3. 📋 DATA QUALITY CHECK")
        print("-" * 40)

        try:
            # Get recent data
            end_date = datetime.now()
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)

            if df.empty:
                print("   ❌ CRITICAL: No data available")
                self.debug_results.append("NO_DATA: No price data available")
                return

            print(f"   ✅ Data Range: {df.index[0].date()} to {df.index[-1].date()}")
            print(f"   ✅ Total Days: {len(df)}")

            # Check for data gaps
            expected_days = (end_date - pd.Timestamp(start_date)).days
            actual_days = len(df)
            gap_percentage = (expected_days - actual_days) / expected_days * 100

            if gap_percentage > 30:
                print(f"   ⚠️ WARNING: {gap_percentage:.1f}% data missing")
                self.debug_results.append(f"DATA_GAPS: {gap_percentage:.1f}% missing data")

            # Check volume data
            zero_volume_days = (df['Volume'] == 0).sum()
            if zero_volume_days > 0:
                print(f"   ⚠️ WARNING: {zero_volume_days} days with zero volume")
                self.debug_results.append(f"VOLUME_ISSUES: {zero_volume_days} zero volume days")

            # Check price volatility
            returns = df['Close'].pct_change().dropna()
            avg_volatility = returns.std() * 100

            print(f"   📊 Average Volatility: {avg_volatility:.2f}%")

            if avg_volatility < 1.0:
                print("   ⚠️ WARNING: Very low volatility might reduce signal strength")
                self.debug_results.append("LOW_VOLATILITY: Price movements too small for strong signals")

        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            self.debug_results.append(f"DATA_ERROR: {str(e)}")

    def test_different_time_periods(self, symbol):
        """
        Test algorithm performance across different time periods
        """
        print("\n4. ⏰ TIME PERIOD TESTING")
        print("-" * 40)

        test_periods = [1, 3, 7, 14, 30, 60]
        results = []

        for days in test_periods:
            try:
                # Temporarily change investment days
                original_days = self.advisor.investment_days
                self.advisor.investment_days = days

                # Run quick analysis
                test_date = datetime.now().date()
                result = self.run_quick_analysis(symbol, test_date)

                if result:
                    results.append({
                        'days': days,
                        'action': result.get('action', 'UNKNOWN'),
                        'confidence': result.get('confidence', 0),
                        'score': result.get('final_score', 0)
                    })

                # Restore original days
                self.advisor.investment_days = original_days

            except Exception as e:
                print(f"   ⚠️ Error testing {days} days: {e}")

        if results:
            print("   Time Period Analysis:")
            for r in results:
                print(
                    f"   {r['days']:2d} days: {r['action']:12} (Score: {r['score']:5.2f}, Conf: {r['confidence']:4.1f}%)")

            # Check if any timeframe gives better results
            buy_signals = [r for r in results if r['action'] == 'BUY']
            if buy_signals:
                best_period = max(buy_signals, key=lambda x: x['confidence'])
                print(f"   💡 SUGGESTION: Try {best_period['days']} day timeframe for better results")
                self.debug_results.append(
                    f"BETTER_TIMEFRAME: {best_period['days']} days shows {best_period['action']} at {best_period['confidence']:.1f}%")
            else:
                print("   ⚠️ No BUY signals found across any timeframes")
                self.debug_results.append("NO_BUY_SIGNALS: No timeframe produces BUY recommendations")

    def check_model_availability(self, symbol):
        """
        Check ML model availability and performance
        """
        print("\n5. 🤖 MODEL AVAILABILITY CHECK")
        print("-" * 40)

        if hasattr(self.advisor, 'models') and symbol in self.advisor.models:
            print(f"   ✅ ML Model available for {symbol}")

            # Test model prediction
            try:
                model = self.advisor.models[symbol]
                test_features = [1.2, 2.0, 45, 0.05, 0.3, 100, 1.0, 1, 0]  # Mock features
                prediction = model.predict([test_features])[0]

                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba([test_features])[0]
                    confidence = max(proba)
                    print(f"   🤖 Model Test: Prediction={prediction}, Confidence={confidence:.1%}")

                    if confidence < 0.7:
                        print("   ⚠️ WARNING: Model confidence is low")
                        self.debug_results.append("LOW_MODEL_CONFIDENCE: ML model predictions are uncertain")
                else:
                    print(f"   🤖 Model Test: Prediction={prediction}")

            except Exception as e:
                print(f"   ❌ Model Error: {e}")
                self.debug_results.append(f"MODEL_ERROR: {str(e)}")
        else:
            print(f"   ⚠️ No ML model available for {symbol}")
            print("   💡 Using technical analysis only")
            self.debug_results.append("NO_MODEL: Only technical analysis available")

    def calculate_quick_indicators(self, df, target_date):
        """
        Quick indicator calculation for debugging
        """
        try:
            historical_data = df[df.index <= target_date].copy()
            if len(historical_data) < 20:
                return None

            current_price = historical_data['Close'].iloc[-1]

            indicators = {
                'current_price': current_price,
                'sma_20': historical_data['Close'].rolling(20).mean().iloc[-1],
                'rsi_14': ta.momentum.RSIIndicator(historical_data['Close'], window=14).rsi().iloc[-1],
                'volume_relative': historical_data['Volume'].iloc[-1] /
                                   historical_data['Volume'].rolling(20).mean().iloc[-1],
                'volatility': historical_data['Close'].pct_change().std() * 100
            }

            # Handle NaN values
            for key, value in indicators.items():
                if pd.isna(value):
                    if 'price' in key:
                        indicators[key] = current_price
                    elif 'rsi' in key:
                        indicators[key] = 50
                    else:
                        indicators[key] = 1.0

            return indicators

        except Exception as e:
            print(f"Error calculating indicators: {e}")
            return None

    def analyze_signals_quickly(self, indicators):
        """
        Quick signal analysis for debugging
        """
        current_price = indicators['current_price']
        sma_20 = indicators['sma_20']
        rsi_14 = indicators['rsi_14']
        volume_relative = indicators['volume_relative']

        # Trend score
        trend_score = 2 if current_price > sma_20 * 1.02 else -2 if current_price < sma_20 * 0.98 else 0

        # Momentum score
        momentum_score = 2 if rsi_14 < 30 else -2 if rsi_14 > 70 else 0

        # Volume score
        volume_score = 1 if volume_relative > 1.5 else 0

        final_score = trend_score + momentum_score + volume_score

        return {
            'final_score': final_score,
            'trend_score': trend_score,
            'momentum_score': momentum_score,
            'volume_score': volume_score
        }

    def run_quick_analysis(self, symbol, target_date):
        """
        Run quick analysis for testing
        """
        try:
            df = self.advisor.get_stock_data(symbol, target_date, days_back=60)
            if df is None or df.empty:
                return None

            indicators = self.advisor.calculate_enhanced_indicators(df, pd.Timestamp(target_date))
            if indicators is None:
                return None

            result = self.advisor.generate_enhanced_recommendation(indicators, symbol)
            return result

        except Exception as e:
            return None


# SOLUTION IMPLEMENTATIONS
def fix_weak_signals_issue(advisor):
    """
    Fix 1: Adjust signal sensitivity for better detection
    """
    print("🔧 FIXING WEAK SIGNALS ISSUE")
    print("-" * 40)

    # Store original analyze_trend method
    original_analyze_trend = advisor.analyze_trend

    def enhanced_analyze_trend(indicators, current_price):
        """Enhanced trend analysis with more sensitive scoring"""
        score = 0
        signals = []

        sma_5 = indicators.get('sma_5', current_price)
        sma_10 = indicators.get('sma_10', current_price)
        sma_20 = indicators.get('sma_20', current_price)
        sma_50 = indicators.get('sma_50', current_price)

        # ENHANCED: More granular and sensitive trend scoring
        if current_price > sma_5 > sma_10 > sma_20:
            score += 3.5  # Increased from 3
            signals.append("📈 Strong SMA uptrend")
        elif current_price > sma_5 > sma_10:
            score += 2.8  # New case
            signals.append("📈 Good SMA uptrend")
        elif current_price > sma_10 > sma_20:
            score += 2.2  # New case
            signals.append("📈 Moderate SMA uptrend")
        elif current_price > sma_20:
            score += 1.8  # Increased from 1.5
            signals.append("📈 Price above SMA20")
        elif current_price > sma_50:
            score += 1.2  # New positive case
            signals.append("📈 Price above SMA50")
        elif current_price < sma_5 < sma_10 < sma_20:
            score -= 3.5
            signals.append("📉 Strong SMA downtrend")
        else:
            score -= 0.8  # Reduced penalty from -1
            signals.append("📉 Price below key SMAs")

        # EMA confirmation (enhanced)
        ema_12 = indicators.get('ema_12', current_price)
        ema_26 = indicators.get('ema_26', current_price)

        if ema_12 > ema_26:
            score += 1.2  # Increased from 1
            signals.append("🔄 Bullish EMA crossover")
        else:
            score -= 0.8  # Reduced penalty from -1
            signals.append("🔄 Bearish EMA crossover")

        return score, signals

    # Replace the method
    advisor.analyze_trend = enhanced_analyze_trend
    print("✅ Enhanced trend analysis applied")
    return advisor


def fix_strategy_thresholds(advisor, strategy_type="Aggressive"):
    """
    Fix 2: Apply more aggressive strategy thresholds
    """
    print(f"🔧 APPLYING {strategy_type.upper()} STRATEGY")
    print("-" * 40)

    strategy_multipliers = {
        "Conservative": {"profit": 0.8, "risk": 0.8, "confidence_req": 85},
        "Balanced": {"profit": 1.0, "risk": 1.0, "confidence_req": 75},
        "Aggressive": {"profit": 1.4, "risk": 1.3, "confidence_req": 60},  # Lower confidence requirement
        "Swing Trading": {"profit": 1.8, "risk": 1.5, "confidence_req": 65},  # Lower confidence requirement
        "Ultra Aggressive": {"profit": 2.0, "risk": 1.8, "confidence_req": 55}  # New ultra-aggressive option
    }

    if strategy_type not in strategy_multipliers:
        strategy_type = "Aggressive"

    advisor.strategy_settings = strategy_multipliers[strategy_type]
    advisor.current_strategy = strategy_type

    print(f"✅ {strategy_type} strategy applied:")
    print(f"   Profit Multiplier: {advisor.strategy_settings['profit']:.1f}x")
    print(f"   Risk Multiplier: {advisor.strategy_settings['risk']:.1f}x")
    print(f"   Confidence Requirement: {advisor.strategy_settings['confidence_req']}%")

    return advisor


def fix_date_parsing_issue(date_input):
    """
    Fix 3: Proper date parsing
    """
    print("🔧 FIXING DATE PARSING")
    print("-" * 40)

    try:
        # Handle various date formats
        if '/' in date_input:
            parts = date_input.strip().split('/')
            if len(parts) == 3:
                month, day, year = parts
                # Handle 2-digit years
                if len(year) == 2:
                    year_int = int(year)
                    if year_int < 50:
                        year = "20" + year
                    else:
                        year = "19" + year

                # Create proper date
                target_date = datetime(int(year), int(month), int(day)).date()
                print(f"✅ Parsed date: {target_date}")
                return target_date

        # If parsing fails, use today
        target_date = datetime.now().date()
        print(f"⚠️ Using today's date: {target_date}")
        return target_date

    except Exception as e:
        print(f"❌ Date parsing error: {e}")
        target_date = datetime.now().date()
        print(f"⚠️ Using today's date: {target_date}")
        return target_date


def apply_enhanced_confidence_calculation(advisor):
    """
    Fix 4: More lenient confidence calculation
    """
    print("🔧 APPLYING ENHANCED CONFIDENCE CALCULATION")
    print("-" * 40)

    original_calculate_enhanced_confidence = advisor.calculate_enhanced_confidence

    def lenient_confidence_calculation(indicators, final_score, strategy_settings, investment_days):
        """More lenient confidence calculation"""

        # Start with higher base confidence
        base_confidence = 65.0  # Increased from typical 50-60

        # More generous signal strength bonuses
        if abs(final_score) >= 2.5:
            signal_boost = 20.0  # Increased
        elif abs(final_score) >= 2.0:
            signal_boost = 16.0  # Increased
        elif abs(final_score) >= 1.5:
            signal_boost = 12.0  # Increased
        elif abs(final_score) >= 1.0:
            signal_boost = 8.0  # Increased
        else:
            signal_boost = 4.0  # Increased

        # Technical confirmation boost (more lenient)
        rsi_14 = indicators.get('rsi_14', 50)
        volume_relative = indicators.get('volume_relative', 1.0)
        volatility = indicators.get('volatility', 2.0)

        tech_boost = 0
        if 25 <= rsi_14 <= 50:  # Broader range
            tech_boost += 8
        if volume_relative > 1.2:  # Lower threshold
            tech_boost += 6
        if 1.5 <= volatility <= 4.0:  # Broader acceptable range
            tech_boost += 4

        # Strategy-specific adjustments (more lenient)
        strategy_type = getattr(advisor, 'current_strategy', 'Balanced')
        if strategy_type in ["Aggressive", "Ultra Aggressive"]:
            base_confidence += 10  # Bonus for aggressive strategies
            signal_boost *= 1.2  # Multiply signal boost

        # Calculate final confidence
        final_confidence = base_confidence + signal_boost + tech_boost

        # Apply strategy bounds (more lenient)
        if strategy_type == "Conservative":
            min_conf, max_conf = 70, 95  # Lowered minimum
        elif strategy_type in ["Aggressive", "Ultra Aggressive"]:
            min_conf, max_conf = 55, 92  # Lowered minimum
        else:
            min_conf, max_conf = 60, 93  # Lowered minimum

        final_confidence = max(min_conf, min(final_confidence, max_conf))

        return final_confidence

    advisor.calculate_enhanced_confidence = lenient_confidence_calculation
    print("✅ Enhanced confidence calculation applied")
    return advisor


# TEST FUNCTION
def test_fixes_comprehensive(advisor, symbol="NVDA"):
    """
    Test all fixes comprehensively
    """
    print("🧪 COMPREHENSIVE TESTING OF FIXES")
    print("=" * 60)

    # Test different scenarios
    test_scenarios = [
        {"strategy": "Aggressive", "days": 7, "expected": "More BUY signals"},
        {"strategy": "Ultra Aggressive", "days": 14, "expected": "Maximum BUY sensitivity"},
        {"strategy": "Swing Trading", "days": 21, "expected": "Medium-term BUY signals"},
    ]

    for scenario in test_scenarios:
        print(f"\n📊 Testing {scenario['strategy']} + {scenario['days']} days:")

        # Apply fixes
        advisor = fix_strategy_thresholds(advisor, scenario['strategy'])
        advisor = fix_weak_signals_issue(advisor)
        advisor = apply_enhanced_confidence_calculation(advisor)
        advisor.investment_days = scenario['days']

        # Run analysis
        try:
            target_date = datetime.now().date()
            result = advisor.analyze_stock_enhanced(symbol, target_date)

            if result:
                action = result.get('action', 'UNKNOWN')
                confidence = result.get('confidence', 0)
                score = result.get('final_score', 0)

                print(f"   Result: {action} (Score: {score:.2f}, Confidence: {confidence:.1f}%)")

                if action == "BUY":
                    print(f"   ✅ SUCCESS: {scenario['expected']}")
                else:
                    print(f"   ⚠️ Still {action}: May need further adjustment")
            else:
                print("   ❌ Analysis failed")

        except Exception as e:
            print(f"   ❌ Error: {e}")


print("🚀 Trading Algorithm Debug and Fix System Ready!")
print("Use the functions above to diagnose and fix your WAIT signal issues.")


# Add this debug section to your Streamlit interface

def add_debug_section_to_streamlit():
    """
    Add comprehensive debug section to your Streamlit app
    """

    # Add this in your sidebar after the main controls
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 Debug & Fix Tools")

    # Quick fix buttons
    if st.sidebar.button("🚀 Apply Aggressive Strategy"):
        st.session_state.enhanced_advisor = fix_strategy_thresholds(
            st.session_state.enhanced_advisor, "Aggressive"
        )
        st.sidebar.success("✅ Aggressive strategy applied!")

    if st.sidebar.button("⚡ Apply Ultra Aggressive"):
        st.session_state.enhanced_advisor = fix_strategy_thresholds(
            st.session_state.enhanced_advisor, "Ultra Aggressive"
        )
        st.sidebar.success("✅ Ultra Aggressive strategy applied!")

    if st.sidebar.button("🔧 Fix Signal Sensitivity"):
        st.session_state.enhanced_advisor = fix_weak_signals_issue(
            st.session_state.enhanced_advisor
        )
        st.sidebar.success("✅ Enhanced signal sensitivity applied!")

    # Signal threshold override
    st.sidebar.markdown("#### Manual Threshold Override")
    override_buy_threshold = st.sidebar.slider(
        "BUY Threshold Override",
        min_value=0.1,
        max_value=2.0,
        value=1.2,
        step=0.1,
        help="Lower = more BUY signals"
    )

    override_confidence = st.sidebar.slider(
        "Min Confidence Override",
        min_value=50,
        max_value=95,
        value=75,
        help="Lower = more signals accepted"
    )

    # Apply overrides
    if st.sidebar.button("Apply Overrides"):
        advisor = st.session_state.enhanced_advisor

        # Create custom strategy with overrides
        advisor.strategy_settings = {
            "profit": 1.4,  # Aggressive multiplier
            "risk": 1.3,
            "confidence_req": override_confidence
        }
        advisor.current_strategy = f"Custom (BUY≥{override_buy_threshold})"

        # Override the recommendation method to use custom thresholds
        original_method = advisor.generate_enhanced_recommendation

        def custom_recommendation(indicators, symbol):
            result = original_method(indicators, symbol)

            # Apply custom threshold logic
            final_score = result.get('final_score', 0)
            confidence = result.get('confidence', 0)

            if final_score >= override_buy_threshold and confidence >= override_confidence:
                result['action'] = 'BUY'
                result['reasons'].append(f"🔧 Custom threshold: {final_score:.2f} ≥ {override_buy_threshold}")
            elif final_score <= -override_buy_threshold and confidence >= override_confidence:
                result['action'] = 'SELL/AVOID'
            else:
                result['action'] = 'WAIT'

            return result

        advisor.generate_enhanced_recommendation = custom_recommendation
        st.sidebar.success("✅ Custom thresholds applied!")


def add_signal_analysis_section():
    """
    Add detailed signal analysis to main interface
    """
    if 'last_analysis_result' in st.session_state:
        result = st.session_state.last_analysis_result

        with st.expander("🔍 Advanced Signal Debugging", expanded=False):
            st.markdown("### 📊 Signal Component Analysis")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Individual Signal Scores:**")
                signal_breakdown = result.get('signal_breakdown', {})

                for signal_type, score in signal_breakdown.items():
                    # Color code based on strength
                    if abs(score) >= 2.0:
                        color = "🟢" if score > 0 else "🔴"
                        strength = "STRONG"
                    elif abs(score) >= 1.0:
                        color = "🟡" if score > 0 else "🟠"
                        strength = "MODERATE"
                    else:
                        color = "⚪"
                        strength = "WEAK"

                    st.write(f"{color} {signal_type.replace('_', ' ').title()}: {score:.2f} ({strength})")

            with col2:
                st.markdown("**Signal Conflicts Analysis:**")
                signals = signal_breakdown.values() if signal_breakdown else []

                positive_signals = [s for s in signals if s > 0]
                negative_signals = [s for s in signals if s < 0]

                st.write(f"✅ Positive signals: {len(positive_signals)}")
                st.write(f"❌ Negative signals: {len(negative_signals)}")

                if len(positive_signals) > 0 and len(negative_signals) > 0:
                    st.warning("⚠️ **SIGNAL CONFLICT DETECTED**")
                    st.write("This explains the WAIT recommendation")

                    # Suggest solutions
                    st.markdown("**Suggested Solutions:**")
                    st.write("• Try Aggressive strategy (lower thresholds)")
                    st.write("• Wait for clearer directional signals")
                    st.write("• Analyze different timeframe")
                elif len(positive_signals) > len(negative_signals):
                    st.info("📈 **BULLISH BIAS** but signals may be weak")
                elif len(negative_signals) > len(positive_signals):
                    st.info("📉 **BEARISH BIAS** but signals may be weak")

            with col3:
                st.markdown("**Threshold Analysis:**")
                final_score = result.get('final_score', 0)
                confidence = result.get('confidence', 0)

                # Show current strategy requirements
                advisor = st.session_state.enhanced_advisor
                strategy_type = getattr(advisor, 'current_strategy', 'Unknown')

                # Define thresholds based on strategy
                if strategy_type == "Conservative":
                    buy_threshold = 1.8
                    confidence_req = 85
                elif strategy_type == "Aggressive":
                    buy_threshold = 0.6
                    confidence_req = 60
                elif strategy_type == "Swing Trading":
                    buy_threshold = 0.9
                    confidence_req = 70
                else:  # Balanced
                    buy_threshold = 1.2
                    confidence_req = 75

                st.write(f"**Current Score:** {final_score:.2f}")
                st.write(f"**Required for BUY:** ≥{buy_threshold}")

                if final_score >= buy_threshold:
                    st.success("✅ Score meets BUY threshold")
                else:
                    gap = buy_threshold - final_score
                    st.error(f"❌ Need +{gap:.2f} more for BUY")

                st.write(f"**Current Confidence:** {confidence:.1f}%")
                st.write(f"**Required Confidence:** ≥{confidence_req}%")

                if confidence >= confidence_req:
                    st.success("✅ Confidence meets requirement")
                else:
                    gap = confidence_req - confidence
                    st.error(f"❌ Need +{gap:.1f}% more confidence")


def add_historical_signal_analysis():
    """
    Add historical signal strength analysis
    """
    st.markdown("### 📈 Historical Signal Strength Analysis")

    if st.button("🔍 Analyze Signal History (Last 30 Days)"):
        advisor = st.session_state.enhanced_advisor
        stock_symbol = st.session_state.get('current_symbol', 'NVDA')

        with st.spinner("Analyzing historical signals..."):
            # Analyze signals for past 30 days
            end_date = datetime.now()
            dates_to_test = []

            # Generate test dates (every 3 days for past 30 days)
            for i in range(10):
                test_date = (end_date - timedelta(days=i * 3)).date()
                dates_to_test.append(test_date)

            historical_results = []

            for test_date in dates_to_test:
                try:
                    result = advisor.analyze_stock_enhanced(stock_symbol, test_date)
                    if result:
                        historical_results.append({
                            'date': test_date,
                            'action': result.get('action', 'UNKNOWN'),
                            'confidence': result.get('confidence', 0),
                            'final_score': result.get('final_score', 0),
                            'signal_breakdown': result.get('signal_breakdown', {})
                        })
                except Exception as e:
                    st.write(f"⚠️ Error analyzing {test_date}: {e}")

            if historical_results:
                # Create DataFrame for analysis
                df_history = pd.DataFrame(historical_results)

                # Display results
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**📊 Signal Distribution:**")
                    action_counts = df_history['action'].value_counts()

                    for action, count in action_counts.items():
                        percentage = (count / len(df_history)) * 100
                        st.write(f"• {action}: {count}/10 ({percentage:.1f}%)")

                    # Average scores
                    avg_score = df_history['final_score'].mean()
                    avg_confidence = df_history['confidence'].mean()

                    st.write(f"**Average Score:** {avg_score:.2f}")
                    st.write(f"**Average Confidence:** {avg_confidence:.1f}%")

                with col2:
                    st.markdown("**🎯 Insights & Recommendations:**")

                    buy_signals = len(df_history[df_history['action'] == 'BUY'])
                    wait_signals = len(df_history[df_history['action'] == 'WAIT'])

                    if buy_signals == 0:
                        st.error("❌ **NO BUY SIGNALS** in past 30 days")
                        st.markdown("**Recommended Actions:**")
                        st.write("• Switch to Aggressive strategy")
                        st.write("• Try different stocks (AAPL, MSFT, GOOGL)")
                        st.write("• Extend analysis period to 14-21 days")
                    elif buy_signals <= 2:
                        st.warning("⚠️ **FEW BUY SIGNALS** detected")
                        st.write("• Consider Swing Trading strategy")
                        st.write("• Look for stocks with stronger trends")
                    else:
                        st.success("✅ **GOOD SIGNAL FREQUENCY**")
                        st.write("• Current settings appear appropriate")

                # Show detailed history table
                st.markdown("**📋 Detailed History:**")
                display_df = df_history[['date', 'action', 'final_score', 'confidence']].copy()
                display_df['final_score'] = display_df['final_score'].round(2)
                display_df['confidence'] = display_df['confidence'].round(1)
                st.dataframe(display_df, hide_index=True)
            else:
                st.error("❌ Could not retrieve historical data for analysis")


def add_quick_stock_comparison():
    """
    Add quick comparison with other popular stocks
    """
    st.markdown("### 🆚 Quick Stock Comparison")

    if st.button("🔍 Compare with Popular Stocks"):
        advisor = st.session_state.enhanced_advisor
        current_symbol = st.session_state.get('current_symbol', 'NVDA')

        # Popular stocks to compare
        comparison_stocks = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
        if current_symbol in comparison_stocks:
            comparison_stocks.remove(current_symbol)

        with st.spinner("Comparing stocks..."):
            comparison_results = []
            target_date = datetime.now().date()

            # Analyze each stock
            for symbol in comparison_stocks[:4]:  # Limit to 4 for speed
                try:
                    result = advisor.analyze_stock_enhanced(symbol, target_date)
                    if result:
                        comparison_results.append({
                            'Symbol': symbol,
                            'Action': result.get('action', 'UNKNOWN'),
                            'Confidence': f"{result.get('confidence', 0):.1f}%",
                            'Score': f"{result.get('final_score', 0):.2f}",
                            'Expected Profit': f"{result.get('expected_profit_pct', 0):.1f}%"
                        })
                except Exception as e:
                    comparison_results.append({
                        'Symbol': symbol,
                        'Action': 'ERROR',
                        'Confidence': '0%',
                        'Score': '0.00',
                        'Expected Profit': '0%'
                    })

            if comparison_results:
                # Add current symbol for comparison
                if 'last_analysis_result' in st.session_state:
                    current_result = st.session_state.last_analysis_result
                    comparison_results.insert(0, {
                        'Symbol': f"{current_symbol} (Current)",
                        'Action': current_result.get('action', 'UNKNOWN'),
                        'Confidence': f"{current_result.get('confidence', 0):.1f}%",
                        'Score': f"{current_result.get('final_score', 0):.2f}",
                        'Expected Profit': f"{current_result.get('expected_profit_pct', 0):.1f}%"
                    })

                # Display comparison table
                df_comparison = pd.DataFrame(comparison_results)
                st.dataframe(df_comparison, hide_index=True)

                # Highlight better opportunities
                buy_stocks = [r for r in comparison_results if r['Action'] == 'BUY']
                if buy_stocks:
                    st.success(f"✅ **BUY opportunities found:** {', '.join([s['Symbol'] for s in buy_stocks])}")
                else:
                    st.info("ℹ️ No strong BUY signals in comparison stocks either - market may be in neutral phase")


# Integration function
def integrate_debug_features():
    """
    Complete integration of all debug features into your Streamlit app
    """
    # Add to your main Streamlit file after the analyze button section:

    # Store result for debugging
    if analyze_btn and result:
        st.session_state.last_analysis_result = result
        st.session_state.current_symbol = stock_symbol

    # Add debug sections
    add_signal_analysis_section()
    add_historical_signal_analysis()
    add_quick_stock_comparison()


# Add this to your sidebar
add_debug_section_to_streamlit()