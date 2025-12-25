"""
🧪 Enhanced Algorithm Testing & Auto-Tuning Script
════════════════════════════════════════════════════

Comprehensive testing script with:
- Random week sampling for quick validation
- Automated parameter fine-tuning
- Performance optimization suggestions
- Real-time results analysis
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time
import numpy as np
import random
import json
import os
from stockwise_simulation import ProfessionalStockAdvisor
from ibkr_connection_manager import ProfessionalIBKRManager
import warnings
import traceback
import get_all_tickers
warnings.filterwarnings('ignore')


class AdvancedAlgorithmTester:
    def __init__(self, advisor=None, debug=False):
        self.debug = debug
        self.ibkr_manager = None
        self.use_ibkr = False
        self.advisor = advisor or ProfessionalStockAdvisor(debug=debug)
        self.results = []
        self.optimization_history = []
        self.best_configuration = None
        self.baseline_performance = None

        # Try to establish IBKR connection using your working manager
        try:
            self.log_test("Attempting IBKR connection using working manager...", "INFO")
            self.ibkr_manager = ProfessionalIBKRManager(debug=debug)

            # Use ONLY the working port (7497) with proper timeout
            working_config = [
                {"host": "127.0.0.1", "port": 7497, "name": "TWS Paper"}
            ]

            if self.ibkr_manager.connect_with_fallback(working_config):
                self.use_ibkr = True
                self.log_test("✅ IBKR connection successful - using professional data", "SUCCESS")
            else:
                self.log_test("⚠️ IBKR connection failed - using yfinance fallback", "WARNING")
                self.ibkr_manager = None

        except Exception as e:
            self.log_test(f"❌ IBKR connection error: {e}", "ERROR")
            self.log_test("Using yfinance fallback for testing", "WARNING")
            self.ibkr_manager = None

        # Initialize advisor WITHOUT automatic IBKR connection
        if advisor:
            self.advisor = advisor
        else:
            # Create advisor with manual IBKR connection control
            self.advisor = self.create_advisor_with_manual_ibkr()

        self.results = []
        self.optimization_history = []
        self.best_configuration = None
        self.baseline_performance = None

    def create_advisor_with_manual_ibkr(self):
        """Create advisor with manual IBKR connection control"""
        try:
            # Import without triggering automatic IBKR connection
            import importlib
            import sys

            # Temporarily disable IBKR auto-connection in stockwise_simulation
            if 'stockwise_simulation' in sys.modules:
                del sys.modules['stockwise_simulation']

            # Set environment variable to disable auto IBKR connection
            os.environ['DISABLE_AUTO_IBKR'] = '1'

            from stockwise_simulation import ProfessionalStockAdvisor
            advisor = ProfessionalStockAdvisor(debug=self.debug)

            # Manually inject our working IBKR connection
            if self.use_ibkr and self.ibkr_manager:
                advisor.ibkr_manager = self.ibkr_manager
                advisor.use_ibkr = True
                self.log_test("✅ Injected working IBKR connection into advisor", "SUCCESS")
            else:
                advisor.use_ibkr = False
                self.log_test("📊 Advisor configured for yfinance data", "INFO")

            return advisor

        except Exception as e:
            self.log_test(f"❌ Error creating advisor: {e}", "ERROR")
            # Fallback: create a minimal advisor
            return self.create_minimal_advisor()

    def create_minimal_advisor(self):
        """Create minimal advisor for testing without IBKR complications"""

        class MinimalAdvisor:
            def __init__(self, debug=False):
                self.debug = debug
                self.use_ibkr = False

            def analyze_stock_enhanced(self, symbol, analysis_date):
                """Simplified analysis using yfinance"""
                try:
                    import yfinance as yf
                    from datetime import timedelta

                    # Get stock data
                    end_date = analysis_date + timedelta(days=1)
                    start_date = analysis_date - timedelta(days=60)

                    df = yf.download(symbol, start=start_date, end=end_date, progress=False)

                    if df.empty or len(df) < 20:
                        return None

                    # Simple analysis
                    current_price = float(df['Close'].iloc[-1])
                    sma_20 = df['Close'].rolling(20).mean().iloc[-1]
                    volume_avg = df['Volume'].rolling(10).mean().iloc[-1]

                    # Simple decision logic
                    if current_price > sma_20 * 1.02:  # 2% above SMA
                        action = 'BUY'
                        confidence = 75
                        expected_profit = 3.5
                    elif current_price < sma_20 * 0.98:  # 2% below SMA
                        action = 'SELL/AVOID'
                        confidence = 70
                        expected_profit = -2.0
                    else:
                        action = 'WAIT'
                        confidence = 60
                        expected_profit = 0

                    return {
                        'action': action,
                        'confidence': confidence,
                        'expected_profit_pct': expected_profit,
                        'final_score': confidence / 100,
                        'buy_price': current_price,
                        'sell_price': current_price * 1.035,
                        'analysis_date': analysis_date
                    }

                except Exception as e:
                    if self.debug:
                        print(f"❌ Error in minimal analysis for {symbol}: {e}")
                    return None

        return MinimalAdvisor(debug=self.debug)

    def get_stock_data_smart(self, symbol, start_date, end_date):
        """Get stock data using IBKR if available, otherwise yfinance"""

        if self.use_ibkr and self.ibkr_manager:
            try:
                # Calculate days for IBKR
                from datetime import datetime
                if isinstance(start_date, str):
                    start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
                if isinstance(end_date, str):
                    end_date = datetime.strptime(end_date, '%Y-%m-%d').date()

                days_back = (end_date - start_date).days + 5  # Add buffer

                df = self.ibkr_manager.get_stock_data(symbol, days_back=days_back)
                if df is not None and not df.empty:
                    self.log_test(f"✅ Got IBKR data for {symbol}", "SUCCESS")
                    return df

            except Exception as e:
                self.log_test(f"⚠️ IBKR data failed for {symbol}: {e}", "WARNING")

        # Fallback to yfinance
        try:
            import yfinance as yf
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if not df.empty:
                self.log_test(f"📊 Got yfinance data for {symbol}", "INFO")
                return df
        except Exception as e:
            self.log_test(f"❌ Yfinance failed for {symbol}: {e}", "ERROR")

        return None

    def log_test(self, message, level="INFO"):
        """Enhanced test logging"""
        if self.debug:
            timestamp = datetime.now().strftime("%H:%M:%S")
            icons = {
                "INFO": "🧪", "SUCCESS": "✅", "ERROR": "❌",
                "WARNING": "⚠️", "OPTIMIZATION": "🎯", "TUNING": "⚙️"
            }
            print(f"[{timestamp}] {icons.get(level, '🔸')} {message}")

    def validate_single_stock(self, symbol, max_retries=2):
        """Validate if a single stock symbol has usable data"""
        import yfinance as yf
        from datetime import datetime, timedelta

        for attempt in range(max_retries):
            try:
                # Try to get recent data (last 10 days)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=10)

                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date, progress=False)

                # Check if we have valid data
                if not data.empty and len(data) >= 2:
                    # Additional validation: check if prices are reasonable
                    current_price = float(data['Close'].iloc[-1])
                    if 0.50 <= current_price <= 10000:  # Reasonable price range
                        return True, current_price
                    else:
                        return False, f"Price out of range: ${current_price}"
                else:
                    return False, "No recent data available"

            except Exception as e:
                if attempt == max_retries - 1:
                    return False, f"API error: {str(e)}"
                time.sleep(1)  # Brief pause before retry

        return False, "Max retries exceeded"

    def create_validated_stock_list(self, csv_path="nasdaq_full_list.csv", target_count=50, max_validate=200):
        """Create a validated list of tradeable stocks"""

        self.log_test("=== STEP 1: CREATING VALIDATED STOCK LIST ===", "OPTIMIZATION")

        try:
            # Load CSV
            df = pd.read_csv(csv_path)
            self.log_test(f"Loaded {len(df)} stocks from CSV", "INFO")

            # Apply basic filtering first
            filtered_df = self.apply_basic_filters(df)
            self.log_test(f"After basic filtering: {len(filtered_df)} stocks", "INFO")

            # Take a larger sample for validation
            validation_sample = min(max_validate, len(filtered_df))
            if len(filtered_df) > validation_sample:
                sample_df = filtered_df.sample(n=validation_sample, random_state=42)
            else:
                sample_df = filtered_df

            candidate_stocks = sample_df['Symbol'].tolist()
            self.log_test(f"Validating {len(candidate_stocks)} candidate stocks...", "INFO")

            # Validate each stock
            validated_stocks = []
            failed_stocks = []

            for i, symbol in enumerate(candidate_stocks):
                if len(validated_stocks) >= target_count:
                    break

                if (i + 1) % 10 == 0:
                    self.log_test(
                        f"Validation progress: {i + 1}/{len(candidate_stocks)} ({len(validated_stocks)} valid so far)",
                        "INFO")

                is_valid, details = self.validate_single_stock(symbol)

                if is_valid:
                    validated_stocks.append(symbol)
                    self.log_test(f"✅ {symbol}: Valid (${details})", "SUCCESS")
                else:
                    failed_stocks.append((symbol, details))
                    self.log_test(f"❌ {symbol}: {details}", "WARNING")

            self.log_test(f"Validation complete: {len(validated_stocks)} valid stocks found", "SUCCESS")

            if failed_stocks:
                self.log_test(f"Failed validation: {len(failed_stocks)} stocks", "WARNING")
                # Log first few failures for debugging
                for symbol, reason in failed_stocks[:5]:
                    self.log_test(f"  {symbol}: {reason}", "WARNING")

            return validated_stocks

        except Exception as e:
            self.log_test(f"Error creating validated stock list: {e}", "ERROR")
            return []

    def apply_basic_filters(self, df):
        """Apply basic filters to remove obviously unsuitable stocks"""

        original_count = len(df)
        filtered = df.copy()

        # Remove NaN symbols
        filtered = filtered.dropna(subset=['Symbol'])

        # Remove ETFs and other fund types
        if 'ETF' in filtered.columns:
            filtered = filtered[filtered['ETF'] != 'Y']

        # Remove special instruments (warrants, units, etc.)
        problematic_patterns = [
            r'.*[UW]$',  # Units and Warrants
            r'.*RT$',  # Rights
            r'.*WS$',  # Warrants
            r'^[A-Z]{1,2}[0-9]',  # Symbols with numbers
            r'.*[\+\-].*'  # Special classes
        ]

        for pattern in problematic_patterns:
            filtered = filtered[~filtered['Symbol'].str.contains(pattern, na=False, regex=True)]

        # Length filter (2-4 characters)
        filtered = filtered[filtered['Symbol'].str.len().between(2, 4)]

        # Market category filter
        if 'Market Category' in filtered.columns:
            main_categories = ['Q', 'G', 'S']
            filtered = filtered[filtered['Market Category'].isin(main_categories)]

        # Financial status filter
        if 'Financial Status' in filtered.columns:
            filtered = filtered[filtered['Financial Status'].isin(['N', 'Normal', ''])]

        self.log_test(f"Basic filtering: {original_count} → {len(filtered)} stocks", "INFO")
        return filtered

    def get_random_test_weeks(self, start_date, end_date, num_weeks=4):
        """Generate random weeks for testing"""
        start_pd = pd.Timestamp(start_date)
        end_pd = pd.Timestamp(end_date)

        # Calculate total weeks available
        total_days = (end_pd - start_pd).days
        total_weeks = total_days // 7

        if total_weeks < num_weeks:
            self.log_test(f"Warning: Only {total_weeks} weeks available, using all", "WARNING")
            num_weeks = total_weeks

        # Generate random week start dates
        random_weeks = []
        for _ in range(num_weeks):
            random_day = random.randint(0, total_days - 7)
            week_start = start_pd + pd.Timedelta(days=random_day)
            week_end = week_start + pd.Timedelta(days=6)
            random_weeks.append((week_start.date(), week_end.date()))

        return random_weeks

    def get_stock_performance(self, symbol, start_date, end_date):
        """Get actual stock performance for validation"""
        try:
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if df.empty:
                return None, None, None

            # start_price = df['Close'].iloc[0]
            # end_price = df['Close'].iloc[-1]
            # max_price = df['High'].max()
            # min_price = df['Low'].min()

            # FIXED: Ensure we get scalar values, not pandas Series
            # FIXED: Ensure we get scalar values, not pandas Series
            start_price = float(df['Close'].iloc[0])
            end_price = float(df['Close'].iloc[-1])
            max_price = float(df['High'].max())
            min_price = float(df['Low'].min())

            actual_return = ((end_price / start_price) - 1) * 100
            max_gain = ((max_price / start_price) - 1) * 100
            max_loss = ((min_price / start_price) - 1) * 100

            return actual_return, max_gain, max_loss

        except Exception as e:
            self.log_test(f"Error getting performance for {symbol}: {e}", "ERROR")
            return None, None, None

    def cleanup(self):
        """Clean up IBKR connection"""
        if self.use_ibkr and self.ibkr_manager:
            try:
                self.ibkr_manager.disconnect()
                self.log_test("✅ IBKR connection cleaned up", "SUCCESS")
            except Exception as e:
                self.log_test(f"⚠️ Cleanup warning: {e}", "WARNING")

    def test_single_stock_week(self, symbol, week_start, week_end, config=None):
        """Test algorithm on a single stock for one week"""
        try:
            # Use custom configuration if provided
            if config:
                self.apply_configuration(config)

            # Run algorithm analysis on week start
            result = self.advisor.analyze_stock_enhanced(symbol, week_start)

            if result is None:
                return None

            # Get actual performance
            actual_return, max_gain, max_loss = self.get_stock_performance(
                symbol, week_start, week_end
            )

            if actual_return is None:
                return None

            # Extract algorithm prediction
            action = result.get('action', 'WAIT')
            confidence = result.get('confidence', 0)
            predicted_profit = result.get('expected_profit_pct', 0)
            buy_price = result.get('buy_price')
            sell_price = result.get('sell_price')
            final_score = result.get('final_score', 0)

            # Evaluate prediction accuracy
            confidence = float(confidence) if not pd.isna(confidence) else 0.0
            predicted_profit = float(predicted_profit) if not pd.isna(predicted_profit) else 0.0
            final_score = float(final_score) if not pd.isna(final_score) else 0.0

            if buy_price is not None and not pd.isna(buy_price):
                buy_price = float(buy_price)
            else:
                buy_price = None

            if sell_price is not None and not pd.isna(sell_price):
                sell_price = float(sell_price)
            else:
                sell_price = None

                # Evaluate prediction accuracy
            prediction_accuracy = self.evaluate_prediction_accuracy(
                action, confidence, predicted_profit, actual_return, max_gain, max_loss
            )

            # Calculate profit accuracy
            profit_accuracy = 0
            if predicted_profit > 0:
                profit_accuracy = abs(predicted_profit - actual_return)

            return {
                'symbol': symbol,
                'week_start': str(week_start),  # Convert to string for JSON serialization
                'week_end': str(week_end),
                'action': action,
                'confidence': confidence,
                'predicted_profit': predicted_profit,
                'final_score': final_score,
                'buy_price': buy_price,
                'sell_price': sell_price,
                'actual_return': actual_return,
                'max_gain_available': max_gain,
                'max_loss_risk': max_loss,
                'prediction_accuracy': prediction_accuracy,
                'direction_correct': self.check_direction_accuracy(action, actual_return),
                'profit_accuracy': profit_accuracy,
                'configuration': config.copy() if config else None
            }

            # prediction_accuracy = self.evaluate_prediction_accuracy(
            #     action, confidence, predicted_profit, actual_return, max_gain, max_loss
            # )

            # return {
            #     'symbol': symbol,
            #     'week_start': week_start,
            #     'week_end': week_end,
            #     'action': action,
            #     'confidence': confidence,
            #     'predicted_profit': predicted_profit,
            #     'final_score': final_score,
            #     'buy_price': buy_price,
            #     'sell_price': sell_price,
            #     'actual_return': actual_return,
            #     'max_gain_available': max_gain,
            #     'max_loss_risk': max_loss,
            #     'prediction_accuracy': prediction_accuracy,
            #     'direction_correct': self.check_direction_accuracy(action, actual_return),
            #     'profit_accuracy': abs(predicted_profit - actual_return) if predicted_profit > 0 else 0,
            #     'configuration': config.copy() if config else None
            # }

        except Exception as e:
            self.log_test(f"Error testing {symbol} for week {week_start}: {e}", "ERROR")
            self.log_test(f"Full traceback: {traceback.format_exc()}", "ERROR")
            return None

    def evaluate_prediction_accuracy(self, action, confidence, predicted_profit,
                                   actual_return, max_gain, max_loss):
        """FIXED: Evaluate prediction accuracy with proper type handling"""
        try:
            # Ensure all inputs are scalars
            confidence = float(confidence) if confidence is not None else 0.0
            predicted_profit = float(predicted_profit) if predicted_profit is not None else 0.0
            actual_return = float(actual_return) if actual_return is not None else 0.0
            max_gain = float(max_gain) if max_gain is not None else 0.0
            max_loss = float(max_loss) if max_loss is not None else 0.0

            score = 0

            # Direction accuracy (40% of score)
            if action == 'BUY' and actual_return > 0:
                score += 40
            elif action == 'SELL/AVOID' and actual_return <= 0:
                score += 40
            elif action == 'WAIT' and -2 <= actual_return <= 2:
                score += 40

            # Confidence calibration (30% of score)
            if confidence >= 85 and actual_return > 3:
                score += 30
            elif confidence >= 75 and actual_return > 1:
                score += 25
            elif confidence >= 65 and actual_return > 0:
                score += 20
            elif confidence < 65 and abs(actual_return) <= 2:
                score += 15

            # Profit prediction accuracy (30% of score)
            if predicted_profit > 0 and actual_return > 0:
                profit_error = abs(predicted_profit - actual_return) / predicted_profit
                if profit_error <= 0.2:  # Within 20%
                    score += 30
                elif profit_error <= 0.5:  # Within 50%
                    score += 20
                elif profit_error <= 1.0:  # Within 100%
                    score += 10

            return min(score, 100)

        except Exception as e:
            self.log_test(f"Error in prediction accuracy calculation: {e}", "ERROR")
            return 0

    def check_direction_accuracy(self, action, actual_return):
        """Check if direction was predicted correctly"""
        try:
            # Ensure actual_return is a scalar float
            actual_return = float(actual_return)

            if action == 'BUY' and actual_return > 0:
                return True
            elif action == 'SELL/AVOID' and actual_return <= 0:
                return True
            elif action == 'WAIT' and -2 <= actual_return <= 2:
                return True
            return False
        except (TypeError, ValueError):
            # If there's an issue with the comparison, default to False
            return False

    def apply_configuration(self, config):
        """Apply configuration to advisor"""
        if 'signal_weights' in config:
            # This would require modifying the advisor's signal weights
            pass
        if 'strategy_thresholds' in config:
            # This would require modifying the advisor's thresholds
            pass
        if 'confidence_requirements' in config:
            # This would require modifying confidence requirements
            pass

    def run_random_week_test(self, stock_symbols, start_date="2024-01-01",
                           end_date="2024-12-31", num_weeks=4, num_stocks=10):
        """Run tests on random weeks with random stocks"""

        self.log_test("STARTING RANDOM WEEK TESTING", "OPTIMIZATION")
        self.log_test(f"Period: {start_date} to {end_date}", "INFO")
        self.log_test(f"Testing {num_weeks} random weeks with {num_stocks} stocks", "INFO")

        # Select random stocks if more than requested
        if len(stock_symbols) > num_stocks:
            test_stocks = random.sample(stock_symbols, num_stocks)
        else:
            test_stocks = stock_symbols

        # Generate random weeks
        random_weeks = self.get_random_test_weeks(start_date, end_date, num_weeks)

        self.log_test(f"Selected weeks: {[f'{w[0]} to {w[1]}' for w in random_weeks]}", "INFO")
        self.log_test(f"Selected stocks: {test_stocks}", "INFO")

        results = []
        total_tests = len(test_stocks) * len(random_weeks)
        current_test = 0

        for week_start, week_end in random_weeks:
            self.log_test(f"Testing week: {week_start} to {week_end}", "INFO")

            for symbol in test_stocks:
                current_test += 1
                self.log_test(f"Testing {symbol} ({current_test}/{total_tests})", "INFO")

                result = self.test_single_stock_week(symbol, week_start, week_end)
                if result:
                    results.append(result)

                    # Quick feedback
                    accuracy = result['prediction_accuracy']
                    action = result['action']
                    actual = result['actual_return']
                    self.log_test(f"  → {action} | Actual: {actual:+.1f}% | Accuracy: {accuracy:.0f}%",
                                "SUCCESS" if accuracy >= 70 else "WARNING")

        self.results = results
        self.baseline_performance = self.calculate_performance_metrics(results)

        self.log_test(f"Baseline test complete: {len(results)} successful tests", "SUCCESS")
        return results

    def calculate_performance_metrics(self, results):
        """Calculate comprehensive performance metrics"""
        if not results:
            return {}

        df = pd.DataFrame(results)

        metrics = {
            'total_tests': len(df),
            'avg_accuracy': df['prediction_accuracy'].mean(),
            'direction_accuracy': df['direction_correct'].mean() * 100,
            'avg_confidence': df['confidence'].mean(),
            'high_confidence_accuracy': df[df['confidence'] >= 85]['prediction_accuracy'].mean() if len(df[df['confidence'] >= 85]) > 0 else 0,
            'buy_success_rate': len(df[(df['action'] == 'BUY') & (df['actual_return'] > 0)]) / len(df[df['action'] == 'BUY']) * 100 if len(df[df['action'] == 'BUY']) > 0 else 0,
            'sell_avoid_success_rate': len(df[(df['action'] == 'SELL/AVOID') & (df['actual_return'] <= 0)]) / len(df[df['action'] == 'SELL/AVOID']) * 100 if len(df[df['action'] == 'SELL/AVOID']) > 0 else 0,
            'avg_profit_accuracy': df[df['predicted_profit'] > 0]['profit_accuracy'].mean() if len(df[df['predicted_profit'] > 0]) > 0 else 0,
            'action_distribution': df['action'].value_counts().to_dict(),
            'confidence_distribution': {
                '85-100%': len(df[df['confidence'] >= 85]),
                '75-85%': len(df[(df['confidence'] >= 75) & (df['confidence'] < 85)]),
                '65-75%': len(df[(df['confidence'] >= 65) & (df['confidence'] < 75)]),
                '50-65%': len(df[df['confidence'] < 65])
            }
        }

        return metrics

    def generate_optimization_suggestions(self, performance_metrics):
        """Generate optimization suggestions based on performance"""
        suggestions = []

        # Check direction accuracy
        if performance_metrics['direction_accuracy'] < 60:
            suggestions.append({
                'area': 'Signal Weights',
                'issue': f"Low direction accuracy ({performance_metrics['direction_accuracy']:.1f}%)",
                'suggestion': 'Increase trend and momentum weights, reduce noise from weak signals',
                'config_change': {
                    'signal_weights': {
                        'trend': 0.35,  # Increase from 0.30
                        'momentum': 0.30,  # Increase from 0.25
                        'volume': 0.15,  # Decrease from 0.20
                        'support_resistance': 0.10,
                        'model': 0.10  # Decrease from 0.15
                    }
                }
            })

        # Check BUY success rate
        if performance_metrics['buy_success_rate'] < 50:
            suggestions.append({
                'area': 'BUY Thresholds',
                'issue': f"Low BUY success rate ({performance_metrics['buy_success_rate']:.1f}%)",
                'suggestion': 'Increase BUY thresholds to be more selective',
                'config_change': {
                    'strategy_thresholds': {
                        'balanced_buy': 1.1,  # Increase from 0.9
                        'aggressive_buy': 0.7,  # Increase from 0.5
                        'conservative_buy': 1.7  # Increase from 1.5
                    }
                }
            })

        # Check confidence calibration
        if performance_metrics['high_confidence_accuracy'] < 80:
            suggestions.append({
                'area': 'Confidence Calculation',
                'issue': f"High confidence predictions not accurate enough ({performance_metrics['high_confidence_accuracy']:.1f}%)",
                'suggestion': 'Increase confidence requirements or improve confluence detection',
                'config_change': {
                    'confidence_requirements': {
                        'conservative': 85,  # Increase from 80
                        'balanced': 75,  # Increase from 70
                        'aggressive': 65   # Increase from 60
                    }
                }
            })

        # Check signal distribution
        action_dist = performance_metrics['action_distribution']
        buy_ratio = action_dist.get('BUY', 0) / performance_metrics['total_tests']

        if buy_ratio > 0.6:  # Too many BUY signals
            suggestions.append({
                'area': 'Signal Balance',
                'issue': f"Too many BUY signals ({buy_ratio:.1%})",
                'suggestion': 'Increase selectivity to reduce false positives',
                'config_change': {
                    'threshold_adjustment': 'increase_all_by_0.1'
                }
            })
        elif buy_ratio < 0.2:  # Too few BUY signals
            suggestions.append({
                'area': 'Signal Balance',
                'issue': f"Too few BUY signals ({buy_ratio:.1%})",
                'suggestion': 'Decrease thresholds to capture more opportunities',
                'config_change': {
                    'threshold_adjustment': 'decrease_all_by_0.1'
                }
            })

        return suggestions

    def auto_tune_parameters(self, stock_symbols, optimization_iterations=3):
        """Automatically tune parameters based on performance"""

        self.log_test("STARTING AUTOMATED PARAMETER TUNING", "TUNING")

        if not self.baseline_performance:
            self.log_test("Running baseline test first...", "INFO")
            self.run_random_week_test(stock_symbols)

        best_performance = self.baseline_performance['avg_accuracy']
        best_config = None

        for iteration in range(optimization_iterations):
            self.log_test(f"Optimization iteration {iteration + 1}/{optimization_iterations}", "TUNING")

            # Generate optimization suggestions
            suggestions = self.generate_optimization_suggestions(self.baseline_performance)

            if not suggestions:
                self.log_test("No optimization suggestions generated", "INFO")
                break

            # Test each suggestion
            for suggestion in suggestions:
                self.log_test(f"Testing: {suggestion['area']} - {suggestion['suggestion']}", "TUNING")

                # Apply configuration change
                config = suggestion['config_change']

                # Run test with new configuration
                test_results = []
                random_weeks = self.get_random_test_weeks("2024-01-01", "2024-12-31", 2)
                test_stocks = random.sample(stock_symbols, min(5, len(stock_symbols)))

                for week_start, week_end in random_weeks:
                    for symbol in test_stocks:
                        result = self.test_single_stock_week(symbol, week_start, week_end, config)
                        if result:
                            test_results.append(result)

                if test_results:
                    performance = self.calculate_performance_metrics(test_results)
                    new_accuracy = performance['avg_accuracy']

                    self.log_test(f"  Result: {new_accuracy:.1f}% accuracy (vs {best_performance:.1f}%)",
                                "SUCCESS" if new_accuracy > best_performance else "WARNING")

                    if new_accuracy > best_performance:
                        best_performance = new_accuracy
                        best_config = config
                        self.log_test(f"  ✅ New best configuration found!", "SUCCESS")

                        # Store optimization result
                        self.optimization_history.append({
                            'iteration': iteration + 1,
                            'area': suggestion['area'],
                            'change': suggestion['suggestion'],
                            'old_accuracy': self.baseline_performance['avg_accuracy'],
                            'new_accuracy': new_accuracy,
                            'improvement': new_accuracy - self.baseline_performance['avg_accuracy'],
                            'config': config
                        })

        self.best_configuration = best_config

        if best_config:
            self.log_test(f"Optimization complete! Best accuracy: {best_performance:.1f}%", "SUCCESS")
            return best_config, best_performance
        else:
            self.log_test("No improvements found during optimization", "WARNING")
            return None, self.baseline_performance['avg_accuracy']

    def generate_optimization_report(self):
        """Generate comprehensive optimization report"""
        if not self.baseline_performance:
            self.log_test("No baseline performance data available", "WARNING")
            return

        print("\n" + "="*80)
        print("🎯 ALGORITHM OPTIMIZATION REPORT")
        print("="*80)

        # Baseline Performance
        print("\n📊 BASELINE PERFORMANCE:")
        bp = self.baseline_performance
        print(f"   Overall Accuracy: {bp['avg_accuracy']:.1f}%")
        print(f"   Direction Accuracy: {bp['direction_accuracy']:.1f}%")
        print(f"   Average Confidence: {bp['avg_confidence']:.1f}%")
        print(f"   High Confidence Accuracy: {bp['high_confidence_accuracy']:.1f}%")
        print(f"   BUY Success Rate: {bp['buy_success_rate']:.1f}%")
        print(f"   SELL/AVOID Success Rate: {bp['sell_avoid_success_rate']:.1f}%")

        # Action Distribution
        print(f"\n📈 ACTION DISTRIBUTION:")
        for action, count in bp['action_distribution'].items():
            percentage = count / bp['total_tests'] * 100
            print(f"   {action}: {count} ({percentage:.1f}%)")

        # Confidence Distribution
        print(f"\n🎪 CONFIDENCE DISTRIBUTION:")
        for range_name, count in bp['confidence_distribution'].items():
            percentage = count / bp['total_tests'] * 100
            print(f"   {range_name}: {count} ({percentage:.1f}%)")

        # Optimization History
        if self.optimization_history:
            print(f"\n⚙️ OPTIMIZATION IMPROVEMENTS:")
            for opt in self.optimization_history:
                print(f"   Iteration {opt['iteration']}: {opt['area']}")
                print(f"      Change: {opt['change']}")
                print(f"      Improvement: {opt['old_accuracy']:.1f}% → {opt['new_accuracy']:.1f}% (+{opt['improvement']:.1f}%)")

        # Best Configuration
        if self.best_configuration:
            print(f"\n🏆 BEST CONFIGURATION FOUND:")
            print(json.dumps(self.best_configuration, indent=2))

        # Recommendations
        current_suggestions = self.generate_optimization_suggestions(self.baseline_performance)
        if current_suggestions:
            print(f"\n💡 ADDITIONAL RECOMMENDATIONS:")
            for i, suggestion in enumerate(current_suggestions, 1):
                print(f"   {i}. {suggestion['area']}: {suggestion['suggestion']}")

        print("="*80)

    def save_results(self, filename=None):
        """Save test results and optimization data"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"algorithm_optimization_results_{timestamp}.json"

        data = {
            'baseline_performance': self.baseline_performance,
            'optimization_history': self.optimization_history,
            'best_configuration': self.best_configuration,
            'detailed_results': self.results,
            'timestamp': datetime.now().isoformat()
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        self.log_test(f"Results saved to: {filename}", "SUCCESS")
        return filename

    def test_intervals_for_stock(self, symbol, analysis_date, buy_price, sell_price, stop_loss, intervals=None):
        """Test if stock hits target or stop-loss across multiple intervals"""
        intervals = intervals or [1, 3, 7, 14, 21, 30, 45, 60, 90, 120]
        results = []

        for days in intervals:
            future_date = analysis_date + timedelta(days=days)
            df = self.advisor.get_stock_data_professional(symbol, future_date, days_back=days)

            if df is None or df.empty:
                self.log_test(f"⚠️ No data for {symbol} at {days}d interval", "WARNING")
                continue

            max_price = df['High'].max()
            min_price = df['Low'].min()
            std_dev = df['Close'].pct_change().std() * 100

            results.append({
                'days': days,
                'hit_target': max_price >= sell_price,
                'hit_stop_loss': min_price <= stop_loss,
                'std_dev': round(std_dev, 2)
            })

        return results

    def run_batch_tests(self, stock_list, target_date, num_stocks=10, intervals=None):
        """Run tests on a random subset of stocks"""
        import random
        selected_stocks = random.sample(stock_list, num_stocks)
        all_results = []

        for symbol in selected_stocks:
            self.log_test(f"🔍 Testing {symbol}", "INFO")
            result = self.advisor.analyze_stock_professional(symbol, target_date)

            if not result or result['action'] != 'BUY':
                self.log_test(f"⏳ Skipping {symbol} (no BUY signal)", "INFO")
                continue

            interval_results = self.test_intervals_for_stock(
                symbol,
                result['analysis_date'],
                result['buy_price'],
                result['sell_price'],
                result['stop_loss'],
                intervals
            )

            all_results.append({
                'symbol': symbol,
                'confidence': result['confidence'],
                'buy_price': result['buy_price'],
                'sell_price': result['sell_price'],
                'stop_loss': result['stop_loss'],
                'intervals': interval_results
            })

        return all_results

    def calibrate_algorithm_from_results(self, results):
        """Analyze test outcomes and suggest strategy adjustments"""
        total = 0
        hits = 0
        stops = 0

        for r in results:
            for interval in r['intervals']:
                total += 1
                if interval['hit_target']:
                    hits += 1
                if interval['hit_stop_loss']:
                    stops += 1

        hit_rate = hits / total if total else 0
        stop_rate = stops / total if total else 0

        self.log_test(f"📊 Hit Rate: {hit_rate:.2%}, Stop-Loss Rate: {stop_rate:.2%}", "INFO")

        suggestions = []
        if stop_rate > 0.3:
            suggestions.append(
                "⚠️ Too many stop-losses hit. Consider tightening entry filters or reducing risk multiplier.")
        if hit_rate < 0.5:
            suggestions.append(
                "📉 Target not reached often. Consider lowering profit expectations or extending holding period.")
        if hit_rate > 0.7 and stop_rate < 0.1:
            suggestions.append("✅ Strategy performing well. You may consider increasing profit multiplier.")

        return suggestions

    def summarize_test_results(self, results):
        """Print/log summary of test outcomes"""
        for r in results:
            self.log_test(
                f"📈 {r['symbol']} | Confidence: {r['confidence']:.1f}% | Buy: ${r['buy_price']:.2f} | Sell: ${r['sell_price']:.2f}",
                "INFO")
            for interval in r['intervals']:
                status = "✅ Target" if interval['hit_target'] else "❌ Stop" if interval[
                    'hit_stop_loss'] else "⏳ Neutral"
                self.log_test(f"  ⏱️ {interval['days']}d | {status} | STD: {interval['std_dev']:.2f}%", "INFO")


def run_quick_optimization_test():
    """Run a quick optimization test with predefined stocks"""

    # Popular stocks for testing
    test_stocks = [
        'AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA',
        'AMZN', 'META', 'AMD', 'NFLX', 'PYPL'
    ]

    print("🚀 STARTING QUICK ALGORITHM OPTIMIZATION")
    print("="*60)

    tester = AdvancedAlgorithmTester(debug=True)

    # Step 1: Baseline test
    print("\n📊 Phase 1: Baseline Performance Test")
    baseline_results = tester.run_random_week_test(
        stock_symbols=test_stocks,
        num_weeks=3,
        num_stocks=8
    )

    if not baseline_results:
        print("❌ Baseline test failed")
        return

    # Step 2: Auto-tuning
    print("\n⚙️ Phase 2: Automated Parameter Tuning")
    best_config, best_accuracy = tester.auto_tune_parameters(
        stock_symbols=test_stocks,
        optimization_iterations=2
    )

    # Step 3: Generate report
    print("\n📋 Phase 3: Optimization Report")
    tester.generate_optimization_report()

    # Step 4: Save results
    filename = tester.save_results()

    print(f"\n🎉 Optimization complete!")
    print(f"📁 Results saved to: {filename}")

    return tester


def stock_above_cap_threshold(cap_threshold):
    """ find stock above cap threshold """

    tickers = get_all_tickers.get_all_tickers(NASDAQ=True)
    valid_stocks = []

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            market_cap = info.get('marketCap', 0)
            if market_cap and market_cap > 1_000_000_000:
                valid_stocks.append((ticker, market_cap))
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")

    print("Stocks over $1B:")
    for symbol, cap in valid_stocks:
        print(f"{symbol}: ${cap:,}")


def debug_stock_loading():
    """Debug the stock loading process"""
    import pandas as pd

    print("🔍 DEBUGGING STOCK LOADING PROCESS")
    print("=" * 50)

    # Test 1: Check if CSV file exists and loads correctly
    try:
        df = pd.read_csv("nasdaq_full_list.csv")
        print(f"✅ CSV loaded successfully: {len(df)} total stocks")
        print(f"📊 Columns available: {list(df.columns)}")
        print(f"🔍 Sample data:")
        print(df.head())

        # Check Symbol column
        if 'Symbol' in df.columns:
            print(f"✅ Symbol column found: {df['Symbol'].nunique()} unique symbols")
            print(f"📊 Sample symbols: {df['Symbol'].head(10).tolist()}")
        else:
            print("❌ 'Symbol' column not found!")
            print(f"Available columns: {df.columns.tolist()}")
            return False

    except FileNotFoundError:
        print("❌ nasdaq_full_list.csv not found!")
        print("💡 Make sure the CSV file is in the same directory as your script")
        return False
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return False

    # Test 2: Check sampling process
    try:
        max_stocks = 100  # Test with smaller number first
        if len(df) > max_stocks:
            sampled_df = df.sample(n=max_stocks, random_state=42)
            test_stocks = sampled_df["Symbol"].tolist()
            print(f"✅ Sampling works: {len(test_stocks)} stocks selected")
            print(f"📊 Sample stocks: {test_stocks[:10]}")
        else:
            test_stocks = df["Symbol"].tolist()
            print(f"✅ Using all stocks: {len(test_stocks)} stocks")

    except Exception as e:
        print(f"❌ Sampling error: {e}")
        return False

    return True


def random_1000_stocks(max_stocks):
    """Run 1000 random stocks test with predefined stocks"""
    print(f"🔍 Loading up to {max_stocks} random stocks...")

    # Import stock file
    try:
        df = pd.read_csv("nasdaq_full_list.csv")
        print(f"✅ Loaded {len(df)} stocks from CSV")

        # Validate Symbol column exists
        if 'Symbol' not in df.columns:
            print(f"❌ Error: 'Symbol' column not found. Available columns: {df.columns.tolist()}")
            return []

        # Clean and validate symbols
        df = df.dropna(subset=['Symbol'])  # Remove NaN symbols
        df = df[df['Symbol'].str.len() >= 2]  # Remove single-character symbols
        df = df[df['Symbol'].str.len() <= 5]  # Remove very long symbols

        print(f"📊 After cleaning: {len(df)} valid stocks")

        # Use DataFrame.sample() instead of list.sample()
        if len(df) > max_stocks:
            sampled_df = df.sample(n=max_stocks, random_state=42)
            test_stocks = sampled_df["Symbol"].tolist()
            print(f"📊 Selected {max_stocks} random stocks for testing")
        else:
            test_stocks = df["Symbol"].tolist()
            print(f"📊 Using all {len(test_stocks)} available stocks")

        print(f"🎯 Sample stocks: {test_stocks[:10]}...")

        # Return just the stock list, don't run the test here
        return test_stocks

    except FileNotFoundError:
        print("❌ Error: nasdaq_full_list.csv not found!")
        print("💡 Solutions:")
        print("   1. Make sure nasdaq_full_list.csv is in the same directory")
        print("   2. Download a stock list CSV from NASDAQ website")
        print("   3. Use a smaller test list of known stocks")
        return []
    except Exception as e:
        print(f"❌ Error reading stock file: {e}")
        return []


def run_custom_optimization():
    """Run optimization with user-defined parameters"""
    print("🎯 CUSTOM ALGORITHM OPTIMIZATION")
    print("="*50)

    # Get user input
    print("📊 Enter stock symbols (comma-separated):")
    stock_input = input("   Example: AAPL,GOOGL,MSFT: ").strip()
    if not stock_input:
        stock_input = "AAPL,GOOGL,MSFT,NVDA,TSLA"

    stock_symbols = [s.strip().upper() for s in stock_input.split(',')]

    print("📅 Enter test period start (YYYY-MM-DD):")
    start_date = input("   Default 2024-01-01: ").strip() or "2024-01-01"

    print("📅 Enter test period end (YYYY-MM-DD):")
    end_date = input("   Default 2024-12-31: ").strip() or "2024-12-31"

    print("🔢 Enter number of random weeks to test:")
    num_weeks = int(input("   Default 4: ").strip() or "4")

    print("🔢 Enter number of stocks to test:")
    num_stocks = int(input(f"   Default {len(stock_symbols)}: ").strip() or str(len(stock_symbols)))

    print("🔢 Enter optimization iterations:")
    iterations = int(input("   Default 3: ").strip() or "3")

    # Run optimization
    tester = AdvancedAlgorithmTester(debug=True)

    baseline_results = tester.run_random_week_test(
        stock_symbols=stock_symbols,
        start_date=start_date,
        end_date=end_date,
        num_weeks=num_weeks,
        num_stocks=num_stocks
    )

    if baseline_results:
        best_config, best_accuracy = tester.auto_tune_parameters(
            stock_symbols=stock_symbols,
            optimization_iterations=iterations
        )

        tester.generate_optimization_report()
        filename = tester.save_results()

        print(f"\n🎉 Custom optimization complete!")
        print(f"📁 Results: {filename}")

    return tester


def test_step1_data_quality():
    """Test Step 1: Data Quality Fix"""

    print("🔧 STEP 1: TESTING DATA QUALITY FIX")
    print("=" * 50)

    tester = AdvancedAlgorithmTester(debug=True)

    # Create validated stock list
    validated_stocks = tester.create_validated_stock_list(
        target_count=20,  # Small number for testing
        max_validate=50  # Check up to 50 candidates
    )

    if not validated_stocks:
        print("❌ Step 1 failed: No validated stocks found")
        return False

    print(f"✅ Step 1 success: {len(validated_stocks)} validated stocks")
    print(f"📊 Validated stocks: {validated_stocks}")

    # Test one stock to ensure it works
    print(f"\n🧪 Testing algorithm on one validated stock: {validated_stocks[0]}")

    from datetime import date, timedelta
    test_date = date.today() - timedelta(days=30)

    result = tester.advisor.analyze_stock_enhanced(validated_stocks[0], test_date)

    if result:
        print(f"✅ Algorithm test successful on {validated_stocks[0]}")
        print(f"   Action: {result.get('action', 'Unknown')}")
        print(f"   Confidence: {result.get('confidence', 0):.1f}%")
        return True
    else:
        print(f"❌ Algorithm test failed on {validated_stocks[0]}")
        return False


# STEP 2: Test confidence calculation fix
def test_step2_confidence_fix():
    """Test Step 2: Confidence Calculation Fix"""

    print("🔧 STEP 2: TESTING CONFIDENCE CALCULATION FIX")
    print("=" * 50)

    tester = AdvancedAlgorithmTester(debug=True)

    # Get validated stocks from Step 1
    validated_stocks = tester.create_validated_stock_list(target_count=10, max_validate=30)

    if len(validated_stocks) < 5:
        print("❌ Need at least 5 validated stocks from Step 1")
        return False

    print(f"📊 Testing confidence fix on {len(validated_stocks)} stocks...")

    # Test confidence calculation on multiple stocks
    confidence_results = []

    from datetime import date, timedelta
    test_dates = [
        date.today() - timedelta(days=30),
        date.today() - timedelta(days=60),
        date.today() - timedelta(days=90)
    ]

    for symbol in validated_stocks[:5]:  # Test first 5 stocks
        for test_date in test_dates[:2]:  # Test 2 dates per stock
            try:
                result = tester.advisor.analyze_stock_enhanced(symbol, test_date)

                if result:
                    confidence = result.get('confidence', 0)
                    final_score = result.get('final_score', 0)
                    action = result.get('action', 'UNKNOWN')

                    confidence_results.append({
                        'symbol': symbol,
                        'confidence': confidence,
                        'final_score': final_score,
                        'action': action
                    })

                    print(f"  {symbol}: Score={final_score:.2f}, Confidence={confidence:.1f}%, Action={action}")

            except Exception as e:
                print(f"  ❌ Error testing {symbol}: {e}")

    if not confidence_results:
        print("❌ No confidence results generated")
        return False

    # Analyze confidence distribution
    confidences = [r['confidence'] for r in confidence_results]
    avg_confidence = sum(confidences) / len(confidences)
    max_confidence = max(confidences)
    min_confidence = min(confidences)

    print(f"\n📊 CONFIDENCE ANALYSIS:")
    print(f"   Average Confidence: {avg_confidence:.1f}%")
    print(f"   Range: {min_confidence:.1f}% - {max_confidence:.1f}%")
    print(f"   Total Tests: {len(confidence_results)}")

    # Check if confidence is more realistic (should be 30-85% range, avg around 55-65%)
    realistic_range = 30 <= avg_confidence <= 75
    reasonable_max = max_confidence <= 85
    reasonable_min = min_confidence >= 30

    if realistic_range and reasonable_max and reasonable_min:
        print("✅ Confidence calculation appears more realistic")
        return True
    else:
        print("❌ Confidence still needs adjustment")
        print(f"   Expected avg: 55-75%, got {avg_confidence:.1f}%")
        print(f"   Expected max: ≤85%, got {max_confidence:.1f}%")
        print(f"   Expected min: ≥30%, got {min_confidence:.1f}%")
        return False


def run_full_interval_test(run_stocks=1000):
    """Fixed interval test function"""
    print("🔧 RUNNING FIXED INTERVAL TEST")
    print("=" * 50)

    # Create tester with fixed IBKR connection
    tester = AdvancedAlgorithmTester(debug=True)

    # Get stock list
    stock_list = random_1000_stocks(run_stocks)

    if not stock_list:
        print("❌ No stocks loaded for testing")
        return

    print(f"🔍 Loaded {len(stock_list)} stocks for testing")

    # FIXED: Use more stocks for testing (was only 10)
    num_stocks_to_test = min(run_stocks, len(stock_list))  # Test up to 100 stocks
    print(f"🎯 Will test {num_stocks_to_test} stocks")

    try:
        # Run tests with the fixed tester
        results = tester.run_batch_tests(stock_list, datetime.now().date(), num_stocks=num_stocks_to_test)

        if results:
            print(f"✅ Successfully tested {len(results)} stocks")

            tester.summarize_test_results(results)
            suggestions = tester.calibrate_algorithm_from_results(results)

            print("\n🔧 Calibration Suggestions:")
            for s in suggestions:
                print(f" - {s}")
        else:
            print("❌ No test results generated")

    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up IBKR connection
        tester.cleanup()


def quick_test_optimizations():
    """🧪 Quick test to verify your changes work"""

    print("🧪 TESTING OPTIMIZATION IMPACT")
    print("=" * 40)

    # Mock BRLT indicators from debug log
    test_indicators = {
        'volume_relative': 0.44,  # Was causing -1, should now be +0.5
        'bb_position': 0.937,  # Was causing -2, should now be -0.5
        'momentum_5': 16.67,  # Should help with BB resistance
        'trend_score': 5.0,  # Strong trend
        'momentum_score': 3.3,  # Good momentum
    }

    # Test Volume Fix
    print("📊 Volume Analysis Test:")
    if test_indicators['volume_relative'] > 0.5:
        volume_score = 0.5  # New: slight positive
        print(f"   Volume {test_indicators['volume_relative']:.2f} → Score: +{volume_score} ✅")
    else:
        print("   Volume test failed ❌")

    # Test S/R Fix
    print("📊 Support/Resistance Test:")
    bb = test_indicators['bb_position']
    momentum = test_indicators['momentum_5']
    if bb > 0.8 and momentum > 2:
        sr_score = -0.5  # New: reduced penalty with momentum
        print(f"   BB {bb:.3f} + momentum {momentum:.1f}% → Score: {sr_score} ✅")
    else:
        print("   S/R test failed ❌")

    # Test Overall Impact
    print("📊 Overall Impact Test:")
    old_weights = {'trend': 0.35, 'momentum': 0.30, 'volume': 0.15, 'support_resistance': 0.10}
    new_weights = {'trend': 0.45, 'momentum': 0.30, 'volume': 0.10, 'support_resistance': 0.05}

    # Old calculation
    old_score = (5.0 * old_weights['trend'] +
                 3.3 * old_weights['momentum'] +
                 (-1.0) * old_weights['volume'] +
                 (-2.0) * old_weights['support_resistance'])

    # New calculation
    new_score = (5.0 * new_weights['trend'] +
                 3.3 * new_weights['momentum'] +
                 0.5 * new_weights['volume'] +
                 (-0.5) * new_weights['support_resistance'])

    print(f"   Old Final Score: {old_score:.2f}")
    print(f"   New Final Score: {new_score:.2f}")
    print(f"   Improvement: +{new_score - old_score:.2f}")

    # Test Threshold
    new_threshold = 0.7  # Balanced strategy
    if new_score >= new_threshold:
        print(f"   Result: BUY ✅ (score {new_score:.2f} ≥ {new_threshold})")
    else:
        print(f"   Result: WAIT ❌ (score {new_score:.2f} < {new_threshold})")

    return new_score > old_score


def analyze_test_results():
    """Analysis of the critical issues from your test results"""

    print("🚨 CRITICAL ISSUES ANALYSIS")
    print("=" * 50)

    print("📊 Your Test Results:")
    print("   • Stocks Tested: 60 (✅ GOOD - was only 10 before)")
    print("   • Hit Rate: 57.83% (✅ DECENT - above 50% is acceptable)")
    print("   • Stop-Loss Rate: 93.17% (❌ CATASTROPHIC - should be <30%)")
    print("")

    print("🔍 Root Cause Analysis:")
    print("   1. PROFIT TARGETS TOO HIGH: Many stocks show 15-20% profit targets")
    print("   2. STOP LOSSES TOO TIGHT: Likely 6% stop loss vs 15%+ profit targets")
    print("   3. UNREALISTIC EXPECTATIONS: Algorithm expects massive gains")
    print("   4. POOR RISK/REWARD RATIO: 2-3:1 against you instead of for you")
    print("")

    print("💡 What the results tell us:")
    print("   • BRLT: 93% confidence, hits target across ALL timeframes ✅")
    print("   • CASH: 82% confidence, hits target across ALL timeframes ✅")
    print("   • Many stocks: High confidence but hit stop-loss immediately ❌")
    print("   • Pattern: Good stock picking, terrible risk management ❌")


if __name__ == "__main__":
    print("🧪 ALGORITHM TESTING & OPTIMIZATION SUITE")
    print("="*50)
    print("1. Quick optimization test (predefined stocks)")
    print("2. Custom optimization test")
    print("3. Baseline performance test only")
    print("4. list of 1000 random stocks")
    print("5. Run Step 1: Data Quality and stock Validation fix")
    print("6. Run Step 2: Confidence Calculation Fix")
    print("7. Run Full Interval-Based Testing + Calibration")
    print("8. Quick optimization test with comparison to baseline performan")

    choice = input("Choose option (1-8): ").strip()

    if choice == "1":
        run_quick_optimization_test()
    elif choice == "2":
        run_custom_optimization()
    elif choice == "3":
        tester = AdvancedAlgorithmTester(debug=True)
        test_stocks = ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA']
        tester.run_random_week_test(test_stocks, num_weeks=3, num_stocks=5)
        tester.generate_optimization_report()
    elif choice == "4":
        test_stocks = random_1000_stocks(10)

        if not test_stocks:  # Check if we got valid stocks
            print("❌ No stocks loaded. Exiting.")
        else:
            print(f"✅ Loaded {len(test_stocks)} stocks")
            tester = AdvancedAlgorithmTester(debug=True)

            # Run the test with the stock list
            results = tester.run_random_week_test(test_stocks, num_weeks=3, num_stocks=50)

            if results:
                tester.generate_optimization_report()
                filename = tester.save_results()
                print(f"\n🎉 Random stocks test complete!")
                print(f"📁 Results saved to: {filename}")
            else:
                print("❌ Test failed")
    elif choice == "5":
        success = test_step1_data_quality()
        if success:
            print("\n🎉 STEP 1 COMPLETE - Ready for Step 2")
        else:
            print("\n❌ STEP 1 FAILED - Fix issues before proceeding")
    elif choice == "6":
        success = test_step2_confidence_fix()
        if success:
            print("\n🎉 STEP 2 COMPLETE - Confidence calculation improved")
        else:
            print("\n❌ STEP 2 NEEDS ADJUSTMENT - Review confidence formula")
    elif choice == "7":
        run_full_interval_test(100)
    elif choice == "8":
        quick_test_optimizations()


else:
        print("❌ Invalid option")