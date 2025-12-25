"""
🔬 ADVANCED TESTING FEATURES
════════════════════════════════════════════════════════════════

Additional testing capabilities to make your testing more comprehensive.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import datetime
import pandas as pd

from algo_testing_script import AlgorithmTester
import numpy as np


class AdvancedAlgorithmTester(AlgorithmTester):
    """Extended testing class with advanced analysis features"""
    
    def __init__(self, debug=False):
        super().__init__(debug)
        self.benchmark_results = {}

    def run_sanity_tests(self):
        print("Running sanity tests...")

        # Step 1: Set parameters
        tickers = ['AAPL', 'MSFT', 'GOOGL']
        start_date = '2024-01-01'
        end_date = '2024-02-29'
        frequency = 'weekly'
        holding_period = 7

        # Step 2: Run predictions
        try:
            results = self.run(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                frequency=frequency,
                holding_days=holding_period,
                chart=False,
                verbose=False
            )
            print("✅ Predictions completed.")
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return False

        # Step 3: Export and verify
        try:
            self.export_to_json(results, "sanity_results.json")
            self.export_to_csv(results, "sanity_results.csv")
            print("✅ Export succeeded.")
        except Exception as e:
            print(f"❌ Export error: {e}")
            return False

        print("🎉 Sanity tests passed.")
        return True

    def run_confidence_calibration_test(self):
        """Test if confidence levels are properly calibrated"""
        if not self.results:
            print("❌ No results to analyze")
            return
        
        df = pd.DataFrame(self.results)
        success_df = df[df['status'] == 'SUCCESS']
        
        print("\n🎯 CONFIDENCE CALIBRATION ANALYSIS")
        print("=" * 50)
        
        # Group by confidence ranges
        confidence_ranges = [(85, 100), (75, 85), (65, 75), (50, 65)]
        
        for min_conf, max_conf in confidence_ranges:
            range_df = success_df[
                (success_df['confidence'] >= min_conf) & 
                (success_df['confidence'] < max_conf)
            ]
            
            if len(range_df) > 0:
                success_rate = len(range_df[range_df['test_result'] == 'PASS']) / len(range_df)
                avg_confidence = range_df['confidence'].mean()
                
                # Calibration score (closer to 1.0 is better)
                calibration_score = success_rate / (avg_confidence / 100)
                
                print(f"📊 {min_conf}-{max_conf}% Confidence:")
                print(f"   Predictions: {len(range_df)}")
                print(f"   Success Rate: {success_rate:.1%}")
                print(f"   Avg Confidence: {avg_confidence:.1f}%")
                print(f"   Calibration Score: {calibration_score:.2f} {'✅' if 0.8 <= calibration_score <= 1.2 else '⚠️'}")
                print()
    
    def compare_with_benchmark(self, benchmark_strategy='buy_and_hold'):
        """Compare algorithm performance with benchmark strategies"""
        if not self.results:
            print("❌ No results to compare")
            return
        
        df = pd.DataFrame(self.results)
        success_df = df[df['status'] == 'SUCCESS']
        
        print(f"\n📈 BENCHMARK COMPARISON: {benchmark_strategy.upper()}")
        print("=" * 50)
        
        symbols = success_df['symbol'].unique()
        
        for symbol in symbols:
            symbol_df = success_df[success_df['symbol'] == symbol]
            
            # Algorithm performance
            algo_profits = symbol_df['actual_profit_pct'].tolist()
            algo_avg = np.mean(algo_profits)
            algo_success_rate = len(symbol_df[symbol_df['test_result'] == 'PASS']) / len(symbol_df)
            
            # Benchmark performance (buy and hold)
            if benchmark_strategy == 'buy_and_hold':
                benchmark_profits = [p for p in algo_profits]  # Same periods
                benchmark_avg = np.mean(benchmark_profits)
                benchmark_success_rate = len([p for p in benchmark_profits if p > 0]) / len(benchmark_profits)
            
            print(f"📊 {symbol}:")
            print(f"   Algorithm: {algo_avg:.2f}% avg, {algo_success_rate:.1%} success")
            print(f"   Benchmark: {benchmark_avg:.2f}% avg, {benchmark_success_rate:.1%} success")
            print(f"   Outperformance: {algo_avg - benchmark_avg:.2f}% {'🟢' if algo_avg > benchmark_avg else '🔴'}")
            print()
    
    def analyze_market_conditions(self):
        """Analyze algorithm performance under different market conditions"""
        if not self.results:
            print("❌ No results to analyze")
            return
        
        df = pd.DataFrame(self.results)
        success_df = df[df['status'] == 'SUCCESS']
        
        print("\n🌤️ MARKET CONDITIONS ANALYSIS")
        print("=" * 50)
        
        # Categorize by market performance
        success_df['market_condition'] = success_df['actual_profit_pct'].apply(
            lambda x: 'Bull Market' if x > 2 else 'Bear Market' if x < -2 else 'Sideways'
        )
        
        for condition in ['Bull Market', 'Sideways', 'Bear Market']:
            condition_df = success_df[success_df['market_condition'] == condition]
            
            if len(condition_df) > 0:
                success_rate = len(condition_df[condition_df['test_result'] == 'PASS']) / len(condition_df)
                avg_confidence = condition_df['confidence'].mean()
                
                print(f"📊 {condition}:")
                print(f"   Tests: {len(condition_df)}")
                print(f"   Success Rate: {success_rate:.1%}")
                print(f"   Avg Confidence: {avg_confidence:.1f}%")
                print(f"   Avg Profit: {condition_df['actual_profit_pct'].mean():.2f}%")
                print()
    
    def generate_performance_charts(self, save_charts=True):
        """Generate visual performance analysis charts"""
        if not self.results:
            print("❌ No results to visualize")
            return
        
        df = pd.DataFrame(self.results)
        success_df = df[df['status'] == 'SUCCESS']
        
        print("\n📊 GENERATING PERFORMANCE CHARTS")
        print("=" * 50)
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Algorithm Performance Analysis', fontsize=16)
        
        # Chart 1: Confidence vs Success Rate
        confidence_bins = np.arange(50, 101, 10)
        success_df['confidence_bin'] = pd.cut(success_df['confidence'], confidence_bins)
        conf_analysis = success_df.groupby('confidence_bin').agg({
            'test_result': lambda x: (x == 'PASS').mean(),
            'confidence': 'count'
        }).reset_index()
        
        axes[0,0].bar(range(len(conf_analysis)), conf_analysis['test_result'], 
                     color='skyblue', alpha=0.7)
        axes[0,0].set_title('Success Rate by Confidence Level')
        axes[0,0].set_xlabel('Confidence Range')
        axes[0,0].set_ylabel('Success Rate')
        axes[0,0].set_xticks(range(len(conf_analysis)))
        axes[0,0].set_xticklabels([str(x) for x in conf_analysis['confidence_bin']], rotation=45)
        
        # Chart 2: Predicted vs Actual Profit
        axes[0,1].scatter(success_df['predicted_profit_pct'], success_df['actual_profit_pct'], 
                         alpha=0.6, color='green')
        axes[0,1].plot([-10, 15], [-10, 15], 'r--', alpha=0.8)  # Perfect prediction line
        axes[0,1].set_title('Predicted vs Actual Profit')
        axes[0,1].set_xlabel('Predicted Profit (%)')
        axes[0,1].set_ylabel('Actual Profit (%)')
        axes[0,1].grid(True, alpha=0.3)
        
        # Chart 3: Action Distribution
        action_counts = success_df['action'].value_counts()
        axes[1,0].pie(action_counts.values, labels=action_counts.index, autopct='%1.1f%%')
        axes[1,0].set_title('Algorithm Action Distribution')
        
        # Chart 4: Profit Distribution by Action
        for action in success_df['action'].unique():
            action_data = success_df[success_df['action'] == action]['actual_profit_pct']
            axes[1,1].hist(action_data, alpha=0.6, label=action, bins=20)
        axes[1,1].set_title('Profit Distribution by Action')
        axes[1,1].set_xlabel('Actual Profit (%)')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_charts:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chart_filename = f"algorithm_performance_charts_{timestamp}.png"
            plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
            print(f"📊 Charts saved to: {chart_filename}")
        
        plt.show()
    
    def run_stress_test(self, stock_symbols, volatile_periods=None):
        """Run stress test during volatile market periods"""
        if volatile_periods is None:
            # Default volatile periods (major market events)
            volatile_periods = [
                ('2020-02-20', '2020-04-30'),  # COVID crash
                ('2022-01-01', '2022-06-30'),  # 2022 bear market
                ('2008-09-01', '2008-12-31'),  # Financial crisis
            ]
        
        print("\n🌪️ STRESS TEST - VOLATILE PERIODS")
        print("=" * 50)
        
        stress_results = []
        
        for start_date, end_date in volatile_periods:
            print(f"🔥 Testing volatile period: {start_date} to {end_date}")
            
            period_results = self.run_comprehensive_test(
                stock_symbols, start_date, end_date, 'weekly', 7
            )
            
            # Analyze this period
            period_df = pd.DataFrame(period_results)
            success_df = period_df[period_df['status'] == 'SUCCESS']
            
            if len(success_df) > 0:
                success_rate = len(success_df[success_df['test_result'] == 'PASS']) / len(success_df)
                avg_profit = success_df['actual_profit_pct'].mean()
                avg_confidence = success_df['confidence'].mean()
                
                period_summary = {
                    'period': f"{start_date} to {end_date}",
                    'total_tests': len(success_df),
                    'success_rate': success_rate,
                    'avg_profit': avg_profit,
                    'avg_confidence': avg_confidence
                }
                
                stress_results.append(period_summary)
                
                print(f"   📊 Success Rate: {success_rate:.1%}")
                print(f"   💰 Avg Profit: {avg_profit:.2f}%")
                print(f"   🎯 Avg Confidence: {avg_confidence:.1f}%")
                print()
        
        return stress_results
    
    def save_detailed_report(self, filename=None):
        """Save comprehensive JSON report"""
        if not self.results:
            print("❌ No results to save")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detailed_test_report_{timestamp}.json"
        
        # Create comprehensive report
        df = pd.DataFrame(self.results)
        success_df = df[df['status'] == 'SUCCESS']
        
        report = {
            'test_summary': {
                'total_tests': len(df),
                'successful_tests': len(success_df),
                'success_rate': len(success_df) / len(df) if len(df) > 0 else 0,
                'test_date_range': {
                    'start': str(df['test_date'].min()) if len(df) > 0 else None,
                    'end': str(df['test_date'].max()) if len(df) > 0 else None
                }
            },
            'performance_metrics': {
                'overall_accuracy': len(success_df[success_df['test_result'] == 'PASS']) / len(success_df) if len(success_df) > 0 else 0,
                'high_confidence_accuracy': 0,
                'direction_accuracy': success_df['direction_correct'].mean() if len(success_df) > 0 else 0,
                'average_profit': {
                    'predicted': success_df['predicted_profit_pct'].mean() if len(success_df) > 0 else 0,
                    'actual': success_df['actual_profit_pct'].mean() if len(success_df) > 0 else 0
                }
            },
            'action_breakdown': {},
            'confidence_analysis': {},
            'individual_results': self.results
        }
        
        # High confidence accuracy
        if len(success_df) > 0:
            high_conf_df = success_df[success_df['confidence'] >= 85]
            if len(high_conf_df) > 0:
                report['performance_metrics']['high_confidence_accuracy'] = len(high_conf_df[high_conf_df['test_result'] == 'PASS']) / len(high_conf_df)
        
        # Action breakdown
        if len(success_df) > 0:
            for action in success_df['action'].unique():
                action_df = success_df[success_df['action'] == action]
                report['action_breakdown'][action] = {
                    'count': len(action_df),
                    'success_rate': len(action_df[action_df['test_result'] == 'PASS']) / len(action_df),
                    'avg_profit': action_df['actual_profit_pct'].mean()
                }
        
        # Confidence analysis
        if len(success_df) > 0:
            confidence_ranges = [(85, 100), (75, 85), (65, 75), (50, 65)]
            for min_conf, max_conf in confidence_ranges:
                range_df = success_df[(success_df['confidence'] >= min_conf) & (success_df['confidence'] < max_conf)]
                if len(range_df) > 0:
                    report['confidence_analysis'][f'{min_conf}-{max_conf}%'] = {
                        'count': len(range_df),
                        'success_rate': len(range_df[range_df['test_result'] == 'PASS']) / len(range_df),
                        'avg_confidence': range_df['confidence'].mean(),
                        'avg_profit': range_df['actual_profit_pct'].mean()
                    }
        
        # Save report
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📄 Detailed report saved to: {filename}")
        return report


# Enhanced interactive testing function
def run_enhanced_interactive_test():
    """Enhanced interactive testing with advanced features"""
    print("🧪 ENHANCED ALGORITHM TESTING SUITE")
    print("=" * 50)
    
    # Test type selection
    print("📋 Select test type:")
    print("1. Basic Performance Test")
    print("2. Confidence Calibration Test")
    print("3. Stress Test (Volatile Periods)")
    print("4. Comprehensive Analysis (All Tests)")
    
    test_type = input("Enter choice (1-4): ").strip()
    
    # Get basic parameters
    print("📊 Stock Input Options:")
    print("1. Upload txt file")
    print("2. Type manually")

    choice = input("Choose option (1-2): ").strip()

    if choice == "1":
        filename = input("📁 Enter txt file path: ").strip()
        try:
            with open(filename, 'r') as f:
                stock_symbols = [line.strip().upper() for line in f if line.strip()]
            print(f"✅ Loaded {len(stock_symbols)} stocks from {filename}")
            print(f"📋 First 5 stocks: {stock_symbols[:5]}")
        except FileNotFoundError:
            print(f"❌ File not found: {filename}")
            return
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return
    else:
        print("📊 Enter stock symbols (comma-separated):")
        stock_input = input("   Example: AAPL,GOOGL,MSFT,NVDA: ").strip()
        stock_symbols = [s.strip().upper() for s in stock_input.split(',') if s.strip()]

    if not stock_symbols:
        print("❌ No valid stock symbols entered")
        return
    
    # Initialize enhanced tester
    tester = AdvancedAlgorithmTester(debug=True)
    
    if test_type == "1":
        # Basic test
        print("\n📅 Enter start date (YYYY-MM-DD):")
        start_date = input("   Example: 2024-01-01: ").strip()
        print("📅 Enter end date (YYYY-MM-DD):")
        end_date = input("   Example: 2024-12-31: ").strip()
        
        results = tester.run_comprehensive_test(stock_symbols, start_date, end_date, 'weekly', 7)
        tester.generate_test_report(save_to_csv=True)
        
    elif test_type == "2":
        # Confidence calibration test
        print("\n📅 Enter start date for calibration test:")
        start_date = input("   Example: 2024-01-01: ").strip()
        print("📅 Enter end date:")
        end_date = input("   Example: 2024-12-31: ").strip()
        
        results = tester.run_comprehensive_test(stock_symbols, start_date, end_date, 'weekly', 7)
        tester.generate_test_report(save_to_csv=True)
        tester.run_confidence_calibration_test()
        
    elif test_type == "3":
        # Stress test
        print("\n🌪️ Running stress test on volatile periods...")
        stress_results = tester.run_stress_test(stock_symbols)
        
        print("\n📊 STRESS TEST SUMMARY:")
        for result in stress_results:
            print(f"Period: {result['period']}")
            print(f"  Success Rate: {result['success_rate']:.1%}")
            print(f"  Avg Profit: {result['avg_profit']:.2f}%")
            print()
            
    elif test_type == "4":
        # Comprehensive analysis
        print("\n🔬 Running comprehensive analysis...")
        print("📅 Enter start date:")
        start_date = input("   Example: 2024-01-01: ").strip()
        print("📅 Enter end date:")
        end_date = input("   Example: 2024-12-31: ").strip()
        
        # Run all tests
        results = tester.run_comprehensive_test(stock_symbols, start_date, end_date, 'weekly', 7)
        
        # Generate all reports
        tester.generate_test_report(save_to_csv=True)
        tester.run_confidence_calibration_test()
        tester.compare_with_benchmark('buy_and_hold')
        tester.analyze_market_conditions()
        
        # Generate charts
        try:
            tester.generate_performance_charts(save_charts=True)
        except Exception as e:
            print(f"⚠️ Could not generate charts: {e}")
        
        # Save detailed report
        tester.save_detailed_report()
        
        print("\n✅ COMPREHENSIVE ANALYSIS COMPLETE!")
        print("📁 Check your directory for all generated reports and charts.")


# Quick test function for specific scenarios
def quick_test_scenario():
    """Quick test for specific trading scenarios"""
    print("⚡ QUICK SCENARIO TEST")
    print("=" * 30)
    
    scenarios = {
        '1': {
            'name': 'Tech Stock Bull Run',
            'stocks': ['AAPL', 'GOOGL', 'MSFT', 'NVDA'],
            'period': ('2023-01-01', '2023-06-30')
        },
        '2': {
            'name': 'Market Correction',
            'stocks': ['SPY', 'QQQ', 'TSLA', 'AMZN'],
            'period': ('2022-01-01', '2022-06-30')
        },
        '3': {
            'name': 'Mixed Market Conditions',
            'stocks': ['AAPL', 'TSLA', 'META', 'NFLX'],
            'period': ('2024-01-01', '2024-10-31')
        }
    }
    
    print("📋 Select scenario:")
    for key, scenario in scenarios.items():
        print(f"{key}. {scenario['name']} ({scenario['period'][0]} to {scenario['period'][1]})")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice in scenarios:
        scenario = scenarios[choice]
        print(f"\n🚀 Running scenario: {scenario['name']}")
        
        tester = AdvancedAlgorithmTester(debug=True)
        results = tester.run_comprehensive_test(
            scenario['stocks'], 
            scenario['period'][0], 
            scenario['period'][1], 
            'weekly', 7
        )
        
        tester.generate_test_report(save_to_csv=True)
        
        # Quick analysis
        df = pd.DataFrame(results)
        success_df = df[df['status'] == 'SUCCESS']
        
        if len(success_df) > 0:
            print(f"\n🎯 SCENARIO RESULTS:")
            print(f"   Success Rate: {len(success_df[success_df['test_result'] == 'PASS']) / len(success_df):.1%}")
            print(f"   Avg Profit: {success_df['actual_profit_pct'].mean():.2f}%")
            print(f"   Best Stock: {success_df.loc[success_df['actual_profit_pct'].idxmax(), 'symbol']}")
    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    print("🧪 ALGORITHM TESTING SUITE")
    print("=" * 40)
    print("1. Enhanced Interactive Test")
    print("2. Quick Scenario Test")

    advanced_algorithm = AdvancedAlgorithmTester()
    advanced_algorithm.run_sanity_tests()
    # choice = input("Select option (1-2): ").strip()


    # if choice == "1":
    #     run_enhanced_interactive_test()
    # elif choice == "2":
    #     quick_test_scenario()
    # else:
    #     print("❌ Invalid choice, running basic test...")
    #     # run_interactive_test()
                    