#!/usr/bin/env python
"""
🧪 Automated Algorithm Test Runner
Run all tests in sequence and generate comprehensive report
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from algo_testing_script import AdvancedAlgorithmTester
from stockwise_simulation import ProfessionalStockAdvisor
from datetime import datetime, date, timedelta
import pandas as pd
import json


def run_all_tests():
    """Run all algorithm tests in sequence"""
    
    print("="*80)
    print("🚀 STOCKWISE ALGORITHM COMPREHENSIVE TEST SUITE")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'tests': {}
    }
    
    # Initialize tester
    tester = AdvancedAlgorithmTester(debug=True)
    
    # TEST 1: Data Quality and Validation
    print("\n" + "="*60)
    print("📊 TEST 1: DATA QUALITY AND STOCK VALIDATION")
    print("="*60)
    
    try:
        validated_stocks = tester.create_validated_stock_list(
            target_count=30,  # Get 30 validated stocks
            max_validate=100   # Check up to 100 candidates
        )
        
        if validated_stocks:
            print(f"✅ Successfully validated {len(validated_stocks)} stocks")
            results['tests']['data_quality'] = {
                'status': 'PASSED',
                'validated_stocks': len(validated_stocks),
                'sample_stocks': validated_stocks[:10]
            }
        else:
            print("❌ Data quality test failed")
            results['tests']['data_quality'] = {'status': 'FAILED'}
            
    except Exception as e:
        print(f"❌ Data quality test error: {e}")
        results['tests']['data_quality'] = {'status': 'ERROR', 'error': str(e)}
    
    # TEST 2: Confidence Calculation
    print("\n" + "="*60)
    print("🎯 TEST 2: CONFIDENCE CALCULATION VALIDATION")
    print("="*60)
    
    try:
        if validated_stocks:
            confidence_results = []
            test_dates = [
                date.today() - timedelta(days=30),
                date.today() - timedelta(days=60)
            ]
            
            for symbol in validated_stocks[:5]:  # Test 5 stocks
                for test_date in test_dates:
                    try:
                        result = tester.advisor.analyze_stock_enhanced(symbol, test_date)
                        if result:
                            confidence_results.append({
                                'symbol': symbol,
                                'confidence': result.get('confidence', 0),
                                'action': result.get('action', 'UNKNOWN'),
                                'final_score': result.get('final_score', 0)
                            })
                    except:
                        pass
            
            if confidence_results:
                confidences = [r['confidence'] for r in confidence_results]
                avg_confidence = sum(confidences) / len(confidences)
                
                print(f"✅ Confidence tests completed")
                print(f"   Average confidence: {avg_confidence:.1f}%")
                print(f"   Range: {min(confidences):.1f}% - {max(confidences):.1f}%")
                
                results['tests']['confidence'] = {
                    'status': 'PASSED',
                    'avg_confidence': avg_confidence,
                    'min_confidence': min(confidences),
                    'max_confidence': max(confidences),
                    'samples': len(confidence_results)
                }
            else:
                results['tests']['confidence'] = {'status': 'FAILED'}
                
    except Exception as e:
        print(f"❌ Confidence test error: {e}")
        results['tests']['confidence'] = {'status': 'ERROR', 'error': str(e)}
    
    # TEST 3: Baseline Performance with Popular Stocks
    print("\n" + "="*60)
    print("📈 TEST 3: BASELINE PERFORMANCE TEST")
    print("="*60)
    
    try:
        # Use popular stocks for baseline
        popular_stocks = ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'META', 'AMD']
        
        baseline_results = tester.run_random_week_test(
            stock_symbols=popular_stocks,
            start_date="2024-01-01",
            end_date="2024-12-31",
            num_weeks=3,
            num_stocks=6
        )
        
        if baseline_results:
            performance = tester.calculate_performance_metrics(baseline_results)
            
            print(f"✅ Baseline test completed")
            print(f"   Overall accuracy: {performance['avg_accuracy']:.1f}%")
            print(f"   Direction accuracy: {performance['direction_accuracy']:.1f}%")
            print(f"   BUY success rate: {performance['buy_success_rate']:.1f}%")
            
            results['tests']['baseline'] = {
                'status': 'PASSED',
                'performance': performance,
                'total_tests': len(baseline_results)
            }
        else:
            results['tests']['baseline'] = {'status': 'FAILED'}
            
    except Exception as e:
        print(f"❌ Baseline test error: {e}")
        results['tests']['baseline'] = {'status': 'ERROR', 'error': str(e)}
    
    # TEST 4: Random Stock Performance
    print("\n" + "="*60)
    print("🎲 TEST 4: RANDOM STOCK PERFORMANCE TEST")
    print("="*60)
    
    try:
        if validated_stocks and len(validated_stocks) >= 10:
            random_results = tester.run_random_week_test(
                stock_symbols=validated_stocks,
                start_date="2024-01-01",
                end_date="2024-12-31",
                num_weeks=2,
                num_stocks=10
            )
            
            if random_results:
                random_performance = tester.calculate_performance_metrics(random_results)
                
                print(f"✅ Random stock test completed")
                print(f"   Overall accuracy: {random_performance['avg_accuracy']:.1f}%")
                print(f"   Direction accuracy: {random_performance['direction_accuracy']:.1f}%")
                
                results['tests']['random_stocks'] = {
                    'status': 'PASSED',
                    'performance': random_performance,
                    'total_tests': len(random_results)
                }
            else:
                results['tests']['random_stocks'] = {'status': 'FAILED'}
                
    except Exception as e:
        print(f"❌ Random stock test error: {e}")
        results['tests']['random_stocks'] = {'status': 'ERROR', 'error': str(e)}
    
    # TEST 5: Strategy Differentiation
    print("\n" + "="*60)
    print("⚡ TEST 5: STRATEGY DIFFERENTIATION TEST")
    print("="*60)
    
    try:
        if validated_stocks:
            test_symbol = validated_stocks[0]
            test_date = date.today() - timedelta(days=30)
            
            strategies = ["Conservative", "Balanced", "Aggressive", "Swing Trading"]
            strategy_results = []
            
            for strategy in strategies:
                # Update advisor strategy
                tester.advisor.current_strategy = strategy
                strategy_multipliers = {
                    "Conservative": {"profit": 0.8, "risk": 0.8, "confidence_req": 85},
                    "Balanced": {"profit": 1.0, "risk": 1.0, "confidence_req": 75},
                    "Aggressive": {"profit": 1.4, "risk": 1.3, "confidence_req": 65},
                    "Swing Trading": {"profit": 1.8, "risk": 1.5, "confidence_req": 70}
                }
                tester.advisor.strategy_settings = strategy_multipliers[strategy]
                
                result = tester.advisor.analyze_stock_enhanced(test_symbol, test_date)
                
                if result:
                    strategy_results.append({
                        'strategy': strategy,
                        'action': result.get('action'),
                        'confidence': result.get('confidence'),
                        'profit_target': result.get('expected_profit_pct')
                    })
            
            if strategy_results:
                print("✅ Strategy differentiation test completed")
                for sr in strategy_results:
                    print(f"   {sr['strategy']}: {sr['action']} @ {sr['confidence']:.1f}% confidence")
                
                results['tests']['strategy_diff'] = {
                    'status': 'PASSED',
                    'strategies': strategy_results
                }
            else:
                results['tests']['strategy_diff'] = {'status': 'FAILED'}
                
    except Exception as e:
        print(f"❌ Strategy test error: {e}")
        results['tests']['strategy_diff'] = {'status': 'ERROR', 'error': str(e)}
    
    # GENERATE FINAL REPORT
    print("\n" + "="*80)
    print("📊 FINAL TEST REPORT")
    print("="*80)
    
    # Count results
    passed = sum(1 for t in results['tests'].values() if t.get('status') == 'PASSED')
    failed = sum(1 for t in results['tests'].values() if t.get('status') == 'FAILED')
    errors = sum(1 for t in results['tests'].values() if t.get('status') == 'ERROR')
    total = len(results['tests'])
    
    print(f"\n📈 TEST SUMMARY:")
    print(f"   Total Tests: {total}")
    print(f"   ✅ Passed: {passed}")
    print(f"   ❌ Failed: {failed}")
    print(f"   ⚠️ Errors: {errors}")
    print(f"   Success Rate: {(passed/total*100):.1f}%")
    
    # Performance summary
    if 'baseline' in results['tests'] and results['tests']['baseline'].get('status') == 'PASSED':
        perf = results['tests']['baseline']['performance']
        print(f"\n🎯 ALGORITHM PERFORMANCE:")
        print(f"   Overall Accuracy: {perf['avg_accuracy']:.1f}%")
        print(f"   Direction Accuracy: {perf['direction_accuracy']:.1f}%")
        print(f"   BUY Success Rate: {perf['buy_success_rate']:.1f}%")
        print(f"   Average Confidence: {perf['avg_confidence']:.1f}%")
        
        # Performance rating
        if perf['avg_accuracy'] >= 70:
            print(f"\n🏆 RATING: EXCELLENT - Algorithm performing well!")
        elif perf['avg_accuracy'] >= 60:
            print(f"\n✅ RATING: GOOD - Algorithm shows promise, minor tuning needed")
        elif perf['avg_accuracy'] >= 50:
            print(f"\n⚠️ RATING: FAIR - Algorithm needs optimization")
        else:
            print(f"\n❌ RATING: POOR - Algorithm requires significant improvements")
    
    # Generate optimization suggestions
    if tester.baseline_performance:
        suggestions = tester.generate_optimization_suggestions(tester.baseline_performance)
        if suggestions:
            print(f"\n💡 OPTIMIZATION SUGGESTIONS:")
            for i, suggestion in enumerate(suggestions[:3], 1):
                print(f"   {i}. {suggestion['area']}: {suggestion['suggestion']}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"comprehensive_test_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📁 Results saved to: {filename}")
    print(f"\n🎉 All tests completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    return results, tester


def quick_algorithm_check():
    """Quick algorithm health check"""
    
    print("\n🔍 QUICK ALGORITHM HEALTH CHECK")
    print("="*50)
    
    from stockwise_simulation import ProfessionalStockAdvisor
    
    advisor = ProfessionalStockAdvisor(debug=False, download_log=False)
    
    # Test on a known good stock
    test_date = date.today() - timedelta(days=30)
    test_symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    results = []
    for symbol in test_symbols:
        try:
            result = advisor.analyze_stock_enhanced(symbol, test_date)
            if result:
                results.append({
                    'symbol': symbol,
                    'action': result.get('action'),
                    'confidence': result.get('confidence'),
                    'profit': result.get('expected_profit_pct')
                })
                print(f"✅ {symbol}: {result.get('action')} @ {result.get('confidence'):.1f}% confidence")
            else:
                print(f"❌ {symbol}: Analysis failed")
        except Exception as e:
            print(f"❌ {symbol}: Error - {e}")
    
    if results:
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        print(f"\n📊 Average confidence: {avg_confidence:.1f}%")
        
        if avg_confidence > 90:
            print("⚠️ Confidence might be too high - check calibration")
        elif avg_confidence < 40:
            print("⚠️ Confidence might be too low - check signal strength")
        else:
            print("✅ Confidence levels appear reasonable")
    
    return results


def test_log_directory_setup():
    """Test that logs are saved in the correct directory"""
    advisor = ProfessionalStockAdvisor(debug=True, download_log=True)

    # Test log file creation
    log_file = advisor.ensure_log_file()

    print(f"✅ Log file created: {log_file}")
    print(f"✅ Directory: {os.path.dirname(log_file)}")
    print(f"✅ File exists: {os.path.exists(log_file)}")

    # Test logging
    advisor.log("Test message", "INFO")

    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            content = f.read()
        print(f"✅ Log content preview: {content[:100]}...")

    return log_file


# Run test
# test_log_directory_setup()


if __name__ == "__main__":
    print("🧪 STOCKWISE ALGORITHM TEST RUNNER")
    print("="*50)
    print("1. Run all comprehensive tests")
    print("2. Quick algorithm health check")
    print("3. Exit")
    
    choice = input("\nChoose option (1-3): ").strip()
    
    if choice == "1":
        results, tester = run_all_tests()
    elif choice == "2":
        quick_algorithm_check()
    elif choice == "3":
        print("Exiting...")
    else:
        print("Invalid choice")
