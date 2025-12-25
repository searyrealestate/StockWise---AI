"""
NVDA Stock Debugging Implementation Guide
========================================

This script shows you exactly how to debug NVDA using your Debug_algo.py
"""

import sys
import os
from datetime import datetime, timedelta

# Import your modules
from Debug_algo import TradingAlgorithmDebugger, fix_weak_signals_issue, fix_strategy_thresholds, apply_enhanced_confidence_calculation
from stockwise_simulation import EnhancedStockAdvisor


def debug_nvda_comprehensive():
    """
    Complete NVDA debugging workflow
    """
    print("🚀 STARTING NVDA COMPREHENSIVE DEBUG")
    print("=" * 60)
    
    # Step 1: Initialize your advisor
    print("\n1️⃣ INITIALIZING ADVISOR")
    advisor = EnhancedStockAdvisor(debug=True, download_log=True)
    advisor.investment_days = 7  # Start with 7 days
    advisor.strategy_settings = {"profit": 1.0, "risk": 1.0, "confidence_req": 75}  # Balanced
    advisor.current_strategy = "Balanced"
    
    print(f"✅ Advisor initialized with {advisor.current_strategy} strategy")
    print(f"✅ Investment days: {advisor.investment_days}")
    
    # Step 2: Initialize debugger
    print("\n2️⃣ INITIALIZING DEBUGGER")
    debugger = TradingAlgorithmDebugger(advisor)
    
    # Step 3: Run comprehensive diagnosis
    print("\n3️⃣ RUNNING COMPREHENSIVE DIAGNOSIS")
    debug_results = debugger.diagnose_wait_signals_issue(symbol="NVDA", start_date="2025-04-21")
    
    print(f"\n📋 DIAGNOSIS SUMMARY:")
    print(f"Found {len(debug_results)} issues:")
    for i, issue in enumerate(debug_results, 1):
        print(f"   {i}. {issue}")
    
    return advisor, debugger, debug_results

def test_nvda_before_fixes():
    """
    Test NVDA with current settings before applying fixes
    """
    print("\n4️⃣ TESTING NVDA BEFORE FIXES")
    print("-" * 40)
    
    advisor = EnhancedStockAdvisor(debug=True)
    advisor.investment_days = 7
    advisor.strategy_settings = {"profit": 1.0, "risk": 1.0, "confidence_req": 75}
    advisor.current_strategy = "Balanced"
    
    # Test current date
    target_date = datetime.now().date()
    
    try:
        result = advisor.analyze_stock_enhanced("NVDA", target_date)
        
        if result:
            print(f"🎯 NVDA Analysis Results (BEFORE fixes):")
            print(f"   Action: {result.get('action', 'UNKNOWN')}")
            print(f"   Confidence: {result.get('confidence', 0):.1f}%")
            print(f"   Final Score: {result.get('final_score', 0):.2f}")
            print(f"   Current Price: ${result.get('current_price', 0):.2f}")
            
            # Show signal breakdown
            signal_breakdown = result.get('signal_breakdown', {})
            if signal_breakdown:
                print(f"   📊 Signal Breakdown:")
                for signal_type, score in signal_breakdown.items():
                    print(f"      {signal_type}: {score:.2f}")
            
            return result
        else:
            print("❌ Analysis failed")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def apply_fixes_step_by_step():
    """
    Apply fixes one by one and test each step
    """
    print("\n5️⃣ APPLYING FIXES STEP BY STEP")
    print("-" * 40)
    
    # Initialize fresh advisor
    advisor = EnhancedStockAdvisor(debug=True)
    advisor.investment_days = 7
    advisor.strategy_settings = {"profit": 1.0, "risk": 1.0, "confidence_req": 75}
    advisor.current_strategy = "Balanced"
    
    target_date = datetime.now().date()
    
    # Fix 1: Apply Aggressive Strategy
    print("\n🔧 FIX 1: Applying Aggressive Strategy")
    advisor = fix_strategy_thresholds(advisor, "Aggressive")
    
    result1 = test_single_fix(advisor, "NVDA", target_date, "Aggressive Strategy")
    
    # Fix 2: Enhanced Signal Sensitivity
    print("\n🔧 FIX 2: Enhancing Signal Sensitivity")
    advisor = fix_weak_signals_issue(advisor)
    
    result2 = test_single_fix(advisor, "NVDA", target_date, "Enhanced Signals")
    
    # Fix 3: Lenient Confidence Calculation
    print("\n🔧 FIX 3: Applying Lenient Confidence")
    advisor = apply_enhanced_confidence_calculation(advisor)
    
    result3 = test_single_fix(advisor, "NVDA", target_date, "Lenient Confidence")
    
    # Fix 4: Ultra Aggressive Strategy
    print("\n🔧 FIX 4: Ultra Aggressive Strategy")
    advisor = fix_strategy_thresholds(advisor, "Ultra Aggressive")
    
    result4 = test_single_fix(advisor, "NVDA", target_date, "Ultra Aggressive")
    
    # Fix 5: Different Time Periods
    print("\n🔧 FIX 5: Testing Different Time Periods")
    
    time_results = []
    for days in [14, 21, 30, 45]:
        advisor.investment_days = days
        result = test_single_fix(advisor, "NVDA", target_date, f"{days} Days")
        time_results.append((days, result))
    
    return result4, time_results

def test_single_fix(advisor, symbol, target_date, fix_name):
    """
    Test a single fix and report results
    """
    try:
        result = advisor.analyze_stock_enhanced(symbol, target_date)
        
        if result:
            action = result.get('action', 'UNKNOWN')
            confidence = result.get('confidence', 0)
            score = result.get('final_score', 0)
            
            print(f"   {fix_name}: {action} (Score: {score:.2f}, Conf: {confidence:.1f}%)")
            
            if action == "BUY":
                print(f"   ✅ SUCCESS! {fix_name} generated BUY signal")
                expected_profit = result.get('expected_profit_pct', 0)
                print(f"   💰 Expected Profit: {expected_profit:.1f}%")
            elif action == "WAIT":
                print(f"   ⏳ Still WAIT with {fix_name}")
            else:
                print(f"   🔴 SELL/AVOID with {fix_name}")
                
            return result
        else:
            print(f"   ❌ {fix_name}: Analysis failed")
            return None
            
    except Exception as e:
        print(f"   ❌ {fix_name}: Error - {e}")
        return None

def compare_nvda_with_other_stocks():
    """
    Compare NVDA with other popular stocks to see if issue is stock-specific
    """
    print("\n6️⃣ COMPARING NVDA WITH OTHER STOCKS")
    print("-" * 40)
    
    # Use best settings from previous tests
    advisor = EnhancedStockAdvisor(debug=False)  # Disable debug for speed
    advisor = fix_strategy_thresholds(advisor, "Aggressive")
    advisor = fix_weak_signals_issue(advisor)
    advisor = apply_enhanced_confidence_calculation(advisor)
    advisor.investment_days = 14  # Use 14 days
    
    stocks_to_test = ["NVDA", "AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
    target_date = datetime.now().date()
    
    results = []
    
    for symbol in stocks_to_test:
        try:
            result = advisor.analyze_stock_enhanced(symbol, target_date)
            
            if result:
                results.append({
                    'Symbol': symbol,
                    'Action': result.get('action', 'UNKNOWN'),
                    'Confidence': result.get('confidence', 0),
                    'Score': result.get('final_score', 0),
                    'Expected_Profit': result.get('expected_profit_pct', 0)
                })
            else:
                results.append({
                    'Symbol': symbol,
                    'Action': 'FAILED',
                    'Confidence': 0,
                    'Score': 0,
                    'Expected_Profit': 0
                })
                
        except Exception as e:
            print(f"   ⚠️ Error with {symbol}: {e}")
            results.append({
                'Symbol': symbol,
                'Action': 'ERROR',
                'Confidence': 0,
                'Score': 0,
                'Expected_Profit': 0
            })
    
    # Display results
    print(f"\n📊 STOCK COMPARISON RESULTS:")
    print(f"{'Symbol':<8} {'Action':<12} {'Score':<8} {'Conf':<6} {'Profit':<8}")
    print("-" * 50)
    
    buy_stocks = []
    for r in results:
        symbol = r['Symbol']
        action = r['Action']
        score = r['Score']
        confidence = r['Confidence']
        profit = r['Expected_Profit']
        
        print(f"{symbol:<8} {action:<12} {score:<8.2f} {confidence:<6.1f}% {profit:<8.1f}%")
        
        if action == "BUY":
            buy_stocks.append(symbol)
    
    # Analysis
    print(f"\n🎯 COMPARISON ANALYSIS:")
    if buy_stocks:
        print(f"✅ BUY signals found in: {', '.join(buy_stocks)}")
        if "NVDA" in buy_stocks:
            print("✅ NVDA issue RESOLVED with enhanced settings!")
        else:
            print("⚠️ NVDA still problematic - may be stock-specific issue")
    else:
        print("❌ NO BUY signals in any stocks - may be market-wide issue")
        print("💡 Consider:")
        print("   • Market may be in neutral/uncertain phase")
        print("   • Try different analysis dates")
        print("   • Check for market-wide events affecting sentiment")
    
    return results

def deep_dive_nvda_technicals():
    """
    Deep dive into NVDA's technical indicators to understand the issue
    """
    print("\n7️⃣ NVDA TECHNICAL DEEP DIVE")
    print("-" * 40)
    
    advisor = EnhancedStockAdvisor(debug=True)
    target_date = datetime.now().date()
    
    try:
        # Get raw data
        df = advisor.get_stock_data("NVDA", target_date, days_back=60)
        
        if df is None or df.empty:
            print("❌ Could not get NVDA data")
            return
        
        print(f"✅ Retrieved {len(df)} days of NVDA data")
        print(f"📅 Data range: {df.index[0].date()} to {df.index[-1].date()}")
        
        # Calculate indicators
        indicators = advisor.calculate_enhanced_indicators(df, pd.Timestamp(target_date))
        
        if indicators is None:
            print("❌ Could not calculate indicators")
            return
        
        # Display key indicators
        current_price = indicators['current_price']
        print(f"\n📊 NVDA TECHNICAL INDICATORS:")
        print(f"Current Price: ${current_price:.2f}")
        
        # Moving Averages
        print(f"\n📈 MOVING AVERAGES:")
        print(f"   SMA 5:  ${indicators.get('sma_5', 0):.2f}")
        print(f"   SMA 10: ${indicators.get('sma_10', 0):.2f}")
        print(f"   SMA 20: ${indicators.get('sma_20', 0):.2f}")
        print(f"   SMA 50: ${indicators.get('sma_50', 0):.2f}")
        
        # Check trend alignment
        sma_5 = indicators.get('sma_5', current_price)
        sma_10 = indicators.get('sma_10', current_price)
        sma_20 = indicators.get('sma_20', current_price)
        sma_50 = indicators.get('sma_50', current_price)
        
        if current_price > sma_5 > sma_10 > sma_20:
            print("   ✅ STRONG bullish alignment")
        elif current_price > sma_20:
            print("   📈 MILD bullish (above SMA20)")
        elif current_price < sma_20:
            print("   📉 BEARISH (below SMA20)")
        else:
            print("   ⚖️ NEUTRAL trend")
        
        # Momentum Indicators
        print(f"\n🚀 MOMENTUM INDICATORS:")
        print(f"   RSI 14: {indicators.get('rsi_14', 50):.1f}")
        print(f"   MACD: {indicators.get('macd', 0):.4f}")
        print(f"   MACD Signal: {indicators.get('macd_signal', 0):.4f}")
        print(f"   MACD Histogram: {indicators.get('macd_histogram', 0):.4f}")
        
        rsi_14 = indicators.get('rsi_14', 50)
        if rsi_14 < 30:
            print("   🔥 RSI OVERSOLD - Strong BUY signal")
        elif rsi_14 < 40:
            print("   📈 RSI favorable for buying")
        elif rsi_14 > 70:
            print("   🚨 RSI OVERBOUGHT - Sell signal")
        else:
            print("   ⚖️ RSI in neutral zone")
        
        # Volume Analysis
        print(f"\n📊 VOLUME ANALYSIS:")
        print(f"   Current Volume: {indicators.get('volume_current', 0):,.0f}")
        print(f"   20-day Avg: {indicators.get('volume_avg_20', 0):,.0f}")
        print(f"   Volume Ratio: {indicators.get('volume_relative', 1.0):.2f}x")
        
        volume_ratio = indicators.get('volume_relative', 1.0)
        if volume_ratio > 2.0:
            print("   🔊 HIGH volume spike - Strong confirmation")
        elif volume_ratio > 1.5:
            print("   📢 Above average volume - Good confirmation")
        elif volume_ratio < 0.8:
            print("   🔇 LOW volume - Weak confirmation")
        else:
            print("   📊 Normal volume levels")
        
        # Identify the main issue
        print(f"\n🔍 ISSUE IDENTIFICATION:")
        
        issues = []
        if rsi_14 > 60:
            issues.append(f"RSI too high ({rsi_14:.1f}) - reducing momentum score")
        if volume_ratio < 1.2:
            issues.append(f"Low volume ({volume_ratio:.2f}x) - weak confirmation")
        if current_price < sma_20:
            issues.append("Price below SMA20 - bearish trend")
        
        macd_hist = indicators.get('macd_histogram', 0)
        if macd_hist < 0:
            issues.append("MACD histogram negative - bearish momentum")
        
        if issues:
            print("   🚨 IDENTIFIED ISSUES:")
            for i, issue in enumerate(issues, 1):
                print(f"      {i}. {issue}")
        else:
            print("   ✅ No obvious technical issues found")
            print("   💡 Issue may be in algorithm sensitivity or thresholds")
        
        return indicators
        
    except Exception as e:
        print(f"❌ Error in technical analysis: {e}")
        return None

def generate_recommendations():
    """
    Generate final recommendations based on debug results
    """
    print("\n8️⃣ FINAL RECOMMENDATIONS")
    print("-" * 40)
    
    print("Based on the comprehensive debug analysis:")
    print("\n🎯 IMMEDIATE ACTIONS:")
    print("1. Switch to 'Aggressive' strategy (lower thresholds)")
    print("2. Increase timeframe to 14-21 days")
    print("3. Apply enhanced signal sensitivity")
    print("4. Use lenient confidence calculation")
    
    print("\n⚙️ ALGORITHM IMPROVEMENTS:")
    print("1. Add market regime detection")
    print("2. Implement adaptive thresholds based on volatility")
    print("3. Add multi-timeframe confirmation")
    print("4. Consider fundamental factors")
    
    print("\n📊 FOR NVDA SPECIFICALLY:")
    print("1. Monitor for breakout above key resistance levels")
    print("2. Wait for RSI to cool down if overbought")
    print("3. Look for volume confirmation on moves")
    print("4. Consider sector rotation effects")
    
    print("\n🔄 ONGOING MONITORING:")
    print("1. Run this debug weekly to track improvements")
    print("2. Compare signal frequency across different stocks")
    print("3. Adjust strategy based on market conditions")
    print("4. Keep log of successful vs failed signals")

def main():
    """
    Main debugging workflow for NVDA
    """
    print("🎯 NVDA STOCK DEBUGGING WORKFLOW")
    print("="*60)
    
    # Run complete workflow
    try:
        # Step 1-3: Initial diagnosis
        advisor, debugger, debug_results = debug_nvda_comprehensive()
        
        # Step 4: Test before fixes
        before_result = test_nvda_before_fixes()
        
        # Step 5: Apply fixes step by step
        after_result, time_results = apply_fixes_step_by_step()
        
        # Step 6: Compare with other stocks
        comparison_results = compare_nvda_with_other_stocks()
        
        # Step 7: Technical deep dive
        technical_indicators = deep_dive_nvda_technicals()
        
        # Step 8: Generate recommendations
        generate_recommendations()
        
        print("\n" + "="*60)
        print("🎉 NVDA DEBUG ANALYSIS COMPLETE!")
        print("="*60)
        
        # Summary
        if before_result and after_result:
            before_action = before_result.get('action', 'UNKNOWN')
            after_action = after_result.get('action', 'UNKNOWN')
            
            print(f"\n📊 SUMMARY:")
            print(f"Before fixes: {before_action}")
            print(f"After fixes:  {after_action}")
            
            if after_action == "BUY" and before_action != "BUY":
                print("✅ SUCCESS: Fixes converted WAIT/SELL to BUY!")
            elif after_action != "WAIT" and before_action == "WAIT":
                print("✅ IMPROVEMENT: Fixes made the algorithm more decisive!")
            else:
                print("⚠️ Limited improvement - may need further adjustments")
        
    except Exception as e:
        print(f"❌ Error in main workflow: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
