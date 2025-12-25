#!/usr/bin/env python3
"""
🚀 Complete Algorithm Optimization Workflow
═══════════════════════════════════════════

This script runs the complete optimization workflow:
1. Tests current algorithm performance on random weeks
2. Identifies optimization opportunities
3. Auto-tunes parameters
4. Applies best configuration
5. Validates improvements

Usage: python run_optimization.py
"""

import sys
import os
from datetime import datetime
import json


def run_complete_optimization():
    """Run the complete optimization workflow"""

    print("🚀 STOCKWISE ALGORITHM OPTIMIZATION SUITE")
    print("=" * 60)
    print("This will:")
    print("1. 📊 Test current performance on random market weeks")
    print("2. 🎯 Identify optimization opportunities")
    print("3. ⚙️ Auto-tune parameters for better accuracy")
    print("4. 🔧 Apply the best configuration found")
    print("5. ✅ Validate improvements")
    print("=" * 60)

    # Get user confirmation
    confirm = input("🚀 Start optimization? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ Optimization cancelled")
        return

    try:
        # FIXED: Import from the correct file names
        from algo_testing_script import AdvancedAlgorithmTester
        from optimization_integration import OptimizationIntegrator

        print("\n" + "=" * 60)
        print("📊 PHASE 1: BASELINE PERFORMANCE TESTING")
        print("=" * 60)

        # Step 1: Run baseline performance test
        tester = AdvancedAlgorithmTester(debug=True)

        # Define test stocks (popular, liquid stocks)
        test_stocks = [
            'AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA',
            'AMZN', 'META', 'AMD', 'NFLX', 'CRM'
        ]

        print(f"Testing with stocks: {test_stocks}")

        # Run baseline test on 4 random weeks with 8 stocks
        baseline_results = tester.run_random_week_test(
            stock_symbols=test_stocks,
            start_date="2024-01-01",
            end_date="2024-12-31",
            num_weeks=4,
            num_stocks=8
        )

        if not baseline_results:
            print("❌ Baseline test failed - cannot continue")
            return

        baseline_accuracy = tester.baseline_performance['avg_accuracy']
        print(f"\n📊 Baseline Performance: {baseline_accuracy:.1f}% accuracy")

        # Step 2: Auto-tune parameters
        print("\n" + "=" * 60)
        print("⚙️ PHASE 2: AUTOMATED PARAMETER TUNING")
        print("=" * 60)

        best_config, best_accuracy = tester.auto_tune_parameters(
            stock_symbols=test_stocks,
            optimization_iterations=3
        )

        improvement = best_accuracy - baseline_accuracy if best_config else 0

        if improvement > 0:
            print(f"\n🎉 Optimization successful!")
            print(f"   Baseline: {baseline_accuracy:.1f}%")
            print(f"   Optimized: {best_accuracy:.1f}%")
            print(f"   Improvement: +{improvement:.1f}%")
        else:
            print(f"\n⚠️ No significant improvements found")
            print(f"   Current performance: {baseline_accuracy:.1f}%")
            print(f"   Algorithm may already be well-tuned")

        # Step 3: Generate and save detailed report
        print("\n" + "=" * 60)
        print("📋 PHASE 3: GENERATING OPTIMIZATION REPORT")
        print("=" * 60)

        tester.generate_optimization_report()
        results_file = tester.save_results()

        # Step 4: Apply optimizations if found
        if best_config and improvement > 1.0:  # Only apply if >1% improvement
            print("\n" + "=" * 60)
            print("🔧 PHASE 4: APPLYING OPTIMIZATIONS")
            print("=" * 60)

            print("Found significant improvements. Applying optimizations...")

            integrator = OptimizationIntegrator()
            integrator.create_backup()

            try:
                integrator.apply_optimization_config(best_config)

                if integrator.validate_changes():
                    print("✅ Optimizations applied successfully!")

                    # Step 5: Quick validation test
                    print("\n" + "=" * 60)
                    print("✅ PHASE 5: VALIDATION TEST")
                    print("=" * 60)

                    print("Running quick validation test...")

                    # Test with 2 random weeks, 5 stocks
                    validation_tester = AdvancedAlgorithmTester(debug=False)
                    validation_results = validation_tester.run_random_week_test(
                        stock_symbols=test_stocks[:5],
                        num_weeks=2,
                        num_stocks=5
                    )

                    if validation_results:
                        validation_accuracy = validation_tester.baseline_performance['avg_accuracy']
                        print(f"   Validation accuracy: {validation_accuracy:.1f}%")

                        if validation_accuracy > baseline_accuracy:
                            print("🎉 Validation successful! Improvements confirmed.")
                        else:
                            print("⚠️ Validation shows mixed results. Monitor performance.")

                else:
                    print("❌ Validation failed. Restoring backup...")
                    integrator.restore_backup()

            except Exception as e:
                print(f"❌ Error applying optimizations: {e}")
                integrator.restore_backup()

        else:
            print("\n" + "=" * 60)
            print("📊 PHASE 4: RECOMMENDATIONS")
            print("=" * 60)

            if improvement <= 1.0:
                print("Current algorithm performance is already strong.")
                print("Consider these manual optimizations:")
                print("• Monitor performance over longer periods")
                print("• Test with different market conditions")
                print("• Adjust strategy based on your risk tolerance")

        # Final summary
        print("\n" + "=" * 60)
        print("🎯 OPTIMIZATION COMPLETE")
        print("=" * 60)

        print(f"📊 Baseline Performance: {baseline_accuracy:.1f}%")
        if best_config:
            print(f"🚀 Best Performance Found: {best_accuracy:.1f}%")
            print(f"📈 Total Improvement: +{improvement:.1f}%")
        print(f"📁 Detailed Results: {results_file}")

        if 'integrator' in locals() and hasattr(integrator, 'backup_file'):
            print(f"💾 Backup File: {integrator.backup_file}")

        print("\n💡 Next Steps:")
        print("1. Monitor algorithm performance over next few days")
        print("2. Run additional tests if needed")
        print("3. Keep backup file in case rollback is needed")
        print("4. Consider periodic re-optimization")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure algo_testing_script.py and optimization_integration.py are in the same directory")
        print("\n🔍 Checking files in current directory:")

        # List relevant Python files in current directory
        python_files = [f for f in os.listdir('.') if f.endswith('.py')]
        print(f"Python files found: {python_files}")

        # Check specifically for required files
        required_files = ['algo_testing_script.py', 'optimization_integration.py', 'stockwise_simulation.py']
        missing_files = [f for f in required_files if f not in python_files]

        if missing_files:
            print(f"❌ Missing required files: {missing_files}")
        else:
            print("✅ All required files are present")
            print("💡 The import error might be due to other dependencies")

        return False

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("Check error details above and try again")
        return False


def run_quick_test_only():
    """Run just a quick performance test without optimization"""

    print("📊 QUICK PERFORMANCE TEST")
    print("=" * 40)

    try:
        # FIXED: Import from correct file name
        from algo_testing_script import AdvancedAlgorithmTester

        tester = AdvancedAlgorithmTester(debug=True)

        test_stocks = ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA']

        results = tester.run_random_week_test(
            stock_symbols=test_stocks,
            num_weeks=2,
            num_stocks=5
        )

        if results:
            tester.generate_optimization_report()
            print("\n🎯 Quick test complete!")

            accuracy = tester.baseline_performance['avg_accuracy']
            if accuracy >= 75:
                print(f"✅ Algorithm performing well: {accuracy:.1f}%")
            elif accuracy >= 65:
                print(f"⚠️ Algorithm performing adequately: {accuracy:.1f}%")
            else:
                print(f"❌ Algorithm needs optimization: {accuracy:.1f}%")
                print("   Consider running full optimization")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure algo_testing_script.py is in the same directory")
        return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def restore_from_backup():
    """Restore algorithm from backup file"""

    print("🔄 RESTORE FROM BACKUP")
    print("=" * 30)

    backup_files = [f for f in os.listdir('.') if f.startswith('stockwise_simulation.py.backup_')]

    if not backup_files:
        print("❌ No backup files found")
        return False

    print("📁 Available backups:")
    for i, file in enumerate(backup_files, 1):
        # Extract timestamp from filename
        timestamp = file.split('backup_')[1]
        print(f"   {i}. {timestamp}")

    try:
        choice = int(input(f"Select backup (1-{len(backup_files)}): ")) - 1
        if choice < 0 or choice >= len(backup_files):
            raise ValueError()

        selected_backup = backup_files[choice]

        import shutil
        shutil.copy2(selected_backup, "stockwise_simulation.py")
        print(f"✅ Restored from {selected_backup}")
        return True

    except (ValueError, IndexError):
        print("❌ Invalid selection")
        return False


def check_dependencies():
    """Check if all required dependencies are available"""

    print("🔍 CHECKING DEPENDENCIES...")
    print("=" * 40)

    required_modules = [
        'pandas', 'yfinance', 'numpy', 'ta', 'streamlit',
        'plotly', 'joblib', 'sklearn'
    ]

    missing_modules = []

    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module} - MISSING")
            missing_modules.append(module)

    if missing_modules:
        print(f"\n❌ Missing modules: {missing_modules}")
        print("Install them with: pip install " + " ".join(missing_modules))
        return False
    else:
        print("\n✅ All dependencies are available")
        return True


def main():
    """Main menu with enhanced error checking"""

    print("🎯 STOCKWISE OPTIMIZATION SUITE")
    print("=" * 40)
    print("1. 🚀 Complete optimization workflow")
    print("2. 📊 Quick performance test only")
    print("3. 🔄 Restore from backup")
    print("4. 🔍 Check dependencies")
    print("5. ❌ Exit")

    choice = input("Choose option (1-5): ").strip()

    if choice == "1":
        success = run_complete_optimization()
        if success:
            print("\n🎉 Optimization workflow completed successfully!")
        else:
            print("\n❌ Optimization workflow failed")

    elif choice == "2":
        success = run_quick_test_only()
        if success:
            print("\n✅ Quick test completed")

    elif choice == "3":
        success = restore_from_backup()
        if success:
            print("\n✅ Restore completed")

    elif choice == "4":
        check_dependencies()

    elif choice == "5":
        print("👋 Goodbye!")

    else:
        print("❌ Invalid option")


if __name__ == "__main__":
    main()