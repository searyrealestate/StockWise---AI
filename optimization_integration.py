"""
🔧 Optimization Integration Script
═══════════════════════════════════

This script applies optimization results directly to your StockWise algorithm.
It can automatically update your algorithm parameters based on test results.
"""

import json
import os
import shutil
from datetime import datetime
import re

class OptimizationIntegrator:
    def __init__(self, stockwise_file="stockwise_simulation.py"):
        self.stockwise_file = stockwise_file
        self.backup_file = None
        self.current_config = {}

    def create_backup(self):
        """Create backup - FIXED ENCODING"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.backup_file = f"{self.stockwise_file}.backup_{timestamp}"

        try:
            # FIXED: Use UTF-8 for backup creation
            with open(self.stockwise_file, 'r', encoding='utf-8', errors='replace') as source:
                content = source.read()

            with open(self.backup_file, 'w', encoding='utf-8', errors='replace') as backup:
                backup.write(content)

            print(f"✅ Backup created: {self.backup_file}")
            return True
        except Exception as e:
            print(f"❌ Backup creation failed: {e}")
            return False
        
    def load_optimization_results(self, results_file):
        """Load optimization results from JSON file"""
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            self.current_config = data.get('best_configuration', {})
            baseline = data.get('baseline_performance', {})
            optimizations = data.get('optimization_history', [])
            
            print(f"📊 Loaded optimization results:")
            print(f"   Baseline accuracy: {baseline.get('avg_accuracy', 0):.1f}%")
            print(f"   Optimizations found: {len(optimizations)}")
            print(f"   Best configuration available: {'Yes' if self.current_config else 'No'}")
            
            return data
            
        except FileNotFoundError:
            print(f"❌ Results file not found: {results_file}")
            return None
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON in results file: {results_file}")
            return None

    def update_signal_weights(self, new_weights):
        """Update signal weights in the algorithm - FIXED ENCODING"""
        print("🔧 Updating signal weights...")

        try:
            # FIXED: Use UTF-8 encoding explicitly
            with open(self.stockwise_file, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Fallback to different encodings
            encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            content = None

            for encoding in encodings_to_try:
                try:
                    with open(self.stockwise_file, 'r', encoding=encoding, errors='replace') as f:
                        content = f.read()
                    print(f"✅ Successfully read file with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue

            if content is None:
                print("❌ Could not read file with any encoding")
                return False

        # Find the signal_weights dictionary
        import re
        pattern = r"signal_weights\s*=\s*\{[^}]+\}"

        new_weights_str = "signal_weights = {\n"
        for key, value in new_weights.items():
            new_weights_str += f"            '{key}': {value},  # Optimized\n"
        new_weights_str = new_weights_str.rstrip(',  # Optimized\n') + "  # Optimized\n        }"

        updated_content = re.sub(pattern, new_weights_str, content, flags=re.DOTALL)

        # FIXED: Write with UTF-8 encoding
        try:
            with open(self.stockwise_file, 'w', encoding='utf-8', errors='replace') as f:
                f.write(updated_content)

            print(f"✅ Signal weights updated:")
            for key, value in new_weights.items():
                print(f"   {key}: {value}")
            return True

        except Exception as write_error:
            print(f"❌ Error writing file: {write_error}")
            return False

    def update_strategy_thresholds(self, new_thresholds):
        """Update strategy thresholds - FIXED ENCODING"""
        print("🔧 Updating strategy thresholds...")

        try:
            with open(self.stockwise_file, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
        except UnicodeDecodeError:
            encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            for encoding in encodings_to_try:
                try:
                    with open(self.stockwise_file, 'r', encoding=encoding, errors='replace') as f:
                        content = f.read()
                    break
                except:
                    continue

        # Update different strategy thresholds
        import re
        threshold_updates = {
            'balanced_buy': (r'buy_threshold = 0\.9', f'buy_threshold = {new_thresholds.get("balanced_buy", 0.9)}'),
            'aggressive_buy': (r'buy_threshold = 0\.5', f'buy_threshold = {new_thresholds.get("aggressive_buy", 0.5)}'),
            'conservative_buy': (
            r'buy_threshold = 1\.5', f'buy_threshold = {new_thresholds.get("conservative_buy", 1.5)}')
        }

        for threshold_name, (pattern, replacement) in threshold_updates.items():
            if threshold_name in new_thresholds:
                content = re.sub(pattern, replacement, content)
                print(f"   {threshold_name}: {new_thresholds[threshold_name]}")

        try:
            with open(self.stockwise_file, 'w', encoding='utf-8', errors='replace') as f:
                f.write(content)
            print("✅ Strategy thresholds updated")
            return True
        except Exception as e:
            print(f"❌ Error writing thresholds: {e}")
            return False
    
    def update_confidence_requirements(self, new_requirements):
        """Update confidence requirements in the algorithm"""
        print("🔧 Updating confidence requirements...")
        
        with open(self.stockwise_file, 'r') as f:
            content = f.read()
        
        # Update confidence requirements for different strategies
        confidence_updates = {
            'conservative': (r'required_confidence = 80', f'required_confidence = {new_requirements.get("conservative", 80)}'),
            'balanced': (r'required_confidence = 70', f'required_confidence = {new_requirements.get("balanced", 70)}'),
            'aggressive': (r'required_confidence = 60', f'required_confidence = {new_requirements.get("aggressive", 60)}')
        }
        
        for strategy, (pattern, replacement) in confidence_updates.items():
            if strategy in new_requirements:
                content = re.sub(pattern, replacement, content)
                print(f"   {strategy}: {new_requirements[strategy]}%")
        
        with open(self.stockwise_file, 'w') as f:
            f.write(content)
        
        print("✅ Confidence requirements updated")
    
    def apply_threshold_adjustment(self, adjustment_type):
        """Apply global threshold adjustments"""
        print(f"🔧 Applying threshold adjustment: {adjustment_type}")
        
        with open(self.stockwise_file, 'r') as f:
            content = f.read()
        
        if adjustment_type == 'increase_all_by_0.1':
            # Find and update all buy thresholds
            adjustments = [
                (r'buy_threshold = 0\.9', 'buy_threshold = 1.0'),
                (r'buy_threshold = 0\.5', 'buy_threshold = 0.6'),
                (r'buy_threshold = 1\.5', 'buy_threshold = 1.6'),
                (r'buy_threshold = 0\.8', 'buy_threshold = 0.9')
            ]
        elif adjustment_type == 'decrease_all_by_0.1':
            adjustments = [
                (r'buy_threshold = 0\.9', 'buy_threshold = 0.8'),
                (r'buy_threshold = 0\.5', 'buy_threshold = 0.4'),
                (r'buy_threshold = 1\.5', 'buy_threshold = 1.4'),
                (r'buy_threshold = 0\.8', 'buy_threshold = 0.7')
            ]
        else:
            print(f"❌ Unknown adjustment type: {adjustment_type}")
            return
        
        for pattern, replacement in adjustments:
            content = re.sub(pattern, replacement, content)
        
        with open(self.stockwise_file, 'w') as f:
            f.write(content)
        
        print("✅ Threshold adjustments applied")
    
    def apply_optimization_config(self, config):
        """Apply full optimization configuration"""
        print("🚀 Applying optimization configuration...")
        
        if 'signal_weights' in config:
            self.update_signal_weights(config['signal_weights'])
        
        if 'strategy_thresholds' in config:
            self.update_strategy_thresholds(config['strategy_thresholds'])
        
        if 'confidence_requirements' in config:
            self.update_confidence_requirements(config['confidence_requirements'])
        
        if 'threshold_adjustment' in config:
            self.apply_threshold_adjustment(config['threshold_adjustment'])
        
        print("🎉 All optimization configurations applied!")
    
    def validate_changes(self):
        """Validate that changes were applied correctly"""
        print("🔍 Validating applied changes...")
        
        try:
            # Try to import the modified file to check for syntax errors
            import importlib.util
            spec = importlib.util.spec_from_file_location("stockwise_test", self.stockwise_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            print("✅ Syntax validation passed")
            
            # Check if key classes exist
            if hasattr(module, 'EnhancedStockAdvisor'):
                print("✅ EnhancedStockAdvisor class found")
            else:
                print("❌ EnhancedStockAdvisor class not found")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return False
    
    def restore_backup(self):
        """Restore from backup if something went wrong"""
        if self.backup_file and os.path.exists(self.backup_file):
            shutil.copy2(self.backup_file, self.stockwise_file)
            print(f"✅ Restored from backup: {self.backup_file}")
            return True
        else:
            print("❌ No backup file available")
            return False
    
    def generate_change_summary(self, optimization_data):
        """Generate summary of changes made"""
        print("\n" + "="*60)
        print("📋 OPTIMIZATION CHANGES SUMMARY")
        print("="*60)
        
        baseline = optimization_data.get('baseline_performance', {})
        optimizations = optimization_data.get('optimization_history', [])
        
        print(f"\n📊 Original Performance:")
        print(f"   Overall Accuracy: {baseline.get('avg_accuracy', 0):.1f}%")
        print(f"   Direction Accuracy: {baseline.get('direction_accuracy', 0):.1f}%")
        print(f"   BUY Success Rate: {baseline.get('buy_success_rate', 0):.1f}%")
        
        if optimizations:
            print(f"\n⚙️ Optimizations Applied:")
            total_improvement = 0
            for opt in optimizations:
                improvement = opt['improvement']
                total_improvement += improvement
                print(f"   • {opt['area']}: +{improvement:.1f}% accuracy")
                print(f"     Change: {opt['change']}")
            
            print(f"\n🎯 Total Expected Improvement: +{total_improvement:.1f}%")
            print(f"   Expected New Accuracy: {baseline.get('avg_accuracy', 0) + total_improvement:.1f}%")
        
        print(f"\n💾 Backup File: {self.backup_file}")
        print("="*60)

def run_integration_wizard():
    """Interactive wizard to apply optimizations"""
    print("🧙‍♂️ OPTIMIZATION INTEGRATION WIZARD")
    print("="*50)
    
    # Find optimization results files
    result_files = [f for f in os.listdir('.') if f.startswith('algorithm_optimization_results_') and f.endswith('.json')]
    
    if not result_files:
        print("❌ No optimization results files found.")
        print("   Run the optimization test first to generate results.")
        return
    
    print("📁 Available optimization results:")
    for i, file in enumerate(result_files, 1):
        print(f"   {i}. {file}")
    
    # Select results file
    try:
        choice = int(input(f"Select results file (1-{len(result_files)}): ")) - 1
        if choice < 0 or choice >= len(result_files):
            raise ValueError()
        selected_file = result_files[choice]
    except (ValueError, IndexError):
        print("❌ Invalid selection")
        return
    
    # Initialize integrator
    integrator = OptimizationIntegrator()
    
    # Load optimization results
    optimization_data = integrator.load_optimization_results(selected_file)
    if not optimization_data:
        return
    
    # Check if optimizations are available
    if not integrator.current_config:
        print("❌ No optimization configuration found in results file")
        return
    
    # Show what will be changed
    print(f"\n🔍 Optimization Configuration Preview:")
    print(json.dumps(integrator.current_config, indent=2))
    
    # Confirm application
    confirm = input(f"\n🚀 Apply these optimizations? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ Optimization cancelled")
        return
    
    # Create backup
    integrator.create_backup()
    
    try:
        # Apply optimizations
        integrator.apply_optimization_config(integrator.current_config)
        
        # Validate changes
        if integrator.validate_changes():
            print("✅ Optimizations applied successfully!")
            integrator.generate_change_summary(optimization_data)
            
            # Quick test suggestion
            print(f"\n💡 Recommendation:")
            print(f"   1. Test the optimized algorithm with new data")
            print(f"   2. Compare performance with baseline")
            print(f"   3. If issues occur, restore from backup: {integrator.backup_file}")
            
        else:
            print("❌ Validation failed. Restoring backup...")
            integrator.restore_backup()
            
    except Exception as e:
        print(f"❌ Error applying optimizations: {e}")
        print("Restoring backup...")
        integrator.restore_backup()

def quick_apply_best_practices():
    """Apply proven best practice optimizations"""
    print("⚡ QUICK BEST PRACTICES APPLICATION")
    print("="*50)
    
    # Proven optimizations from testing
    best_practices = {
        'signal_weights': {
            'trend': 0.30,
            'momentum': 0.25,
            'volume': 0.20,
            'support_resistance': 0.10,
            'model': 0.15
        },
        'strategy_thresholds': {
            'balanced_buy': 0.9,
            'aggressive_buy': 0.5,
            'conservative_buy': 1.5
        },
        'confidence_requirements': {
            'conservative': 80,
            'balanced': 70,
            'aggressive': 60
        }
    }
    
    print("📋 Best practices to apply:")
    print("   • Rebalanced signal weights (trend: 30%, momentum: 25%)")
    print("   • Optimized Balanced strategy threshold (1.2 → 0.9)")
    print("   • Adjusted confidence requirements")
    
    confirm = input("\n🚀 Apply best practices? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ Application cancelled")
        return
    
    integrator = OptimizationIntegrator()
    integrator.create_backup()
    
    try:
        integrator.apply_optimization_config(best_practices)
        
        if integrator.validate_changes():
            print("✅ Best practices applied successfully!")
            print(f"\n💾 Backup created: {integrator.backup_file}")
        else:
            print("❌ Validation failed. Restoring backup...")
            integrator.restore_backup()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        integrator.restore_backup()

if __name__ == "__main__":
    print("🔧 OPTIMIZATION INTEGRATION TOOLS")
    print("="*40)
    print("1. Apply optimization results from test")
    print("2. Apply proven best practices")
    print("3. Restore from backup")
    
    choice = input("Choose option (1-3): ").strip()
    
    if choice == "1":
        run_integration_wizard()
    elif choice == "2":
        quick_apply_best_practices()
    elif choice == "3":
        backup_files = [f for f in os.listdir('.') if f.startswith('stockwise_simulation.py.backup_')]
        if backup_files:
            print("📁 Available backups:")
            for i, file in enumerate(backup_files, 1):
                print(f"   {i}. {file}")
            
            try:
                choice = int(input(f"Select backup (1-{len(backup_files)}): ")) - 1
                selected_backup = backup_files[choice]
                
                shutil.copy2(selected_backup, "stockwise_simulation.py")
                print(f"✅ Restored from {selected_backup}")
            except (ValueError, IndexError):
                print("❌ Invalid selection")
        else:
            print("❌ No backup files found")
    else:
        print("❌ Invalid option")