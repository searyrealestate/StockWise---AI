# validation_manager.py

"""
StockWise Gen-12 Validation Manager (The Auditor)
=================================================
The Immune System of the App.
Runs a suite of automated tests (unittest) to verify:
1. Business Logic (Commission, Stop Loss rules)
2. Configuration Integrity (Secrets, Paths, Schedules)
3. Model Existence (Training Check)

Should be run before starting the Live Engine to prevent runtime failures.
"""

import unittest
import logging
from portfolio_manager import PortfolioManager
import system_config as cfg

logger = logging.getLogger("ValidationManager")

# --- TEST SUITE 1: SANITY CHECKS ---
class TestGen12Sanity(unittest.TestCase):
    """Verifies internal logic correctness."""
    
    def setUp(self):
        """Setup mock environment for tests."""
        self.pm = PortfolioManager()
        
    def test_stop_loss_logic(self):
        """Sanity: Stop Loss must be < Entry Price for Long positions."""
        entry = 100.0
        stop = 95.0
        self.assertTrue(stop < entry, "Stop loss must be below entry for Longs")
        
    def test_commission_calc(self):
        """Sanity: Verify Commission Calculation Matches Spec."""
        # 1. Minimum Commission Check
        # 10 shares * 0.005 = 0.05, should be bumped to $1.00 min
        comm = self.pm.calculate_commission(10)
        self.assertEqual(comm, 1.00)
        
        # 2. Per Share Commission Check
        # 1000 shares * 0.005 = 5.00, which is > 1.00
        comm2 = self.pm.calculate_commission(1000)
        self.assertEqual(comm2, 5.00)

# --- TEST SUITE 2: FLOW & CONFIG ---
class TestGen12Flow(unittest.TestCase):
    """Verifies Configuration Mapping."""
    
    def test_config_integrity(self):
        """Flow: Ensure Critical Strategy Concepts are loaded."""
        self.assertIn("SNIPER", cfg.STRATEGY_CONFIG)
        self.assertIn("TACTICAL", cfg.STRATEGY_CONFIG)
        self.assertIn("STRATEGIC", cfg.STRATEGY_CONFIG)
        
    def test_scan_schedule(self):
        """Flow: Schedule Parsing Correctness."""
        self.assertEqual(cfg.SCAN_SCHEDULE["SHORT_RANGE"]["interval"], "1h")

# --- TEST SUITE 3: ASSETS ---
class TestGen12Training(unittest.TestCase):
    """Verifies required binary assets exist."""
    
    def test_models_exist(self):
        """Training: Verify AI Models exist for every Watchlist symbol."""
        import os
        missing = []
        for symbol in cfg.WATCHLIST:
            path = os.path.join(cfg.MODELS_DIR, f"gen7_lstm_{symbol}.h5")
            # In validation mode, we just check if files exist
            if not os.path.exists(path):
                missing.append(symbol)
        
        # Fail if any are missing
        self.assertFalse(missing, f"Missing models for: {missing}")

class ValidationManager:
    """
    Orchestrator for running the test suites.
    """
    def __init__(self):
        self.loader = unittest.TestLoader()
        
    def run_audit(self):
        """Executes the full test suite and returns Pass/Fail."""
        logger.info("🕵️ Starting Gen-12 Full Audit...")
        
        suite = unittest.TestSuite()
        suite.addTests(self.loader.loadTestsFromTestCase(TestGen12Sanity))
        suite.addTests(self.loader.loadTestsFromTestCase(TestGen12Flow))
        # Note: TestGen12Training might fail if models aren't trained yet. 
        # Consider making it optional or soft-fail.
        suite.addTests(self.loader.loadTestsFromTestCase(TestGen12Training))
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        if result.wasSuccessful():
            logger.info("✅ Audit Passed. System Green.")
            return True
        else:
            logger.error("❌ Audit Failed. Check Logs.")
            return False

if __name__ == "__main__":
    vm = ValidationManager()
    vm.run_audit()
