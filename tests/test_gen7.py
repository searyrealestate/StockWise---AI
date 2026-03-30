import unittest
import pandas as pd
import asyncio
import os
import sys

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# MOCK DEPENDENCIES
from unittest.mock import MagicMock
sys.modules['pandas_ta_classic'] = MagicMock()
sys.modules['stockstats'] = MagicMock()
sys.modules['tensorflow'] = MagicMock() # Mock TF for other imports if needed

from feature_engine import calculate_ground_truth
from strategy_engine import StrategyOrchestra

class TestGen7Upgrade(unittest.TestCase):
    
    def test_ground_truth(self):
        print("\nTesting Ground Truth Logic...")
        # Mock DF: Drop then rally
        data = {
            'close': [100, 95, 90, 88, 92, 98, 105, 110, 112, 115, 120, 122, 125, 128, 130],
            'high':  [102, 97, 92, 90, 94, 100, 108, 112, 114, 118, 122, 125, 128, 130, 135],
            'low':   [99,  94, 89, 87, 85, 90, 100, 108, 110, 112, 118, 120, 122, 124, 128],
        }
        df = pd.DataFrame(data)
        # Pad with enough rows for lookahead
        df = pd.concat([df] * 10, ignore_index=True)
        
        df = calculate_ground_truth(df, lookahead=5)
        
        # Check if we have targets
        if 'target_ground_truth' in df.columns:
            print("✅ Ground Truth Column Created")
            print(df[['close', 'target_ground_truth']].head(10))
        else:
            self.fail("Ground Truth column missing")

    def test_strategy_momentum(self):
        print("\nTesting Momentum Agent...")
        strat = StrategyOrchestra()
        
        # Mock Bullish Setup
        df = pd.DataFrame([{
            'close': 100,
            'supertrend_direction': 1,
            'adx': 40,
            'rsi_14': 75, # Overbought but High ADX -> Should Buy
            'atr_14': 2.0
        }])
        
        decision = strat.decide_action("TEST", df, {}, 0.9, allowed_agents=["MOMENTUM"])
        
        if decision and decision[3] == "MOMENTUM":
            print(f"✅ Momentum Logic Passed (Captured High ADX Overbought): {decision}")
        else:
            self.fail(f"Momentum Logic Failed: {decision}")

    def test_strategy_reversion(self):
        print("\nTesting Reversion Agent...")
        strat = StrategyOrchestra()
        
        # Mock Knife Catch Setup
        df = pd.DataFrame([{
            'close': 100,
            'rsi_14': 20,
            'wt1': -70,
            'slope_angle': -50, # Falling Knife
            'atr_14': 2.0
        }])
        
        # 1. Low Confidence -> Should Fail (Vetoed)
        decision_fail = strat._reversion_agent(df, 0.4)
        if decision_fail is not None and decision_fail['score'] < 0:
             print("✅ Reversion Safety Passed (Vetoed low confidence knife)")
        
        # 2. High Confidence -> Should Buy
        decision_pass = strat._reversion_agent(df, 0.9)
        if decision_pass and decision_pass['score'] > 0:
            print("✅ Reversion Override Passed (Caught knife with high AI conf)")
        else:
             self.fail("Reversion Override Failed")

if __name__ == '__main__':
    unittest.main()
