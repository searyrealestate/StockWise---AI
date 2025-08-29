import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock

from stockwise_simulation_gen2 import ProfessionalStockAdvisor, FeatureCalculator

class TestStockwiseGen2(unittest.TestCase):
    def setUp(self):
        dates = pd.to_datetime(pd.date_range(start='2024-01-01', periods=400))
        self.sample_df = pd.DataFrame({
            'Open': np.random.uniform(100, 102, size=400),
            'High': np.random.uniform(102, 104, size=400),
            'Low': np.random.uniform(98, 100, size=400),
            'Close': np.random.uniform(100, 103, size=400),
            'Volume': np.random.randint(1_000_000, 5_000_000, size=400)
        }, index=dates)
        self.sample_df.index.name = 'Date'
        self.mock_data_manager = MagicMock()
        self.mock_data_manager.get_stock_data.return_value = self.sample_df.copy()

    def test_feature_calculator(self):
        print("\n--- Running Test: Feature Calculation ---")
        calculator = FeatureCalculator()
        featured_df = calculator.calculate_all_features(self.sample_df)
        self.assertFalse(featured_df.empty)
        print("✅ Test Passed")

    def test_israeli_tax_and_fees_calculation(self):
        print("\n--- Running Test: Israeli Tax & Fees Calculation ---")
        advisor = ProfessionalStockAdvisor(testing_mode=True)
        net_profit, _ = advisor.apply_israeli_fees_and_tax(10.0)
        self.assertAlmostEqual(net_profit, 7.2, places=2)
        print("✅ Test Passed")

    def test_run_analysis_for_buy_signal(self):
        print("\n--- Running Test: Main Analysis Loop (BUY Signal) ---")
        advisor = ProfessionalStockAdvisor(data_source_manager=self.mock_data_manager, testing_mode=True)
        mock_model = MagicMock()
        mock_model.predict.return_value = [1]
        mock_model.predict_proba.return_value = [[0.1, 0.9]]
        advisor.model = mock_model
        # CRITICAL FIX: The mocked feature names MUST match the features the model would be trained on
        advisor.feature_names = ['Volume_MA_20', 'RSI_14', 'Momentum_5', 'MACD', 'MACD_Signal', 'MACD_Histogram',
                                 'BB_Upper', 'BB_Lower', 'BB_Middle', 'BB_Position', 'Daily_Return',
                                 'Volatility_20D', 'ATR_14', 'ADX', 'ADX_pos', 'ADX_neg', 'OBV',
                                 'RSI_28', 'Dominant_Cycle_126D']
        advisor.model_filename = 'mock_model.pkl'

        _, result = advisor.run_analysis("TEST", analysis_date=datetime(2025, 1, 15))

        self.assertIsNotNone(result, "Analysis result should not be None.")
        self.assertEqual(result['action'], "BUY")
        self.assertAlmostEqual(result['confidence'], 90.0)
        print("✅ Test Passed")

    def test_profit_is_hardcoded_at_5_percent(self):
        print("\n--- Running Test: Confirming Hardcoded 5% Profit ---")
        advisor = ProfessionalStockAdvisor(data_source_manager=self.mock_data_manager, testing_mode=True)
        mock_model = MagicMock()
        mock_model.predict.return_value = [1]
        mock_model.predict_proba.return_value = [[0.1, 0.9]]
        advisor.model = mock_model
        # CRITICAL FIX: The mocked feature names MUST match
        advisor.feature_names = ['Volume_MA_20', 'RSI_14', 'Momentum_5', 'MACD', 'MACD_Signal', 'MACD_Histogram',
                                 'BB_Upper', 'BB_Lower', 'BB_Middle', 'BB_Position', 'Daily_Return',
                                 'Volatility_20D', 'ATR_14', 'ADX', 'ADX_pos', 'ADX_neg', 'OBV',
                                 'RSI_28', 'Dominant_Cycle_126D']
        advisor.model_filename = 'mock_model.pkl'

        _, result = advisor.run_analysis("TEST", analysis_date=datetime(2025, 1, 15))

        self.assertIsNotNone(result)
        self.assertEqual(result['action'], "BUY")
        self.assertAlmostEqual(result['gross_profit_pct'], 5.0)
        print("✅ Test Passed")


if __name__ == '__main__':
    unittest.main()