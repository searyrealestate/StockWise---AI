import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import logging
from unittest.mock import MagicMock, patch

# Configure a basic logging setup for the test runner itself
# This should be done carefully to not interfere with file handlers of algo_configurator
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

# Add parent directory to sys.path to allow imports from stockwise_simulation and algo_configurator
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from stockwise_simulation import ProfessionalStockAdvisor
    from algo_configurator import StockWiseAutoCalibrator

    logging.info("✅ Successfully imported ProfessionalStockAdvisor and StockWiseAutoCalibrator.")
except ImportError as e:
    logging.error(f"❌ Error importing modules: {e}")
    logging.error("Please ensure 'stockwise_simulation.py' and 'algo_configurator.py' are in the same directory.")
    sys.exit(1)


class TestStockWiseAutoCalibrator(unittest.TestCase):
    """
    Unit tests for the StockWiseAutoCalibrator class in algo_configurator.py.
    Mocks external dependencies like yfinance and ProfessionalStockAdvisor for isolated testing.
    """

    @classmethod
    def setUpClass(cls):
        """Set up class-level resources (e.g., suppress some logging during tests)."""
        # Suppress logging from yfinance and other noisy modules during tests
        logging.getLogger('yfinance').setLevel(logging.CRITICAL)
        logging.getLogger('urllib3').setLevel(logging.CRITICAL)
        logging.getLogger('numexpr').setLevel(logging.CRITICAL)
        logging.getLogger('stockwise_simulation').setLevel(
            logging.WARNING)  # Allow warnings/errors from advisor if needed for test cases

    def setUp(self):
        """
        Set up for each test method.
        Initializes a mock ProfessionalStockAdvisor and a StockWiseAutoCalibrator instance.
        Ensures existing log file handlers are closed before potential cleanup.
        """
        # Determine the configuration files directory
        self.config_files_dir = os.path.join(current_dir, 'configuration_files')
        os.makedirs(self.config_files_dir, exist_ok=True)

        # Close and remove any existing file handlers from the root logger
        # This is crucial to release file locks from previous runs or global setup
        for handler in list(logging.getLogger().handlers):
            if isinstance(handler, logging.FileHandler):
                try:
                    handler.close()
                    logging.getLogger().removeHandler(handler)
                except Exception as e:
                    logging.warning(f"Error closing/removing handler in setUp: {e}")

        # Clean up any potential log files from previous failed tests
        for f_name in os.listdir(self.config_files_dir):
            if f_name.startswith('calibration_progress_') or f_name.startswith('stockwise_partial_calibration_'):
                file_path = os.path.join(self.config_files_dir, f_name)
                try:
                    os.remove(file_path)
                    # logging.info(f"Cleaned up old log file: {file_path}") # Optional: keep for verbose cleanup
                except PermissionError as e:
                    logging.warning(
                        f"Could not remove old log file {file_path}: {e}. It might be in use by another process or previous run.")
                except Exception as e:
                    logging.error(f"Error removing file {file_path}: {e}")

        # Mock the ProfessionalStockAdvisor instance
        self.mock_advisor = MagicMock(spec=ProfessionalStockAdvisor)
        # Set default return values for methods the calibrator will call
        self.mock_advisor.analyze_stock_enhanced.return_value = {
            "action": "BUY",
            "confidence": 80.0,
            "expected_profit_pct": 5.0,
            "reasons": ["Mocked positive trend"],
            "trading_plan": {},
            "signal_strengths": {},
            "current_strategy": "Balanced",
            "final_score": 1.5
        }
        # Mock specific methods/attributes of the advisor that the calibrator interacts with
        self.mock_advisor.get_default_strategy_settings.return_value = {
            "Conservative": {"profit": 0.8, "risk": 0.8, "confidence_req": 85, "buy_threshold": 1.0,
                             "sell_threshold": -1.0},
            "Balanced": {"profit": 1.0, "risk": 1.0, "confidence_req": 75, "buy_threshold": 0.9,
                         "sell_threshold": -0.9},
            "Aggressive": {"profit": 1.4, "risk": 1.3, "confidence_req": 65, "buy_threshold": 0.6,
                           "sell_threshold": -0.6},
            "Swing Trading": {"profit": 1.8, "risk": 1.5, "confidence_req": 70, "buy_threshold": 0.8,
                              "sell_threshold": -0.8}
        }
        # Initialize attributes that algo_configurator.py attempts to modify
        self.mock_advisor.strategy_settings = self.mock_advisor.get_default_strategy_settings()
        self.mock_advisor.signal_weights = {
            'trend': 0.45, 'momentum': 0.30, 'volume': 0.10,
            'support_resistance': 0.05, 'ai_model': 0.10
        }
        self.mock_advisor.confidence_params = {
            'base_multiplier': 1.0, 'confluence_weight': 1.0, 'penalty_strength': 1.0
        }
        self.mock_advisor.investment_days = 7
        self.mock_advisor.current_strategy = "Balanced"
        self.mock_advisor.current_buy_threshold = 0.9
        self.mock_advisor.current_sell_threshold = -0.9

        # Initialize the calibrator with the mock advisor
        # This will trigger algo_configurator's global logging setup for the current test run
        self.calibrator = StockWiseAutoCalibrator(self.mock_advisor)

    def tearDown(self):
        """
        Clean up after each test method.
        Ensures log file handlers opened during the test are closed to release file locks.
        """
        # Close any file handlers that might have been opened during the test
        # (e.g., by algo_configurator's global logging setup)
        for handler in list(logging.getLogger().handlers):
            if isinstance(handler, logging.FileHandler):
                try:
                    handler.close()
                    logging.getLogger().removeHandler(handler)
                except Exception as e:
                    logging.warning(f"Error closing/removing handler in tearDown: {e}")

        # Re-run cleanup for good measure, after handlers are guaranteed closed
        for f_name in os.listdir(self.config_files_dir):
            if f_name.startswith('calibration_progress_') or f_name.startswith('stockwise_partial_calibration_'):
                file_path = os.path.join(self.config_files_dir, f_name)
                try:
                    os.remove(file_path)
                except Exception as e:
                    logging.warning(f"Failed to remove file during tearDown cleanup: {file_path} - {e}")

    def test_get_default_config(self):
        """Test if the default configuration is loaded correctly."""
        config = self.calibrator.get_default_config()
        self.assertIsInstance(config, dict)
        self.assertIn('stock_universe', config)
        self.assertIn('advisor_params', config)
        self.assertIn('walk_forward', config)
        self.assertIn('prediction_window_days', config)
        self.assertGreater(len(config['advisor_params']), 0)
        self.assertEqual(config['walk_forward']['train_months'], 12)
        self.assertEqual(config['prediction_window_days'], 7)
        logging.info("✅ test_get_default_config passed.")

    @patch('algo_configurator.yf.download')
    def test_calculate_actual_return_positive(self, mock_yfinance_download):
        """Test calculate_actual_return for a positive return scenario."""
        # Mock yfinance.download to return a DataFrame
        mock_yfinance_download.return_value = pd.DataFrame({
            'Close': [100.0, 101.0, 102.0, 103.0, 105.0]
        }, index=pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']))

        # Use a timestamp that exists in the mocked data
        actual_return = self.calibrator.calculate_actual_return('AAPL', datetime(2024, 1, 1).date(), 3)
        self.assertIsNotNone(actual_return)
        # Expected return: (103 - 100) / 100 * 100 = 3.0
        self.assertAlmostEqual(actual_return, 3.0)
        logging.info("✅ test_calculate_actual_return_positive passed.")

    @patch('algo_configurator.yf.download')
    def test_calculate_actual_return_negative(self, mock_yfinance_download):
        """Test calculate_actual_return for a negative return scenario."""
        mock_yfinance_download.return_value = pd.DataFrame({
            'Close': [100.0, 99.0, 98.0, 97.0, 95.0]
        }, index=pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']))

        actual_return = self.calibrator.calculate_actual_return('AAPL', datetime(2024, 1, 1).date(), 3)
        self.assertIsNotNone(actual_return)
        # Expected return: (97 - 100) / 100 * 100 = -3.0
        self.assertAlmostEqual(actual_return, -3.0)
        logging.info("✅ test_calculate_actual_return_negative passed.")

    @patch('algo_configurator.yf.download')
    def test_calculate_actual_return_no_future_data(self, mock_yfinance_download):
        """Test calculate_actual_return when future data is not available."""
        mock_yfinance_download.return_value = pd.DataFrame({
            'Close': [100.0, 101.0, 102.0]
        }, index=pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']))

        actual_return = self.calibrator.calculate_actual_return('AAPL', datetime(2024, 1, 1).date(),
                                                                5)  # Needs 5 days, only 3 available
        self.assertIsNone(actual_return)
        logging.info("✅ test_calculate_actual_return_no_future_data passed.")

    @patch('algo_configurator.yf.download')
    def test_calculate_actual_return_empty_df(self, mock_yfinance_download):
        """Test calculate_actual_return with an empty DataFrame from yfinance."""
        mock_yfinance_download.return_value = pd.DataFrame()  # Empty DataFrame

        actual_return = self.calibrator.calculate_actual_return('AAPL', datetime(2024, 1, 1).date(), 3)
        self.assertIsNone(actual_return)
        logging.info("✅ test_calculate_actual_return_empty_df passed.")

    def test_evaluate_prediction(self):
        """Test evaluate_prediction logic for different scenarios."""
        # Scenario 1: BUY, actual profit > 3% (correct)
        rec = {"action": "BUY", "confidence": 80, "expected_profit_pct": 5}
        eval_result = self.calibrator.evaluate_prediction(rec, 4.0)
        self.assertTrue(eval_result['correct'])
        self.assertTrue(eval_result['direction_correct'])
        self.assertTrue(eval_result['profitable'])
        logging.info("✅ test_evaluate_prediction: BUY, profitable passed.")

        # Scenario 2: SELL/AVOID, actual loss < -2% (correct)
        rec = {"action": "SELL/AVOID", "confidence": 70, "expected_profit_pct": -3}
        eval_result = self.calibrator.evaluate_prediction(rec, -3.5)
        self.assertTrue(eval_result['correct'])
        self.assertTrue(eval_result['direction_correct'])
        self.assertTrue(eval_result['profitable'])  # 'Profitable' means avoided loss here
        logging.info("✅ test_evaluate_prediction: SELL/AVOID, avoided loss passed.")

        # Scenario 3: WAIT, actual sideways (correct)
        rec = {"action": "WAIT", "confidence": 60, "expected_profit_pct": 0}
        eval_result = self.calibrator.evaluate_prediction(rec, 1.5)
        self.assertTrue(eval_result['correct'])
        self.assertTrue(eval_result['direction_correct'])
        self.assertTrue(eval_result['profitable'])
        logging.info("✅ test_evaluate_prediction: WAIT, sideways passed.")

        # Scenario 4: BUY, but small loss (incorrect, not profitable)
        rec = {"action": "BUY", "confidence": 80, "expected_profit_pct": 5}
        eval_result = self.calibrator.evaluate_prediction(rec, -1.0)
        self.assertFalse(eval_result['correct'])
        self.assertFalse(eval_result['profitable'])
        self.assertFalse(eval_result['direction_correct'])  # Predicted up, went down
        logging.info("✅ test_evaluate_prediction: BUY, small loss passed.")

        # Scenario 5: Actual return is None
        rec = {"action": "BUY", "confidence": 80, "expected_profit_pct": 5}
        eval_result = self.calibrator.evaluate_prediction(rec, None)
        self.assertFalse(eval_result['correct'])
        self.assertIsNone(eval_result['actual_return'])
        logging.info("✅ test_evaluate_prediction: Actual return None passed.")

    def test_generate_param_grid(self):
        """Test generation of parameter combinations."""
        # Use a simplified advisor_params for testing the grid generation logic
        simple_advisor_params = {
            'investment_days': [7, 14],
            'profit_threshold': [0.03, 0.04],
            'weight_trend': [1.0, 1.2],
        }
        param_grid = self.calibrator._generate_param_grid(simple_advisor_params)

        # Expected number of combinations: 2 * 2 * 2 (for params) * 4 (for strategies) = 32
        self.assertEqual(len(param_grid), 32)

        # Check structure of a sample combination
        sample_param = param_grid[0]
        self.assertIn('investment_days', sample_param)
        self.assertIn('profit_threshold', sample_param)
        self.assertIn('weight_trend', sample_param)
        self.assertIn('strategy_type', sample_param)
        self.assertIn(sample_param['strategy_type'], ["Conservative", "Balanced", "Aggressive", "Swing Trading"])
        logging.info("✅ test_generate_param_grid passed.")

    def test_create_walk_forward_windows(self):
        """Test walk-forward window creation."""
        # Temporarily modify config for simpler test case
        original_config = self.calibrator.config.copy()
        self.calibrator.config['start_date'] = '2023-01-01'
        self.calibrator.config['end_date'] = '2024-06-01'
        self.calibrator.config['walk_forward']['train_months'] = 6
        self.calibrator.config['walk_forward']['test_months'] = 3
        self.calibrator.config['walk_forward']['step_months'] = 3

        windows = self.calibrator.create_walk_forward_windows()
        self.assertIsInstance(windows, list)
        self.assertGreater(len(windows), 0)

        # Expected windows:
        # 1. Train: 2023-01-01 to 2023-07-01; Test: 2023-07-01 to 2023-10-01
        # 2. Train: 2023-04-01 to 2023-10-01; Test: 2023-10-01 to 2024-01-01
        # 3. Train: 2023-07-01 to 2024-01-01; Test: 2024-01-01 to 2024-04-01
        # 4. Train: 2023-10-01 to 2024-04-01; Test: 2024-04-01 to 2024-07-01 (This one will be excluded as end_date is 2024-06-01)
        # So, we expect 3 windows.

        self.assertEqual(len(windows), 3)
        self.assertEqual(windows[0]['train_start'].date(), datetime(2023, 1, 1).date())
        self.assertEqual(windows[0]['test_end'].date(), datetime(2023, 10, 1).date())
        self.assertEqual(windows[2]['train_start'].date(), datetime(2023, 7, 1).date())
        self.assertEqual(windows[2]['test_end'].date(), datetime(2024, 4, 1).date())

        # Restore original config
        self.calibrator.config = original_config
        logging.info("✅ test_create_walk_forward_windows passed.")

    def test_apply_parameters_to_advisor(self):
        """Test if parameters are correctly applied to the mock advisor."""
        test_params = {
            'investment_days': 30,
            'profit_threshold': 0.05,
            'stop_loss_threshold': 0.07,
            'confidence_req_balanced': 85,
            'weight_trend': 0.6,
            'weight_momentum': 0.2,
            'strategy_type': 'Balanced',
            'base_multiplier': 1.1,
            'confluence_weight': 0.9,
            'penalty_strength': 1.1,
            'buy_threshold': 1.5,
            'sell_threshold': -1.5
        }
        self.calibrator.apply_parameters_to_advisor(test_params)

        self.assertEqual(self.mock_advisor.investment_days, 30)
        self.assertEqual(self.mock_advisor.current_strategy, 'Balanced')

        self.assertAlmostEqual(self.mock_advisor.strategy_settings['profit'], 0.05)
        self.assertAlmostEqual(self.mock_advisor.strategy_settings['risk'], 0.07)
        self.assertEqual(self.mock_advisor.strategy_settings['confidence_req'], 85)
        self.assertAlmostEqual(self.mock_advisor.strategy_settings['buy_threshold'], 1.5)
        self.assertAlmostEqual(self.mock_advisor.strategy_settings['sell_threshold'], -1.5)

        self.assertAlmostEqual(self.mock_advisor.signal_weights['trend'], 0.6)
        self.assertAlmostEqual(self.mock_advisor.signal_weights['momentum'], 0.2)
        # Check that other weights not in test_params retain their default mock values
        self.assertAlmostEqual(self.mock_advisor.signal_weights['volume'], 0.10)

        self.assertAlmostEqual(self.mock_advisor.confidence_params['base_multiplier'], 1.1)
        self.assertAlmostEqual(self.mock_advisor.confidence_params['confluence_weight'], 0.9)
        self.assertAlmostEqual(self.mock_advisor.confidence_params['penalty_strength'], 1.1)

        # Check direct thresholds if advisor has them
        self.assertAlmostEqual(self.mock_advisor.current_buy_threshold, 1.5)
        self.assertAlmostEqual(self.mock_advisor.current_sell_threshold, -1.5)

        logging.info("✅ test_apply_parameters_to_advisor passed.")

    def test_calculate_performance_metrics(self):
        """Test calculation of aggregated performance metrics."""
        mock_results = [
            {'stock': 'AAPL', 'timestamp': '2023-01-01', 'recommendation': {'action': 'BUY', 'confidence': 85},
             'actual_return': 5.0, 'performance': {'correct': True, 'direction_correct': True, 'profitable': True}},
            {'stock': 'GOOG', 'timestamp': '2023-01-01', 'recommendation': {'action': 'SELL/AVOID', 'confidence': 70},
             'actual_return': -3.0, 'performance': {'correct': True, 'direction_correct': True, 'profitable': True}},
            {'stock': 'MSFT', 'timestamp': '2023-01-01', 'recommendation': {'action': 'WAIT', 'confidence': 60},
             'actual_return': 1.0, 'performance': {'correct': True, 'direction_correct': True, 'profitable': True}},
            {'stock': 'AMZN', 'timestamp': '2023-01-01', 'recommendation': {'action': 'BUY', 'confidence': 75},
             'actual_return': -2.0, 'performance': {'correct': False, 'direction_correct': False, 'profitable': False}}
        ]
        metrics = self.calibrator.calculate_performance_metrics(mock_results)

        self.assertGreater(metrics['overall_accuracy'], 0)
        self.assertGreater(metrics['direction_accuracy'], 0)
        self.assertGreater(metrics['avg_return'], 0)
        self.assertGreater(metrics['sharpe_ratio'], 0)
        self.assertLess(metrics['max_drawdown'], 0)  # Drawdown is typically negative or zero

        self.assertEqual(metrics['total_trades'], 4)
        self.assertAlmostEqual(metrics['confidence_avg'], (85 + 70 + 60 + 75) / 4)

        # Check buy success rate (1 correct BUY out of 2 BUY signals)
        self.assertAlmostEqual(metrics['buy_success_rate'], 50.0)
        logging.info("✅ test_calculate_performance_metrics passed.")

    def test_calculate_fitness_score(self):
        """Test the fitness score calculation."""
        mock_performance = {
            'overall_accuracy': 80.0,
            'direction_accuracy': 90.0,
            'buy_success_rate': 70.0,
            'avg_return': 5.0,
            'volatility': 2.0,
            'sharpe_ratio': 2.5,  # 5.0 / 2.0
            'max_drawdown': -10.0,
            'total_trades': 100,
            'signal_distribution': {'BUY': 40, 'SELL/AVOID': 30, 'WAIT': 30},
            'confidence_avg': 75
        }
        fitness_score = self.calibrator.calculate_fitness_score(mock_performance)
        self.assertIsInstance(fitness_score, float)
        self.assertGreater(fitness_score, 0)
        logging.info("✅ test_calculate_fitness_score passed.")

    def test_run_calibration_sanity_mode_config_adjustment(self):
        """
        Test that run_calibration correctly adjusts walk_forward config for sanity mode
        and restores it afterward.
        """
        # Store original walk_forward config BEFORE calling run_calibration
        # The default is 12 months for train, 3 for test, 3 for step
        original_train_months = self.calibrator.config['walk_forward']['train_months']
        original_test_months = self.calibrator.config['walk_forward']['test_months']
        original_step_months = self.calibrator.config['walk_forward']['step_months']

        # Mock methods to allow run_calibration to proceed without full execution
        self.calibrator.get_stock_universe = MagicMock(return_value=['AAPL', 'MSFT'])
        self.calibrator.generate_test_timestamps = MagicMock(return_value=[datetime(2023, 1, 1).date()])
        self.calibrator.create_walk_forward_windows = MagicMock(return_value=[
            {'train_start': datetime(2023, 1, 1), 'train_end': datetime(2023, 3, 1), 'test_start': datetime(2023, 3, 1),
             'test_end': datetime(2023, 4, 1), 'window_id': 1}
        ])
        self.calibrator.optimize_strategy = MagicMock(
            return_value={'parameters': {}, 'performance': self.calibrator.get_empty_metrics(), 'fitness_score': 0})

        # Run the calibration in sanity mode
        self.calibrator.run_calibration(test_size='sanity', strategies=['balanced'])

        # Assert that walk_forward config was RESTORED to its original values
        self.assertEqual(self.calibrator.config['walk_forward']['train_months'], original_train_months)
        self.assertEqual(self.calibrator.config['walk_forward']['test_months'], original_test_months)
        self.assertEqual(self.calibrator.config['walk_forward']['step_months'], original_step_months)
        logging.info("✅ test_run_calibration_sanity_mode_config_adjustment passed.")

    def test_normalize_numeric_values(self):
        """Test normalization of NumPy types to standard Python types."""
        test_data = {
            'float_val': np.float64(123.45),
            'int_val': np.int64(987),
            'bool_val': np.bool_(True),
            'list_of_floats': [np.float32(1.1), np.float64(2.2)],
            'nested_dict': {
                'np_array': np.array([1, 2, 3]),
                'another_float': np.float16(0.5)
            },
            'regular_string': "hello"
        }
        normalized_data = self.calibrator._normalize_numeric_values(test_data)

        self.assertIsInstance(normalized_data['float_val'], float)
        self.assertIsInstance(normalized_data['int_val'], float)  # np.int64 converts to float by default
        self.assertIsInstance(normalized_data['bool_val'], bool)
        self.assertIsInstance(normalized_data['list_of_floats'][0], float)
        self.assertIsInstance(normalized_data['list_of_floats'][1], float)
        self.assertIsInstance(normalized_data['nested_dict']['np_array'], list)
        self.assertIsInstance(normalized_data['nested_dict']['np_array'][0],
                              int)  # Elements of array can be int if they were int in numpy
        self.assertIsInstance(normalized_data['nested_dict']['another_float'], float)
        self.assertEqual(normalized_data['regular_string'], "hello")
        logging.info("✅ test_normalize_numeric_values passed.")


# This allows running the tests directly from the script
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

