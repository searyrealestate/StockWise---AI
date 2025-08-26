import unittest
import pandas as pd
import numpy as np
import os
import shutil
from unittest.mock import patch, MagicMock

# Import the functions and classes from the script we want to test
from Create_parquet_file_NASDAQ import (
    add_technical_indicators_and_features,
    get_symbols_from_csv,
    process_ticker_list,
    get_dominant_cycle  # Also import the new helper function
)
from data_source_manager import DataSourceManager


class TestCreateParquetFile(unittest.TestCase):
    """
    Unit and integration tests for the Create_parquet_file_NASDAQ.py script.
    """

    def setUp(self):
        """Set up a temporary environment for tests."""
        # Create a dummy DataFrame that our mocked DataSourceManager will return
        dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=200))
        self.sample_df = pd.DataFrame({
            'Open': np.random.uniform(100, 102, size=200),
            'High': np.random.uniform(102, 104, size=200),
            'Low': np.random.uniform(98, 100, size=200),
            'Close': np.random.uniform(100, 103, size=200),
            'Volume': np.random.randint(1_000_000, 5_000_000, size=200)
        }, index=dates)
        self.sample_df.index.name = 'Date'

        # Create temporary directories for test outputs
        self.temp_dir = "temp_test_output"
        os.makedirs(os.path.join(self.temp_dir, "features"), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, "models"), exist_ok=True)

    def tearDown(self):
        """Clean up the temporary environment after tests."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_get_symbols_from_csv_filtering(self):
        """
        Tests if the symbol filtering logic correctly removes unwanted tickers.
        """
        print("\n--- Running Test: Symbol CSV Filtering ---")
        # Create a dummy CSV in memory
        csv_data = "Symbol,Name\nAAPL,Apple Inc.\nGOOG,Google LLC\nAACBR,AACB Rights\nMSFT,Microsoft Corp.\n"
        dummy_csv_path = os.path.join(self.temp_dir, "dummy_symbols.csv")
        with open(dummy_csv_path, "w") as f:
            f.write(csv_data)

        symbols = get_symbols_from_csv(dummy_csv_path)

        # Assert that standard symbols are included
        self.assertIn("AAPL", symbols)
        self.assertIn("MSFT", symbols)
        # Assert that the problematic symbol with 'R' at the end is excluded
        self.assertNotIn("AACBR", symbols)
        print("✅ Test Passed")

    def test_add_all_new_features(self):
        """
        Tests if the feature engineering function adds all the new advanced indicators.
        """
        print("\n--- Running Test: Advanced Feature Engineering ---")
        # Add the 'Datetime' column which is expected by the function
        df_with_datetime = self.sample_df.reset_index().rename(columns={'Date': 'Datetime'})

        # Process the DataFrame
        featured_df = add_technical_indicators_and_features(df_with_datetime)

        # List of all new feature columns we expect to be added
        new_feature_columns = [
            'ATR_14', 'ADX', 'ADX_pos', 'ADX_neg', 'OBV', 'RSI_28', 'Dominant_Cycle_126D'
        ]

        # Assert that all new columns exist in the output
        for col in new_feature_columns:
            self.assertIn(col, featured_df.columns)

        # Assert that the DataFrame is not empty after processing
        self.assertFalse(featured_df.empty)
        # Assert that some rows were dropped due to NaNs from indicator calculations
        self.assertLess(len(featured_df), len(self.sample_df))
        print("✅ Test Passed")

    @patch('Create_parquet_file_NASDAQ.DataSourceManager')
    def test_process_ticker_list_success(self, MockDataSourceManager):
        """
        Tests the main processing loop for a successful case.
        Verifies that a Parquet file is created with the correct data.
        """
        print("\n--- Running Test: Main Processing Loop (Success) ---")
        # --- Arrange ---
        # Configure the mock DataSourceManager
        mock_manager_instance = MockDataSourceManager.return_value
        # When get_stock_data is called, it will return our sample DataFrame
        mock_manager_instance.get_stock_data.return_value = self.sample_df.copy()

        # --- Act ---
        # Run the processing function on a single ticker
        process_ticker_list(
            tickers=["AAPL"],
            output_base_dir=self.temp_dir,
            train=True,  # This determines where the dummy model is saved
            data_source_manager_instance=mock_manager_instance
        )

        # --- Assert ---
        # Check that the Parquet file was created
        output_features_dir = os.path.join(self.temp_dir, "features")
        created_files = os.listdir(output_features_dir)
        self.assertEqual(len(created_files), 1)
        self.assertTrue(created_files[0].startswith("AAPL") and created_files[0].endswith(".parquet"))

        # Load the created file and verify its contents
        result_df = pd.read_parquet(os.path.join(output_features_dir, created_files[0]))
        self.assertFalse(result_df.empty)
        self.assertIn("ATR_14", result_df.columns)  # Check for one of the new features
        self.assertIn("Target", result_df.columns)
        print("✅ Test Passed")


if __name__ == '__main__':
    unittest.main()