import unittest
import pandas as pd
from unittest.mock import patch, MagicMock

# Import the class we want to test
from data_source_manager import DataSourceManager


class TestDataSourceManager(unittest.TestCase):
    """
    Unit tests for the DataSourceManager class.
    It uses mocking to test the logic without making real network calls.
    """

    def setUp(self):
        """This method is called before each test."""
        # We initialize the manager with use_ibkr=False in setUp
        # because most tests will mock the download functions directly.
        # Tests that need the IBKR connection logic can re-initialize it.
        self.manager = DataSourceManager(use_ibkr=False)

    @patch('data_source_manager.DataSourceManager._download_from_yfinance')
    @patch('data_source_manager.DataSourceManager._download_from_ibkr')
    def test_ibkr_success_scenario(self, mock_ibkr_download, mock_yfinance_download):
        """
        Test Case 1: Verify that if IBKR returns data, yfinance is NOT called.
        """
        print("\n--- Running Test: IBKR Success ---")
        # --- Arrange ---
        # Create a sample DataFrame that our mock IBKR function will return
        sample_df = pd.DataFrame({'Close': [150, 151, 152]})
        mock_ibkr_download.return_value = sample_df

        # Create a manager instance that is configured to use IBKR
        ibkr_manager = DataSourceManager(use_ibkr=True)
        # We need to mock the connection logic as well for this test
        ibkr_manager.connect_to_ibkr = MagicMock(return_value=True)

        # --- Act ---
        # Call the main function we want to test
        result_df = ibkr_manager.get_stock_data("AAPL")

        # --- Assert ---
        # Check that the IBKR download function was called exactly once
        mock_ibkr_download.assert_called_once_with("AAPL")
        # CRITICAL: Check that the yfinance download function was NEVER called
        mock_yfinance_download.assert_not_called()
        # Check that the returned DataFrame is the one from our mock IBKR
        self.assertTrue(result_df.equals(sample_df))
        print("✅ Test Passed")

    @patch('data_source_manager.DataSourceManager._download_from_yfinance')
    @patch('data_source_manager.DataSourceManager._download_from_ibkr')
    def test_yfinance_fallback_scenario(self, mock_ibkr_download, mock_yfinance_download):
        """
        Test Case 2: Verify that if IBKR fails (returns empty), the system correctly falls back to yfinance.
        """
        print("\n--- Running Test: YFinance Fallback ---")
        # --- Arrange ---
        # Simulate IBKR failing by having it return an empty DataFrame
        mock_ibkr_download.return_value = pd.DataFrame()

        # Simulate yfinance succeeding by returning a sample DataFrame
        sample_df = pd.DataFrame({'Close': [200, 201, 202]})
        mock_yfinance_download.return_value = sample_df

        # Create a manager instance configured to use IBKR
        ibkr_manager = DataSourceManager(use_ibkr=True)
        ibkr_manager.connect_to_ibkr = MagicMock(return_value=True)

        # --- Act ---
        result_df = ibkr_manager.get_stock_data("GOOG")

        # --- Assert ---
        # Check that both download methods were called exactly once
        mock_ibkr_download.assert_called_once_with("GOOG")
        mock_yfinance_download.assert_called_once_with("GOOG", days_back=5 * 365)
        # Check that the final result is the DataFrame from yfinance
        self.assertTrue(result_df.equals(sample_df))
        print("✅ Test Passed")

    @patch('data_source_manager.DataSourceManager._download_from_yfinance')
    @patch('data_source_manager.DataSourceManager._download_from_ibkr')
    def test_total_failure_scenario(self, mock_ibkr_download, mock_yfinance_download):
        """
        Test Case 3: Verify that if both IBKR and yfinance fail, the system returns an empty DataFrame.
        """
        print("\n--- Running Test: Total Failure ---")
        # --- Arrange ---
        # Simulate both sources failing by returning an empty DataFrame
        mock_ibkr_download.return_value = pd.DataFrame()
        mock_yfinance_download.return_value = pd.DataFrame()

        # Create a manager instance
        ibkr_manager = DataSourceManager(use_ibkr=True)
        ibkr_manager.connect_to_ibkr = MagicMock(return_value=True)

        # --- Act ---
        result_df = ibkr_manager.get_stock_data("TSLA")

        # --- Assert ---
        # Check that both download methods were called
        mock_ibkr_download.assert_called_once_with("TSLA")
        mock_yfinance_download.assert_called_once_with("TSLA", days_back=5 * 365)
        # Check that the final result is an empty DataFrame
        self.assertTrue(result_df.empty)
        print("✅ Test Passed")


if __name__ == '__main__':
    unittest.main()