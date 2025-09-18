import unittest
import pandas as pd
from unittest.mock import patch, MagicMock

# Import the class we want to test
from data_source_manager import DataSourceManager


class TestDataSourceManager(unittest.TestCase):
    """
    Unit tests for the upgraded DataSourceManager class using the Client Portal API.
    It uses mocking to test the logic without making real network calls.
    """

    def setUp(self):
        """This method is called before each test."""
        # We initialize the manager with use_ibkr=False in setUp
        # because most tests will mock the download functions directly.
        self.manager = DataSourceManager(use_ibkr=False)

    @patch('data_source_manager.DataSourceManager._download_from_yfinance')
    @patch('data_source_manager.DataSourceManager._download_from_ibkr')
    def test_ibkr_success_scenario(self, mock_ibkr_download, mock_yfinance_download):
        """
        Test Case 1: Verify that if IBKR returns data, yfinance is NOT called.
        """
        print("\n--- Running Test: IBKR Success ---")
        # --- Arrange ---
        sample_df = pd.DataFrame({'Close': [150, 151, 152]})
        mock_ibkr_download.return_value = sample_df

        ibkr_manager = DataSourceManager(use_ibkr=True)
        # Mock the connection check for the Client Portal API to return True
        ibkr_manager.connect_to_ibkr = MagicMock(return_value=True)

        # --- Act ---
        result_df = ibkr_manager.get_stock_data("AAPL")

        # --- Assert ---
        mock_ibkr_download.assert_called_once_with("AAPL")
        mock_yfinance_download.assert_not_called()
        self.assertTrue(result_df.equals(sample_df))
        print("✅ Test Passed")

    @patch('data_source_manager.DataSourceManager._download_from_yfinance')
    @patch('data_source_manager.DataSourceManager._download_from_ibkr')
    def test_yfinance_fallback_scenario(self, mock_ibkr_download, mock_yfinance_download):
        """
        Test Case 2: Verify that if IBKR fails, the system correctly falls back to yfinance.
        """
        print("\n--- Running Test: YFinance Fallback ---")
        # --- Arrange ---
        mock_ibkr_download.return_value = pd.DataFrame()

        sample_df = pd.DataFrame({'Close': [200, 201, 202]})
        mock_yfinance_download.return_value = sample_df

        ibkr_manager = DataSourceManager(use_ibkr=True)
        ibkr_manager.connect_to_ibkr = MagicMock(return_value=True)

        # --- Act ---
        result_df = ibkr_manager.get_stock_data("GOOG")

        # --- Assert ---
        mock_ibkr_download.assert_called_once_with("GOOG")
        mock_yfinance_download.assert_called_once_with("GOOG", days_back=5 * 365)
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
        mock_ibkr_download.return_value = pd.DataFrame()
        mock_yfinance_download.return_value = pd.DataFrame()

        ibkr_manager = DataSourceManager(use_ibkr=True)
        ibkr_manager.connect_to_ibkr = MagicMock(return_value=True)

        # --- Act ---
        result_df = ibkr_manager.get_stock_data("TSLA")

        # --- Assert ---
        mock_ibkr_download.assert_called_once_with("TSLA")
        mock_yfinance_download.assert_called_once_with("TSLA", days_back=5 * 365)
        self.assertTrue(result_df.empty)
        print("✅ Test Passed")

    @patch('data_source_manager.DataSourceManager.connect_to_ibkr')
    @patch('data_source_manager.DataSourceManager._download_from_yfinance')
    @patch('data_source_manager.DataSourceManager._download_from_ibkr')
    def test_ibkr_connection_failure_scenario(self, mock_ibkr_download, mock_yfinance_download, mock_connect):
        """
        Test Case 4: Verify that if the IBKR connection itself fails, the system falls back to yfinance.
        """
        print("\n--- Running Test: IBKR Connection Failure ---")
        # --- Arrange ---
        # Simulate IBKR connection failing
        mock_connect.return_value = False

        # Simulate yfinance succeeding
        sample_df = pd.DataFrame({'Close': [200, 201, 202]})
        mock_yfinance_download.return_value = sample_df

        ibkr_manager = DataSourceManager(use_ibkr=True)

        # --- Act ---
        result_df = ibkr_manager.get_stock_data("MSFT")

        # --- Assert ---
        # The IBKR download method should NOT be called at all, since the connection failed
        mock_ibkr_download.assert_not_called()
        mock_yfinance_download.assert_called_once_with("MSFT", days_back=5 * 365)
        self.assertTrue(result_df.equals(sample_df))
        print("✅ Test Passed")


if __name__ == '__main__':
    unittest.main()