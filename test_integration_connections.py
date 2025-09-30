"""
Integration Tests for Data Source Connections
=============================================

This script contains integration tests for the `DataSourceManager` class, designed to
verify real, live connections to external data sources. It also includes unit tests
to validate the internal logic of the manager, such as the fallback mechanism.

These tests are critical for ensuring that the application can successfully fetch
data from both its primary (IBKR TWS) and secondary (yfinance) sources.

Pre-conditions for running:
---------------------------
- A live, running instance of Interactive Brokers Trader Workstation (TWS) or
  Gateway must be available and configured to accept API connections on the
  default port (7497).
- An active internet connection is required.

Tests Included:
---------------
- `test_yfinance_real_connection`: Performs a live data download from Yahoo Finance.
- `test_tws_api_real_connection`: Performs a live connection and data download from
  the TWS API.
- `test_ibkr_to_yfinance_fallback`: A unit test that mocks the data download methods
  to confirm that the system correctly falls back to yfinance if an IBKR data
  request fails after a successful connection.

"""

# test_integration_connections.py

import pytest
import pandas as pd
import socket
from unittest.mock import patch
from data_source_manager import DataSourceManager


class TestLiveTwsConnection:
    # In test_integration_connections.py

    @pytest.fixture(scope="class")
    def data_manager(self):
        """Initializes a real DataSourceManager and ensures disconnection."""
        manager = DataSourceManager()
        yield manager
        # --- This cleanup code will now run even if the connection fails ---
        print("\n--- Tearing down test session, ensuring disconnection ---")
        try:
            if manager.isConnected():
                manager.disconnect()
        except Exception as e:
            print(f"Error during disconnection: {e}")

    def test_yfinance_real_connection(self, data_manager):
        """Tests a real data download from yfinance."""
        print("\n--- Running Integration Test: Real YFinance Download ---")
        data_manager.use_ibkr = False
        df = data_manager.get_stock_data('SPY')
        data_manager.use_ibkr = True

        assert isinstance(df, pd.DataFrame)
        assert not df.empty, "yfinance should have returned data for SPY."
        print("✅ Test Passed: Successfully downloaded real data from yfinance.")

    def test_tws_api_real_connection(self, data_manager):
        """Tests a real connection and data download from the TWS API."""
        print("\n--- Running Integration Test: Real TWS API Download ---")
        data_manager.use_ibkr = True
        df = data_manager.get_stock_data('AAPL')

        assert isinstance(df, pd.DataFrame)
        assert not df.empty, "TWS API should have returned data for AAPL."
        print("✅ Test Passed: Successfully connected to and downloaded real data from TWS.")

        # In test_integration_connections.py

    @patch('data_source_manager.DataSourceManager._download_from_yfinance')
    @patch('data_source_manager.DataSourceManager._download_from_ibkr')
    def test_ibkr_to_yfinance_fallback(self, mock_ibkr_download, mock_yfinance_download, data_manager):
        """
        Tests the fallback logic: if IBKR download fails, yfinance should be called.
        """
        print("\n--- Running Unit Test: IBKR to YFinance Fallback ---")
        # --- Arrange ---
        mock_ibkr_download.return_value = pd.DataFrame()
        mock_yfinance_download.return_value = pd.DataFrame({'Close': [100]})

        # --- FIX: Manually set the connection status to True to simulate a live connection ---
        data_manager.use_ibkr = True
        data_manager.connection_event.set()  # This sets isConnected() to True

        # --- Act ---
        df = data_manager.get_stock_data('TSLA')

        # --- Assert ---
        mock_ibkr_download.assert_called_once_with('TSLA', 1825)
        mock_yfinance_download.assert_called_once_with('TSLA', 1825)
        assert not df.empty
        print("✅ Test Passed: Correctly fell back from IBKR to yfinance.")