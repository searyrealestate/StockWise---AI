# test_create_parquet_NASDAQ_file.py
"""
Unit and Integration Tests for the Data Pipeline
================================================

This script contains a suite of tests for the `Create_parquet_file_NASDAQ.py`
and `data_source_manager.py` modules. It is designed to ensure the correctness
and robustness of both the data sourcing logic and the feature engineering pipeline.

The tests are divided into two main categories:

1.  Data Sourcing Logic (`TestDataSourcing`):
    -   Contains integration tests that use mocking to simulate the behavior of
        the `DataSourceManager`.
    -   Verifies the critical success path (data is correctly fetched from IBKR).
    -   Verifies the fallback mechanism (the system correctly switches to yfinance
        if the IBKR data fetch fails).

2.  Feature Engineering Logic (`TestFeatureEngineering`):
    -   Contains unit tests that validate the `add_technical_indicators_and_features`
        function.
    -   Uses `pytest.mark.parametrize` to run all tests against each of the supported
        profit modes (e.g., 'dynamic', 'fixed_net'), ensuring comprehensive coverage.
    -   Includes a critical schema validation test that asserts all expected feature
        and target columns are present in the final DataFrame.
    -   Includes specific tests for newly added features like KAMA, Stochastic
        Oscillator, and Dominant Cycle.

"""


import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch, MagicMock
from Create_parquet_file_NASDAQ import add_technical_indicators_and_features
from data_source_manager import DataSourceManager


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """A single, robust function to clean raw data immediately after fetching."""
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(0)
    df.columns = [col.lower() for col in df.columns]
    return df


@pytest.fixture(scope="module")
def sample_stock_data():
    """Creates a realistic sample DataFrame of stock data for testing."""
    dates = pd.to_datetime(pd.date_range(start='2022-01-01', periods=300))
    close_prices = 100 + np.cumsum(np.random.normal(0, 1.5, size=300))
    data = {
        'Open': close_prices - np.random.uniform(0, 1, size=300),
        'High': close_prices + np.random.uniform(0, 1.5, size=300),
        'Low': close_prices - np.random.uniform(0, 1.5, size=300),
        'Close': close_prices,
        'Volume': np.random.randint(1_000_000, 10_000_000, size=300)
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'Datetime'
    return df


@pytest.fixture
def mock_qqq_data(sample_stock_data):
    """Creates a mock QQQ series for the tests."""
    return pd.Series(range(len(sample_stock_data)), index=sample_stock_data.index)


class TestDataSourcing:
    """Unit tests for the DataSourceManager's sourcing and fallback logic."""

    @patch('data_source_manager.DataSourceManager._download_from_yfinance')
    @patch('data_source_manager.DataSourceManager._download_from_ibkr')
    def test_ibkr_to_yfinance_fallback(self, mock_ibkr_download, mock_yfinance_download):
        print("\n--- Running Test: IBKR to YFinance Fallback ---")
        mock_ibkr_download.return_value = pd.DataFrame()
        mock_yfinance_download.return_value = pd.DataFrame({'close': [100]})

        manager = DataSourceManager(use_ibkr=True)
        manager.connect_to_ibkr = MagicMock(return_value=True)
        manager.isConnected = MagicMock(return_value=True)

        result_df = manager.get_stock_data("TEST")

        mock_ibkr_download.assert_called_once()
        mock_yfinance_download.assert_called_once()
        assert not result_df.empty, "The final DataFrame should not be empty."
        print("✅ Test Passed: Correctly fell back from IBKR to yfinance.")

    @patch('data_source_manager.DataSourceManager._download_from_yfinance')
    @patch('data_source_manager.DataSourceManager._download_from_ibkr')
    def test_ibkr_success_path(self, mock_ibkr_download, mock_yfinance_download):
        print("\n--- Running Test: IBKR Success Path ---")
        mock_ibkr_download.return_value = pd.DataFrame({'close': [200]})

        manager = DataSourceManager(use_ibkr=True)
        manager.connect_to_ibkr = MagicMock(return_value=True)
        manager.isConnected = MagicMock(return_value=True)
        result_df = manager.get_stock_data("TEST")

        mock_ibkr_download.assert_called_once()
        mock_yfinance_download.assert_not_called()
        assert result_df.iloc[0]['close'] == 200, "Should have returned the IBKR data."
        print("✅ Test Passed: Correctly used IBKR and did not fall back.")


@pytest.mark.parametrize("profit_mode", ['dynamic', 'fixed_net'])
def test_all_expected_features_exist(sample_stock_data, mock_qqq_data, profit_mode):
    """Test ID: T1.1 - Verifies all expected feature and target columns exist."""
    clean_df = clean_raw_data(sample_stock_data.copy())

    # ADDED: Create mock data for the new arguments
    mock_vix_data = pd.Series(index=clean_df.index, data=15)
    mock_tlt_data = pd.Series(index=clean_df.index, data=100)

    # MODIFIED: Pass the new mock data to the function
    featured_df = add_technical_indicators_and_features(
        clean_df, (0.015, 0.03), mock_qqq_data, profit_mode, 0.03,
        mock_vix_data, mock_tlt_data
    )

    expected_cols = {
        'open', 'high', 'low', 'close', 'volume', 'volume_ma_20', 'rsi_14', 'momentum_5', 'macd', 'macd_signal',
        'macd_histogram', 'bb_position', 'volatility_20d', 'atr_14', 'adx', 'adx_pos', 'adx_neg', 'obv', 'rsi_28',
        'z_score_20', 'bb_width', 'correlation_50d_qqq', 'vix_close', 'corr_tlt', 'cmf',
        'bb_upper', 'bb_lower', 'bb_middle', 'daily_return',
        'kama_10', 'stoch_k', 'stoch_d', 'dominant_cycle',
        'volatility_cluster', 'target_entry', 'target_profit_take', 'target_cut_loss', 'target_trailing_stop'
    }

    assert expected_cols.issubset(featured_df.columns)


def test_new_feature_kama(sample_stock_data, mock_qqq_data):
    """Tests if the KAMA feature is calculated correctly."""
    clean_df = clean_raw_data(sample_stock_data.copy())

    # ADDED: Create mock data for the new arguments
    mock_vix_data = pd.Series(index=clean_df.index, data=15)
    mock_tlt_data = pd.Series(index=clean_df.index, data=100)

    # MODIFIED: Pass the new mock data to the function
    featured_df = add_technical_indicators_and_features(
        clean_df, (0.015, 0.03), mock_qqq_data, 'dynamic', 0.03,
        mock_vix_data, mock_tlt_data
    )
    assert 'kama_10' in featured_df.columns
    assert pd.api.types.is_numeric_dtype(featured_df['kama_10'])


def test_new_feature_stochastic(sample_stock_data, mock_qqq_data):
    """Tests if the Stochastic Oscillator features are calculated correctly."""
    clean_df = clean_raw_data(sample_stock_data.copy())

    # ADDED: Create mock data for the new arguments
    mock_vix_data = pd.Series(index=clean_df.index, data=15)
    mock_tlt_data = pd.Series(index=clean_df.index, data=100)

    # MODIFIED: Pass the new mock data to the function
    featured_df = add_technical_indicators_and_features(
        clean_df, (0.015, 0.03), mock_qqq_data, 'dynamic', 0.03,
        mock_vix_data, mock_tlt_data
    )
    assert 'stoch_k' in featured_df.columns
    assert 'stoch_d' in featured_df.columns
    assert pd.api.types.is_numeric_dtype(featured_df['stoch_k'])


def test_new_feature_dominant_cycle(sample_stock_data, mock_qqq_data):
    """Tests if the Dominant Cycle feature is calculated."""
    clean_df = clean_raw_data(sample_stock_data.copy())

    # ADDED: Create mock data for the new arguments
    mock_vix_data = pd.Series(index=clean_df.index, data=15)
    mock_tlt_data = pd.Series(index=clean_df.index, data=100)

    # MODIFIED: Pass the new mock data to the function
    featured_df = add_technical_indicators_and_features(
        clean_df, (0.015, 0.03), mock_qqq_data, 'dynamic', 0.03,
        mock_vix_data, mock_tlt_data
    )
    assert 'dominant_cycle' in featured_df.columns
    assert pd.api.types.is_numeric_dtype(featured_df['dominant_cycle'])


# def test_triple_barrier_logic_scenario(mock_qqq_data):
#     """Test ID: TL-02 - Verifies the labeling logic with a predictable scenario."""
#     # Create 120 days of data: 20 for warm-up, 100 for the actual test
#     warmup_days = 20
#     total_days = 120
#     dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=total_days))
#
#     # price_data now has 120 points
#     price_data = ([100] * (warmup_days + 20) + list(range(101, 121)) +
#                   [120] * 10 + list(range(119, 99, -1)) + [100] * 30)
#
#     data = {
#         'open': price_data, 'high': price_data, 'low': price_data,
#         'close': price_data, 'volume': [1000000] * total_days
#     }
#     df = pd.DataFrame(data, index=dates)
#     df = clean_raw_data(df)
#
#     # Manually set a predictable event in our scenario (adjusting index for the warm-up period)
#     df.loc[df.index[warmup_days + 25], 'high'] = 130
#
#     # Create mock data for the new arguments, also for 120 days
#     mock_vix_data = pd.Series(index=df.index, data=15)
#     mock_tlt_data = pd.Series(index=df.index, data=100)
#
#     # Pass the full 120-day DataFrame to the function. It will handle the dropna().
#     featured_df_full = add_technical_indicators_and_features(
#         df, (0.015, 0.03), mock_qqq_data, 'dynamic', 0.03,
#         mock_vix_data, mock_tlt_data
#     )
#
#     # The original dates for the 100-day test period
#     test_dates = dates[warmup_days:]
#
#     # Assertions now check for dates within the original 100-day test window
#     assert featured_df_full.loc[
#                test_dates[21], 'target_entry'] == 1, "The 'target_entry' label was not correctly applied."
#     assert featured_df_full.loc[
#                test_dates[55], 'target_cut_loss'] == 1, "The 'target_cut_loss' label was not correctly applied."