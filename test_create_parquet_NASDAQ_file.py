# test_create_parquet_NASDAQ_file.py

import pytest
import pandas as pd
import numpy as np
import os

# --- Import the function we are testing ---
from Create_parquet_file_NASDAQ import add_technical_indicators_and_features


# --- Test Fixtures ---
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


# NEW: Fixture to provide sample QQQ data for correlation testing
@pytest.fixture(scope="module")
def qqq_data(sample_stock_data):
    """Creates a sample QQQ closing price Series aligned with the stock data."""
    qqq_close_prices = 150 + np.cumsum(np.random.normal(0, 2, size=len(sample_stock_data)))
    return pd.Series(qqq_close_prices, index=sample_stock_data.index, name="QQQ_Close")


# --- Test Cases ---

# NEW: Use parametrize to automatically run tests for both profit modes
@pytest.mark.parametrize("profit_mode", ['dynamic', 'fixed_net'])
def test_all_expected_features_exist(sample_stock_data, qqq_data, profit_mode):
    """
    Test ID: T1.1
    Verifies that the final DataFrame contains ALL expected features.
    This test now runs for both 'dynamic' and 'fixed_net' profit modes.
    """
    # MODIFIED: Updated the set of expected columns to match the new function's output
    expected_columns = {
        'Open', 'High', 'Low', 'Close', 'Volume', 'Volume_MA_20', 'RSI_14',
        'Momentum_5', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'BB_Position',
        'Volatility_20D', 'ATR_14', 'ADX', 'ADX_pos', 'ADX_neg', 'OBV', 'RSI_28',
        'Z_Score_20', 'BB_Width', 'Correlation_50D_QQQ', 'BB_Upper', 'BB_Lower',
        'BB_Middle', 'Daily_Return', 'Volatility_Cluster', 'Target_Entry',
        'Target_Profit_Take', 'Target_Cut_Loss'
    }

    # MODIFIED: Updated the function call with all required arguments
    featured_df = add_technical_indicators_and_features(
        df=sample_stock_data.copy(),
        vol_thresholds=(0.015, 0.03),
        qqq_close=qqq_data,
        profit_mode=profit_mode,
        net_profit_target=0.03,
        debug=False  # NEW: Activate the debug prints for this test
    )

    missing_cols = expected_columns - set(featured_df.columns)
    assert not missing_cols, f"Missing critical columns for mode '{profit_mode}': {missing_cols}"
    assert not featured_df.empty, f"The DataFrame is empty for mode '{profit_mode}'."


@pytest.mark.parametrize("profit_mode", ['dynamic', 'fixed_net'])
def test_feature_z_score(sample_stock_data, qqq_data, profit_mode):
    """Test ID: FE-02 - Z-Score validation."""
    featured_df = add_technical_indicators_and_features(
        sample_stock_data.copy(), (0.015, 0.03), qqq_data, profit_mode, 0.03
    )
    assert 'Z_Score_20' in featured_df.columns
    assert pd.api.types.is_numeric_dtype(featured_df['Z_Score_20'])
    assert not featured_df['Z_Score_20'].isnull().all()


@pytest.mark.parametrize("profit_mode", ['dynamic', 'fixed_net'])
def test_feature_bollinger_band_width(sample_stock_data, qqq_data, profit_mode):
    """Test ID: FE-03 - Bollinger Band Width validation."""
    featured_df = add_technical_indicators_and_features(
        sample_stock_data.copy(), (0.015, 0.03), qqq_data, profit_mode, 0.03
    )
    assert 'BB_Width' in featured_df.columns
    assert featured_df['BB_Width'].min() >= 0


@pytest.mark.parametrize("profit_mode", ['dynamic', 'fixed_net'])
def test_feature_correlation_to_qqq(sample_stock_data, qqq_data, profit_mode):
    """Test ID: FE-04 - Correlation to QQQ validation."""
    featured_df = add_technical_indicators_and_features(
        sample_stock_data.copy(), (0.015, 0.03), qqq_data, profit_mode, 0.03
    )
    assert 'Correlation_50D_QQQ' in featured_df.columns
    assert featured_df['Correlation_50D_QQQ'].min() >= -1.0
    assert featured_df['Correlation_50D_QQQ'].max() <= 1.0


@pytest.mark.parametrize("profit_mode", ['dynamic', 'fixed_net'])
def test_feature_volatility_cluster(sample_stock_data, qqq_data, profit_mode):
    """Test ID: FE-05 - Volatility Cluster validation."""
    featured_df = add_technical_indicators_and_features(
        sample_stock_data.copy(), (0.015, 0.03), qqq_data, profit_mode, 0.03
    )
    assert 'Volatility_Cluster' in featured_df.columns
    valid_clusters = {'low', 'mid', 'high'}
    assert set(featured_df['Volatility_Cluster'].unique()).issubset(valid_clusters)


@pytest.mark.parametrize("profit_mode", ['dynamic', 'fixed_net'])
def test_triple_barrier_labeling(sample_stock_data, qqq_data, profit_mode):
    """Test ID: TL-01 - Triple Barrier Labeling validation."""
    featured_df = add_technical_indicators_and_features(
        sample_stock_data.copy(), (0.015, 0.03), qqq_data, profit_mode, 0.03
    )
    assert 'Target_Entry' in featured_df.columns
    assert 'Target_Profit_Take' in featured_df.columns
    assert 'Target_Cut_Loss' in featured_df.columns
    assert set(featured_df['Target_Entry'].unique()).issubset({0, 1})
    assert set(featured_df['Target_Profit_Take'].unique()).issubset({0, 1})
    assert set(featured_df['Target_Cut_Loss'].unique()).issubset({0, 1})