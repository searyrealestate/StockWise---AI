import pytest
import pandas as pd
import numpy as np
import os
import json
from unittest.mock import patch, MagicMock, call

# --- Import the class we are testing ---
from model_evaluator import Gen3ModelEvaluator
from data_manager import DataManager  # ADDED: Missing import


# --- Test Fixtures ---
@pytest.fixture
def synthetic_test_data():
    """
    Creates a small, controlled synthetic DataFrame with the CORRECTED Gen-3 feature set.
    """
    num_rows = 150
    clusters = ['low', 'mid', 'high']
    data = {
        'Volatility_Cluster': np.random.choice(clusters, num_rows),
        'Target_Entry': np.random.choice([0, 1], num_rows, p=[0.7, 0.3]),
        'Target_Profit_Take': np.random.choice([0, 1], num_rows, p=[0.7, 0.3]),
        'Target_Cut_Loss': np.random.choice([0, 1], num_rows, p=[0.9, 0.1]),
    }

    # --- CORRECTED Gen-3 Feature Columns ---
    feature_columns = [
        'Volume_MA_20', 'RSI_14', 'Momentum_5', 'MACD', 'MACD_Signal',
        'MACD_Histogram', 'BB_Position', 'Volatility_20D', 'ATR_14',
        'ADX', 'ADX_pos', 'ADX_neg', 'OBV', 'RSI_28',
        'Z_Score_20', 'BB_Width', 'Correlation_50D_QQQ',
        'BB_Upper', 'BB_Lower', 'BB_Middle', 'Daily_Return'
    ]
    # Add random data for all feature columns
    for col in feature_columns:
        if col not in data:
            data[col] = np.random.rand(num_rows)

    return pd.DataFrame(data)

@pytest.fixture
def mock_data_manager(synthetic_test_data):
    """Mocks the DataManager to return our synthetic test data."""
    manager = MagicMock()
    manager.combine_feature_files.return_value = synthetic_test_data
    return manager


# --- Unit Test Cases ---

@patch('model_evaluator.glob.glob')
@patch('model_evaluator.joblib.load')
@patch('builtins.open', new_callable=MagicMock)
def test_evaluator_loads_all_models(mock_open, mock_load, mock_glob, mock_data_manager):
    """
    Tests that the evaluator correctly finds and attempts to load all 9 models.
    """
    # Arrange: Simulate finding 9 model files
    mock_model_files = [f"/fake/dir/model_{i}.pkl" for i in range(9)]
    mock_glob.return_value = mock_model_files

    # Mock the return values for joblib and open
    mock_load.return_value = MagicMock()
    mock_open.return_value.__enter__.return_value.read.return_value = '["feature1"]'

    # Act
    evaluator = Gen3ModelEvaluator(model_dir="/fake/dir", test_data_manager=mock_data_manager)

    # Assert
    assert mock_glob.call_count == 1
    assert mock_load.call_count == 9
    assert mock_open.call_count == 9
    assert len(evaluator.models) == 9


@patch('model_evaluator.os.path.exists', return_value=True)
def test_evaluation_orchestration(mock_path_exists, synthetic_test_data, mock_data_manager):
    """
    Tests the main orchestration loop calls the evaluation helper correctly.
    """
    # Arrange
    evaluator = Gen3ModelEvaluator(model_dir="/fake/dir", test_data_manager=mock_data_manager)

    evaluator.models = {f"{mtype}_model_{c}_vol": MagicMock() for c in ['low', 'mid', 'high'] for mtype in
                        ['entry', 'profit_take', 'cut_loss']}
    evaluator.feature_cols = {name: ['Z_Score_20'] for name in evaluator.models.keys()}

    mock_report = {
        '0': {'precision': 0.9, 'recall': 0.9, 'f1-score': 0.9, 'support': 90},
        '1': {'precision': 0.8, 'recall': 0.8, 'f1-score': 0.8, 'support': 10}
    }

    with patch.object(evaluator, '_evaluate_single_model', return_value=mock_report) as mock_helper:
        # Act
        evaluator.evaluate_all_models()

        # Assert
        assert mock_helper.call_count == 9, "Expected the evaluation helper to be called 9 times."

        low_entry_call = next(c for c in mock_helper.call_args_list if c.kwargs['model_name'] == 'entry_model_low_vol')
        call_df = low_entry_call.kwargs['df']
        assert all(call_df['Volatility_Cluster'] == 'low'), "Data for low-vol model was not filtered correctly."


def test_evaluator_handles_missing_cluster_data(synthetic_test_data, mock_data_manager, caplog):
    """
    Tests that the evaluator logs a warning and continues if the test data
    is missing a specific volatility cluster.
    """
    # Arrange
    data_without_high_vol = synthetic_test_data[synthetic_test_data['Volatility_Cluster'] != 'high']
    mock_data_manager.combine_feature_files.return_value = data_without_high_vol

    evaluator = Gen3ModelEvaluator(model_dir="/fake/dir", test_data_manager=mock_data_manager)

    mock_model_instance = MagicMock()
    mock_model_instance.predict.side_effect = lambda X: np.zeros(len(X), dtype=int)
    evaluator.models = {f"{mtype}_model_{c}_vol": mock_model_instance for c in ['low', 'mid', 'high'] for mtype in
                        ['entry', 'profit_take', 'cut_loss']}
    evaluator.feature_cols = {name: ['Z_Score_20'] for name in evaluator.models.keys()}

    with patch('model_evaluator.classification_report', return_value={'0': {}, '1': {}}):
        # Act
        evaluator.evaluate_all_models()

    # Assert
    assert "No test data found for cluster 'high'" in caplog.text


def test_evaluator_handles_empty_test_data(mock_data_manager, caplog):
    """
    Tests that the evaluator handles the critical edge case of receiving no test data
    and exits gracefully without crashing.
    """
    # Arrange
    mock_data_manager.combine_feature_files.return_value = pd.DataFrame()
    evaluator = Gen3ModelEvaluator(model_dir="/fake/dir", test_data_manager=mock_data_manager)

    # Act
    result_df = evaluator.evaluate_all_models()

    # Assert
    assert "Combined test DataFrame is empty" in caplog.text
    assert result_df.empty