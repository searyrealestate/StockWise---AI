import pytest
import pandas as pd
import numpy as np
import os
import json
import lightgbm as lgb
from unittest.mock import patch, MagicMock, call, ANY

# --- Import the class we are testing ---
from model_trainer import Gen3ModelTrainer


# --- Test Fixtures ---

@pytest.fixture
def synthetic_training_data():
    """
    Creates a synthetic DataFrame with the CORRECTED Gen-3 feature set.
    """
    num_rows = 300
    clusters = ['low', 'mid', 'high']
    data = {
        # --- Grouping Column ---
        'Volatility_Cluster': np.random.choice(clusters, num_rows),

        # --- Target Columns ---
        'Target_Entry': np.random.choice([0, 1], num_rows, p=[0.8, 0.2]),
        'Target_Profit_Take': np.random.choice([0, 1], num_rows, p=[0.8, 0.2]),
        'Target_Cut_Loss': np.random.choice([0, 1], num_rows, p=[0.9, 0.1]),

        # --- CORRECTED Gen-3 Feature Columns ---
        'Volume_MA_20': np.random.rand(num_rows) * 1e6,
        'RSI_14': np.random.uniform(30, 70, num_rows),
        'Momentum_5': np.random.randn(num_rows),
        'MACD': np.random.randn(num_rows),
        'MACD_Signal': np.random.randn(num_rows),
        'MACD_Histogram': np.random.randn(num_rows),
        'BB_Upper': np.random.uniform(100, 110, num_rows),
        'BB_Lower': np.random.uniform(90, 100, num_rows),
        'BB_Middle': np.random.uniform(95, 105, num_rows),
        'BB_Position': np.random.rand(num_rows),
        'Daily_Return': np.random.normal(0, 0.02, num_rows),
        'Volatility_20D': np.random.rand(num_rows) * 0.05,
        'ATR_14': np.random.uniform(1, 5, num_rows),
        'ADX': np.random.uniform(10, 50, num_rows),
        'ADX_pos': np.random.uniform(10, 40, num_rows),
        'ADX_neg': np.random.uniform(10, 40, num_rows),
        'OBV': np.random.randint(1e6, 1e8, num_rows),
        'RSI_28': np.random.uniform(30, 70, num_rows),
        'Z_Score_20': np.random.randn(num_rows),
        'BB_Width': np.random.rand(num_rows),
        'Correlation_50D_QQQ': np.random.uniform(-1, 1, num_rows)
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_lgbm_classifier():
    """
    Mocks the LightGBM classifier to prevent actual training and ensures
    its predict method returns a dynamically sized, valid NumPy array.
    """
    with patch('model_trainer.lgb.LGBMClassifier', autospec=True) as mock:
        instance = mock.return_value
        instance.fit.return_value = instance
        instance.best_iteration_ = 100

        # Instead of a fixed return value, use a side_effect function.
        # This function will be called with the same arguments as model.predict(X_val).
        # It creates a prediction array of the CORRECT size every time.
        def dynamic_predict(X_data):
            return np.random.choice([0, 1], size=len(X_data))

        instance.predict.side_effect = dynamic_predict
        # --- END FIX ---

        yield mock

@pytest.fixture
def mock_joblib_dump():
    """Mocks joblib.dump to prevent file saving."""
    with patch('model_trainer.joblib.dump') as mock:
        yield mock


# --- Unit Test Cases ---

def test_trainer_initialization(tmpdir):
    """Tests if the trainer initializes correctly and creates the model directory."""
    model_dir = str(tmpdir.mkdir("test_models"))
    trainer = Gen3ModelTrainer(model_dir=model_dir, custom_params={})
    assert os.path.isdir(model_dir)


def test_train_specialist_models_orchestration(synthetic_training_data):
    """Tests the main orchestration loop to ensure it calls the helper for all 9 models."""
    trainer = Gen3ModelTrainer(model_dir="/fake/dir", custom_params={})

    with patch.object(trainer, '_train_single_model') as mock_helper:
        trainer.train_specialist_models(synthetic_training_data)
        assert mock_helper.call_count == 9, "Expected the training helper to be called 9 times (3 clusters * 3 model types)."

        model_names_called = {c.kwargs['model_name'] for c in mock_helper.call_args_list}
        expected_names = {
            'entry_model_low_vol.pkl', 'profit_take_model_low_vol.pkl', 'cut_loss_model_low_vol.pkl',
            'entry_model_mid_vol.pkl', 'profit_take_model_mid_vol.pkl', 'cut_loss_model_mid_vol.pkl',
            'entry_model_high_vol.pkl', 'profit_take_model_high_vol.pkl', 'cut_loss_model_high_vol.pkl'
        }
        assert model_names_called == expected_names


def test_train_single_model_saves_correctly(synthetic_training_data, mock_lgbm_classifier, mock_joblib_dump, tmpdir):
    """Tests the `_train_single_model` helper to ensure it saves the model and features file."""
    # Arrange
    model_dir = str(tmpdir)
    trainer = Gen3ModelTrainer(model_dir=model_dir, custom_params={'n_estimators': 10})
    feature_cols = [col for col in synthetic_training_data.columns if
                    'Target' not in col and 'Volatility_Cluster' not in col]
    label_col = 'Target_Entry'
    model_name = 'entry_model_low_vol.pkl'
    low_vol_data = synthetic_training_data[synthetic_training_data['Volatility_Cluster'] == 'low'].copy()

    # Act: Patch json.dump directly and call the function
    with patch('model_trainer.json.dump') as mock_json_dump:
        trainer._train_single_model(
            df=low_vol_data,
            feature_cols=feature_cols,
            label_col=label_col,
            model_name=model_name
        )

    # Assert
    expected_model_path = os.path.join(model_dir, model_name)
    mock_joblib_dump.assert_called_once_with(mock_lgbm_classifier.return_value, expected_model_path)

    # Check that json.dump was called with the correct features and a file handle
    mock_json_dump.assert_called_once()
    args, kwargs = mock_json_dump.call_args
    assert args[0] == feature_cols
    assert kwargs.get('indent') == 4


def test_trainer_handles_empty_cluster_data(synthetic_training_data, caplog):
    """
    Tests that the trainer handles the case where a volatility cluster has no data,
    and logs a warning without crashing.
    """
    trainer = Gen3ModelTrainer(model_dir="/tmp/models", custom_params={})
    data_without_high_vol = synthetic_training_data[synthetic_training_data['Volatility_Cluster'] != 'high']

    trainer.train_specialist_models(data_without_high_vol)

    assert "No data found for cluster 'high'" in caplog.text


def test_trainer_handles_insufficient_labels(synthetic_training_data, caplog):
    """
    Tests that the trainer skips training if a target column has fewer than
    two unique classes (i.e., all 0s or all 1s).
    """
    trainer = Gen3ModelTrainer(model_dir="/tmp/models", custom_params={})
    modified_data = synthetic_training_data.copy()
    modified_data.loc[modified_data['Volatility_Cluster'] == 'low', 'Target_Cut_Loss'] = 0

    trainer.train_specialist_models(modified_data)

    assert "Target 'Target_Cut_Loss' has fewer than 2 unique classes" in caplog.text


@patch('model_trainer.accuracy_score', return_value=0.9)
@patch('model_trainer.precision_score', return_value=0.8)
@patch('model_trainer.recall_score', return_value=0.85)
@patch('model_trainer.f1_score', return_value=0.75)
@patch('model_trainer.lgb.LGBMClassifier')
# Add the mock_joblib_dump fixture here
def test_trained_model_meets_kpi(mock_lgbm_classifier, mock_f1_score, mock_recall_score, mock_precision_score, mock_accuracy_score, synthetic_training_data, tmpdir, mock_joblib_dump):
    """
    Test ID: T4.1
    Tests that a trained model for the 'entry' task meets the F1-Score KPI of > 0.60.
    This verifies the training pipeline's output quality against a key metric.
    """
    # Arrange:
    model_dir = str(tmpdir)
    trainer = Gen3ModelTrainer(model_dir=model_dir, custom_params={'n_estimators': 10})
    feature_cols = [col for col in synthetic_training_data.columns if
                    'Target' not in col and 'Volatility_Cluster' not in col]
    label_col = 'Target_Entry'
    model_name = 'entry_model_mid_vol.pkl'

    # We mock the prediction return value to ensure the metric functions work
    mock_lgbm_classifier.return_value.predict.return_value = np.random.choice([0, 1], size=len(synthetic_training_data[synthetic_training_data['Volatility_Cluster'] == 'mid']))

    mid_vol_data = synthetic_training_data[synthetic_training_data['Volatility_Cluster'] == 'mid'].copy()

    # Act
    metrics = trainer._train_single_model(
        df=mid_vol_data,
        feature_cols=feature_cols,
        label_col=label_col,
        model_name=model_name
    )

    # Assert
    assert metrics['f1'] > 0.60
    assert metrics['f1'] == mock_f1_score.return_value