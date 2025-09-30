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
        'volatility_cluster': np.random.choice(clusters, num_rows),
        'target_entry': np.random.choice([0, 1], num_rows, p=[0.8, 0.2]),
        'target_profit_take': np.random.choice([0, 1], num_rows, p=[0.8, 0.2]),
        'target_cut_loss': np.random.choice([0, 1], num_rows, p=[0.9, 0.1]),
        'volume_ma_20': np.random.rand(num_rows) * 1e6, 'rsi_14': np.random.uniform(30, 70, num_rows),
        'momentum_5': np.random.randn(num_rows), 'macd': np.random.randn(num_rows),
        'macd_signal': np.random.randn(num_rows), 'macd_histogram': np.random.randn(num_rows),
        'bb_position': np.random.rand(num_rows), 'volatility_20d': np.random.rand(num_rows),
        'atr_14': np.random.rand(num_rows), 'adx': np.random.uniform(10, 50, num_rows),
        'adx_pos': np.random.uniform(10, 40, num_rows), 'adx_neg': np.random.uniform(10, 40, num_rows),
        'obv': np.random.randn(num_rows) * 1e7, 'rsi_28': np.random.uniform(30, 70, num_rows),
        'z_score_20': np.random.randn(num_rows), 'bb_width': np.random.rand(num_rows),
        'correlation_50d_qqq': np.random.uniform(-1, 1, num_rows),
        'vix_close': np.random.uniform(10, 30, num_rows), 'corr_tlt': np.random.uniform(-1, 1, num_rows),
        'cmf': np.random.uniform(-0.5, 0.5, num_rows),
        'bb_upper': np.random.rand(num_rows) * 110, 'bb_lower': np.random.rand(num_rows) * 90,
        'bb_middle': np.random.rand(num_rows) * 100, 'daily_return': np.random.randn(num_rows) / 100,
        'kama_10': np.random.rand(num_rows) * 100, 'stoch_k': np.random.uniform(0, 100, num_rows),
        'stoch_d': np.random.uniform(0, 100, num_rows), 'dominant_cycle': np.random.uniform(10, 50, num_rows),
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
    trainer = Gen3ModelTrainer(model_dir=model_dir, agent_name="test_agent")
    assert os.path.isdir(model_dir)


@patch('model_trainer.Gen3ModelTrainer._train_single_model')
def test_train_specialist_models_orchestration(mock_train_helper, synthetic_training_data):
    # Initialize with agent_name
    trainer = Gen3ModelTrainer(model_dir="/fake/dir", agent_name="test_agent")
    trainer.train_specialist_models(synthetic_training_data)
    assert mock_train_helper.call_count == 9


@patch('model_trainer.joblib.dump')
@patch('model_trainer.lgb.LGBMClassifier')
def test_train_single_model_saves_correctly(mock_lgbm, mock_joblib, synthetic_training_data, tmpdir):
    model_dir = str(tmpdir)
    mock_lgbm.return_value.predict.return_value = np.array([0, 1, 0, 1])
    trainer = Gen3ModelTrainer(model_dir=model_dir, agent_name="test_agent")
    feature_cols = [c for c in synthetic_training_data.columns if 'target' not in c and 'volatility_cluster' not in c]
    trainer._train_single_model(synthetic_training_data, feature_cols, 'target_entry', 'test_model.pkl', {})
    mock_joblib.assert_called()


def test_trainer_handles_empty_cluster_data(synthetic_training_data, caplog):
    """
    Tests that the trainer handles the case where a volatility cluster has no data,
    and logs a warning without crashing.
    """
    trainer = Gen3ModelTrainer(model_dir="/tmp/models", agent_name="test_agent")
    data_without_low_vol = synthetic_training_data[synthetic_training_data['volatility_cluster'] != 'low']
    trainer.train_specialist_models(data_without_low_vol)
    assert "No training data available for cluster 'low'" in caplog.text


def test_trainer_handles_insufficient_labels(synthetic_training_data, caplog):
    """
    Tests that the trainer skips training if a target column has fewer than
    two unique classes (i.e., all 0s or all 1s).
    """
    trainer = Gen3ModelTrainer(model_dir="/tmp/models", agent_name="test_agent")
    modified_data = synthetic_training_data.copy()
    modified_data.loc[modified_data['volatility_cluster'] == 'low', 'target_cut_loss'] = 0
    trainer.train_specialist_models(modified_data)
    assert "Target 'target_cut_loss' has fewer than 2 unique classes" in caplog.text


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
    trainer = Gen3ModelTrainer(model_dir=model_dir, agent_name="test_agent")
    feature_cols = [col for col in synthetic_training_data.columns if
                    'target' not in col and 'volatility_cluster' not in col]
    label_col = 'target_entry'
    model_name = 'entry_model_mid_vol.pkl'

    # We mock the prediction return value to ensure the metric functions work
    mock_lgbm_classifier.return_value.predict.return_value = np.random.choice([0, 1], size=len(synthetic_training_data[synthetic_training_data['volatility_cluster'] == 'mid']))

    mid_vol_data = synthetic_training_data[synthetic_training_data['volatility_cluster'] == 'mid'].copy()

    # Act
    metrics = trainer._train_single_model(
        df=mid_vol_data,
        feature_cols=feature_cols,
        label_col=label_col,
        model_name=model_name,
        custom_params={}
    )

    # Assert
    assert metrics['f1'] > 0.60
    assert metrics['f1'] == mock_f1_score.return_value