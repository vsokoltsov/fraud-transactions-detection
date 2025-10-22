# tests/test_fraud_predictor.py
import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock

from api.predictor.fraud import FraudPredictor, LINEAR_REGRESSION, RANDOM_FOREST, XGBOOST
from api.config import Settings
from api.types import DataFrame


class TestFraudPredictor:
    """Test suite for FraudPredictor class."""

    @pytest.fixture
    def mock_settings(self) -> Mock:
        """Create a mock Settings object for testing."""
        settings = Mock(spec=Settings)
        settings.model = LINEAR_REGRESSION
        settings.models_path = "/fake/path"
        return settings

    @pytest.fixture
    def sample_dataframe(self) -> pd.DataFrame:
        """Create a sample DataFrame for testing predictions."""
        return pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [0.5, 1.5, 2.5],
            'feature3': [10.0, 20.0, 30.0]
        })

    @pytest.fixture
    def mock_model(self) -> Mock:
        """Create a mock model for testing predictions."""
        model = Mock()
        model.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
        return model

    def test_init_linear_regression(self, mock_settings: Mock) -> None:
        """Test initialization with linear regression model."""
        mock_settings.model = LINEAR_REGRESSION
        
        with patch('joblib.load') as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model
            
            predictor = FraudPredictor(mock_settings)
            
            assert predictor.settings == mock_settings
            assert predictor.model == mock_model
            mock_load.assert_called_once_with(
                os.path.join(mock_settings.models_path, f"{mock_settings.model}.pkl")
            )

    def test_init_random_forest(self, mock_settings: Mock) -> None:
        """Test initialization with random forest model."""
        mock_settings.model = RANDOM_FOREST
        
        with patch('joblib.load') as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model
            
            predictor = FraudPredictor(mock_settings)
            
            assert predictor.settings == mock_settings
            assert predictor.model == mock_model
            mock_load.assert_called_once_with(
                os.path.join(mock_settings.models_path, f"{mock_settings.model}.pkl")
            )

    def test_init_xgboost(self, mock_settings: Mock) -> None:
        """Test initialization with XGBoost model."""
        mock_settings.model = XGBOOST
        
        with patch('xgboost.XGBClassifier') as mock_xgb_class:
            mock_model = Mock()
            mock_xgb_class.return_value = mock_model
            
            predictor = FraudPredictor(mock_settings)
            
            assert predictor.settings == mock_settings
            assert predictor.model == mock_model
            mock_xgb_class.assert_called_once()
            mock_model.load_model.assert_called_once_with(
                os.path.join(mock_settings.models_path, f"{mock_settings.model}.json")
            )

    def test_init_unknown_model(self, mock_settings: Mock) -> None:
        """Test initialization with unknown model raises ValueError."""
        mock_settings.model = "unknown_model"
        
        with pytest.raises(ValueError, match="Model is unknown"):
            FraudPredictor(mock_settings)

    def test_predict_proba(self, mock_settings: Mock, sample_dataframe: pd.DataFrame, mock_model: Mock) -> None:
        """Test predict_proba method calls model correctly."""
        predictor = FraudPredictor.__new__(FraudPredictor)
        predictor.settings = mock_settings
        predictor.model = mock_model
        
        result = predictor.predict_proba(sample_dataframe)
        
        mock_model.predict_proba.assert_called_once_with(sample_dataframe)
        np.testing.assert_array_equal(result, np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]]))

    def test_predict_proba_with_different_data(self, mock_settings: Mock, mock_model: Mock) -> None:
        """Test predict_proba with different DataFrame shapes."""
        predictor = FraudPredictor.__new__(FraudPredictor)
        predictor.settings = mock_settings
        predictor.model = mock_model
        
        # Test with single row
        single_row_df = pd.DataFrame({'feature1': [1.0], 'feature2': [0.5]})
        expected_result = np.array([[0.9, 0.1]])
        mock_model.predict_proba.return_value = expected_result
        
        result = predictor.predict_proba(single_row_df)
        
        mock_model.predict_proba.assert_called_with(single_row_df)
        np.testing.assert_array_equal(result, expected_result)

    def test_model_loading_file_not_found(self, mock_settings: Mock) -> None:
        """Test handling of file not found errors during model loading."""
        mock_settings.model = LINEAR_REGRESSION
        
        with patch('joblib.load', side_effect=FileNotFoundError("Model file not found")):
            with pytest.raises(FileNotFoundError, match="Model file not found"):
                FraudPredictor(mock_settings)

    def test_xgboost_model_loading_file_not_found(self, mock_settings: Mock) -> None:
        """Test handling of file not found errors for XGBoost model loading."""
        mock_settings.model = XGBOOST
        
        with patch('xgboost.XGBClassifier') as mock_xgb_class:
            mock_model = Mock()
            mock_xgb_class.return_value = mock_model
            mock_model.load_model.side_effect = FileNotFoundError("XGBoost model file not found")
            
            with pytest.raises(FileNotFoundError, match="XGBoost model file not found"):
                FraudPredictor(mock_settings)

    def test_model_loading_joblib_error(self, mock_settings: Mock) -> None:
        """Test handling of joblib loading errors."""
        mock_settings.model = RANDOM_FOREST
        
        with patch('joblib.load', side_effect=Exception("Joblib loading error")):
            with pytest.raises(Exception, match="Joblib loading error"):
                FraudPredictor(mock_settings)

    def test_predict_proba_model_error(self, mock_settings: Mock, sample_dataframe: pd.DataFrame) -> None:
        """Test handling of model prediction errors."""
        predictor = FraudPredictor.__new__(FraudPredictor)
        predictor.settings = mock_settings
        predictor.model = Mock()
        predictor.model.predict_proba.side_effect = Exception("Prediction error")
        
        with pytest.raises(Exception, match="Prediction error"):
            predictor.predict_proba(sample_dataframe)

    def test_constants_are_defined(self) -> None:
        """Test that model type constants are properly defined."""
        assert LINEAR_REGRESSION == "linear_regression"
        assert RANDOM_FOREST == "random_forest"
        assert XGBOOST == "xgboost"

    def test_model_path_construction(self, mock_settings: Mock) -> None:
        """Test that model paths are constructed correctly."""
        mock_settings.model = LINEAR_REGRESSION
        mock_settings.models_path = "/test/models"
        
        with patch('joblib.load') as mock_load:
            FraudPredictor(mock_settings)
            expected_path = os.path.join("/test/models", "linear_regression.pkl")
            mock_load.assert_called_once_with(expected_path)

    def test_xgboost_model_path_construction(self, mock_settings: Mock) -> None:
        """Test that XGBoost model paths are constructed correctly."""
        mock_settings.model = XGBOOST
        mock_settings.models_path = "/test/models"
        
        with patch('xgboost.XGBClassifier') as mock_xgb_class:
            mock_model = Mock()
            mock_xgb_class.return_value = mock_model
            FraudPredictor(mock_settings)
            expected_path = os.path.join("/test/models", "xgboost.json")
            mock_model.load_model.assert_called_once_with(expected_path)

    @pytest.mark.parametrize("model_type", [LINEAR_REGRESSION, RANDOM_FOREST, XGBOOST])
    def test_all_model_types_initialization(self, mock_settings: Mock, model_type: str) -> None:
        """Test initialization for all supported model types."""
        mock_settings.model = model_type
        
        if model_type == XGBOOST:
            with patch('xgboost.XGBClassifier') as mock_xgb_class:
                mock_model = Mock()
                mock_xgb_class.return_value = mock_model
                predictor = FraudPredictor(mock_settings)
                assert predictor.model == mock_model
        else:
            with patch('joblib.load') as mock_load:
                mock_model = Mock()
                mock_load.return_value = mock_model
                predictor = FraudPredictor(mock_settings)
                assert predictor.model == mock_model

    def test_predict_proba_return_type(self, mock_settings: Mock, sample_dataframe: pd.DataFrame, mock_model: Mock) -> None:
        """Test that predict_proba returns correct numpy array type."""
        predictor = FraudPredictor.__new__(FraudPredictor)
        predictor.settings = mock_settings
        predictor.model = mock_model
        
        result = predictor.predict_proba(sample_dataframe)
        
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64

    def test_empty_dataframe_prediction(self, mock_settings: Mock, mock_model: Mock) -> None:
        """Test prediction with empty DataFrame."""
        predictor = FraudPredictor.__new__(FraudPredictor)
        predictor.settings = mock_settings
        predictor.model = mock_model
        
        empty_df = pd.DataFrame()
        mock_model.predict_proba.return_value = np.array([])
        
        result = predictor.predict_proba(empty_df)
        
        mock_model.predict_proba.assert_called_once_with(empty_df)
        assert len(result) == 0