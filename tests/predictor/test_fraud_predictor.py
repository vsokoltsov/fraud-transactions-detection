# tests/test_fraud_predictor.py
import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, mock_open

from api.predictor.fraud import (
    FraudPredictor,
    LINEAR_REGRESSION,
    RANDOM_FOREST,
    XGBOOST,
)
from api.config import Settings


class TestFraudPredictor:
    """Test suite for FraudPredictor class."""

    @pytest.fixture
    def mock_settings(self) -> Mock:
        """Create a mock Settings object for testing."""
        settings = Mock(spec=Settings)
        settings.model = LINEAR_REGRESSION
        settings.models_path = "/fake/path"
        settings.features_path = "/fake/path/features.yaml"
        return settings

    @pytest.fixture
    def sample_dataframe(self) -> pd.DataFrame:
        """Create a sample DataFrame for testing predictions."""
        return pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0],
                "feature2": [0.5, 1.5, 2.5],
                "feature3": [10.0, 20.0, 30.0],
            }
        )

    @pytest.fixture
    def mock_model(self) -> Mock:
        """Create a mock model for testing predictions."""
        model = Mock()
        model.predict_proba.return_value = np.array(
            [[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]]
        )
        return model

    @pytest.fixture
    def mock_features(self) -> list[str]:
        """Create mock features list for testing."""
        return ["feature1", "feature2", "feature3"]

    @pytest.fixture
    def mock_thresholds(self) -> dict[str, float]:
        """Create mock thresholds for testing."""
        return {"linear_regression": 0.5, "random_forest": 0.3, "xgboost": 0.4}

    def test_init_linear_regression(
        self,
        mock_settings: Mock,
        mock_features: list[str],
        mock_thresholds: dict[str, float],
    ) -> None:
        """Test initialization with linear regression model."""
        mock_settings.model = LINEAR_REGRESSION

        with (
            patch("joblib.load") as mock_load,
            patch("yaml.safe_load") as mock_yaml_load,
            patch("builtins.open", create=True),
        ):
            mock_model = Mock()
            mock_load.return_value = mock_model

            # Mock features loading
            mock_yaml_load.side_effect = [mock_features, mock_thresholds]

            predictor = FraudPredictor(mock_settings)

            assert predictor.settings == mock_settings
            assert predictor.model == mock_model
            assert predictor.features == mock_features
            assert predictor.threshold == mock_thresholds["linear_regression"]
            mock_load.assert_called_once_with(
                os.path.join(mock_settings.models_path, f"{mock_settings.model}.pkl")
            )

    def test_init_random_forest(
        self,
        mock_settings: Mock,
        mock_features: list[str],
        mock_thresholds: dict[str, float],
    ) -> None:
        """Test initialization with random forest model."""
        mock_settings.model = RANDOM_FOREST

        with (
            patch("joblib.load") as mock_load,
            patch("yaml.safe_load") as mock_yaml_load,
            patch("builtins.open", create=True),
        ):
            mock_model = Mock()
            mock_load.return_value = mock_model

            # Mock features loading
            mock_yaml_load.side_effect = [mock_features, mock_thresholds]

            predictor = FraudPredictor(mock_settings)

            assert predictor.settings == mock_settings
            assert predictor.model == mock_model
            assert predictor.features == mock_features
            assert predictor.threshold == mock_thresholds["random_forest"]
            mock_load.assert_called_once_with(
                os.path.join(mock_settings.models_path, f"{mock_settings.model}.pkl")
            )

    def test_init_xgboost(
        self,
        mock_settings: Mock,
        mock_features: list[str],
        mock_thresholds: dict[str, float],
    ) -> None:
        """Test initialization with XGBoost model."""
        mock_settings.model = XGBOOST

        with (
            patch("xgboost.XGBClassifier") as mock_xgb_class,
            patch("yaml.safe_load") as mock_yaml_load,
            patch("builtins.open", create=True),
        ):
            mock_model = Mock()
            mock_xgb_class.return_value = mock_model

            # Mock features loading
            mock_yaml_load.side_effect = [mock_features, mock_thresholds]

            predictor = FraudPredictor(mock_settings)

            assert predictor.settings == mock_settings
            assert predictor.model == mock_model
            assert predictor.features == mock_features
            assert predictor.threshold == mock_thresholds["xgboost"]
            mock_xgb_class.assert_called_once()
            mock_model.load_model.assert_called_once_with(
                os.path.join(mock_settings.models_path, f"{mock_settings.model}.json")
            )

    def test_init_unknown_model(self, mock_settings: Mock) -> None:
        """Test initialization with unknown model raises ValueError."""
        mock_settings.model = "unknown_model"

        with pytest.raises(ValueError, match="Model is unknown"):
            FraudPredictor(mock_settings)

    def test_predict_proba(
        self, mock_settings: Mock, sample_dataframe: pd.DataFrame, mock_model: Mock
    ) -> None:
        """Test predict_proba method calls model correctly."""
        predictor = FraudPredictor.__new__(FraudPredictor)
        predictor.settings = mock_settings
        predictor.model = mock_model

        result = predictor.predict_proba(sample_dataframe)

        mock_model.predict_proba.assert_called_once_with(sample_dataframe)
        np.testing.assert_array_equal(
            result, np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
        )

    def test_predict_proba_with_different_data(
        self, mock_settings: Mock, mock_model: Mock
    ) -> None:
        """Test predict_proba with different DataFrame shapes."""
        predictor = FraudPredictor.__new__(FraudPredictor)
        predictor.settings = mock_settings
        predictor.model = mock_model

        # Test with single row
        single_row_df = pd.DataFrame({"feature1": [1.0], "feature2": [0.5]})
        expected_result = np.array([[0.9, 0.1]])
        mock_model.predict_proba.return_value = expected_result

        result = predictor.predict_proba(single_row_df)

        mock_model.predict_proba.assert_called_with(single_row_df)
        np.testing.assert_array_equal(result, expected_result)

    def test_model_loading_file_not_found(self, mock_settings: Mock) -> None:
        """Test handling of file not found errors during model loading."""
        mock_settings.model = LINEAR_REGRESSION

        with patch(
            "joblib.load", side_effect=FileNotFoundError("Model file not found")
        ):
            with pytest.raises(FileNotFoundError, match="Model file not found"):
                FraudPredictor(mock_settings)

    def test_xgboost_model_loading_file_not_found(self, mock_settings: Mock) -> None:
        """Test handling of file not found errors for XGBoost model loading."""
        mock_settings.model = XGBOOST

        with patch("xgboost.XGBClassifier") as mock_xgb_class:
            mock_model = Mock()
            mock_xgb_class.return_value = mock_model
            mock_model.load_model.side_effect = FileNotFoundError(
                "XGBoost model file not found"
            )

            with pytest.raises(FileNotFoundError, match="XGBoost model file not found"):
                FraudPredictor(mock_settings)

    def test_model_loading_joblib_error(self, mock_settings: Mock) -> None:
        """Test handling of joblib loading errors."""
        mock_settings.model = RANDOM_FOREST

        with patch("joblib.load", side_effect=Exception("Joblib loading error")):
            with pytest.raises(Exception, match="Joblib loading error"):
                FraudPredictor(mock_settings)

    def test_predict_proba_model_error(
        self, mock_settings: Mock, sample_dataframe: pd.DataFrame
    ) -> None:
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

    def test_model_path_construction(
        self,
        mock_settings: Mock,
        mock_features: list[str],
        mock_thresholds: dict[str, float],
    ) -> None:
        """Test that model paths are constructed correctly."""
        mock_settings.model = LINEAR_REGRESSION
        mock_settings.models_path = "/test/models"

        with (
            patch("joblib.load") as mock_load,
            patch("yaml.safe_load") as mock_yaml_load,
            patch("builtins.open", create=True),
        ):
            # Mock features and thresholds loading
            mock_yaml_load.side_effect = [mock_features, mock_thresholds]

            FraudPredictor(mock_settings)
            expected_path = os.path.join("/test/models", "linear_regression.pkl")
            mock_load.assert_called_once_with(expected_path)

    def test_xgboost_model_path_construction(
        self,
        mock_settings: Mock,
        mock_features: list[str],
        mock_thresholds: dict[str, float],
    ) -> None:
        """Test that XGBoost model paths are constructed correctly."""
        mock_settings.model = XGBOOST
        mock_settings.models_path = "/test/models"

        with (
            patch("xgboost.XGBClassifier") as mock_xgb_class,
            patch("yaml.safe_load") as mock_yaml_load,
            patch("builtins.open", create=True),
        ):
            mock_model = Mock()
            mock_xgb_class.return_value = mock_model

            # Mock features and thresholds loading
            mock_yaml_load.side_effect = [mock_features, mock_thresholds]

            FraudPredictor(mock_settings)
            expected_path = os.path.join("/test/models", "xgboost.json")
            mock_model.load_model.assert_called_once_with(expected_path)

    @pytest.mark.parametrize("model_type", [LINEAR_REGRESSION, RANDOM_FOREST, XGBOOST])
    def test_all_model_types_initialization(
        self,
        mock_settings: Mock,
        model_type: str,
        mock_features: list[str],
        mock_thresholds: dict[str, float],
    ) -> None:
        """Test initialization for all supported model types."""
        mock_settings.model = model_type

        with (
            patch("yaml.safe_load") as mock_yaml_load,
            patch("builtins.open", create=True),
        ):
            # Mock features loading
            mock_yaml_load.side_effect = [mock_features, mock_thresholds]

            if model_type == XGBOOST:
                with patch("xgboost.XGBClassifier") as mock_xgb_class:
                    mock_model = Mock()
                    mock_xgb_class.return_value = mock_model
                    predictor = FraudPredictor(mock_settings)
                    assert predictor.model == mock_model
                    assert predictor.features == mock_features
                    assert predictor.threshold == mock_thresholds[model_type]
            else:
                with patch("joblib.load") as mock_load:
                    mock_model = Mock()
                    mock_load.return_value = mock_model
                    predictor = FraudPredictor(mock_settings)
                    assert predictor.model == mock_model
                    assert predictor.features == mock_features
                    assert predictor.threshold == mock_thresholds[model_type]

    def test_predict_proba_return_type(
        self, mock_settings: Mock, sample_dataframe: pd.DataFrame, mock_model: Mock
    ) -> None:
        """Test that predict_proba returns correct numpy array type."""
        predictor = FraudPredictor.__new__(FraudPredictor)
        predictor.settings = mock_settings
        predictor.model = mock_model

        result = predictor.predict_proba(sample_dataframe)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64

    def test_empty_dataframe_prediction(
        self, mock_settings: Mock, mock_model: Mock
    ) -> None:
        """Test prediction with empty DataFrame."""
        predictor = FraudPredictor.__new__(FraudPredictor)
        predictor.settings = mock_settings
        predictor.model = mock_model

        empty_df = pd.DataFrame()
        mock_model.predict_proba.return_value = np.array([])

        result = predictor.predict_proba(empty_df)

        mock_model.predict_proba.assert_called_once_with(empty_df)
        assert len(result) == 0

    def test_load_features(self, mock_settings: Mock, mock_features: list[str]) -> None:
        """Test _load_features method."""
        predictor = FraudPredictor.__new__(FraudPredictor)
        predictor.settings = mock_settings

        with (
            patch("yaml.safe_load", return_value=mock_features),
            patch("builtins.open", create=True) as mock_open,
        ):
            result = predictor._load_features()

            assert result == mock_features
            mock_open.assert_called_once_with(
                mock_settings.features_path, "r", encoding="utf-8"
            )

    def test_load_model_thresholds(
        self, mock_settings: Mock, mock_thresholds: dict[str, float]
    ) -> None:
        """Test _load_model_thresholds method."""
        predictor = FraudPredictor.__new__(FraudPredictor)
        predictor.settings = mock_settings

        with (
            patch("yaml.safe_load", return_value=mock_thresholds),
            patch("builtins.open", create=True) as mock_open,
        ):
            result = predictor._load_model_thresholds("/fake/thresholds.yaml")

            assert result == mock_thresholds
            mock_open.assert_called_once_with(
                "/fake/thresholds.yaml", "r", encoding="utf-8"
            )

    def test_threshold_for_model(
        self, mock_settings: Mock, mock_thresholds: dict[str, float]
    ) -> None:
        """Test _threshold_for_model method."""
        predictor = FraudPredictor.__new__(FraudPredictor)
        predictor.settings = mock_settings

        with patch.object(
            predictor, "_load_model_thresholds", return_value=mock_thresholds
        ):
            result = predictor._threshold_for_model("/fake/thresholds.yaml")

            assert result == mock_thresholds["linear_regression"]

    def test_threshold_for_model_not_found(self, mock_settings: Mock) -> None:
        """Test _threshold_for_model method when model threshold not found."""
        predictor = FraudPredictor.__new__(FraudPredictor)
        predictor.settings = mock_settings
        predictor.settings.model = "unknown_model"

        with patch.object(
            predictor, "_load_model_thresholds", return_value={"linear_regression": 0.5}
        ):
            with pytest.raises(
                ValueError, match="There is no threshold for model 'unknown_model'"
            ):
                predictor._threshold_for_model("/fake/thresholds.yaml")

    def test_features_loading_error(self, mock_settings: Mock) -> None:
        """Test handling of features loading errors."""
        with (
            patch(
                "builtins.open",
                side_effect=FileNotFoundError("Features file not found"),
            ),
            patch("joblib.load"),
        ):
            with pytest.raises(FileNotFoundError, match="Features file not found"):
                FraudPredictor(mock_settings)

    def test_thresholds_loading_error(
        self, mock_settings: Mock, mock_features: list[str]
    ) -> None:
        """Test handling of thresholds loading errors."""
        with (
            patch("joblib.load"),
            patch("yaml.safe_load") as mock_yaml_load,
            patch("builtins.open", create=True),
        ):
            # First call returns features, second call raises error
            mock_yaml_load.side_effect = [
                mock_features,
                FileNotFoundError("Thresholds file not found"),
            ]

            with pytest.raises(FileNotFoundError, match="Thresholds file not found"):
                FraudPredictor(mock_settings)

    def test_yaml_loading_error(self, mock_settings: Mock) -> None:
        """Test handling of YAML parsing errors."""
        m = mock_open(read_data="invalid: [yaml")
        with (
            patch("builtins.open", m),
            patch("yaml.safe_load", side_effect=Exception("YAML parsing error")),
            patch("joblib.load"),
        ):
            with pytest.raises(Exception, match="YAML parsing error"):
                FraudPredictor(mock_settings)
