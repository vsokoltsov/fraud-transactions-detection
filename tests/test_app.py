import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

from api.app import api, get_db, get_model, PredictResponse
from api.config import Settings
from api.db.storage import Storage
from api.predictor.fraud import FraudPredictor


class TestApp:
    """Test suite for FastAPI application."""

    @pytest.fixture
    def mock_settings(self) -> Mock:
        """Create a mock Settings object for testing."""
        settings = Mock(spec=Settings)
        settings.storage = "csv"
        settings.storage_path = "/fake/path/to/data.csv"
        settings.features_path = "/fake/path/to/features.yaml"
        settings.models_path = "/fake/path/to/models"
        return settings

    @pytest.fixture
    def mock_storage(self) -> Mock:
        """Create a mock Storage object for testing."""
        storage = Mock(spec=Storage)
        storage.prepare_for_verification.return_value = pd.DataFrame(
            {
                "customer_id": [1001, 1001],
                "tx_amount_log": [5.0, 4.3],
                "tx_amount_log_deviates": [0, 1],
                "feature1": [1.0, 2.0],
                "feature2": [0.5, 1.5],
            }
        )
        return storage

    @pytest.fixture
    def mock_model(self) -> Mock:
        """Create a mock FraudPredictor object for testing."""
        model = Mock(spec=FraudPredictor)
        model.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])
        model.features = ["feature1", "feature2"]
        model.threshold = 0.5
        return model

    @pytest.fixture
    def client(self) -> TestClient:
        """Create a test client for the FastAPI app."""
        return TestClient(api)

    def test_get_db_function(self, mock_storage: Mock) -> None:
        """Test get_db dependency function."""
        with patch.object(api.state, "db", mock_storage, create=True):
            result = get_db()

            assert result == mock_storage
            assert isinstance(result, Storage)

    def test_get_model_function(self, mock_model: Mock) -> None:
        """Test get_model dependency function."""
        with patch.object(api.state, "model", mock_model, create=True):
            result = get_model()

            assert result == mock_model
            assert isinstance(result, FraudPredictor)

    def test_predict_response_model(self) -> None:
        """Test PredictResponse model validation."""
        response = PredictResponse(proba=0.75, is_fraud=1)

        assert response.proba == 0.75
        assert response.is_fraud == 1
        assert isinstance(response.proba, float)
        assert isinstance(response.is_fraud, int)

    def test_predict_response_model_validation(self) -> None:
        """Test PredictResponse model validation with invalid data."""
        with pytest.raises(ValueError):  # Pydantic validation error
            PredictResponse(proba=-1, is_fraud=1)

    def test_predict_endpoint_success(
        self, client: TestClient, mock_storage: Mock, mock_model: Mock
    ) -> None:
        """Test successful prediction endpoint."""
        with (
            patch.object(api.state, "db", mock_storage, create=True),
            patch.object(api.state, "model", mock_model, create=True),
        ):
            response = client.post("/model/v0/prediction/0")

            assert response.status_code == 200
            data = response.json()
            assert "proba" in data
            assert "is_fraud" in data
            assert isinstance(data["proba"], float)
            assert isinstance(data["is_fraud"], int)

            # Verify storage was called with correct parameters
            mock_storage.prepare_for_verification.assert_called_once_with(
                trx_id=0, features=mock_model.features
            )

    def test_predict_endpoint_with_different_trx_id(
        self, client: TestClient, mock_storage: Mock, mock_model: Mock
    ) -> None:
        """Test prediction endpoint with different transaction IDs."""
        with (
            patch.object(api.state, "db", mock_storage, create=True),
            patch.object(api.state, "model", mock_model, create=True),
        ):
            # Test with different transaction IDs
            for trx_id in [0, 1]:
                response = client.post(f"/model/v0/prediction/{trx_id}")
                assert response.status_code == 200

                # Verify storage was called with correct trx_id and features
                mock_storage.prepare_for_verification.assert_called_with(
                    trx_id=trx_id, features=mock_model.features
                )

    def test_predict_endpoint_storage_error(
        self, client: TestClient, mock_model: Mock
    ) -> None:
        """Test prediction endpoint when storage raises an error."""
        mock_storage = Mock(spec=Storage)
        mock_storage.prepare_for_verification.side_effect = IndexError(
            "Invalid transaction ID"
        )

        with (
            patch.object(api.state, "db", mock_storage, create=True),
            patch.object(api.state, "model", mock_model, create=True),
        ):
            with pytest.raises(IndexError, match="Invalid transaction ID"):
                client.post("/model/v0/prediction/999")

    def test_predict_endpoint_transaction_not_found(
        self, client: TestClient, mock_model: Mock
    ) -> None:
        """Test prediction endpoint when transaction is not found (returns 404)."""
        mock_storage = Mock(spec=Storage)
        mock_storage.prepare_for_verification.return_value = (
            None  # Transaction not found
        )

        with (
            patch.object(api.state, "db", mock_storage, create=True),
            patch.object(api.state, "model", mock_model, create=True),
        ):
            response = client.post("/model/v0/prediction/999")

            assert response.status_code == 404
            assert response.json()["detail"] == "Transaction not found"

    def test_predict_endpoint_model_error(
        self, client: TestClient, mock_storage: Mock, mock_model: Mock
    ) -> None:
        """Test prediction endpoint when model raises an error."""
        mock_model.predict_proba.side_effect = Exception("Model prediction error")

        with (
            patch.object(api.state, "db", mock_storage, create=True),
            patch.object(api.state, "model", mock_model, create=True),
        ):
            with pytest.raises(Exception, match="Model prediction error"):
                client.post("/model/v0/prediction/0")

    def test_predict_endpoint_invalid_trx_id_type(self, client: TestClient) -> None:
        """Test prediction endpoint with invalid transaction ID type."""
        with (
            patch.object(api.state, "db", Mock(), create=True),
            patch.object(api.state, "model", Mock(), create=True),
            patch.object(api.state, "features", [], create=True),
        ):
            response = client.post("/model/v0/prediction/invalid")

            assert response.status_code == 422  # Validation error

    def test_predict_endpoint_negative_trx_id(
        self, client: TestClient, mock_storage: Mock, mock_model: Mock
    ) -> None:
        """Test prediction endpoint with negative transaction ID."""
        with (
            patch.object(api.state, "db", mock_storage, create=True),
            patch.object(api.state, "model", mock_model, create=True),
        ):
            response = client.post("/model/v0/prediction/-1")

            # Should still work (pandas allows negative indexing)
            assert response.status_code == 200

    def test_predict_endpoint_data_processing(
        self, client: TestClient, mock_model: Mock
    ) -> None:
        """Test that data processing works correctly in prediction endpoint."""
        # Create a more realistic mock storage response
        mock_storage = Mock(spec=Storage)
        mock_df = pd.DataFrame(
            {
                "customer_id": [1001, 1001, 1002],
                "tx_amount_log": [5.0, 4.3, 6.1],
                "tx_amount_log_deviates": [0, 1, 0],
                "feature1": [1.0, 2.0, 3.0],
                "feature2": [0.5, 1.5, 2.5],
            }
        )
        mock_storage.prepare_for_verification.return_value = (
            mock_df.iloc[0].to_frame().T
        )

        with (
            patch.object(api.state, "db", mock_storage, create=True),
            patch.object(api.state, "model", mock_model, create=True),
        ):
            response = client.post("/model/v0/prediction/1")

            assert response.status_code == 200

            # Verify that the correct row was selected and processed
            mock_model.predict_proba.assert_called_once()
            called_df = mock_model.predict_proba.call_args[0][0]
            assert called_df.shape[0] == 1  # Should be a single row
            assert called_df.iloc[0]["feature1"] == 1.0
            assert called_df.iloc[0]["feature2"] == 0.5

    def test_predict_endpoint_threshold_logic(
        self, client: TestClient, mock_storage: Mock
    ) -> None:
        """Test threshold logic in prediction endpoint."""
        # Test with different probability values
        test_cases = [
            (0.3, 0.5, 0),  # Below threshold
            (0.5, 0.5, 1),  # At threshold
            (0.7, 0.5, 1),  # Above threshold
            (0.9, 0.5, 1),  # High probability
        ]

        for proba_value, threshold, expected_fraud in test_cases:
            mock_model = Mock(spec=FraudPredictor)
            mock_model.predict_proba.return_value = np.array(
                [[1 - proba_value, proba_value]]
            )
            mock_model.threshold = threshold
            mock_model.features = ["feature1", "feature2"]

            with (
                patch.object(api.state, "db", mock_storage, create=True),
                patch.object(api.state, "model", mock_model, create=True),
            ):
                response = client.post("/model/v0/prediction/0")

                assert response.status_code == 200
                data = response.json()
                assert data["proba"] == proba_value
                assert data["is_fraud"] == expected_fraud

    def test_app_initialization(self) -> None:
        """Test that the FastAPI app is properly initialized."""
        assert api.title == "Fraud Detection API"
        assert api.description == "Application for detection fraud transactions"
        assert api.version == "0.1.0"
        assert hasattr(api, "router")

    def test_app_routes(self) -> None:
        """Test that the app has the expected routes."""
        routes = [route.path for route in api.routes if hasattr(route, "path")]
        assert "/model/v0/prediction/{trx_id}" in routes

    def test_predict_endpoint_method(self) -> None:
        """Test that the predict endpoint only accepts POST requests."""
        # Find the predict route
        predict_route = None
        for route in api.routes:
            if (
                hasattr(route, "path")
                and hasattr(route, "methods")
                and "/model/v0/prediction/{trx_id}" in route.path
            ):
                predict_route = route
                break

        assert predict_route is not None
        assert "POST" in predict_route.methods

    def test_lifespan_context_manager(self, mock_settings: Mock) -> None:
        """Test the lifespan context manager."""
        with (
            patch("api.app.Storage", return_value=Mock()),
            patch("api.app.FraudPredictor", return_value=Mock()),
        ):
            # Test that lifespan can be called directly
            async def test_lifespan() -> None:
                from api.app import lifespan

                async with lifespan(api):
                    pass

            import asyncio

            asyncio.run(test_lifespan())

    def test_predict_endpoint_response_model_validation(
        self, client: TestClient, mock_storage: Mock, mock_model: Mock
    ) -> None:
        """Test that response model validation works correctly."""
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])

        with (
            patch.object(api.state, "db", mock_storage, create=True),
            patch.object(api.state, "model", mock_model, create=True),
        ):
            response = client.post("/model/v0/prediction/0")

            assert response.status_code == 200
            data = response.json()

            # Verify response structure matches PredictResponse model
            assert "proba" in data
            assert "is_fraud" in data
            assert isinstance(data["proba"], (int, float))
            assert isinstance(data["is_fraud"], int)

    def test_predict_endpoint_with_model_features(
        self, client: TestClient, mock_storage: Mock, mock_model: Mock
    ) -> None:
        """Test prediction endpoint uses model's features."""
        # Set model with specific features
        mock_model.features = ["custom_feature1", "custom_feature2"]

        with (
            patch.object(api.state, "db", mock_storage, create=True),
            patch.object(api.state, "model", mock_model, create=True),
        ):
            response = client.post("/model/v0/prediction/0")

            # Verify storage was called with model's features
            mock_storage.prepare_for_verification.assert_called_once_with(
                trx_id=0, features=mock_model.features
            )
            assert response.status_code == 200
