import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from typing import List, Optional
from pathlib import Path

from api.db.csv import CSVStorage


class TestCSVStorage:
    """Test suite for CSVStorage class."""

    @pytest.fixture
    def sample_csv_content(self) -> str:
        """Create sample CSV content for testing."""
        return """TX_DATETIME,CUSTOMER_ID,SECTOR_ID,TX_AMOUNT,TX_FRAUD
2023-01-01 10:00:00,1001,1,150.50,0
2023-01-01 11:30:00,1001,1,75.25,0
2023-01-01 12:45:00,1002,2,200.00,1
2023-01-01 14:20:00,1001,1,300.75,0
2023-01-01 15:10:00,1003,3,50.00,0"""

    @pytest.fixture
    def sample_csv_file(self, tmp_path: Path, sample_csv_content: str) -> str:
        """Create a temporary CSV file for testing."""
        csv_file = tmp_path / "test_transactions.csv"
        csv_file.write_text(sample_csv_content)
        return str(csv_file)

    @pytest.fixture
    def sample_dataframe(self) -> pd.DataFrame:
        """Create a sample DataFrame matching the expected structure."""
        return pd.DataFrame(
            {
                "tx_datetime": pd.to_datetime(
                    [
                        "2023-01-01 10:00:00",
                        "2023-01-01 11:30:00",
                        "2023-01-01 12:45:00",
                    ]
                ),
                "customer_id": [1001, 1001, 1002],
                "sector_id": [1, 1, 2],
                "tx_amount": [150.50, 75.25, 200.00],
                "tx_fraud": [0, 0, 1],
            }
        )

    @pytest.fixture
    def mock_generate_features(self) -> Mock:
        """Create a mock for generate_features function."""
        mock = Mock()
        mock.return_value = pd.DataFrame(
            {
                "customer_id": [1001, 1001],
                "tx_amount_log": [5.0, 4.3],
                "tx_amount_log_deviates": [0, 1],
            }
        )
        return mock

    def test_init_loads_csv_correctly(self, sample_csv_file: str) -> None:
        """Test that CSV is loaded correctly with proper data types and column formatting."""
        with patch("pandas.read_csv") as mock_read_csv:
            mock_df = pd.DataFrame(
                {
                    "TX_DATETIME": pd.to_datetime(["2023-01-01 10:00:00"]),
                    "CUSTOMER_ID": [1001],
                    "SECTOR_ID": [1],
                    "TX_AMOUNT": [150.50],
                    "TX_FRAUD": [0],
                }
            )
            mock_read_csv.return_value = mock_df

            storage = CSVStorage(sample_csv_file)

            # Verify pandas.read_csv was called with correct parameters
            mock_read_csv.assert_called_once_with(
                sample_csv_file,
                parse_dates=["TX_DATETIME"],
                dtype={
                    "CUSTOMER_ID": np.int64,
                    "SECTOR_ID": np.int64,
                    "TX_FRAUD": np.int8,
                },
            )

            # Verify column names are converted to lowercase with underscores
            assert storage.data.columns.tolist() == [
                "tx_datetime",
                "customer_id",
                "sector_id",
                "tx_amount",
                "tx_fraud",
            ]

    def test_init_with_default_features_version(self, sample_csv_file: str) -> None:
        """Test initialization with default features_version parameter."""
        with patch("pandas.read_csv") as mock_read_csv:
            mock_df = pd.DataFrame(
                {
                    "TX_DATETIME": [],
                    "CUSTOMER_ID": [],
                    "SECTOR_ID": [],
                    "TX_AMOUNT": [],
                    "TX_FRAUD": [],
                }
            )
            mock_read_csv.return_value = mock_df

            storage = CSVStorage(sample_csv_file)

            assert storage.file_path == sample_csv_file
            # features_version is stored but not used in current implementation
            # This test documents the parameter exists

    def test_init_with_custom_features_version(self, sample_csv_file: str) -> None:
        """Test initialization with custom features_version parameter."""
        with patch("pandas.read_csv") as mock_read_csv:
            mock_df = pd.DataFrame(
                {
                    "TX_DATETIME": [],
                    "CUSTOMER_ID": [],
                    "SECTOR_ID": [],
                    "TX_AMOUNT": [],
                    "TX_FRAUD": [],
                }
            )
            mock_read_csv.return_value = mock_df

            storage = CSVStorage(sample_csv_file, features_version="v1")

            assert storage.file_path == sample_csv_file

    def test_prepare_for_verification_returns_correct_data(
        self, sample_csv_file: str, mock_generate_features: Mock
    ) -> None:
        """Test prepare_for_verification returns correct transaction data."""
        with (
            patch("pandas.read_csv") as mock_read_csv,
            patch("api.db.csv.generate_features", mock_generate_features),
        ):
            # Create mock data
            mock_df = pd.DataFrame(
                {
                    "TX_DATETIME": pd.to_datetime(
                        [
                            "2023-01-01 10:00:00",
                            "2023-01-01 11:30:00",
                            "2023-01-01 12:45:00",
                        ]
                    ),
                    "CUSTOMER_ID": [1001, 1001, 1002],
                    "SECTOR_ID": [1, 1, 2],
                    "TX_AMOUNT": [150.50, 75.25, 200.00],
                    "TX_FRAUD": [0, 0, 1],
                }
            )
            mock_read_csv.return_value = mock_df

            # Mock generate_features to return data with row_id
            mock_generate_features.return_value = pd.DataFrame(
                {
                    "customer_id": [1001, 1001],
                    "tx_amount_log": [5.0, 4.3],
                    "tx_amount_log_deviates": [0, 1],
                    "row_id": [0, 1],
                }
            )

            storage = CSVStorage(sample_csv_file)
            features = ["tx_amount_log", "tx_amount_log_deviates"]
            result = storage.prepare_for_verification(
                0, features
            )  # First transaction (customer_id: 1001)

            # Verify generate_features was called with correct user transactions
            mock_generate_features.assert_called_once()
            called_df = mock_generate_features.call_args[0][0]
            assert len(called_df) == 2  # Two transactions for customer 1001
            assert called_df["customer_id"].iloc[0] == 1001
            assert called_df["customer_id"].iloc[1] == 1001
            assert "row_id" in called_df.columns  # Verify row_id was added

            # Verify result contains only the specified features for the correct row
            assert result is not None
            assert list(result.columns) == features
            assert len(result) == 1  # Only one row (the requested transaction)

    def test_prepare_for_verification_with_different_trx_id(
        self, sample_csv_file: str, mock_generate_features: Mock
    ) -> None:
        """Test prepare_for_verification with different transaction IDs."""
        with (
            patch("pandas.read_csv") as mock_read_csv,
            patch("api.db.csv.generate_features", mock_generate_features),
        ):
            mock_df = pd.DataFrame(
                {
                    "TX_DATETIME": pd.to_datetime(
                        [
                            "2023-01-01 10:00:00",
                            "2023-01-01 11:30:00",
                            "2023-01-01 12:45:00",
                        ]
                    ),
                    "CUSTOMER_ID": [1001, 1002, 1002],
                    "SECTOR_ID": [1, 2, 2],
                    "TX_AMOUNT": [150.50, 75.25, 200.00],
                    "TX_FRAUD": [0, 0, 1],
                }
            )
            mock_read_csv.return_value = mock_df

            # Mock generate_features to return data with row_id
            mock_generate_features.return_value = pd.DataFrame(
                {
                    "customer_id": [1002, 1002],
                    "tx_amount_log": [4.3, 5.3],
                    "tx_amount_log_deviates": [0, 1],
                    "row_id": [1, 2],
                }
            )

            storage = CSVStorage(sample_csv_file)
            features = ["tx_amount_log", "tx_amount_log_deviates"]

            # Test with transaction ID 1 (customer_id: 1002)
            result = storage.prepare_for_verification(1, features)
            called_df = mock_generate_features.call_args[0][0]
            assert len(called_df) == 2  # Two transactions for customer 1002
            assert all(called_df["customer_id"] == 1002)
            assert result is not None
            assert list(result.columns) == features

    def test_prepare_for_verification_single_customer_transaction(
        self, sample_csv_file: str, mock_generate_features: Mock
    ) -> None:
        """Test prepare_for_verification when customer has only one transaction."""
        with (
            patch("pandas.read_csv") as mock_read_csv,
            patch("api.db.csv.generate_features", mock_generate_features),
        ):
            mock_df = pd.DataFrame(
                {
                    "TX_DATETIME": pd.to_datetime(
                        ["2023-01-01 10:00:00", "2023-01-01 11:30:00"]
                    ),
                    "CUSTOMER_ID": [1001, 1002],
                    "SECTOR_ID": [1, 2],
                    "TX_AMOUNT": [150.50, 75.25],
                    "TX_FRAUD": [0, 0],
                }
            )
            mock_read_csv.return_value = mock_df

            # Mock generate_features to return data with row_id
            mock_generate_features.return_value = pd.DataFrame(
                {
                    "customer_id": [1002],
                    "tx_amount_log": [4.3],
                    "tx_amount_log_deviates": [0],
                    "row_id": [1],
                }
            )

            storage = CSVStorage(sample_csv_file)
            features = ["tx_amount_log", "tx_amount_log_deviates"]

            # Test with transaction ID 1 (customer_id: 1002, single transaction)
            result = storage.prepare_for_verification(1, features)
            called_df = mock_generate_features.call_args[0][0]
            assert len(called_df) == 1
            assert called_df["customer_id"].iloc[0] == 1002
            assert result is not None
            assert list(result.columns) == features

    def test_prepare_for_verification_invalid_trx_id(
        self, sample_csv_file: str
    ) -> None:
        """Test prepare_for_verification with invalid transaction ID."""
        with patch("pandas.read_csv") as mock_read_csv:
            mock_df = pd.DataFrame(
                {
                    "TX_DATETIME": pd.to_datetime(["2023-01-01 10:00:00"]),
                    "CUSTOMER_ID": [1001],
                    "SECTOR_ID": [1],
                    "TX_AMOUNT": [150.50],
                    "TX_FRAUD": [0],
                }
            )
            mock_read_csv.return_value = mock_df

            storage = CSVStorage(sample_csv_file)
            features = ["tx_amount_log"]

            # Should return None for invalid transaction ID
            result = storage.prepare_for_verification(5, features)  # Invalid index
            assert result is None

    def test_prepare_for_verification_negative_trx_id(
        self, sample_csv_file: str, mock_generate_features: Mock
    ) -> None:
        """Test prepare_for_verification with negative transaction ID."""
        with (
            patch("pandas.read_csv") as mock_read_csv,
            patch("api.db.csv.generate_features", mock_generate_features),
        ):
            mock_df = pd.DataFrame(
                {
                    "TX_DATETIME": pd.to_datetime(["2023-01-01 10:00:00"]),
                    "CUSTOMER_ID": [1001],
                    "SECTOR_ID": [1],
                    "TX_AMOUNT": [150.50],
                    "TX_FRAUD": [0],
                }
            )
            mock_read_csv.return_value = mock_df

            # Mock generate_features to return data with row_id
            mock_generate_features.return_value = pd.DataFrame(
                {
                    "customer_id": [1001],
                    "tx_amount_log": [5.0],
                    "tx_amount_log_deviates": [0],
                    "row_id": [0],
                }
            )

            storage = CSVStorage(sample_csv_file)
            features = ["tx_amount_log", "tx_amount_log_deviates"]

            # Negative index wraps around in pandas, so -1 becomes the last row
            # This should not raise an error, but should return the last transaction
            result = storage.prepare_for_verification(-1, features)
            # Verify it returns a DataFrame with the correct features
            assert result is not None
            assert isinstance(result, pd.DataFrame)
            assert list(result.columns) == features

    def test_csv_loading_error_handling(self, tmp_path: Path) -> None:
        """Test handling of CSV loading errors."""
        non_existent_file = tmp_path / "non_existent.csv"

        with pytest.raises(FileNotFoundError):
            CSVStorage(str(non_existent_file))

    def test_csv_parsing_error_handling(self, tmp_path: Path) -> None:
        """Test handling of CSV parsing errors."""
        invalid_csv_file = tmp_path / "invalid.csv"
        invalid_csv_file.write_text("invalid,csv,content\nwith,missing,columns")

        with pytest.raises((KeyError, ValueError)):
            CSVStorage(str(invalid_csv_file))

    def test_data_types_are_correct(self, sample_csv_file: str) -> None:
        """Test that data types are correctly set after loading."""
        with patch("pandas.read_csv") as mock_read_csv:
            # Create mock DataFrame with the correct dtypes that would be set by CSV loading
            mock_df = pd.DataFrame(
                {
                    "TX_DATETIME": pd.to_datetime(["2023-01-01 10:00:00"]),
                    "CUSTOMER_ID": [1001],
                    "SECTOR_ID": [1],
                    "TX_AMOUNT": [150.50],
                    "TX_FRAUD": [0],
                }
            )
            # Set the dtypes as they would be after CSV loading
            mock_df["CUSTOMER_ID"] = mock_df["CUSTOMER_ID"].astype(np.int64)
            mock_df["SECTOR_ID"] = mock_df["SECTOR_ID"].astype(np.int64)
            mock_df["TX_FRAUD"] = mock_df["TX_FRAUD"].astype(np.int8)
            mock_read_csv.return_value = mock_df

            storage = CSVStorage(sample_csv_file)

            # Verify the data types are preserved
            assert storage.data["customer_id"].dtype == np.int64
            assert storage.data["sector_id"].dtype == np.int64
            assert storage.data["tx_fraud"].dtype == np.int8
            assert pd.api.types.is_datetime64_any_dtype(storage.data["tx_datetime"])

    def test_column_name_transformation(self, sample_csv_file: str) -> None:
        """Test that column names are properly transformed to lowercase with underscores."""
        with patch("pandas.read_csv") as mock_read_csv:
            mock_df = pd.DataFrame(
                {
                    "TX_DATETIME": pd.to_datetime(["2023-01-01 10:00:00"]),
                    "CUSTOMER_ID": [1001],
                    "SECTOR_ID": [1],
                    "TX_AMOUNT": [150.50],
                    "TX_FRAUD": [0],
                    "SOME COLUMN": [1],  # Column with space
                    "ANOTHER_COLUMN": [2],  # Column with underscore
                }
            )
            mock_read_csv.return_value = mock_df

            storage = CSVStorage(sample_csv_file)

            expected_columns = [
                "tx_datetime",
                "customer_id",
                "sector_id",
                "tx_amount",
                "tx_fraud",
                "some_column",
                "another_column",
            ]
            assert storage.data.columns.tolist() == expected_columns

    def test_generate_features_error_handling(
        self, sample_csv_file: str, mock_generate_features: Mock
    ) -> None:
        """Test error handling when generate_features fails."""
        with (
            patch("pandas.read_csv") as mock_read_csv,
            patch("api.db.csv.generate_features", mock_generate_features),
        ):
            mock_df = pd.DataFrame(
                {
                    "TX_DATETIME": pd.to_datetime(["2023-01-01 10:00:00"]),
                    "CUSTOMER_ID": [1001],
                    "SECTOR_ID": [1],
                    "TX_AMOUNT": [150.50],
                    "TX_FRAUD": [0],
                }
            )
            mock_read_csv.return_value = mock_df
            mock_generate_features.side_effect = Exception("Feature generation failed")

            storage = CSVStorage(sample_csv_file)
            features = ["tx_amount_log"]

            with pytest.raises(Exception, match="Feature generation failed"):
                storage.prepare_for_verification(0, features)

    def test_empty_dataframe_handling(self, sample_csv_file: str) -> None:
        """Test handling of empty DataFrame."""
        with (
            patch("pandas.read_csv") as mock_read_csv,
            patch("api.db.csv.generate_features") as mock_generate_features,
        ):
            mock_df = pd.DataFrame(
                columns=[
                    "TX_DATETIME",
                    "CUSTOMER_ID",
                    "SECTOR_ID",
                    "TX_AMOUNT",
                    "TX_FRAUD",
                ]
            )
            mock_read_csv.return_value = mock_df
            mock_generate_features.return_value = pd.DataFrame()

            storage = CSVStorage(sample_csv_file)
            features = ["tx_amount_log"]

            # Should return None for empty DataFrame
            result = storage.prepare_for_verification(0, features)
            assert result is None

    def test_storage_backend_protocol_compliance(self, sample_csv_file: str) -> None:
        """Test that CSVStorage implements StorageBackend protocol correctly."""
        with (
            patch("pandas.read_csv") as mock_read_csv,
            patch("api.db.csv.generate_features") as mock_generate_features,
        ):
            mock_df = pd.DataFrame(
                {
                    "TX_DATETIME": pd.to_datetime(["2023-01-01 10:00:00"]),
                    "CUSTOMER_ID": [1001],
                    "SECTOR_ID": [1],
                    "TX_AMOUNT": [150.50],
                    "TX_FRAUD": [0],
                }
            )
            mock_read_csv.return_value = mock_df
            mock_generate_features.return_value = pd.DataFrame()

            storage = CSVStorage(sample_csv_file)

            # Verify it has the required method
            assert hasattr(storage, "prepare_for_verification")
            assert callable(getattr(storage, "prepare_for_verification"))

            # Verify method signature matches updated protocol
            import inspect

            sig = inspect.signature(storage.prepare_for_verification)
            assert list(sig.parameters.keys()) == ["trx_id", "features"]
            assert sig.return_annotation == Optional[pd.DataFrame]

    def test_multiple_customers_data_handling(
        self, sample_csv_file: str, mock_generate_features: Mock
    ) -> None:
        """Test handling of data with multiple customers."""
        with (
            patch("pandas.read_csv") as mock_read_csv,
            patch("api.db.csv.generate_features", mock_generate_features),
        ):
            mock_df = pd.DataFrame(
                {
                    "TX_DATETIME": pd.to_datetime(
                        [
                            "2023-01-01 10:00:00",
                            "2023-01-01 11:00:00",
                            "2023-01-01 12:00:00",
                            "2023-01-01 13:00:00",
                            "2023-01-01 14:00:00",
                        ]
                    ),
                    "CUSTOMER_ID": [1001, 1001, 1002, 1002, 1003],
                    "SECTOR_ID": [1, 1, 2, 2, 3],
                    "TX_AMOUNT": [150.50, 75.25, 200.00, 100.00, 50.00],
                    "TX_FRAUD": [0, 0, 1, 0, 0],
                }
            )
            mock_read_csv.return_value = mock_df

            # Mock generate_features to return data with row_id
            mock_generate_features.return_value = pd.DataFrame(
                {
                    "customer_id": [1001, 1001],
                    "tx_amount_log": [5.0, 4.3],
                    "tx_amount_log_deviates": [0, 1],
                    "row_id": [0, 1],
                }
            )

            storage = CSVStorage(sample_csv_file)
            features = ["tx_amount_log", "tx_amount_log_deviates"]

            # Test with customer 1001 (2 transactions)
            result = storage.prepare_for_verification(0, features)
            called_df = mock_generate_features.call_args[0][0]
            assert len(called_df) == 2
            assert all(called_df["customer_id"] == 1001)
            assert result is not None
            assert list(result.columns) == features

            # Test with customer 1002 (2 transactions)
            mock_generate_features.return_value = pd.DataFrame(
                {
                    "customer_id": [1002, 1002],
                    "tx_amount_log": [5.3, 4.6],
                    "tx_amount_log_deviates": [1, 0],
                    "row_id": [2, 3],
                }
            )
            result = storage.prepare_for_verification(2, features)
            called_df = mock_generate_features.call_args[0][0]
            assert len(called_df) == 2
            assert all(called_df["customer_id"] == 1002)
            assert result is not None

            # Test with customer 1003 (1 transaction)
            mock_generate_features.return_value = pd.DataFrame(
                {
                    "customer_id": [1003],
                    "tx_amount_log": [3.9],
                    "tx_amount_log_deviates": [0],
                    "row_id": [4],
                }
            )
            result = storage.prepare_for_verification(4, features)
            called_df = mock_generate_features.call_args[0][0]
            assert len(called_df) == 1
            assert all(called_df["customer_id"] == 1003)
            assert result is not None

    def test_file_path_storage(self, sample_csv_file: str) -> None:
        """Test that file_path is stored correctly."""
        with patch("pandas.read_csv") as mock_read_csv:
            mock_df = pd.DataFrame(
                {
                    "TX_DATETIME": [],
                    "CUSTOMER_ID": [],
                    "SECTOR_ID": [],
                    "TX_AMOUNT": [],
                    "TX_FRAUD": [],
                }
            )
            mock_read_csv.return_value = mock_df

            storage = CSVStorage(sample_csv_file)

            assert storage.file_path == sample_csv_file

    def test_data_attribute_type(self, sample_csv_file: str) -> None:
        """Test that data attribute is a pandas DataFrame."""
        with patch("pandas.read_csv") as mock_read_csv:
            mock_df = pd.DataFrame(
                {
                    "TX_DATETIME": [],
                    "CUSTOMER_ID": [],
                    "SECTOR_ID": [],
                    "TX_AMOUNT": [],
                    "TX_FRAUD": [],
                }
            )
            mock_read_csv.return_value = mock_df

            storage = CSVStorage(sample_csv_file)

            assert isinstance(storage.data, pd.DataFrame)

    def test_prepare_for_verification_return_type(
        self, sample_csv_file: str, mock_generate_features: Mock
    ) -> None:
        """Test that prepare_for_verification returns pandas DataFrame."""
        with (
            patch("pandas.read_csv") as mock_read_csv,
            patch("api.db.csv.generate_features", mock_generate_features),
        ):
            mock_df = pd.DataFrame(
                {
                    "TX_DATETIME": pd.to_datetime(["2023-01-01 10:00:00"]),
                    "CUSTOMER_ID": [1001],
                    "SECTOR_ID": [1],
                    "TX_AMOUNT": [150.50],
                    "TX_FRAUD": [0],
                }
            )
            mock_read_csv.return_value = mock_df

            # Mock generate_features to return data with row_id
            mock_generate_features.return_value = pd.DataFrame(
                {
                    "customer_id": [1001],
                    "tx_amount_log": [5.0],
                    "tx_amount_log_deviates": [0],
                    "row_id": [0],
                }
            )

            storage = CSVStorage(sample_csv_file)
            features = ["tx_amount_log", "tx_amount_log_deviates"]
            result = storage.prepare_for_verification(0, features)

            assert result is not None
            assert isinstance(result, pd.DataFrame)
            assert list(result.columns) == features

    def test_iloc_access_behavior(self, sample_csv_file: str) -> None:
        """Test that iloc access works correctly for transaction selection."""
        with patch("pandas.read_csv") as mock_read_csv:
            mock_df = pd.DataFrame(
                {
                    "TX_DATETIME": pd.to_datetime(
                        ["2023-01-01 10:00:00", "2023-01-01 11:00:00"]
                    ),
                    "CUSTOMER_ID": [1001, 1002],
                    "SECTOR_ID": [1, 2],
                    "TX_AMOUNT": [150.50, 200.00],
                    "TX_FRAUD": [0, 1],
                }
            )
            mock_read_csv.return_value = mock_df

            storage = CSVStorage(sample_csv_file)

            # Test that iloc access works
            trx_0 = storage.data.iloc[0]
            trx_1 = storage.data.iloc[1]

            assert trx_0.customer_id == 1001
            assert trx_1.customer_id == 1002

    def test_customer_filtering_behavior(
        self, sample_csv_file: str, mock_generate_features: Mock
    ) -> None:
        """Test that customer filtering works correctly."""
        with (
            patch("pandas.read_csv") as mock_read_csv,
            patch("api.db.csv.generate_features", mock_generate_features),
        ):
            mock_df = pd.DataFrame(
                {
                    "TX_DATETIME": pd.to_datetime(
                        [
                            "2023-01-01 10:00:00",
                            "2023-01-01 11:00:00",
                            "2023-01-01 12:00:00",
                        ]
                    ),
                    "CUSTOMER_ID": [1001, 1001, 1002],
                    "SECTOR_ID": [1, 1, 2],
                    "TX_AMOUNT": [150.50, 200.00, 300.00],
                    "TX_FRAUD": [0, 0, 1],
                }
            )
            mock_read_csv.return_value = mock_df

            # Mock generate_features to return data with row_id
            mock_generate_features.return_value = pd.DataFrame(
                {
                    "customer_id": [1001, 1001],
                    "tx_amount_log": [5.0, 5.3],
                    "tx_amount_log_deviates": [0, 1],
                    "row_id": [0, 1],
                }
            )

            storage = CSVStorage(sample_csv_file)
            features = ["tx_amount_log", "tx_amount_log_deviates"]

            # Test filtering for customer 1001
            result = storage.prepare_for_verification(
                0, features
            )  # First transaction (customer 1001)
            called_df = mock_generate_features.call_args[0][0]
            assert all(called_df["customer_id"] == 1001)
            assert len(called_df) == 2  # Two transactions for customer 1001
            assert result is not None
            assert list(result.columns) == features

    def test_prepare_for_verification_with_empty_features_list(
        self, sample_csv_file: str, mock_generate_features: Mock
    ) -> None:
        """Test prepare_for_verification with empty features list."""
        with (
            patch("pandas.read_csv") as mock_read_csv,
            patch("api.db.csv.generate_features", mock_generate_features),
        ):
            mock_df = pd.DataFrame(
                {
                    "TX_DATETIME": pd.to_datetime(["2023-01-01 10:00:00"]),
                    "CUSTOMER_ID": [1001],
                    "SECTOR_ID": [1],
                    "TX_AMOUNT": [150.50],
                    "TX_FRAUD": [0],
                }
            )
            mock_read_csv.return_value = mock_df

            # Mock generate_features to return data with row_id
            mock_generate_features.return_value = pd.DataFrame(
                {
                    "customer_id": [1001],
                    "tx_amount_log": [5.0],
                    "tx_amount_log_deviates": [0],
                    "row_id": [0],
                }
            )

            storage = CSVStorage(sample_csv_file)
            features: List[str] = []  # Empty features list

            result = storage.prepare_for_verification(0, features)

            assert result is not None
            assert list(result.columns) == features
            assert len(result) == 1

    def test_prepare_for_verification_with_nonexistent_features(
        self, sample_csv_file: str, mock_generate_features: Mock
    ) -> None:
        """Test prepare_for_verification when requested features don't exist in generated data."""
        with (
            patch("pandas.read_csv") as mock_read_csv,
            patch("api.db.csv.generate_features", mock_generate_features),
        ):
            mock_df = pd.DataFrame(
                {
                    "TX_DATETIME": pd.to_datetime(["2023-01-01 10:00:00"]),
                    "CUSTOMER_ID": [1001],
                    "SECTOR_ID": [1],
                    "TX_AMOUNT": [150.50],
                    "TX_FRAUD": [0],
                }
            )
            mock_read_csv.return_value = mock_df

            # Mock generate_features to return data with row_id but without requested features
            mock_generate_features.return_value = pd.DataFrame(
                {
                    "customer_id": [1001],
                    "tx_amount_log": [5.0],
                    "tx_amount_log_deviates": [0],
                    "row_id": [0],
                }
            )

            storage = CSVStorage(sample_csv_file)
            features = ["nonexistent_feature"]  # Feature that doesn't exist

            result = storage.prepare_for_verification(0, features)
            assert result is None

    def test_prepare_for_verification_row_id_filtering(
        self, sample_csv_file: str, mock_generate_features: Mock
    ) -> None:
        """Test that prepare_for_verification correctly filters by row_id."""
        with (
            patch("pandas.read_csv") as mock_read_csv,
            patch("api.db.csv.generate_features", mock_generate_features),
        ):
            mock_df = pd.DataFrame(
                {
                    "TX_DATETIME": pd.to_datetime(
                        ["2023-01-01 10:00:00", "2023-01-01 11:00:00"]
                    ),
                    "CUSTOMER_ID": [1001, 1001],
                    "SECTOR_ID": [1, 1],
                    "TX_AMOUNT": [150.50, 200.00],
                    "TX_FRAUD": [0, 0],
                }
            )
            mock_read_csv.return_value = mock_df

            # Mock generate_features to return data with row_id
            mock_generate_features.return_value = pd.DataFrame(
                {
                    "customer_id": [1001, 1001],
                    "tx_amount_log": [5.0, 5.3],
                    "tx_amount_log_deviates": [0, 1],
                    "row_id": [0, 1],
                }
            )

            storage = CSVStorage(sample_csv_file)
            features = ["tx_amount_log", "tx_amount_log_deviates"]

            # Test with first transaction (row_id=0)
            result = storage.prepare_for_verification(0, features)
            assert result is not None
            assert len(result) == 1
            assert result.iloc[0]["tx_amount_log"] == 5.0

            # Test with second transaction (row_id=1)
            result = storage.prepare_for_verification(1, features)
            assert result is not None
            assert len(result) == 1
            assert result.iloc[0]["tx_amount_log"] == 5.3

    def test_prepare_for_verification_missing_row_id_returns_none(
        self, sample_csv_file: str, mock_generate_features: Mock
    ) -> None:
        """Test that prepare_for_verification returns None when row_id is not found."""
        with (
            patch("pandas.read_csv") as mock_read_csv,
            patch("api.db.csv.generate_features", mock_generate_features),
        ):
            mock_df = pd.DataFrame(
                {
                    "TX_DATETIME": pd.to_datetime(["2023-01-01 10:00:00"]),
                    "CUSTOMER_ID": [1001],
                    "SECTOR_ID": [1],
                    "TX_AMOUNT": [150.50],
                    "TX_FRAUD": [0],
                }
            )
            mock_read_csv.return_value = mock_df

            # Mock generate_features to return data with different row_id
            mock_generate_features.return_value = pd.DataFrame(
                {
                    "customer_id": [1001],
                    "tx_amount_log": [5.0],
                    "tx_amount_log_deviates": [0],
                    "row_id": [999],  # Different row_id than requested
                }
            )

            storage = CSVStorage(sample_csv_file)
            features = ["tx_amount_log"]

            # Should return None when row_id is not found
            result = storage.prepare_for_verification(2, features)
            assert result is None
