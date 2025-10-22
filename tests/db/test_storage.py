import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from api.db.storage import Storage, STORAGE_CSV
from api.config import Settings
from api.db.csv import CSVStorage
from api.db.protocols import StorageBackend


class TestStorage:
    """Test suite for Storage class."""

    @pytest.fixture
    def mock_settings(self) -> Mock:
        """Create a mock Settings object for testing."""
        settings = Mock(spec=Settings)
        settings.storage = STORAGE_CSV
        settings.storage_path = "/fake/path/to/data.csv"
        return settings

    @pytest.fixture
    def mock_csv_storage(self) -> Mock:
        """Create a mock CSVStorage object for testing."""
        mock_storage = Mock(spec=CSVStorage)
        mock_storage.prepare_for_verification.return_value = pd.DataFrame({
            'customer_id': [1001, 1001],
            'tx_amount_log': [5.0, 4.3],
            'tx_amount_log_deviates': [0, 1]
        })
        return mock_storage

    @pytest.fixture
    def sample_dataframe(self) -> pd.DataFrame:
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({
            'customer_id': [1001, 1001],
            'tx_amount_log': [5.0, 4.3],
            'tx_amount_log_deviates': [0, 1]
        })

    def test_init_with_csv_storage(self, mock_settings: Mock) -> None:
        """Test initialization with CSV storage type."""
        with patch('api.db.storage.CSVStorage') as mock_csv_class:
            mock_csv_instance = Mock(spec=CSVStorage)
            mock_csv_class.return_value = mock_csv_instance
            
            storage = Storage(mock_settings)
            
            assert storage.store == mock_csv_instance
            mock_csv_class.assert_called_once_with(mock_settings.storage_path)

    def test_init_with_unknown_storage_type(self, mock_settings: Mock) -> None:
        """Test initialization with unknown storage type raises ValueError."""
        mock_settings.storage = "unknown_storage"
        
        with pytest.raises(ValueError, match="Unknown storage type: unknown_storage"):
            Storage(mock_settings)

    def test_prepare_for_verification_delegates_to_store(
        self, 
        mock_settings: Mock, 
        mock_csv_storage: Mock
    ) -> None:
        """Test that prepare_for_verification delegates to the underlying store."""
        with patch('api.db.storage.CSVStorage', return_value=mock_csv_storage):
            storage = Storage(mock_settings)
            
            result = storage.prepare_for_verification(trx_id=5)
            
            mock_csv_storage.prepare_for_verification.assert_called_once_with(trx_id=5)
            pd.testing.assert_frame_equal(result, mock_csv_storage.prepare_for_verification.return_value)

    def test_prepare_for_verification_with_different_trx_ids(
        self, 
        mock_settings: Mock, 
        mock_csv_storage: Mock
    ) -> None:
        """Test prepare_for_verification with different transaction IDs."""
        with patch('api.db.storage.CSVStorage', return_value=mock_csv_storage):
            storage = Storage(mock_settings)
            
            # Test with different transaction IDs
            storage.prepare_for_verification(0)
            storage.prepare_for_verification(10)
            storage.prepare_for_verification(999)
            
            assert mock_csv_storage.prepare_for_verification.call_count == 3
            mock_csv_storage.prepare_for_verification.assert_any_call(trx_id=0)
            mock_csv_storage.prepare_for_verification.assert_any_call(trx_id=10)
            mock_csv_storage.prepare_for_verification.assert_any_call(trx_id=999)

    def test_prepare_for_verification_return_type(
        self, 
        mock_settings: Mock, 
        mock_csv_storage: Mock
    ) -> None:
        """Test that prepare_for_verification returns pandas DataFrame."""
        with patch('api.db.storage.CSVStorage', return_value=mock_csv_storage):
            storage = Storage(mock_settings)
            
            result = storage.prepare_for_verification(trx_id=0)
            
            assert isinstance(result, pd.DataFrame)

    def test_storage_backend_protocol_compliance(self, mock_settings: Mock) -> None:
        """Test that Storage implements StorageBackend protocol correctly."""
        with patch('api.db.storage.CSVStorage') as mock_csv_class:
            mock_csv_instance = Mock(spec=CSVStorage)
            mock_csv_class.return_value = mock_csv_instance
            
            storage = Storage(mock_settings)
            
            # Verify it has the required method
            assert hasattr(storage, 'prepare_for_verification')
            assert callable(getattr(storage, 'prepare_for_verification'))
            
            # Verify method signature matches protocol
            import inspect
            sig = inspect.signature(storage.prepare_for_verification)
            assert list(sig.parameters.keys()) == ['trx_id']
            assert sig.return_annotation == pd.DataFrame

    def test_storage_constant_definition(self) -> None:
        """Test that STORAGE_CSV constant is properly defined."""
        assert STORAGE_CSV == "csv"
        assert isinstance(STORAGE_CSV, str)

    def test_storage_initialization_with_different_settings(self) -> None:
        """Test storage initialization with different settings configurations."""
        # Test with different storage paths
        settings1 = Mock(spec=Settings)
        settings1.storage = STORAGE_CSV
        settings1.storage_path = "/path1/data.csv"
        
        settings2 = Mock(spec=Settings)
        settings2.storage = STORAGE_CSV
        settings2.storage_path = "/path2/data.csv"
        
        with patch('api.db.storage.CSVStorage') as mock_csv_class:
            mock_csv_instance = Mock(spec=CSVStorage)
            mock_csv_class.return_value = mock_csv_instance
            
            storage1 = Storage(settings1)
            storage2 = Storage(settings2)
            
            # Verify CSVStorage was called with correct paths
            assert mock_csv_class.call_count == 2
            mock_csv_class.assert_any_call("/path1/data.csv")
            mock_csv_class.assert_any_call("/path2/data.csv")

    def test_storage_error_handling_from_underlying_store(
        self, 
        mock_settings: Mock, 
        mock_csv_storage: Mock
    ) -> None:
        """Test error handling when underlying store raises exceptions."""
        with patch('api.db.storage.CSVStorage', return_value=mock_csv_storage):
            storage = Storage(mock_settings)
            
            # Test with various exceptions from underlying store
            mock_csv_storage.prepare_for_verification.side_effect = IndexError("Invalid transaction ID")
            
            with pytest.raises(IndexError, match="Invalid transaction ID"):
                storage.prepare_for_verification(trx_id=999)
            
            mock_csv_storage.prepare_for_verification.side_effect = Exception("Storage error")
            
            with pytest.raises(Exception, match="Storage error"):
                storage.prepare_for_verification(trx_id=0)

    def test_storage_initialization_error_handling(self) -> None:
        """Test error handling during storage initialization."""
        settings = Mock(spec=Settings)
        settings.storage = STORAGE_CSV
        settings.storage_path = "/fake/path"
        
        with patch('api.db.storage.CSVStorage', side_effect=FileNotFoundError("File not found")):
            with pytest.raises(FileNotFoundError, match="File not found"):
                Storage(settings)

    def test_storage_type_validation(self) -> None:
        """Test validation of storage type."""
        settings = Mock(spec=Settings)
        settings.storage = "invalid_storage"
        settings.storage_path = "/fake/path"
        
        with pytest.raises(ValueError, match="Unknown storage type: invalid_storage"):
            Storage(settings)

    def test_storage_type_case_sensitivity(self) -> None:
        """Test that storage type is case sensitive."""
        settings = Mock(spec=Settings)
        settings.storage = "CSV"  # Uppercase
        settings.storage_path = "/fake/path"
        
        with pytest.raises(ValueError, match="Unknown storage type: CSV"):
            Storage(settings)

    def test_storage_type_empty_string(self) -> None:
        """Test handling of empty storage type."""
        settings = Mock(spec=Settings)
        settings.storage = ""
        settings.storage_path = "/fake/path"
        
        with pytest.raises(ValueError, match="Unknown storage type: "):
            Storage(settings)

    def test_storage_type_none(self) -> None:
        """Test handling of None storage type."""
        settings = Mock(spec=Settings)
        settings.storage = None
        settings.storage_path = "/fake/path"
        
        with pytest.raises(ValueError, match="Unknown storage type: None"):
            Storage(settings)

    def test_storage_attributes(self, mock_settings: Mock) -> None:
        """Test that storage attributes are set correctly."""
        with patch('api.db.storage.CSVStorage') as mock_csv_class:
            mock_csv_instance = Mock(spec=CSVStorage)
            mock_csv_class.return_value = mock_csv_instance
            
            storage = Storage(mock_settings)
            
            # Verify store attribute is set
            assert hasattr(storage, 'store')
            assert storage.store == mock_csv_instance
            assert isinstance(storage.store, StorageBackend)

    def test_storage_methods_exist(self, mock_settings: Mock) -> None:
        """Test that all required methods exist."""
        with patch('api.db.storage.CSVStorage') as mock_csv_class:
            mock_csv_instance = Mock(spec=CSVStorage)
            mock_csv_class.return_value = mock_csv_instance
            
            storage = Storage(mock_settings)
            
            # Verify required methods exist
            assert hasattr(storage, 'prepare_for_verification')
            assert hasattr(storage, '_storage')
            
            # Verify methods are callable
            assert callable(storage.prepare_for_verification)
            assert callable(storage._storage)

    def test_storage_private_method_visibility(self, mock_settings: Mock) -> None:
        """Test that private methods are properly encapsulated."""
        with patch('api.db.storage.CSVStorage') as mock_csv_class:
            mock_csv_instance = Mock(spec=CSVStorage)
            mock_csv_class.return_value = mock_csv_instance
            
            storage = Storage(mock_settings)
            
            # Verify _storage method exists but is private
            assert hasattr(storage, '_storage')
            assert storage._storage.__name__.startswith('_')

    def test_storage_with_multiple_instances(self) -> None:
        """Test creating multiple Storage instances."""
        settings1 = Mock(spec=Settings)
        settings1.storage = STORAGE_CSV
        settings1.storage_path = "/path1/data.csv"
        
        settings2 = Mock(spec=Settings)
        settings2.storage = STORAGE_CSV
        settings2.storage_path = "/path2/data.csv"
        
        with patch('api.db.storage.CSVStorage') as mock_csv_class:
            mock_csv_instance1 = Mock(spec=CSVStorage)
            mock_csv_instance2 = Mock(spec=CSVStorage)
            mock_csv_class.side_effect = [mock_csv_instance1, mock_csv_instance2]
            
            storage1 = Storage(settings1)
            storage2 = Storage(settings2)
            
            # Verify they are different instances
            assert storage1 is not storage2
            assert storage1.store is not storage2.store
            assert storage1.store == mock_csv_instance1
            assert storage2.store == mock_csv_instance2

    def test_storage_prepare_for_verification_with_none_trx_id(
        self, 
        mock_settings: Mock, 
        mock_csv_storage: Mock
    ) -> None:
        """Test prepare_for_verification with None transaction ID."""
        with patch('api.db.storage.CSVStorage', return_value=mock_csv_storage):
            storage = Storage(mock_settings)
            
            # This should pass through to the underlying store
            storage.prepare_for_verification(trx_id=-1)
            
            mock_csv_storage.prepare_for_verification.assert_called_once_with(trx_id=-1)

    def test_storage_prepare_for_verification_with_negative_trx_id(
        self, 
        mock_settings: Mock, 
        mock_csv_storage: Mock
    ) -> None:
        """Test prepare_for_verification with negative transaction ID."""
        with patch('api.db.storage.CSVStorage', return_value=mock_csv_storage):
            storage = Storage(mock_settings)
            
            # This should pass through to the underlying store
            storage.prepare_for_verification(trx_id=-1)
            
            mock_csv_storage.prepare_for_verification.assert_called_once_with(trx_id=-1)