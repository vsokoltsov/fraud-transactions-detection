from typing import Optional, List
from api.aggregates.transaction import Transaction
from api.config import Settings
from .csv import CSVStorage
from .protocols import StorageBackend

STORAGE_CSV = "csv"

class Storage:
    def __init__(self, settings: Settings):
        self.store: StorageBackend = self._storage(settings=settings)

    def get_transaction_id(self, trx_id: int) -> Optional[Transaction]:
        return self.store.get_transaction_id(trx_id=trx_id)

    def get_transactions_for_customer(self, customer_id: int) -> List[Transaction]:
        return self.store.get_transactions_for_customer(customer_id=customer_id)

    def _storage(self, settings: Settings) -> StorageBackend:
        if settings.storage == STORAGE_CSV:
            self.store = CSVStorage(settings.storage_path)
        else:
            raise ValueError(f"Unknown storage type: {settings.storage}")
