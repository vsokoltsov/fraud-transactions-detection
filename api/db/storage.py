from typing import List
import pandas as pd
from api.config import Settings
from .csv import CSVStorage
from .protocols import StorageBackend

STORAGE_CSV = "csv"


class Storage(StorageBackend):
    def __init__(self, settings: Settings):
        self.store: StorageBackend = self._storage(settings=settings)

    def prepare_for_verification(
        self, trx_id: int, features: List[str]
    ) -> pd.DataFrame:
        return self.store.prepare_for_verification(trx_id=trx_id, features=features)

    def _storage(self, settings: Settings) -> StorageBackend:
        if settings.storage == STORAGE_CSV:
            return CSVStorage(settings.storage_path)
        else:
            raise ValueError(f"Unknown storage type: {settings.storage}")
