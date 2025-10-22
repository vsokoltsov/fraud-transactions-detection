from typing import Protocol, runtime_checkable, List
import pandas as pd


@runtime_checkable
class StorageBackend(Protocol):
    """Protocol defining the interface for storage backends"""

    def prepare_for_verification(
        self, trx_id: int, features: List[str]
    ) -> pd.DataFrame: ...
