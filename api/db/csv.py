from typing import List, Optional
import pandas as pd
import numpy as np
from .predictions import generate_features
from api.db.protocols import StorageBackend


class CSVStorage(StorageBackend):
    def __init__(self, file_path: str, features_version: str = "v2"):
        self.file_path = file_path
        self.data: pd.DataFrame = pd.read_csv(
            self.file_path,
            parse_dates=["TX_DATETIME"],
            dtype={"CUSTOMER_ID": np.int64, "SECTOR_ID": np.int64, "TX_FRAUD": np.int8},
        )
        self.data.columns = self.data.columns.str.lower().str.replace(" ", "_")

    def prepare_for_verification(
        self, trx_id: int, features: List[str]
    ) -> Optional[pd.DataFrame]:
        try:
            trx = self.data.iloc[trx_id]
        except IndexError:
            return None
        user_trxs = self.data[self.data["customer_id"] == trx.customer_id]
        user_trxs["row_id"] = user_trxs.index
        df = generate_features(user_trxs)
        try:
            return df[df["row_id"] == trx_id][features]
        except KeyError:
            return None
        except IndexError:
            return None
