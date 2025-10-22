import pandas as pd
import numpy as np
from .predictions import generate_features

class CSVStorage:
    def __init__(self, file_path: str, features_version:str = "v2"):
        self.file_path = file_path
        self.data: pd.DataFrame = pd.read_csv(
            self.file_path,
            parse_dates=['TX_DATETIME'],
            dtype = {'CUSTOMER_ID': np.int64,'SECTOR_ID': np.int64,'TX_FRAUD': np.int8}
        )
        self.data.columns = self.data.columns.str.lower().str.replace(" ", "_")

    def prepare_for_verification(self, trx_id: int) -> pd.DataFrame:
        trx = self.data.iloc[trx_id]
        user_trxs = self.data[self.data['customer_id'] == trx.customer_id]
        df = generate_features(user_trxs)
        return df
