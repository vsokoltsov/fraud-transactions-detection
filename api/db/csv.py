import pandas as pd
import numpy as np
from typing import List, Optional
from api.aggregates.transaction import Transaction

class CSVStorage:
    def __init__(self, file_path: str):
        self.data = pd.read_csv(
            file_path,
            parse_dates=['TX_DATETIME'],
            dtype = {'CUSTOMER_ID': np.int64,'SECTOR_ID': np.int64,'TX_FRAUD': np.int8}
        )
        self.data.columns = self.data.columns.str.lower().str.replace(" ", "_")

    def get_transaction_id(self, trx_id: int) -> Optional[Transaction]:
        try:
            row = self.data.iloc[trx_id]
            return Transaction(
                id=trx_id,
                customer_id=row.customer_id,
                sector_id=row.sector_id,
                tx_datetime=row.tx_datetime,
                tx_amount=row.tx_amount,
                tx_fraud=row.tx_fraud
            )
        except IndexError:
            return None

    def get_transactions_for_customer(self, customer_id: int) -> List[Transaction]:
        pass