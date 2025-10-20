import numpy as np
from dataclasses import dataclass

@dataclass
class Transaction:
    id: np.int64
    tx_amount: np.float64
    tx_datetime: np.datetime64
    customer_id: np.int64
    sector_id: np.int64
    tx_fraud: np.int8