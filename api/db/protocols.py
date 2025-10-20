from typing import Protocol, Optional, List
from api.aggregates.transaction import Transaction

class StorageBackend(Protocol):
    """Protocol defining the interface for storage backends"""
    
    def get_transaction_id(self, trx_id: int) -> Optional[Transaction]:
        """Get a transaction by its ID"""
        ...
    
    def get_transactions_for_customer(self, customer_id: int) -> List[Transaction]:
        """Get all transactions for a customer"""
        ...