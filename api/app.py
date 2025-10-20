from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from functools import lru_cache
from .config import settings
from .db.storage import Storage

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db = Storage(settings)
    yield
    app.state.db = None

api = FastAPI(
    title="Fraud Detection API",
    description="Application for detection fraud transactions",
    version="0.1.0",
    lifespan=lifespan
)

@lru_cache
def get_db() -> Storage:
    return api.state.db

class PredictRequest(BaseModel):
    pass

class PredictResponse(BaseModel):
    proba: float
    is_fraud: int

@api.post("/model/v0/prediction/{trx_id}", response_model=PredictResponse)
async def predict(trx_id: int, req: PredictRequest, db = Depends(get_db)) -> PredictResponse:
    """
    Root endpoint that returns a simple JSON response
    """
    trx = db.get_transaction_id(trx_id)
    print(trx)
    return PredictResponse(
        proba=float(0.8),
        is_fraud=True
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.app:api",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )
