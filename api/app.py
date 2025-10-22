from typing import cast, AsyncGenerator, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, PositiveFloat
from .config import settings
from .db.storage import Storage
from api.predictor.fraud import FraudPredictor


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[Any, None]:
    app.state.db = Storage(settings)
    app.state.model = FraudPredictor(settings=settings)
    yield
    app.state.db = None
    app.state.model = None


api = FastAPI(
    title="Fraud Detection API",
    description="Application for detection fraud transactions",
    version="0.1.0",
    lifespan=lifespan,
)


def get_db() -> Storage:
    return cast(Storage, api.state.db)


def get_model() -> FraudPredictor:
    return cast(FraudPredictor, api.state.model)


class PredictResponse(BaseModel):
    proba: PositiveFloat
    is_fraud: int


@api.post("/model/v0/prediction/{trx_id}", response_model=PredictResponse)
async def predict(
    trx_id: int,
    db: Storage = Depends(get_db),
    model: FraudPredictor = Depends(get_model),
) -> PredictResponse:
    df = db.prepare_for_verification(trx_id=trx_id, features=model.features)
    if df is None:
        raise HTTPException(status_code=404, detail="Transaction not found")
    proba = model.predict_proba(df)
    return PredictResponse(proba=proba[0][1], is_fraud=proba[0][1] >= model.threshold)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.app:api", host="0.0.0.0", port=8000, reload=True, log_level="info")
