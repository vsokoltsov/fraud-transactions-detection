from datetime import datetime
from fastapi import FastAPI, Depends
from pydantic import BaseModel

api = FastAPI(
    title="Fraud Detection API",
    description="Application for detection fraud transactions",
    version="0.1.0",
)

class PredictRequest(BaseModel):
    pass

class PredictResponse(BaseModel):
    proba: float
    is_fraud: int

@api.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
    """
    Root endpoint that returns a simple JSON response
    """
    return PredictResponse(
        proba=float(0.8),
        is_fraud=True
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:api",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )
