from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.predict import load_model, predict


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    top_guess: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="Complaint Triage Classifier", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse, response_model_exclude_none=True)
def predict_route(request: PredictRequest):
    result = predict(request.text)
    if result["prediction"] == "invalid_input":
        raise HTTPException(status_code=422, detail=result["error"])
    return result