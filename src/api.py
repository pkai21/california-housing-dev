"""API dự đoán giá nhà + phục vụ giao diện web. Chạy: uvicorn src.api:app"""
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import joblib
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.schema import PRICE_CAP, PRICE_UNIT_USD, HouseFeatures, PredictionResponse

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.joblib"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"
STATIC_DIR = BASE_DIR / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Nạp model 1 lần lúc khởi động, không nạp lại mỗi request
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Chưa có model tại {MODEL_PATH}. Chạy: python -m src.train")
    app.state.model = joblib.load(MODEL_PATH)
    app.state.metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    yield


app = FastAPI(title="California Housing API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "healthy", "model_loaded": hasattr(app.state, "model")}


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    return app.state.metrics


# Dùng `def` thường (không async): FastAPI chạy nó trong thread pool,
# nên tính toán của model không chặn các request khác.
@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseFeatures) -> PredictionResponse:
    raw = float(app.state.model.predict([features.to_row()])[0])
    # Dataset cắt trần ở 5.0 nên model không đáng tin ngoài khoảng [0, 5.0]
    value = min(max(raw, 0.0), PRICE_CAP)
    return PredictionResponse(
        price_usd=round(value * PRICE_UNIT_USD),
        raw_value=round(raw, 4),
        capped=value >= PRICE_CAP,
    )


# PHẢI đặt cuối cùng: mount "/" nuốt mọi đường dẫn chưa khớp,
# đặt trước thì nó che mất /health, /predict.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")