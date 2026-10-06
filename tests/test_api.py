from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import MODEL_PATH, app

# Bỏ qua nếu chưa train. Trong CI phải chạy train trước pytest.
pytestmark = pytest.mark.skipif(not Path(MODEL_PATH).exists(), reason="Chưa có model")

VALID = {
    "MedInc": 3.5, "HouseAge": 29, "AveRooms": 5.4, "AveBedrms": 1.1,
    "Population": 1166, "AveOccup": 2.8, "Latitude": 34.2, "Longitude": -118.5,
}


def test_health_and_predict() -> None:
    # `with` để lifespan chạy và model được nạp
    with TestClient(app) as client:
        assert client.get("/health").json()["model_loaded"] is True
        res = client.post("/predict", json=VALID)
        assert res.status_code == 200
        assert 0 < res.json()["price_usd"] <= 500_000


def test_rejects_out_of_range() -> None:
    with TestClient(app) as client:
        assert client.post("/predict", json={**VALID, "Latitude": 80}).status_code == 422


def test_serves_ui() -> None:
    with TestClient(app) as client:
        assert "text/html" in client.get("/").headers["content-type"]