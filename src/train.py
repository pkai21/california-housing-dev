"""Huấn luyện model dự đoán giá nhà California. Chạy: python -m src.train"""
import json
from pathlib import Path

import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from src.schema import FEATURE_NAMES, PRICE_UNIT_USD

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
SEED = 42


def main() -> None:
    # Lần đầu sẽ tải dữ liệu từ mạng, các lần sau đọc từ cache cục bộ
    data = fetch_california_housing()

    # Nếu sklearn đổi thứ tự cột thì dừng ngay, thay vì train ra model sai âm thầm
    assert list(data.feature_names) == FEATURE_NAMES, "Thứ tự đặc trưng không khớp schema.py"

    x_train, x_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=SEED
    )

    model = HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.1,
        early_stopping=True,  # tự dừng khi chưa hết 500 vòng nếu không cải thiện
        random_state=SEED,
    )
    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    metrics = {
        "r2": round(float(r2_score(y_test, pred)), 4),
        # MAE đổi sang USD để người không chuyên đọc được
        "mae_usd": round(float(mean_absolute_error(y_test, pred)) * PRICE_UNIT_USD),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "n_iter": int(model.n_iter_),
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "model.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Đã lưu model. Kết quả trên tập test:", metrics)


if __name__ == "__main__":
    main()