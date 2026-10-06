"""Định nghĩa đặc trưng và kiểm tra đầu vào, dùng chung cho train và API."""
from pydantic import BaseModel, Field

# Thứ tự này PHẢI khớp với dataset. Model chỉ nhận mảng số, không nhận tên cột,
# nên sai thứ tự sẽ cho kết quả sai mà không báo lỗi.
FEATURE_NAMES: list[str] = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

# Giá nhà trong dataset tính theo đơn vị 100.000 USD và bị cắt trần ở 5.0
PRICE_UNIT_USD: int = 100_000
PRICE_CAP: float = 5.0


class HouseFeatures(BaseModel):
    """8 thông số của một khu dân cư (block group) ở California."""

    # Khoảng giá trị rộng hơn dữ liệu thật một chút để chặn đầu vào vô nghĩa
    # (kể cả NaN/inf, vì so sánh với NaN luôn sai nên bị loại).
    MedInc: float = Field(ge=0, le=20, description="Thu nhập trung vị (đơn vị 10.000 USD)")
    HouseAge: float = Field(ge=1, le=60, description="Tuổi nhà trung vị (năm)")
    AveRooms: float = Field(ge=0.5, le=200, description="Số phòng trung bình mỗi hộ")
    AveBedrms: float = Field(ge=0.2, le=50, description="Số phòng ngủ trung bình mỗi hộ")
    Population: float = Field(ge=1, le=50_000, description="Dân số của khu")
    AveOccup: float = Field(ge=0.5, le=2000, description="Số người trung bình mỗi hộ")
    Latitude: float = Field(ge=32, le=42, description="Vĩ độ")
    Longitude: float = Field(ge=-125, le=-114, description="Kinh độ")

    def to_row(self) -> list[float]:
        """Trả về 1 hàng số theo đúng thứ tự FEATURE_NAMES."""
        return [getattr(self, name) for name in FEATURE_NAMES]


class PredictionResponse(BaseModel):
    price_usd: int = Field(description="Giá nhà trung vị dự đoán (USD)")
    raw_value: float = Field(description="Giá trị gốc của model (đơn vị 100.000 USD)")
    capped: bool = Field(description="True nếu dự đoán chạm trần 500.000 USD của dataset")