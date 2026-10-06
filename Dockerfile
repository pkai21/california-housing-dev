FROM python:3.12-slim

WORKDIR /app

# ============================================================
# Install dependencies
# ============================================================

# Runtime chỉ cần scikit-learn + fastapi + uvicorn (joblib đi kèm scikit-learn).
# Không cài pytest/httpx vì chỉ dùng khi test.
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir scikit-learn fastapi uvicorn

# ============================================================
# Copy application code, UI và model đã train
# ============================================================

COPY src/ ./src/
COPY static/ ./static/
COPY models/ ./models/

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]