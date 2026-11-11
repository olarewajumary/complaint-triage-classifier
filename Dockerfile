FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1

WORKDIR /app

RUN pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY src/api.py src/predict.py src/
COPY models/final_model/ models/final_model/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
