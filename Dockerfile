FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng tesseract-ocr-osd \
    poppler-utils qpdf ghostscript \
    libreoffice \
    libmagic1 curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /opt/docling-models && python - <<'PY'
import os
from pathlib import Path
from docling.utils.model_downloader import download_models

models_dir = Path("/opt/docling-models")
models_dir.mkdir(parents=True, exist_ok=True)

# Совместимость с разными версиями Docling: пробуем разные варианты параметров
downloaded = False
for attempt in [
    lambda: download_models(),  # без параметров (default path)
    lambda: download_models(models_dir),  # Path object
    lambda: download_models(str(models_dir)),  # string path
    lambda: download_models(target_dir=models_dir),  # именованный Path
    lambda: download_models(target_dir=str(models_dir)),  # именованный string
]:
    try:
        attempt()
        downloaded = True
        break
    except (TypeError, AttributeError) as e:
        print(f"Попытка не удалась: {e}")
        continue

if not downloaded:
    print("Все попытки загрузки моделей провалились")
    exit(1)
else:
    print("Prefetched Docling models to:", models_dir)
PY

COPY app ./app
EXPOSE 8088
CMD ["bash", "-lc", "uvicorn app.main:app --host ${UVICORN_HOST:-0.0.0.0} --port ${UVICORN_PORT:-8088} --workers 1"]
