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
from docling.utils.model_downloader import download_models

models_dir = "/opt/docling-models"
os.makedirs(models_dir, exist_ok=True)

# Совместимость с разными версиями Docling: именование аргумента могло меняться.
downloaded = False
for kwargs in (
    {"target_dir": models_dir},
    {"target_path": models_dir},
    {},  # попробуем позиционный вызов
):
    try:
        if kwargs:
            download_models(**kwargs)
        else:
            download_models(models_dir)
        downloaded = True
        break
    except TypeError:
        pass

if not downloaded:
    # Последняя попытка: простой позиционный вызов (на случай других сигнатур)
    download_models(models_dir)

print("Prefetched Docling models to:", models_dir)
PY

COPY app ./app
EXPOSE 8088
CMD ["bash", "-lc", "uvicorn app.main:app --host ${UVICORN_HOST:-0.0.0.0} --port ${UVICORN_PORT:-8088} --workers 1"]
