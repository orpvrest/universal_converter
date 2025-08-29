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

# Prefetch Docling models at build time to speed up first run
RUN python - <<'PY'
from pathlib import Path
from docling.utils.model_downloader import download_models

out = Path('/opt/docling-models')
out.mkdir(parents=True, exist_ok=True)
download_models(
    output_dir=out,
    progress=False,
    with_layout=True,
    with_tableformer=True,
    with_code_formula=True,
    with_picture_classifier=True,
    with_easyocr=True,
)
print('Docling models prefetched to', out)
PY

COPY app ./app
EXPOSE 8088
CMD ["bash", "-lc", "uvicorn app.main:app --host ${UVICORN_HOST:-0.0.0.0} --port ${UVICORN_PORT:-8088} --workers 1"]
