# Docling FastAPI Microservice

A Dockerized FastAPI microservice that converts office documents, PDFs, and images into Markdown and/or JSON using Docling. It supports:

- Autodetection of input type
- Legacy conversion of `.doc/.xls/.ppt` via LibreOffice to OOXML
- OCR strategies for PDF/images using Tesseract with configurable PSM sweep
- Table structure extraction (ACCURATE mode)

## Features

- Single universal endpoint: `POST /convert`
- Health check: `GET /health`
- Configurable via environment variables
- Pre-fetches Docling models at build time for faster cold starts

## Project Structure

```
.
├─ docker-compose.yml
├─ Dockerfile
├─ requirements.txt
└─ app/
   └─ main.py
```

## Requirements

- Docker and Docker Compose (recommended), or Python 3.11+ with system packages:
  - tesseract-ocr (+ language packs), poppler-utils, qpdf, ghostscript, libreoffice, libmagic1

## Quick Start (Docker Compose)

```bash
# Build and run
docker compose up --build -d

# Check health
curl http://localhost:8088/health

# Convert a file (example)
curl -X POST \
  -F "file=@/path/to/input.pdf" \
  -F "out=both" \
  -F "langs=rus,eng" \
  -F "psm_list=6,4,11" \
  http://localhost:8088/convert | jq '.meta, .best_strategy, (.content_markdown | tostring) | .[0:2000]'
```

## Environment Variables

- `UVICORN_HOST` (default: `0.0.0.0`)
- `UVICORN_PORT` (default: `8088`)
- `OMP_NUM_THREADS` (default: `4`)
- `DOCLING_ARTIFACTS_PATH` (default: `/opt/docling-models`)
- `DOCLING_TABLE_MODE` (default: `ACCURATE`)
- `DOCLING_FORCE_OCR` (default: `true`)
- `DOCLING_LANGS` (default: `rus,eng`)
- `MAX_FILE_SIZE_MB` (default: `80`)

Set these in `docker-compose.yml` as shown, or pass them to the container runtime.

## API

### GET /health

- Returns `{ "ok": true }`.

### POST /convert

Form fields:
- `file` (required): file upload.
- `out` (optional): `markdown` | `json` | `both` (default: `both`).
- `langs` (optional): e.g., `rus,eng` (defaults to env `DOCLING_LANGS`).
- `psm_list` (optional): Tesseract PSM candidates, e.g. `6,4,11`.
- `max_pages` (optional): integer page limit for conversion.

Behavior:
- `.doc/.xls/.ppt` → LibreOffice to OOXML → Docling (no OCR)
- PDF/images → try no-OCR; then force-OCR with provided PSMs; choose best by heuristic
- OOXML/HTML/MD/CSV → Docling directly (no OCR)

Response (example):
```json
{
  "best_strategy": "psm:6",
  "trials": [
    { "strategy": "no_ocr", "length": 1200, "cyr_ratio": 0.32, "score": 936.0 },
    { "strategy": "psm:6", "length": 1950, "cyr_ratio": 0.48, "score": 1497.0 }
  ],
  "content_markdown": "# ...",
  "content_json": { "pages": [ /* ... */ ] },
  "meta": {
    "filename": "input.pdf",
    "size_bytes": 123456,
    "converted": false,
    "out": "both",
    "langs": "rus,eng",
    "max_pages": null
  }
}
```

## Build Without Compose (Docker)

```bash
docker build -t docling-fastapi:latest .
docker run --rm -p 8088:8088 \
  -e UVICORN_HOST=0.0.0.0 -e UVICORN_PORT=8088 \
  -e DOCLING_ARTIFACTS_PATH=/opt/docling-models \
  -e DOCLING_TABLE_MODE=ACCURATE \
  -e DOCLING_FORCE_OCR=true \
  -e DOCLING_LANGS=rus,eng \
  -e MAX_FILE_SIZE_MB=80 \
  -v docling-models:/opt/docling-models \
  -v $(pwd)/data:/data \
  --name docling docling-fastapi:latest
```

## Local Development (Python)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8088 --reload
```

Note: You must install system packages mentioned in Requirements when running outside Docker.

## Notes on OCR and PSM

- PSM values (e.g., 6, 4, 11) control page segmentation in Tesseract.
- The service computes a simple heuristic based on output length and Cyrillic ratio
  to choose the best strategy.

## Troubleshooting

- Missing docling models: ensure `/opt/docling-models` volume and that models were pre-fetched during image build.
- Poor OCR quality: try different `psm_list` or add appropriate language packs.
- Large PDFs: use `max_pages` to limit processing or increase container resources.

## License

This project contains third-party dependencies under their respective licenses.
