# Docling FastAPI микросервис

Docker-микросервис на FastAPI для конвертации офисных документов, PDF и изображений
в Markdown и/или JSON с помощью Docling. Поддерживает:

- Автоопределение типа входного файла
- Конвертацию устаревших форматов `.doc/.xls/.ppt` через LibreOffice в OOXML
- Стратегии OCR для PDF/изображений на Tesseract с перебором PSM
- Извлечение структуры таблиц (режим ACCURATE)

## Возможности

- Один универсальный эндпоинт: `POST /convert`
- Проверка состояния: `GET /health`
- Гибкая настройка через переменные окружения
- Забор моделей Docling на этапе сборки образа для ускорения старта

## Структура проекта

```
.
├─ docker-compose.yml
├─ Dockerfile
├─ requirements.txt
└─ app/
   └─ main.py
```

## Требования

- Docker и Docker Compose (рекомендуется), либо Python 3.11+ и системные пакеты:
  - tesseract-ocr (+ языковые пакеты), poppler-utils, qpdf, ghostscript, libreoffice, libmagic1

## Быстрый старт (Docker Compose)

```bash
# Сборка и запуск
docker compose up --build -d

# Проверка доступности
curl http://localhost:8088/health

# Конвертация файла (пример)
curl -X POST \
  -F "file=@/path/to/input.pdf" \
  -F "out=both" \
  -F "langs=rus,eng" \
  -F "psm_list=6,4,11" \
  http://localhost:8088/convert | jq '.meta, .best_strategy, (.content_markdown | tostring) | .[0:2000]'
```

## Переменные окружения

- `UVICORN_HOST` (по умолчанию: `0.0.0.0`)
- `UVICORN_PORT` (по умолчанию: `8088`)
- `OMP_NUM_THREADS` (по умолчанию: `4`)
- `DOCLING_ARTIFACTS_PATH` (по умолчанию: `/opt/docling-models`)
- `DOCLING_TABLE_MODE` (по умолчанию: `ACCURATE`)
- `DOCLING_FORCE_OCR` (по умолчанию: `true`)
- `DOCLING_LANGS` (по умолчанию: `rus,eng`)
- `MAX_FILE_SIZE_MB` (по умолчанию: `80`)

Можно задать в `docker-compose.yml` либо передать в рантайм контейнера.

## API

### GET /health

- Возвращает `{ "ok": true }`.

### POST /convert

Поля формы:
- `file` (обязательно): загружаемый файл.
- `out` (необязательно): `markdown` | `json` | `both` (по умолчанию: `both`).
- `langs` (необязательно): напр. `rus,eng` (по умолчанию берётся из `DOCLING_LANGS`).
- `psm_list` (необязательно): кандидаты PSM, напр. `6,4,11`.
- `max_pages` (необязательно): ограничение страниц для обработки.

Поведение:
- `.doc/.xls/.ppt` → LibreOffice → OOXML → Docling (без OCR)
- PDF/изображения → сначала без OCR; затем принудительный OCR с PSM; выбор лучшего по эвристике
- OOXML/HTML/MD/CSV → напрямую в Docling (без OCR)

Пример ответа:
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

## Сборка без Compose (Docker)

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

## Локальная разработка (Python)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8088 --reload
```

Примечание: при запуске вне Docker необходимо установить системные пакеты из
раздела «Требования».

## Замечания по OCR и PSM

- PSM (например, 6, 4, 11) управляет сегментацией страниц в Tesseract.
- Сервис использует простую эвристику (длина текста и доля кириллицы) для выбора
  лучшей стратегии.

## Диагностика проблем

- Нет моделей Docling: проверьте том `/opt/docling-models` и предзагрузку моделей
  во время сборки образа.
- Низкое качество OCR: попробуйте другой `psm_list` или установите нужные языковые пакеты.
- Большие PDF: используйте `max_pages` или выделите больше ресурсов контейнеру.

## Лицензия

Проект использует сторонние зависимости, распространяемые по их лицензиям.
