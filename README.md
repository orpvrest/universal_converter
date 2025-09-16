# Universal Document Converter

Микросервис на FastAPI, который конвертирует офисные документы, PDF и изображения в Markdown/JSON с помощью Docling. Сервис включает интеллектуальный выбор OCR-стратегии (Tesseract CLI + EasyOCR), очистку повторяющихся шапок и гибридный чанкинг текста.

## 💡 Основные возможности

- **Автоопределение стратегии**: выбор пайплайна Docling в зависимости от типа входного файла (legacy форматы через LibreOffice, PDF/изображения через OCR-пайплайн, "родные" форматы напрямую).
- **Мульти-OCR**: принудительный OCR выполняется двумя движками (Tesseract CLI и EasyOCR). Лучший результат определяется эвристикой `score = длина × (0.6 + 0.4 × доля кириллицы)`.
- **Нормализация языков**: алиасы `ru`, `en`, `uk`, `kz` и др. преобразуются в коды Tesseract (`rus`, `eng`, …) и дедуплируются, что повышает качество распознавания русского текста.
- **OCRmyPDF препроцессинг**: при наличии утилиты запускается deskew/denoise/rotate-cleanup перед OCR.
- **Удаление повторяющихся шапок**: эвристики для повторяющихся заголовков и "первой страницы" с конфигурируемыми параметрами.
- **Гибридный чанкинг**: деление Markdown на перекрывающиеся чанки с учётом структуры.

## 🧱 Зависимости

### Системные пакеты

Список установлен в `Dockerfile` и включает:

- `tesseract-ocr`, `tesseract-ocr-rus`, `tesseract-ocr-eng`, `tesseract-ocr-osd`
- `libreoffice`, `poppler-utils`, `qpdf`, `ghostscript`
- `libgl1`, `libglib2.0-0`, `libsm6`, `libxext6`, `libxrender1`, `libgomp1` — необходимые для EasyOCR/OpenCV
- `libmagic1`, `curl`, `ca-certificates`

### Python-пакеты

Устанавливаются через `requirements.txt`:

```text
fastapi>=0.110.0
uvicorn[standard]>=0.24.0
python-multipart>=0.0.7
pydantic>=2.6.0
pydantic-settings>=2.2.0
docling
easyocr>=1.7.1
ocrmypdf>=16.0.0
```

`easyocr` добавляет поддержку кириллических моделей и автоматически подтягивает `torch`, `opencv-python-headless`, `scikit-image` и другие необходимые зависимости.

Для разработки см. `requirements-dev.txt` (pytest, mypy, pre-commit и др.).

## 🚀 Быстрый старт

### Запуск в Docker

```bash
docker build -t universal-converter .
docker run -p 8088:8088 -e DOCLING_LANGS="rus,eng" universal-converter
```

Docker-образ заранее предзагружает модели Docling (включая EasyOCR), поэтому первый запрос к `/convert` не требует скачивания моделей в рантайме.

### Локальный запуск

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # по желанию
uvicorn app.main:app --reload --host 0.0.0.0 --port 8088
```

## ⚙️ Конфигурация

| Переменная | Значение по умолчанию | Описание |
|------------|----------------------|----------|
| `UVICORN_HOST` | `0.0.0.0` | Хост uvicorn |
| `UVICORN_PORT` | `8088` | Порт uvicorn |
| `DOCLING_ARTIFACTS_PATH` | не задано | Путь к артефактам Docling (скачивание моделей при старте) |
| `DOCLING_TABLE_MODE` | `ACCURATE` | Режим извлечения таблиц |
| `DOCLING_FORCE_OCR` | `false` | Базовое значение для принудительного OCR |
| `DOCLING_LANGS` | `rus,eng` | Языки OCR по умолчанию; алиасы нормализуются (например, `ru` → `rus`) |
| `MAX_FILE_SIZE_MB` | `80` | Максимальный размер загружаемого файла |
| `HEADER_FILTER_CONFIG` | `config/header_filter.json` | Конфигурация фильтров шапок |
| `OMP_NUM_THREADS` | `4` | Количество потоков OpenMP для Tesseract |

## 🔍 OCR-пайплайны

1. **`no_ocr`** – Docling без принудительного OCR; подходит для PDF с текстовым слоем.
2. **`force_ocr`** – Docling + Tesseract CLI (`force_full_page_ocr=True`).
3. **`easyocr`** – Docling + EasyOCR (russian+english модели).

Каждый пайплайн проходит постобработку (удаление повторяющихся шапок, чистка первой страницы). Результаты сравниваются по длине текста и доле кириллицы; лучший результат возвращается в ответе (`best_strategy`).

## 🧪 Тестирование

```bash
python -m pytest -q
```

Основные тесты:

- `tests/test_ocr_pdf.py::test_convert_ocr_image_pdf` – synthetic PDF (изображение + русская надпись) с проверкой работы Tesseract и EasyOCR.
- `tests/test_ocr_pdf.py::test_convert_real_russian_pdf` – загрузка и распознавание реального документа МАИ [[ссылка](https://mai.ru/upload/iblock/6f4/nqun8lcnq727pkbgla33zzusnlxpupns/2025_08_25_221_ORG_Ob_utverzhdenii_instruktsii_o_vzyatii_pomeshchenii_.pdf)]. Тест автоматически пропускается при отсутствии сети или зависимостей.

## 📑 API

### `GET /health`
Простая проверка доступности. Возвращает `{ "ok": true }`.

### `POST /convert`
Основной эндпоинт.

- `file`: загружаемый документ.
- `out`: `markdown` | `json` | `both` (по умолчанию `both`).
- `langs`: строка с языками OCR (например, `ru,eng`).
- `max_pages`: ограничение числа страниц.

Ответ содержит лучший Markdown/JSON, список `trials` с метриками для каждой стратегии и метаданные (название файла, размер, выбранные языки и т.д.).

### `POST /chunk`
Чанкинг произвольного текста (с сохранением Markdown структуры и оверлапами).

## 🧭 Структура репозитория

```
universal_converter/
├── Dockerfile
├── requirements*.txt
├── app/
│   └── main.py
├── config/
│   └── header_filter.json (пример)
└── tests/
    ├── test_health.py
    ├── test_langs_normalization.py
    └── test_ocr_pdf.py
```

## 📌 Дополнительно

- Файл `config/header_filter.json` позволяет настраивать эвристики удаления шапок.
- При первом запуске Docling скачивает модели (если путь не задан). Для ускорения используйте переменную `DOCLING_ARTIFACTS_PATH` или заранее прогрейте кэш.
- При необходимости можно подключить S3/MinIO для хранения временных файлов через монтирование внешнего тома.
