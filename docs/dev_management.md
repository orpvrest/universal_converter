# Управление разработкой: Docling Converter

## Обзор проекта
- **Назначение**: HTTP-сервис на FastAPI для конвертации офисных документов и изображений в структурированный Markdown/JSON с использованием Docling и OCR (Tesseract/EasyOCR).
- **Ключевые сценарии**: загрузка документов через `/convert`, чанкинг текста через `/chunk`, мониторинг готовности через `/health`.
- **Важные зависимости**: Docling, LibreOffice CLI, OCRmyPDF, Tesseract CLI, EasyOCR.

## Архитектура и модули
- `app/main.py` — точка входа FastAPI, регистрация роутеров и инициализация моделей.
- `app/routes/` — обработчики HTTP (`convert`, `chunk`, `health`).
- `app/langs.py`, `app/markdown_filters.py`, `app/chunking.py`, `app/ocr_utils.py` — сервисные утилиты для языков, очистки Markdown, чанкинга и работы с Docling.
- `app/config.py` — загрузка конфигурации и переменных окружения.
- `config/header_filter.json` — настройки фильтров для повторяющихся шапок и заголовков.
- `tests/` — проверка нормализации языков, эндпоинта `/health`, интеграционный OCR-тест.

## Среда разработки
- Минимальный Python: 3.11.
- Пакеты: описаны в `pyproject.toml`, dev-зависимости в `requirements-dev.txt`.
- Контейнерный запуск: `Dockerfile` и `docker-compose.yml` (пример с переменными окружения в `docker-compose.yml.example`).

## Процесс разработки
1. Создать и активировать виртуальное окружение `python -m venv .venv`.
2. Установить зависимости `pip install -r requirements.txt && pip install -r requirements-dev.txt`.
3. Запуск тестов `python -m pytest -q`.
4. Для интеграционного теста OCR требуется локальный Tesseract CLI и доступ к документации Docling.

## Контроль качества
- Покрытие кода хранится в `htmlcov/` (генерируется через `pytest --cov`).
- Линтеры/форматтеры подключены через `pre-commit` (см. `.pre-commit-config.yaml`).
- Для ручных проверок OCR: каталог `data/` содержит примеры PDF.

## Наблюдение и эксплуатация
- Конфигурируемые переменные окружения: `DOCLING_ARTIFACTS_PATH`, `DOCLING_LANGS`, `DOCLING_FORCE_OCR`, `MAX_FILE_SIZE_MB` и др. (подробности в docstring `app/main.py`).
- Логи модели: `ensure_models` печатает состояние артефактов при старте.
- Готовность проверяется через `GET /health`.

## Текущие риски и долги
- Нет автоматизированного сравнения качества OCR (используется эвристический скор).
- Отсутствует очередь задач — сервис рассчитан на синхронную обработку.
- Для первого запуска требуется загрузка Docling-моделей (>2 ГБ).
