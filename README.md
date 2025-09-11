# Universal Document Converter

Продвинутый Docker-микросервис на FastAPI для конвертации офисных документов, PDF и изображений в структурированный Markdown/JSON с использованием Docling. Включает умную стратегию OCR, фильтрацию заголовков и текстовый чанкинг.

## 🚀 Ключевые возможности

- **Универсальная конвертация**: автоопределение формата + адаптивные стратегии
- **Умный OCR**: сравнение стратегий с эвристическим выбором лучшего результата
- **Legacy поддержка**: `.doc/.xls/.ppt` → LibreOffice → OOXML → Docling
- **Предобработка PDF**: автоматическая очистка через OCRmyPDF (deskew/denoise)
- **Фильтрация заголовков**: удаление повторяющихся "шапок" организаций
- **Текстовый чанкинг**: гибридный алгоритм с сохранением структуры Markdown
- **Извлечение таблиц**: режим ACCURATE с сохранением структуры

## 📁 Структура проекта

```
universal_converter/
├── docker-compose.yml          # Развертывание сервиса
├── Dockerfile                  # Образ с предустановленными моделями
├── requirements.txt            # Python зависимости
├── requirements-dev.txt        # Зависимости для разработки
├── pyproject.toml             # Конфигурация линтеров/тестов
├── .pre-commit-config.yaml    # Git хуки (black/flake8/pytest)
├── app/
│   └── main.py               # Основной FastAPI сервис (1250+ строк)
├── config/
│   ├── header_filter.example.json  # Пример конфигурации фильтров
│   └── header_filter.json          # Боевая конфигурация (в .gitignore)
├── tests/
│   ├── test_health.py        # Тест health endpoint
│   └── test_ocr_pdf.py       # Тест OCR с генерацией PDF
└── data/                     # Рабочая папка (монтируется как том)
```

## 🔄 Архитектура и алгоритмы

### Алгоритм конвертации

1. **Валидация и определение типа**
   - Проверка размера файла (MAX_FILE_SIZE_MB)
   - Определение MIME-type и расширения
   - Валидация поддерживаемых форматов

2. **Legacy форматы** (`.doc/.xls/.ppt`)
   ```
   Input → LibreOffice CLI → OOXML → Docling (do_ocr=False)
           ↓ (при ошибке)
   Fallback: LibreOffice → PDF → OCR стратегии
   ```

3. **PDF/Изображения** (двухстратегическая обработка)
   ```
   Input PDF → OCRmyPDF preprocessing (опционально)
             ↓
   Strategy A: no_ocr (Docling сам определяет необходимость OCR)
   Strategy B: force_ocr (принудительный полностраничный Tesseract)
             ↓
   Эвристический выбор: score = length × (0.6 + 0.4 × cyrillic_ratio)
   ```

4. **OOXML/HTML/MD/CSV**
   ```
   Input → Docling (напрямую, без OCR)
         ↓ (при ошибке)
   Fallback: → PDF → OCR стратегии
   ```

5. **Постобработка**
   - Удаление повторяющихся заголовков (конфигурируемые ключевые слова)
   - Фильтрация "шапки" первой страницы
   - Опциональный чанкинг текста

### Система фильтрации заголовков

**Повторяющиеся заголовки:**
- Анализ первых строк абзацев по всему документу
- Детекция по частоте (≥3 повторений) + ключевые слова организаций
- Нормализация через regex + проверка доли верхнего регистра

**Заголовок первой страницы:**
- Анализ верхнего блока (до 20 строк, макс. 6 строк блока)
- Исключение Markdown заголовков (`# ...`)
- Детекция по ключевым словам + доле верхнего регистра

## 🛠 API Documentation

### GET /health
Проверка доступности сервиса.

**Response:**
```json
{ "ok": true }
```

### POST /convert
Универсальная конвертация документов.

**Request:**
- `file` (file, required): Загружаемый документ
- `out` (string, optional): `markdown` | `json` | `both` (default: `both`)
- `langs` (string, optional): Языки OCR, напр. `rus,eng` (default: из `DOCLING_LANGS`)
- `max_pages` (int, optional): Ограничение страниц

**Response:**
```json
{
  "best_strategy": "force_ocr",
  "trials": [
    {
      "strategy": "no_ocr",
      "length": 1200,
      "cyr_ratio": 0.32,
      "score": 936.0,
      "header_removed": 2,
      "first_header_removed": 3
    },
    {
      "strategy": "force_ocr", 
      "length": 1950,
      "cyr_ratio": 0.48,
      "score": 1497.0,
      "header_removed": 2,
      "first_header_removed": 3
    }
  ],
  "content_markdown": "# Заголовок\n\nТекст документа...",
  "content_json": { "pages": [...] },
  "meta": {
    "filename": "input.pdf",
    "size_bytes": 123456,
    "converted": true,
    "out": "both",
    "langs": "rus,eng",
    "max_pages": null
  }
}
```

### POST /chunk
Гибридный чанкинг текста с сохранением структуры Markdown.

**Request:**
```json
{
  "text": "# Заголовок\n\nАбзац текста...",
  "max_chars": 2000,
  "overlap": 200,
  "preserve_markdown": true
}
```

**Response:**
```json
{
  "chunks": [
    {
      "index": 0,
      "start": 0,
      "end": 150,
      "text": "# Заголовок\n\nАбзац текста..."
    }
  ],
  "meta": {
    "total_chunks": 1,
    "strategy": "hybrid"
  }
}
```

## ⚙️ Конфигурация

### Переменные окружения

| Переменная | Значение по умолчанию | Описание |
|------------|----------------------|----------|
| `UVICORN_HOST` | `0.0.0.0` | Хост сервера |
| `UVICORN_PORT` | `8088` | Порт сервера |
| `DOCLING_ARTIFACTS_PATH` | `/opt/docling-models` | Путь к моделям Docling |
| `DOCLING_TABLE_MODE` | `ACCURATE` | Режим извлечения таблиц |
| `DOCLING_LANGS` | `rus,eng` | Языки OCR по умолчанию |
| `MAX_FILE_SIZE_MB` | `80` | Максимальный размер файла |
| `HEADER_FILTER_CONFIG` | `config/header_filter.json` | Путь к конфигурации фильтров |
| `OMP_NUM_THREADS` | `4` | Количество потоков OpenMP |

### Конфигурация фильтров заголовков

Файл `config/header_filter.json`:
```json
{
  "min_repeats": 3,
  "min_length": 4, 
  "uppercase_ratio": 0.6,
  "keywords": ["ООО", "АО", "МИНИСТЕРСТВО", ...],
  "first_page": {
    "enable": true,
    "lines_limit": 20,
    "max_block_lines": 6,
    "uppercase_ratio": 0.6,
    "keywords": ["ООО", "АО", "ФЕДЕРАЛЬНОЕ", ...]
  }
}
```

## 📋 Предложения по улучшению алгоритмов

### 🎯 Приоритетные улучшения

#### 1. **Семантическая метрика качества OCR**
```python
# Заменить простую эвристику на семантическую оценку
def semantic_quality_score(text: str) -> float:
    # Словарный анализ (доля валидных слов)
    # Энтропия символов (детекция "мусорного" OCR)  
    # Синтаксическая корректность (парсинг предложений)
    # Специфичные паттерны (номера, даты, email)
    pass
```

#### 2. **Адаптивный OCR с несколькими движками**
```python
# Каскадная стратегия OCR
strategies = [
    {"engine": "docling_native", "weight": 1.0},
    {"engine": "tesseract", "psm": [6, 4, 11], "weight": 0.8},
    {"engine": "paddleocr", "lang": ["ru", "en"], "weight": 0.9},
    {"engine": "easyocr", "weight": 0.7}
]
# Ensemble voting для финального результата
```

#### 3. **Структурно-осознанная обработка**
```python
# Использование layout analysis от Docling для умного чанкинга
def structure_aware_processing(doc: DoclingDocument):
    # Сегментация по типам: headers, paragraphs, tables, figures
    # Сохранение иерархии заголовков при чанкинге
    # Отдельная обработка таблиц (OCR + структурное восстановление)
    # Интеграция с изображениями (OCR + caption generation)
    pass
```

#### 4. **Кеширование с умной инвалидацией**
```python
# Redis-based кеширование по хешу файла + параметров
cache_key = hash(file_content + lang + ocr_params + model_version)
# Warm-up cache для популярных документов
# Инкрементальное обновление при изменении настроек
```

### 🔧 Технические улучшения

#### 5. **Асинхронная обработка**
```python
# Celery/RQ для длительных операций
@app.post("/convert/async")
async def convert_async():
    job_id = await queue.enqueue(convert_task, file_data)
    return {"job_id": job_id, "status": "queued"}

@app.get("/convert/status/{job_id}")
async def get_status(job_id: str):
    return await queue.get_job_status(job_id)
```

#### 6. **Умное распараллеливание**
```python
# Параллельная обработка страниц PDF
async def parallel_page_processing(pdf_pages: List[bytes]) -> List[str]:
    tasks = [process_page_ocr(page) for page in pdf_pages]
    results = await asyncio.gather(*tasks)
    return merge_pages_with_structure(results)
```

#### 7. **Динамическая оптимизация ресурсов**
```python
# Автоматическая настройка workers в зависимости от нагрузки
# Мониторинг памяти/CPU и адаптивное масштабирование
# Graceful degradation при высокой нагрузке
```

### 🧠 Алгоритмические усовершенствования

#### 8. **ML-driven постобработка**
```python
# Обучаемая модель для детекции и исправления OCR ошибок
# Контекстное исправление на основе словарей и n-грамм
# Автоматическая детекция языка и переключение моделей
```

#### 9. **Гибридный чанкинг следующего поколения**
```python
def next_gen_chunking(doc: DoclingDocument, embeddings_model: str):
    # Семантическое сходство между соседними абзацами
    # Динамический размер чанков на основе семантической плотности  
    # Сохранение логических границ (не ломать таблицы/списки)
    # Overlap оптимизация через attention механизмы
    pass
```

#### 10. **Мультимодальная обработка**
```python
# Интеграция Computer Vision для анализа диаграмм/графиков
# OCR + Visual Question Answering для сложных макетов
# Автоматическое создание alt-текста для изображений
```

### 📊 Мониторинг и аналитика

#### 11. **Продвинутая метрика качества**
```python
# Confidence scores для каждого элемента документа
# Heatmap качества OCR по регионам страницы  
# A/B тестирование разных стратегий OCR
# Пользовательский feedback loop для улучшения алгоритмов
```

#### 12. **Интеллектуальные предупреждения**
```python
# Автодетекция "проблемных" документов (низкое качество сканирования)
# Рекомендации по улучшению входных данных
# Предупреждения о потенциальных ошибках OCR
```

### 🔒 Безопасность и производительность

#### 13. **Sandboxing и изоляция**
```python
# Выполнение LibreOffice в отдельном контейнере/namespace
# Ограничение ресурсов для каждой операции конвертации
# Антивирусное сканирование загружаемых файлов
```

#### 14. **Потоковая обработка больших файлов**
```python
# Streaming upload/download для файлов >100MB
# Chunked processing с промежуточными результатами
# Прогресс-бар для длительных операций
```

Проект демонстрирует зрелую архитектуру с хорошо продуманными алгоритмами. Основные направления развития: семантическая оценка качества, мультидвижковый OCR, структурно-осознанная обработка и масштабируемость.# Universal Document Converter

Продвинутый Docker-микросервис на FastAPI для конвертации офисных документов, PDF и изображений в структурированный Markdown/JSON с использованием Docling. Включает умную стратегию OCR, фильтрацию заголовков и текстовый чанкинг.

## 🚀 Ключевые возможности

- **Универсальная конвертация**: автоопределение формата + адаптивные стратегии
- **Умный OCR**: сравнение стратегий с эвристическим выбором лучшего результата
- **Legacy поддержка**: `.doc/.xls/.ppt` → LibreOffice → OOXML → Docling
- **Предобработка PDF**: автоматическая очистка через OCRmyPDF (deskew/denoise)
- **Фильтрация заголовков**: удаление повторяющихся "шапок" организаций
- **Текстовый чанкинг**: гибридный алгоритм с сохранением структуры Markdown
- **Извлечение таблиц**: режим ACCURATE с сохранением структуры

## 📁 Структура проекта

```
universal_converter/
├── docker-compose.yml          # Развертывание сервиса
├── Dockerfile                  # Образ с предустановленными моделями
├── requirements.txt            # Python зависимости
├── requirements-dev.txt        # Зависимости для разработки
├── pyproject.toml             # Конфигурация линтеров/тестов
├── .pre-commit-config.yaml    # Git хуки (black/flake8/pytest)
├── app/
│   └── main.py               # Основной FastAPI сервис (1250+ строк)
├── config/
│   ├── header_filter.example.json  # Пример конфигурации фильтров
│   └── header_filter.json          # Боевая конфигурация (в .gitignore)
├── tests/
│   ├── test_health.py        # Тест health endpoint
│   └── test_ocr_pdf.py       # Тест OCR с генерацией PDF
└── data/                     # Рабочая папка (монтируется как том)
```

## 🔄 Архитектура и алгоритмы

### Алгоритм конвертации

1. **Валидация и определение типа**
   - Проверка размера файла (MAX_FILE_SIZE_MB)
   - Определение MIME-type и расширения
   - Валидация поддерживаемых форматов

2. **Legacy форматы** (`.doc/.xls/.ppt`)
   ```
   Input → LibreOffice CLI → OOXML → Docling (do_ocr=False)
           ↓ (при ошибке)
   Fallback: LibreOffice → PDF → OCR стратегии
   ```

3. **PDF/Изображения** (двухстратегическая обработка)
   ```
   Input PDF → OCRmyPDF preprocessing (опционально)
             ↓
   Strategy A: no_ocr (Docling сам определяет необходимость OCR)
   Strategy B: force_ocr (принудительный полностраничный Tesseract)
             ↓
   Эвристический выбор: score = length × (0.6 + 0.4 × cyrillic_ratio)
   ```

4. **OOXML/HTML/MD/CSV**
   ```
   Input → Docling (напрямую, без OCR)
         ↓ (при ошибке)
   Fallback: → PDF → OCR стратегии
   ```

5. **Постобработка**
   - Удаление повторяющихся заголовков (конфигурируемые ключевые слова)
   - Фильтрация "шапки" первой страницы
   - Опциональный чанкинг текста

### Система фильтрации заголовков

**Повторяющиеся заголовки:**
- Анализ первых строк абзацев по всему документу
- Детекция по частоте (≥3 повторений) + ключевые слова организаций
- Нормализация через regex + проверка доли верхнего регистра

**Заголовок первой страницы:**
- Анализ верхнего блока (до 20 строк, макс. 6 строк блока)
- Исключение Markdown заголовков (`# ...`)
- Детекция по ключевым словам + доле верхнего регистра

## 🛠 API Documentation

### GET /health
Проверка доступности сервиса.

**Response:**
```json
{ "ok": true }
```

### POST /convert
Универсальная конвертация документов.

**Request:**
- `file` (file, required): Загружаемый документ
- `out` (string, optional): `markdown` | `json` | `both` (default: `both`)
- `langs` (string, optional): Языки OCR, напр. `rus,eng` (default: из `DOCLING_LANGS`)
- `max_pages` (int, optional): Ограничение страниц

**Response:**
```json
{
  "best_strategy": "force_ocr",
  "trials": [
    {
      "strategy": "no_ocr",
      "length": 1200,
      "cyr_ratio": 0.32,
      "score": 936.0,
      "header_removed": 2,
      "first_header_removed": 3
    },
    {
      "strategy": "force_ocr", 
      "length": 1950,
      "cyr_ratio": 0.48,
      "score": 1497.0,
      "header_removed": 2,
      "first_header_removed": 3
    }
  ],
  "content_markdown": "# Заголовок\n\nТекст документа...",
  "content_json": { "pages": [...] },
  "meta": {
    "filename": "input.pdf",
    "size_bytes": 123456,
    "converted": true,
    "out": "both",
    "langs": "rus,eng",
    "max_pages": null
  }
}
```

### POST /chunk
Гибридный чанкинг текста с сохранением структуры Markdown.

**Request:**
```json
{
  "text": "# Заголовок\n\nАбзац текста...",
  "max_chars": 2000,
  "overlap": 200,
  "preserve_markdown": true
}
```

**Response:**
```json
{
  "chunks": [
    {
      "index": 0,
      "start": 0,
      "end": 150,
      "text": "# Заголовок\n\nАбзац текста..."
    }
  ],
  "meta": {
    "total_chunks": 1,
    "strategy": "hybrid"
  }
}
```

## ⚙️ Конфигурация

### Переменные окружения

| Переменная | Значение по умолчанию | Описание |
|------------|----------------------|----------|
| `UVICORN_HOST` | `0.0.0.0` | Хост сервера |
| `UVICORN_PORT` | `8088` | Порт сервера |
| `DOCLING_ARTIFACTS_PATH` | `/opt/docling-models` | Путь к моделям Docling |
| `DOCLING_TABLE_MODE` | `ACCURATE` | Режим извлечения таблиц |
| `DOCLING_LANGS` | `rus,eng` | Языки OCR по умолчанию |
| `MAX_FILE_SIZE_MB` | `80` | Максимальный размер файла |
| `HEADER_FILTER_CONFIG` | `config/header_filter.json` | Путь к конфигурации фильтров |
| `OMP_NUM_THREADS` | `4` | Количество потоков OpenMP |

### Конфигурация фильтров заголовков

Файл `config/header_filter.json`:
```json
{
  "min_repeats": 3,
  "min_length": 4, 
  "uppercase_ratio": 0.6,
  "keywords": ["ООО", "АО", "МИНИСТЕРСТВО", ...],
  "first_page": {
    "enable": true,
    "lines_limit": 20,
    "max_block_lines": 6,
    "uppercase_ratio": 0.6,
    "keywords": ["ООО", "АО", "ФЕДЕРАЛЬНОЕ", ...]
  }
}
```

## 📋 Предложения по улучшению алгоритмов

### 🎯 Приоритетные улучшения

#### 1. **Семантическая метрика качества OCR**
```python
# Заменить простую эвристику на семантическую оценку
def semantic_quality_score(text: str) -> float:
    # Словарный анализ (доля валидных слов)
    # Энтропия символов (детекция "мусорного" OCR)  
    # Синтаксическая корректность (парсинг предложений)
    # Специфичные паттерны (номера, даты, email)
    pass
```

#### 2. **Адаптивный OCR с несколькими движками**
```python
# Каскадная стратегия OCR
strategies = [
    {"engine": "docling_native", "weight": 1.0},
    {"engine": "tesseract", "psm": [6, 4, 11], "weight": 0.8},
    {"engine": "paddleocr", "lang": ["ru", "en"], "weight": 0.9},
    {"engine": "easyocr", "weight": 0.7}
]
# Ensemble voting для финального результата
```

#### 3. **Структурно-осознанная обработка**
```python
# Использование layout analysis от Docling для умного чанкинга
def structure_aware_processing(doc: DoclingDocument):
    # Сегментация по типам: headers, paragraphs, tables, figures
    # Сохранение иерархии заголовков при чанкинге
    # Отдельная обработка таблиц (OCR + структурное восстановление)
    # Интеграция с изображениями (OCR + caption generation)
    pass
```

#### 4. **Кеширование с умной инвалидацией**
```python
# Redis-based кеширование по хешу файла + параметров
cache_key = hash(file_content + lang + ocr_params + model_version)
# Warm-up cache для популярных документов
# Инкрементальное обновление при изменении настроек
```

### 🔧 Технические улучшения

#### 5. **Асинхронная обработка**
```python
# Celery/RQ для длительных операций
@app.post("/convert/async")
async def convert_async():
    job_id = await queue.enqueue(convert_task, file_data)
    return {"job_id": job_id, "status": "queued"}

@app.get("/convert/status/{job_id}")
async def get_status(job_id: str):
    return await queue.get_job_status(job_id)
```

#### 6. **Умное распараллеливание**
```python
# Параллельная обработка страниц PDF
async def parallel_page_processing(pdf_pages: List[bytes]) -> List[str]:
    tasks = [process_page_ocr(page) for page in pdf_pages]
    results = await asyncio.gather(*tasks)
    return merge_pages_with_structure(results)
```

#### 7. **Динамическая оптимизация ресурсов**
```python
# Автоматическая настройка workers в зависимости от нагрузки
# Мониторинг памяти/CPU и адаптивное масштабирование
# Graceful degradation при высокой нагрузке
```

### 🧠 Алгоритмические усовершенствования

#### 8. **ML-driven постобработка**
```python
# Обучаемая модель для детекции и исправления OCR ошибок
# Контекстное исправление на основе словарей и n-грамм
# Автоматическая детекция языка и переключение моделей
```

#### 9. **Гибридный чанкинг следующего поколения**
```python
def next_gen_chunking(doc: DoclingDocument, embeddings_model: str):
    # Семантическое сходство между соседними абзацами
    # Динамический размер чанков на основе семантической плотности  
    # Сохранение логических границ (не ломать таблицы/списки)
    # Overlap оптимизация через attention механизмы
    pass
```

#### 10. **Мультимодальная обработка**
```python
# Интеграция Computer Vision для анализа диаграмм/графиков
# OCR + Visual Question Answering для сложных макетов
# Автоматическое создание alt-текста для изображений
```

### 📊 Мониторинг и аналитика

#### 11. **Продвинутая метрика качества**
```python
# Confidence scores для каждого элемента документа
# Heatmap качества OCR по регионам страницы  
# A/B тестирование разных стратегий OCR
# Пользовательский feedback loop для улучшения алгоритмов
```

#### 12. **Интеллектуальные предупреждения**
```python
# Автодетекция "проблемных" документов (низкое качество сканирования)
# Рекомендации по улучшению входных данных
# Предупреждения о потенциальных ошибках OCR
```

### 🔒 Безопасность и производительность

#### 13. **Sandboxing и изоляция**
```python
# Выполнение LibreOffice в отдельном контейнере/namespace
# Ограничение ресурсов для каждой операции конвертации
# Антивирусное сканирование загружаемых файлов
```

#### 14. **Потоковая обработка больших файлов**
```python
# Streaming upload/download для файлов >100MB
# Chunked processing с промежуточными результатами
# Прогресс-бар для длительных операций
```
