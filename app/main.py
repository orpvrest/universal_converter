"""FastAPI-сервис конвертации документов на базе Docling."""

from __future__ import annotations

from fastapi import FastAPI

from .chunking import hybrid_chunk  # noqa: F401 - re-export for backwards compat
from .langs import (  # noqa: F401 - re-export для тестов
    langs_join as _langs_join,
    normalize_easyocr_langs as _normalize_easyocr_langs,
    normalize_langs as _normalize_langs,
)
from .markdown_filters import (  # noqa: F401 - re-export для совместимости
    detect_and_remove_repeating_header as _detect_and_remove_repeating_header,
    remove_first_page_header as _remove_first_page_header,
)
from .ocr_utils import ensure_models, try_ocrmypdf_preprocess as _try_ocrmypdf_preprocess
from .routes import chunk, convert, health
from .schemas import ConvertResponse, ChunkItem, ChunkRequest, ChunkResponse

app = FastAPI(title="Docling microservice", version="3.3.0")
app.include_router(health.router)
app.include_router(chunk.router)
app.include_router(convert.router)


@app.on_event("startup")
def init_models() -> None:
    """Подгружает модели Docling при старте, чтобы избежать первых 500."""
    try:
        ensure_models()
    except Exception:
        # Сервис не должен падать на старте — Docling попробует скачать модели
        # при первом запросе, если здесь что-то пошло не так.
        pass


__all__ = [
    "app",
    "ConvertResponse",
    "ChunkItem",
    "ChunkRequest",
    "ChunkResponse",
    "_normalize_langs",
    "_normalize_easyocr_langs",
    "_langs_join",
    "_detect_and_remove_repeating_header",
    "_remove_first_page_header",
    "_try_ocrmypdf_preprocess",
    "hybrid_chunk",
]
