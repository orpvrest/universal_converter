"""Pydantic-модели запросов и ответов сервиса."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class ConvertResponse(BaseModel):
    """Стандартная модель ответа конвертации."""

    content_markdown: Optional[str] = None
    content_json: Optional[dict] = None
    meta: dict


class ChunkItem(BaseModel):
    """Единичный чанк текста с позицией и индексом."""

    index: int
    start: int
    end: int
    text: str


class ChunkRequest(BaseModel):
    """Параметры чанкинга текста."""

    text: str
    max_chars: int = 2000
    overlap: int = 200
    preserve_markdown: bool = True


class ChunkResponse(BaseModel):
    """Ответ с набором чанков и метаданными."""

    chunks: list[ChunkItem]
    meta: dict
