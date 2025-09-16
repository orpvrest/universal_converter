"""Конфигурация и общие настройки сервиса Docling."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

# Параметры среды
DOCLING_ARTIFACTS_PATH = os.getenv("DOCLING_ARTIFACTS_PATH")
DEFAULT_TABLE_MODE = os.getenv("DOCLING_TABLE_MODE", "ACCURATE").upper()
FORCE_OCR_TABLE_MODE = os.getenv(
    "DOCLING_TABLE_MODE_FORCE_OCR", "LERF"
).upper()
DEFAULT_LANGS = [
    x.strip()
    for x in os.getenv("DOCLING_LANGS", "rus,eng").split(",")
    if x.strip()
]
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "80"))
HEADER_CFG_PATH = Path(
    os.getenv(
        "HEADER_FILTER_CONFIG",
        Path(__file__).resolve().parent.parent / "config" / "header_filter.json",
    )
)


@lru_cache()
def load_header_cfg() -> dict[str, Any]:
    """Загружает JSON-конфигурацию фильтра шапок с диска.

    Возвращает словарь с порогами и ключевыми словами.
    Если конфиг отсутствует или повреждён, подставляет консервативные значения
    по умолчанию.
    """
    if HEADER_CFG_PATH.exists():
        try:
            return json.loads(HEADER_CFG_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    # Фолбэк на консервативные настройки
    return {
        "min_repeats": 3,
        "min_length": 4,
        "uppercase_ratio": 0.6,
        "keywords": [
            "ООО",
            "АО",
            "ПАО",
            "ФГУП",
            "ФГБУ",
            "МИНИСТЕРСТВО",
            "АДМИНИСТРАЦИЯ",
            "КОМПАНИЯ",
            "КОРПОРАЦИЯ",
            "УНИВЕРСИТЕТ",
            "ИНСТИТУТ",
            "ФЕДЕРАЛЬНОЕ",
            "ГОСУДАРСТВЕННОЕ",
            "ОБЩЕСТВО",
            "ОРГАНИЗАЦИЯ",
            "РОССИЯ",
            "РОССИЙСКАЯ",
        ],
    }
