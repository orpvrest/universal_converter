"""Утилиты по работе со списками языков для OCR."""

from __future__ import annotations

from typing import Iterable

from .config import DEFAULT_LANGS

# Сопоставление пользовательских сокращений с кодами Tesseract.
_LANG_ALIASES = {
    "ru": "rus",
    "en": "eng",
    "uk": "ukr",
    "kz": "kaz",
    "kk": "kaz",
    "uz": "uzb",
    "be": "bel",
    "bg": "bul",
}

# EasyOCR ожидает двухбуквенные ISO-коды; приводим распространённые варианты.
_EASYOCR_ALIASES = {
    "rus": "ru",
    "ru": "ru",
    "eng": "en",
    "en": "en",
    "ukr": "uk",
    "uk": "uk",
    "bel": "be",
    "be": "be",
    "kaz": "kk",
    "kk": "kk",
    "kz": "kk",
    "uzb": "uz",
    "uz": "uz",
    "bul": "bg",
    "bg": "bg",
}


def normalize_langs(langs: Iterable[str]) -> list[str]:
    """Приводит список языков к формату Tesseract и убирает дубли."""
    normalized: list[str] = []
    for lang in langs:
        key = lang.lower()
        norm = _LANG_ALIASES.get(key, lang)
        if norm not in normalized:
            normalized.append(norm)
    return normalized


def normalize_easyocr_langs(langs: Iterable[str]) -> list[str]:
    """Готовит список языков к формату EasyOCR (двухсимвольные коды)."""
    normalized: list[str] = []
    for lang in langs:
        key = lang.lower()
        if key in _EASYOCR_ALIASES:
            norm = _EASYOCR_ALIASES[key]
        elif len(key) == 3 and key.isalpha():
            norm = key[:2]
        else:
            norm = key
        if norm not in normalized:
            normalized.append(norm)
    return normalized


def langs_join(langs: Iterable[str]) -> str:
    """Формирует строку языков для Tesseract/OCRmyPDF: 'rus+eng'."""
    cleaned = [x for x in langs if x]
    return "+".join(cleaned) if cleaned else "rus+eng"


def resolve_langs_param(langs_value: str | None) -> list[str]:
    """Разбирает строку языков из формы и подставляет значения из ENV."""
    if langs_value is None:
        source = DEFAULT_LANGS
    else:
        source = [x.strip() for x in langs_value.split(",") if x.strip()]
    return normalize_langs(source)


# Обратная совместимость для тестов, которые импортируют приватные функции.
_langs_join = langs_join
_normalize_langs = normalize_langs
_normalize_easyocr_langs = normalize_easyocr_langs
