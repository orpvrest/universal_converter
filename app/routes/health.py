"""Маршрут проверки готовности сервиса."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health() -> dict[str, bool]:
    """Возвращает простой индикатор готовности."""
    return {"ok": True}
