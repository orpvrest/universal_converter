"""Тесты для здоровья API."""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    """Тест работоспособности эндпоинта /health."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_openapi_schema():
    """Проверка доступности OpenAPI схемы."""
    response = client.get("/openapi.json")
    assert response.status_code == 200

    schema = response.json()
    assert "openapi" in schema
    assert "/health" in schema["paths"]
    assert "/convert" in schema["paths"]
