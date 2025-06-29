"""Тесты для API endpoints."""

import pytest
from fastapi.testclient import TestClient
from main import app


client = TestClient(app)


def test_root_endpoint():
    """Тест корневого эндпоинта."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "Mini RAG System" in data["message"]


def test_ping_endpoint():
    """Тест ping эндпоинта."""
    response = client.get("/ping")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "pong"
    assert "timestamp" in data


def test_health_endpoint():
    """Тест health check эндпоинта."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "services" in data
    assert "timestamp" in data


def test_docs_endpoint():
    """Тест доступности документации."""
    response = client.get("/docs")
    assert response.status_code == 200


def test_query_endpoint_validation():
    """Тест валидации запроса."""
    # Пустой запрос
    response = client.post("/api/v1/query", json={})
    assert response.status_code == 422
    
    # Запрос без вопроса
    response = client.post("/api/v1/query", json={"language": "ru"})
    assert response.status_code == 422


def test_document_upload_validation():
    """Тест валидации загрузки документа."""
    # Пустой запрос
    response = client.post("/api/v1/documents", json={})
    assert response.status_code == 422
    
    # Запрос без контента
    response = client.post("/api/v1/documents", json={"title": "Test"})
    assert response.status_code == 422 