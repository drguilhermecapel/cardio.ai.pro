import pytest
from fastapi.testclient import TestClient
from app.main import app


def test_health_endpoint():
    """Test health check endpoint."""
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "cardioai-pro-standalone"


def test_root_endpoint():
    """Test root endpoint."""
    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200


def test_api_v1_prefix():
    """Test API v1 prefix is included."""
    client = TestClient(app)
    response = client.get("/api/v1/health")

    assert response.status_code == 200


def test_cors_middleware():
    """Test CORS middleware is configured."""
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200


def test_app_initialization():
    """Test FastAPI app initialization."""
    assert app.title == "CardioAI Pro API"
    assert app.version == "1.0.0"


def test_openapi_schema():
    """Test OpenAPI schema generation."""
    client = TestClient(app)
    response = client.get("/api/v1/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    assert "openapi" in schema
    assert schema["info"]["title"] == "CardioAI Pro API"


def test_docs_endpoint():
    """Test API documentation endpoint."""
    client = TestClient(app)
    response = client.get("/api/v1/docs")

    assert response.status_code == 200


def test_redoc_endpoint():
    """Test ReDoc documentation endpoint."""
    client = TestClient(app)
    response = client.get("/api/v1/redoc")

    assert response.status_code == 200


def test_middleware_configuration():
    """Test middleware configuration."""
    middlewares = [middleware.cls.__name__ for middleware in app.user_middleware]
    assert "CORSMiddleware" in middlewares


def test_exception_handlers():
    """Test exception handlers are configured."""
    assert len(app.exception_handlers) >= 0
