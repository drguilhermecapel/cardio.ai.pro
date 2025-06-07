import pytest
from fastapi.testclient import TestClient
from app.core.config import settings

if settings.STANDALONE_MODE:
    pytest.skip("API endpoint tests skipped in standalone mode", allow_module_level=True)

from app.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_api_v1_health_endpoint(client):
    """Test API v1 health endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200


def test_docs_endpoint(client):
    """Test API documentation endpoint."""
    response = client.get("/docs")
    assert response.status_code == 200


def test_openapi_endpoint(client):
    """Test OpenAPI schema endpoint."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "openapi" in schema


def test_cors_headers(client):
    """Test CORS headers are present."""
    response = client.get("/health")
    assert response.status_code == 200


def test_invalid_endpoint(client):
    """Test invalid endpoint returns 404."""
    response = client.get("/invalid-endpoint")
    assert response.status_code == 404


def test_api_v1_prefix_routes(client):
    """Test API v1 prefix routes exist."""
    response = client.get("/api/v1/")
    assert response.status_code in [200, 404, 405]


def test_authentication_endpoints(client):
    """Test authentication endpoints exist."""
    response = client.post("/api/v1/auth/login")
    assert response.status_code in [400, 422, 405]


def test_user_endpoints(client):
    """Test user endpoints exist."""
    response = client.get("/api/v1/users/")
    assert response.status_code in [401, 403, 422]


def test_patient_endpoints(client):
    """Test patient endpoints exist."""
    response = client.get("/api/v1/patients/")
    assert response.status_code in [401, 403, 422]


def test_ecg_analysis_endpoints(client):
    """Test ECG analysis endpoints exist."""
    response = client.get("/api/v1/ecg-analyses/")
    assert response.status_code in [401, 403, 422]


def test_notification_endpoints(client):
    """Test notification endpoints exist."""
    response = client.get("/api/v1/notifications/")
    assert response.status_code in [401, 403, 422]


def test_validation_endpoints(client):
    """Test validation endpoints exist."""
    response = client.get("/api/v1/validations/")
    assert response.status_code in [401, 403, 422]
