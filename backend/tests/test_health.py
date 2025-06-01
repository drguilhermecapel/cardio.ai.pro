"""Test health endpoint."""

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


def test_health_endpoint(client):
    """Test health endpoint returns correct response."""
    response = client.get("/health")
    
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "cardioai-pro-api"}
