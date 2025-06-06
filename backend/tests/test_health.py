"""Test health endpoint."""

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.mark.timeout(30)



def test_health_endpoint(client):
    """Test health endpoint returns correct response."""
    response = client.get("/health")
    
    assert response.status_code == 200
    response_data = response.json()
    assert "status" in response_data
    assert response_data["status"] == "healthy"
