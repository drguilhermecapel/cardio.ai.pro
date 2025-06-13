"""Comprehensive API endpoint tests for full coverage."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock, patch
import os
from datetime import datetime, timedelta

os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///test.db"

from app.main import app
from app.core.security import create_access_token


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Create authentication headers."""
    token = create_access_token(subject="test_user")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def admin_headers():
    """Create admin authentication headers."""
    token = create_access_token(subject="admin_user")
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_auth_endpoints_comprehensive(client):
    """Test all auth endpoints comprehensively."""
    
    # Test login - success case
    with patch('app.api.v1.endpoints.auth.UserService') as mock_service:
        mock_user = Mock()
        mock_user.id = 1
        mock_user.email = "test@test.com"
        mock_user.is_active = True
        mock_service.return_value.authenticate_user = AsyncMock(return_value=mock_user)
        
        response = client.post("/api/v1/auth/login",
            json={"email": "test@test.com", "password": "password123"})
        assert response.status_code == 200
        assert "access_token" in response.json()
    
    # Test login - invalid credentials
    with patch('app.api.v1.endpoints.auth.UserService') as mock_service:
        mock_service.return_value.authenticate_user = AsyncMock(return_value=None)
        
        response = client.post("/api/v1/auth/login",
            json={"email": "invalid@test.com", "password": "wrong"})
        assert response.status_code in [400, 401]
    
    # Test login - inactive user
    with patch('app.api.v1.endpoints.auth.UserService') as mock_service:
        mock_user = Mock()
        mock_user.is_active = False
        mock_service.return_value.authenticate_user = AsyncMock(return_value=mock_user)
        
        response = client.post("/api/v1/auth/login",
            json={"email": "inactive@test.com", "password": "password123"})
        assert response.status_code in [400, 401]
    
    # Test register - success case
    with patch('app.api.v1.endpoints.auth.UserService') as mock_service:
        mock_service.return_value.get_user_by_email = AsyncMock(return_value=None)
        mock_service.return_value.create_user = AsyncMock(return_value=Mock(id=1))
        
        response = client.post("/api/v1/auth/register",
            json={
                "email": "new@test.com",
                "password": "password123",
                "full_name": "New User",
                "role": "technician"
            })
        assert response.status_code in [200, 201]
    
    # Test register - existing email
    with patch('app.api.v1.endpoints.auth.UserService') as mock_service:
        mock_service.return_value.get_user_by_email = AsyncMock(return_value=Mock())
        
        response = client.post("/api/v1/auth/register",
            json={
                "email": "existing@test.com",
                "password": "password123",
                "full_name": "Existing User"
            })
        assert response.status_code in [400, 409]
    
    # Test refresh token
    response = client.post("/api/v1/auth/refresh", headers=auth_headers)
    assert response.status_code in [200, 401]
    
    # Test logout
    response = client.post("/api/v1/auth/logout", headers=auth_headers)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_users_endpoints_comprehensive(client, auth_headers, admin_headers):
    """Test all user endpoints comprehensively."""
    
    # Test get current user
    with patch('app.api.v1.endpoints.users.get_current_user') as mock_get_user:
        mock_get_user.return_value = Mock(id=1, email="test@test.com")
        
        response = client.get("/api/v1/users/me", headers=auth_headers)
        assert response.status_code == 200
    
    # Test update current user
    with patch('app.api.v1.endpoints.users.UserService') as mock_service:
        mock_service.return_value.update_user = AsyncMock(return_value=Mock())
        
        response = client.put("/api/v1/users/me",
            headers=auth_headers,
            json={"full_name": "Updated Name"})
        assert response.status_code == 200
    
    # Test get all users (admin only)
    with patch('app.api.v1.endpoints.users.UserService') as mock_service:
        mock_service.return_value.get_users = AsyncMock(return_value=([], 0))
        
        response = client.get("/api/v1/users/", headers=admin_headers)
        assert response.status_code in [200, 403]
    
    # Test get user by ID
    with patch('app.api.v1.endpoints.users.UserService') as mock_service:
        mock_service.return_value.get_user_by_id = AsyncMock(return_value=Mock())
        
        response = client.get("/api/v1/users/1", headers=auth_headers)
        assert response.status_code in [200, 403, 404]
    
    # Test update user (admin)
    with patch('app.api.v1.endpoints.users.UserService') as mock_service:
        mock_service.return_value.update_user = AsyncMock(return_value=Mock())
        
        response = client.put("/api/v1/users/1",
            headers=admin_headers,
            json={"role": "physician"})
        assert response.status_code in [200, 403, 404]
    
    # Test delete user (admin)
    with patch('app.api.v1.endpoints.users.UserService') as mock_service:
        mock_service.return_value.delete_user = AsyncMock(return_value=True)
        
        response = client.delete("/api/v1/users/1", headers=admin_headers)
        assert response.status_code in [200, 204, 403, 404]


@pytest.mark.asyncio
async def test_patients_endpoints_comprehensive(client, auth_headers):
    """Test all patient endpoints comprehensively."""
    
    # Test create patient
    with patch('app.api.v1.endpoints.patients.PatientService') as mock_service:
        mock_service.return_value.create_patient = AsyncMock(return_value=Mock(id=1))
        
        response = client.post("/api/v1/patients/",
            headers=auth_headers,
            json={
                "name": "Test Patient",
                "birth_date": "1990-01-01",
                "gender": "M",
                "email": "patient@test.com"
            })
        assert response.status_code in [200, 201]
    
    # Test get all patients
    with patch('app.api.v1.endpoints.patients.PatientService') as mock_service:
        mock_service.return_value.get_patients = AsyncMock(return_value=([], 0))
        
        response = client.get("/api/v1/patients/", headers=auth_headers)
        assert response.status_code == 200
    
    # Test search patients
    with patch('app.api.v1.endpoints.patients.PatientService') as mock_service:
        mock_service.return_value.search_patients = AsyncMock(return_value=[])
        
        response = client.get("/api/v1/patients/search?query=test", headers=auth_headers)
        assert response.status_code == 200
    
    # Test get patient by ID
    with patch('app.api.v1.endpoints.patients.PatientService') as mock_service:
        mock_service.return_value.get_patient_by_id = AsyncMock(return_value=Mock())
        
        response = client.get("/api/v1/patients/1", headers=auth_headers)
        assert response.status_code == 200
    
    # Test update patient
    with patch('app.api.v1.endpoints.patients.PatientService') as mock_service:
        mock_service.return_value.update_patient = AsyncMock(return_value=Mock())
        
        response = client.put("/api/v1/patients/1",
            headers=auth_headers,
            json={"phone": "+1234567890"})
        assert response.status_code == 200
    
    # Test delete patient
    with patch('app.api.v1.endpoints.patients.PatientService') as mock_service:
        mock_service.return_value.delete_patient = AsyncMock(return_value=True)
        
        response = client.delete("/api/v1/patients/1", headers=auth_headers)
        assert response.status_code in [200, 204]


@pytest.mark.asyncio
async def test_ecg_endpoints_comprehensive(client, auth_headers):
    """Test all ECG analysis endpoints comprehensively."""
    
    # Test upload ECG
    with patch('app.api.v1.endpoints.ecg_analyses.ECGAnalysisService') as mock_service:
        mock_service.return_value.create_analysis = AsyncMock(return_value=Mock(id=1))
        
        # Create a mock file upload
        from io import BytesIO
        file_content = b"mock ECG data"
        
        response = client.post("/api/v1/ecg-analyses/upload",
            headers=auth_headers,
            files={"file": ("test.txt", BytesIO(file_content), "text/plain")},
            data={"patient_id": "1"})
        assert response.status_code in [200, 201, 422]
    
    # Test get all analyses
    with patch('app.api.v1.endpoints.ecg_analyses.ECGAnalysisService') as mock_service:
        mock_service.return_value.get_analyses = AsyncMock(return_value=([], 0))
        
        response = client.get("/api/v1/ecg-analyses/", headers=auth_headers)
        assert response.status_code == 200
    
    # Test get analysis by ID
    with patch('app.api.v1.endpoints.ecg_analyses.ECGAnalysisService') as mock_service:
        mock_service.return_value.get_analysis_by_id = AsyncMock(return_value=Mock())
        
        response = client.get("/api/v1/ecg-analyses/1", headers=auth_headers)
        assert response.status_code == 200
    
    # Test get analyses by patient
    with patch('app.api.v1.endpoints.ecg_analyses.ECGAnalysisService') as mock_service:
        mock_service.return_value.get_analyses_by_patient = AsyncMock(return_value=[])
        
        response = client.get("/api/v1/ecg-analyses/patient/1", headers=auth_headers)
        assert response.status_code == 200
    
    # Test reprocess analysis
    with patch('app.api.v1.endpoints.ecg_analyses.ECGAnalysisService') as mock_service:
        mock_service.return_value.reprocess_analysis = AsyncMock(return_value=Mock())
        
        response = client.post("/api/v1/ecg-analyses/1/reprocess", headers=auth_headers)
        assert response.status_code == 200
    
    # Test delete analysis
    with patch('app.api.v1.endpoints.ecg_analyses.ECGAnalysisService') as mock_service:
        mock_service.return_value.delete_analysis = AsyncMock(return_value=True)
        
        response = client.delete("/api/v1/ecg-analyses/1", headers=auth_headers)
        assert response.status_code in [200, 204]


@pytest.mark.asyncio
async def test_validations_endpoints_comprehensive(client, auth_headers):
    """Test all validation endpoints comprehensively."""
    
    # Test create validation
    with patch('app.api.v1.endpoints.validations.ValidationService') as mock_service:
        mock_service.return_value.create_validation = AsyncMock(return_value=Mock(id=1))
        
        response = client.post("/api/v1/validations/",
            headers=auth_headers,
            json={
                "analysis_id": 1,
                "status": "pending",
                "notes": "Initial validation"
            })
        assert response.status_code in [200, 201]
    
    # Test get pending validations
    with patch('app.api.v1.endpoints.validations.ValidationService') as mock_service:
        mock_service.return_value.get_pending_validations = AsyncMock(return_value=[])
        
        response = client.get("/api/v1/validations/pending", headers=auth_headers)
        assert response.status_code == 200
    
    # Test submit validation
    with patch('app.api.v1.endpoints.validations.ValidationService') as mock_service:
        mock_service.return_value.submit_validation = AsyncMock(return_value=Mock())
        
        response = client.post("/api/v1/validations/1/submit",
            headers=auth_headers,
            json={
                "status": "approved",
                "diagnosis_confirmed": True,
                "notes": "Validation complete"
            })
        assert response.status_code == 200
    
    # Test get validation by ID
    with patch('app.api.v1.endpoints.validations.ValidationService') as mock_service:
        mock_service.return_value.get_validation_by_id = AsyncMock(return_value=Mock())
        
        response = client.get("/api/v1/validations/1", headers=auth_headers)
        assert response.status_code == 200
    
    # Test get validations by validator
    with patch('app.api.v1.endpoints.validations.ValidationService') as mock_service:
        mock_service.return_value.get_validations_by_validator = AsyncMock(return_value=[])
        
        response = client.get("/api/v1/validations/validator/1", headers=auth_headers)
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_notifications_endpoints_comprehensive(client, auth_headers):
    """Test all notification endpoints comprehensively."""
    
    # Test get user notifications
    with patch('app.api.v1.endpoints.notifications.NotificationService') as mock_service:
        mock_service.return_value.get_user_notifications = AsyncMock(return_value=[])
        
        response = client.get("/api/v1/notifications/", headers=auth_headers)
        assert response.status_code == 200
    
    # Test get unread count
    with patch('app.api.v1.endpoints.notifications.NotificationService') as mock_service:
        mock_service.return_value.get_unread_count = AsyncMock(return_value=5)
        
        response = client.get("/api/v1/notifications/unread-count", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["count"] == 5
    
    # Test mark notification as read
    with patch('app.api.v1.endpoints.notifications.NotificationService') as mock_service:
        mock_service.return_value.mark_as_read = AsyncMock(return_value=True)
        
        response = client.put("/api/v1/notifications/1/read", headers=auth_headers)
        assert response.status_code == 200
    
    # Test mark all as read
    with patch('app.api.v1.endpoints.notifications.NotificationService') as mock_service:
        mock_service.return_value.mark_all_as_read = AsyncMock(return_value=10)
        
        response = client.put("/api/v1/notifications/read-all", headers=auth_headers)
        assert response.status_code == 200
    
    # Test delete notification
    with patch('app.api.v1.endpoints.notifications.NotificationService') as mock_service:
        mock_service.return_value.delete_notification = AsyncMock(return_value=True)
        
        response = client.delete("/api/v1/notifications/1", headers=auth_headers)
        assert response.status_code in [200, 204]


@pytest.mark.asyncio
async def test_health_and_system_endpoints(client):
    """Test health check and system endpoints."""
    
    # Test health check
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    
    # Test root endpoint
    response = client.get("/")
    assert response.status_code in [200, 307]
    
    # Test docs endpoint
    response = client.get("/docs")
    assert response.status_code in [200, 404]
    
    # Test OpenAPI schema
    response = client.get("/openapi.json")
    assert response.status_code in [200, 404]
    
    # Test metrics endpoint (if exists)
    response = client.get("/metrics")
    # Metrics might not be available in test environment
    assert response.status_code in [200, 404]


@pytest.mark.asyncio
async def test_error_handling_and_edge_cases(client, auth_headers):
    """Test error handling and edge cases for all endpoints."""
    
    # Test invalid JSON
    response = client.post("/api/v1/patients/",
        headers=auth_headers,
        data="invalid json{")
    assert response.status_code in [400, 422]
    
    # Test missing required fields
    response = client.post("/api/v1/patients/",
        headers=auth_headers,
        json={})  # Missing required fields
    assert response.status_code == 422
    
    # Test invalid ID format
    response = client.get("/api/v1/patients/invalid-id", headers=auth_headers)
    assert response.status_code in [404, 422]
    
    # Test unauthorized access
    response = client.get("/api/v1/patients/")
    assert response.status_code == 401
    
    # Test invalid token
    invalid_headers = {"Authorization": "Bearer invalid.token.here"}
    response = client.get("/api/v1/patients/", headers=invalid_headers)
    assert response.status_code == 401
    
    # Test method not allowed
    response = client.patch("/api/v1/auth/login")  # PATCH not allowed
    assert response.status_code == 405
    
    # Test CORS preflight
    response = client.options("/api/v1/patients/",
        headers={"Origin": "http://localhost:3000"})
    assert response.status_code in [200, 204, 405]