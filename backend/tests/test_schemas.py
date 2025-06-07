import pytest
from datetime import datetime
from app.schemas.user import UserCreate, UserUpdate, UserInDB
from app.schemas.patient import PatientCreate, PatientUpdate, PatientInDB
from app.schemas.ecg_analysis import ECGAnalysisCreate, ECGAnalysisUpdate
from app.schemas.notification import NotificationCreate, NotificationUpdate
from app.schemas.validation import ValidationCreate, ValidationUpdate


def test_user_create_schema():
    """Test UserCreate schema validation."""
    user_data = {
        "username": "testuser",
        "email": "test@example.com",
        "password": "Password123!",
        "first_name": "Test",
        "last_name": "User"
    }
    
    user = UserCreate(**user_data)
    
    assert user.email == "test@example.com"
    assert user.password == "Password123!"
    assert user.first_name == "Test"
    assert user.last_name == "User"


def test_user_update_schema():
    """Test UserUpdate schema validation."""
    user_data = {
        "first_name": "Updated",
        "last_name": "Name"
    }
    
    user = UserUpdate(**user_data)
    
    assert user.first_name == "Updated"
    assert user.last_name == "Name"


def test_user_in_db_schema():
    """Test UserInDB schema validation."""
    user_data = {
        "id": 1,
        "username": "testuser",
        "email": "test@example.com",
        "first_name": "Test",
        "last_name": "User",
        "is_active": True,
        "is_verified": True,
        "is_superuser": False,
        "last_login": None,
        "created_at": datetime.now(),
        "updated_at": datetime.now()
    }
    
    user = UserInDB(**user_data)
    
    assert user.id == 1
    assert user.email == "test@example.com"


def test_patient_create_schema():
    """Test PatientCreate schema validation."""
    from datetime import date
    patient_data = {
        "patient_id": "PAT123",
        "first_name": "John",
        "last_name": "Doe",
        "date_of_birth": date(1990, 1, 1),
        "gender": "male"
    }
    
    patient = PatientCreate(**patient_data)
    
    assert patient.first_name == "John"
    assert patient.last_name == "Doe"
    assert patient.gender == "male"


def test_patient_update_schema():
    """Test PatientUpdate schema validation."""
    patient_data = {
        "first_name": "Jane",
        "last_name": "Doe",
        "phone": "+1234567890"
    }
    
    patient = PatientUpdate(**patient_data)
    
    assert patient.first_name == "Jane"
    assert patient.last_name == "Doe"
    assert patient.phone == "+1234567890"


def test_ecg_analysis_create_schema():
    """Test ECGAnalysisCreate schema validation."""
    from app.schemas.ecg_analysis import ECGAnalysisCreate
    from datetime import datetime
    analysis_data = {
        "patient_id": 1,
        "original_filename": "test_ecg.csv",
        "acquisition_date": datetime.now(),
        "sample_rate": 500,
        "duration_seconds": 10.0,
        "leads_count": 12,
        "leads_names": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    }
    
    analysis = ECGAnalysisCreate(**analysis_data)
    
    assert analysis.patient_id == 1
    assert analysis.original_filename == "test_ecg.csv"


def test_notification_create_schema():
    """Test NotificationCreate schema validation."""
    from app.schemas.notification import NotificationCreate
    from app.core.constants import NotificationType, NotificationPriority
    notification_data = {
        "title": "Test Notification",
        "message": "This is a test",
        "notification_type": NotificationType.SYSTEM_ALERT,
        "priority": NotificationPriority.MEDIUM,
        "user_id": 1
    }
    
    notification = NotificationCreate(**notification_data)
    
    assert notification.title == "Test Notification"
    assert notification.notification_type == NotificationType.SYSTEM_ALERT


def test_validation_create_schema():
    """Test ValidationCreate schema validation."""
    from app.schemas.validation import ValidationCreate
    validation_data = {
        "analysis_id": 1,
        "validator_id": 1
    }
    
    validation = ValidationCreate(**validation_data)
    
    assert validation.analysis_id == 1
    assert validation.validator_id == 1


def test_schema_field_validation():
    """Test schema field validation."""
    with pytest.raises(ValueError):
        UserCreate(email="invalid-email", password="123", full_name="Test")


def test_optional_fields():
    """Test optional fields in schemas."""
    user = UserUpdate(first_name="Test")
    
    assert user.first_name == "Test"
    assert user.email is None
    assert user.last_name is None


def test_schema_serialization():
    """Test schema serialization."""
    user = UserCreate(
        username="testuser",
        email="test@example.com",
        password="Password123!",
        first_name="Test",
        last_name="User"
    )
    
    data = user.model_dump()
    
    assert isinstance(data, dict)
    assert data["email"] == "test@example.com"


def test_schema_json_serialization():
    """Test schema JSON serialization."""
    user = UserCreate(
        username="testuser",
        email="test@example.com",
        password="Password123!",
        first_name="Test",
        last_name="User"
    )
    
    json_str = user.model_dump_json()
    
    assert isinstance(json_str, str)
    assert "test@example.com" in json_str
