"""Testes para garantir cobertura das exceções personalizadas."""

import pytest
from app.core.exceptions import (
    CardioAIException,
    ECGProcessingException,
    ValidationException,
    AuthenticationException,
    AuthorizationException,
    NotFoundException,
    ConflictException,
    PermissionDeniedException,
    ECGNotFoundException,
    PatientNotFoundException,
    UserNotFoundException,
    MLModelException,
    ValidationNotFoundException,
    AnalysisNotFoundException,
    ValidationAlreadyExistsException,
    InsufficientPermissionsException,
    RateLimitExceededException,
    FileProcessingException,
    DatabaseException,
    ExternalServiceException,
    NonECGImageException,
    MultiPathologyException,
    ECGReaderException,
)


class TestExceptions:
    """Test all custom exceptions for coverage."""
    
    def test_base_exception(self):
        """Test base CardioAI exception."""
        exc = CardioAIException("Test error", "TEST_CODE", 400, {"detail": "test"})
        assert exc.message == "Test error"
        assert exc.error_code == "TEST_CODE"
        assert exc.status_code == 400
        assert exc.details == {"detail": "test"}
        assert str(exc) == "Test error"
        
    def test_ecg_processing_exception(self):
        """Test ECG processing exception."""
        exc = ECGProcessingException("Processing failed", "ecg_001")
        assert exc.ecg_id == "ecg_001"
        assert exc.error_code == "ECG_PROCESSING_ERROR"
        assert exc.status_code == 422
        
    def test_validation_exception(self):
        """Test validation exception."""
        errors = [{"field": "age", "message": "Invalid age"}]
        exc = ValidationException("Validation failed", errors)
        assert exc.validation_errors == errors
        assert exc.error_code == "VALIDATION_ERROR"
        
    def test_authentication_exception(self):
        """Test authentication exception."""
        exc = AuthenticationException()
        assert exc.message == "Could not validate credentials"
        assert exc.error_code == "AUTHENTICATION_ERROR"
        assert exc.status_code == 401
        
    def test_authorization_exception(self):
        """Test authorization exception."""
        exc = AuthorizationException()
        assert exc.message == "Not authorized to access this resource"
        assert exc.error_code == "AUTHORIZATION_ERROR"
        assert exc.status_code == 403
        
    def test_not_found_exceptions(self):
        """Test all not found exceptions."""
        ecg_exc = ECGNotFoundException("ecg_123")
        assert "ECG ecg_123 not found" in ecg_exc.message
        
        patient_exc = PatientNotFoundException("patient_456")
        assert "Patient patient_456 not found" in patient_exc.message
        
        user_exc = UserNotFoundException("user_789")
        assert "User user_789 not found" in user_exc.message
        
    def test_ml_model_exception(self):
        """Test ML model exception."""
        exc = MLModelException("Model failed", "model_v1")
        assert exc.details == {"model_name": "model_v1"}
        assert exc.error_code == "ML_MODEL_ERROR"
        
    def test_validation_not_found(self):
        """Test validation not found exception."""
        exc = ValidationNotFoundException("val_001")
        assert "Validation val_001 not found" in exc.message
        
    def test_analysis_not_found(self):
        """Test analysis not found exception."""
        exc = AnalysisNotFoundException("analysis_001")
        assert "Analysis analysis_001 not found" in exc.message
        
    def test_validation_already_exists(self):
        """Test validation already exists exception."""
        exc = ValidationAlreadyExistsException("analysis_001")
        assert "already exists" in exc.message
        
    def test_insufficient_permissions(self):
        """Test insufficient permissions exception."""
        exc = InsufficientPermissionsException("admin")
        assert "Required: admin" in exc.message
        
    def test_rate_limit_exceeded(self):
        """Test rate limit exceeded exception."""
        exc = RateLimitExceededException()
        assert exc.status_code == 429
        assert exc.error_code == "RATE_LIMIT_EXCEEDED"
        
    def test_file_processing_exception(self):
        """Test file processing exception."""
        exc = FileProcessingException("Invalid file", "test.pdf")
        assert exc.details == {"filename": "test.pdf"}
        assert exc.error_code == "FILE_PROCESSING_ERROR"
        
    def test_database_exception(self):
        """Test database exception."""
        exc = DatabaseException("Connection failed")
        assert exc.error_code == "DATABASE_ERROR"
        assert exc.status_code == 500
        
    def test_external_service_exception(self):
        """Test external service exception."""
        exc = ExternalServiceException("Service down", "Redis")
        assert exc.details == {"service_name": "Redis"}
        assert exc.error_code == "EXTERNAL_SERVICE_ERROR"
        
    def test_non_ecg_image_exception(self):
        """Test non-ECG image exception."""
        exc = NonECGImageException()
        assert exc.error_code == "NON_ECG_IMAGE"
        
    def test_multi_pathology_exception(self):
        """Test multi-pathology exception."""
        pathologies = ["AFib", "VTach"]
        exc = MultiPathologyException("Multiple issues", pathologies)
        assert exc.details == {"pathologies": pathologies}
        
    def test_ecg_reader_exception(self):
        """Test ECG reader exception."""
        exc = ECGReaderException("Cannot read file", "EDF")
        assert exc.details == {"file_format": "EDF"}
