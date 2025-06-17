"""
CardioAI Pro exception classes.

This module defines custom exceptions used throughout the application.
"""

from typing import Any, Dict, Optional


class CardioAIException(Exception):
    """Base exception for all CardioAI custom exceptions."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "CARDIOAI_ERROR",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize CardioAI exception.
        
        Args:
            message: Error message
            error_code: Internal error code
            status_code: HTTP status code
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
    
    def __str__(self) -> str:
        """Return string representation."""
        return self.message


class ValidationException(CardioAIException):
    """Validation exception."""

    def __init__(
        self,
        message: str = "Validation error",
        validation_errors: list[dict] = None,
        **kwargs
    ) -> None:
        """Initialize validation exception.
        
        Args:
            message: Error message
            validation_errors: List of validation errors
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(message, "VALIDATION_ERROR", 422)
        self.validation_errors = validation_errors or []
        # Store any additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


class AuthenticationException(CardioAIException):
    """Authentication exception."""
    
    def __init__(self, message: str = "Could not validate credentials") -> None:
        """Initialize authentication exception."""
        super().__init__(message, "AUTHENTICATION_ERROR", 401)


class AuthorizationException(CardioAIException):
    """Authorization exception."""
    
    def __init__(self, message: str = "Not authorized to access this resource") -> None:
        """Initialize authorization exception."""
        super().__init__(message, "AUTHORIZATION_ERROR", 403)


class NotFoundException(CardioAIException):
    """Not found exception."""
    
    def __init__(self, message: str = "Resource not found") -> None:
        """Initialize not found exception."""
        super().__init__(message, "NOT_FOUND", 404)


class ECGProcessingException(CardioAIException):
    """ECG processing exception."""
    
    def __init__(self, message: str, ecg_id: str = None) -> None:
        """Initialize ECG processing exception."""
        details = {"ecg_id": ecg_id} if ecg_id else {}
        super().__init__(message, "ECG_PROCESSING_ERROR", 422, details)
        self.ecg_id = ecg_id


# Exceções Not Found específicas
class ECGNotFoundException(NotFoundException):
    """ECG not found exception."""
    def __init__(self, ecg_id: str) -> None:
        super().__init__(f"ECG {ecg_id} not found")


class PatientNotFoundException(NotFoundException):
    """Patient not found exception."""
    def __init__(self, patient_id: str) -> None:
        super().__init__(f"Patient {patient_id} not found")


class UserNotFoundException(NotFoundException):
    """User not found exception."""
    def __init__(self, user_id: str) -> None:
        super().__init__(f"User {user_id} not found")


class ConflictException(CardioAIException):
    """Exception for conflict errors."""
    def __init__(self, message: str) -> None:
        super().__init__(message, "CONFLICT_ERROR", 409)


class PermissionDeniedException(CardioAIException):
    """Exception for permission denied errors."""
    def __init__(self, message: str) -> None:
        super().__init__(message, "PERMISSION_DENIED", 403)


class MLModelException(CardioAIException):
    """ML Model exception."""
    def __init__(self, message: str, model_name: str = None) -> None:
        details = {"model_name": model_name} if model_name else {}
        super().__init__(message, "ML_MODEL_ERROR", 500, details)


class ValidationNotFoundException(NotFoundException):
    """Validation not found."""
    def __init__(self, validation_id: str) -> None:
        super().__init__(f"Validation {validation_id} not found")


class AnalysisNotFoundException(NotFoundException):
    """Analysis not found."""
    def __init__(self, analysis_id: str) -> None:
        super().__init__(f"Analysis {analysis_id} not found")


class ValidationAlreadyExistsException(ConflictException):
    """Validation already exists."""
    def __init__(self, analysis_id: str) -> None:
        super().__init__(f"Validation for analysis {analysis_id} already exists")


class InsufficientPermissionsException(PermissionDeniedException):
    """Insufficient permissions."""
    def __init__(self, required_permission: str) -> None:
        super().__init__(f"Insufficient permissions. Required: {required_permission}")


class RateLimitExceededException(CardioAIException):
    """Rate limit exceeded."""
    def __init__(self, message: str = "Rate limit exceeded") -> None:
        super().__init__(message, "RATE_LIMIT_EXCEEDED", 429)


class FileProcessingException(CardioAIException):
    """File processing error."""
    def __init__(self, message: str, filename: str = None) -> None:
        details = {"filename": filename} if filename else {}
        super().__init__(message, "FILE_PROCESSING_ERROR", 422, details)


class DatabaseException(CardioAIException):
    """Database error."""
    def __init__(self, message: str) -> None:
        super().__init__(message, "DATABASE_ERROR", 500)


class ExternalServiceException(CardioAIException):
    """External service error."""
    def __init__(self, message: str, service_name: str = None) -> None:
        details = {"service_name": service_name} if service_name else {}
        super().__init__(message, "EXTERNAL_SERVICE_ERROR", 502, details)


class NonECGImageException(CardioAIException):
    """Non-ECG image detected."""
    def __init__(self) -> None:
        super().__init__("Uploaded image is not an ECG", "NON_ECG_IMAGE", 422)


class MultiPathologyException(CardioAIException):
    """Exception for multi-pathology service errors."""
    
    def __init__(self, message: str, pathologies: list[str] | None = None) -> None:
        details = {"pathologies": pathologies} if pathologies else {}
        super().__init__(
            message=message,
            error_code="MULTI_PATHOLOGY_ERROR",
            status_code=500,
            details=details,
        )


class ECGReaderException(CardioAIException):
    """Exception for ECG file reading errors."""
    
    def __init__(self, message: str, file_format: str | None = None) -> None:
        details = {"file_format": file_format} if file_format else {}
        super().__init__(
            message=message,
            error_code="ECG_READER_ERROR",
            status_code=422,
            details=details,
        )
