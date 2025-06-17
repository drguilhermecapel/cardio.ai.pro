"""
Custom exceptions for CardioAI Pro.
"""

from typing import Any


class CardioAIException(Exception):
    """Base exception for CardioAI Pro."""

    def __init__(
        self,
        message: str,
        error_code: str = "CARDIOAI_ERROR",
        status_code: int = 500,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ValidationException(CardioAIException):
    """Validation error exception."""

    def __init__(
        self,
        message: str = "Validation failed",
        errors: list[dict[str, Any]] | None = None,
    ) -> None:
        self.errors = errors or []
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=422,
            details={"errors": self.errors},
        )


class AuthenticationException(CardioAIException):
    """Authentication error exception."""

    def __init__(self, message: str = "Authentication failed") -> None:
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            status_code=401,
        )



class AuthorizationException(CardioAIException):
    """Exception raised for authorization errors."""
    
    def __init__(self, message: str = "Not authorized to access this resource") -> None:
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_ERROR",
            status_code=403
        )

class PermissionDeniedException(CardioAIException):
    """Permission denied exception."""

    def __init__(self, message: str = "Permission denied") -> None:
        super().__init__(
            message=message,
            error_code="PERMISSION_DENIED",
            status_code=403,
        )


class NotFoundException(CardioAIException):
    """Resource not found exception."""

    def __init__(self, message: str = "Resource not found") -> None:
        super().__init__(
            message=message,
            error_code="NOT_FOUND",
            status_code=404,
        )


class ConflictException(CardioAIException):
    """Resource conflict exception."""

    def __init__(self, message: str = "Resource conflict") -> None:
        super().__init__(
            message=message,
            error_code="CONFLICT",
            status_code=409,
        )


class ECGProcessingException(CardioAIException):
    """ECG processing error exception."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            error_code="ECG_PROCESSING_ERROR",
            status_code=422,
            details=details,
        )


class MLModelException(CardioAIException):
    """ML model error exception."""

    def __init__(self, message: str, model_name: str | None = None) -> None:
        details = {"model_name": model_name} if model_name else {}
        super().__init__(
            message=message,
            error_code="ML_MODEL_ERROR",
            status_code=500,
            details=details,
        )


class ValidationNotFoundException(NotFoundException):
    """Validation not found exception."""

    def __init__(self, validation_id: str) -> None:
        super().__init__(f"Validation {validation_id} not found")


class AnalysisNotFoundException(NotFoundException):
    """Analysis not found exception."""

    def __init__(self, analysis_id: str) -> None:
        super().__init__(f"Analysis {analysis_id} not found")


class ValidationAlreadyExistsException(ConflictException):
    """Validation already exists exception."""

    def __init__(self, analysis_id: str) -> None:
        super().__init__(f"Validation for analysis {analysis_id} already exists")


class InsufficientPermissionsException(PermissionDeniedException):
    """Insufficient permissions exception."""

    def __init__(self, required_permission: str) -> None:
        super().__init__(f"Insufficient permissions. Required: {required_permission}")


class RateLimitExceededException(CardioAIException):
    """Rate limit exceeded exception."""

    def __init__(self, message: str = "Rate limit exceeded") -> None:
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            status_code=429,
        )


class FileProcessingException(CardioAIException):
    """File processing error exception."""

    def __init__(self, message: str, filename: str | None = None) -> None:
        details = {"filename": filename} if filename else {}
        super().__init__(
            message=message,
            error_code="FILE_PROCESSING_ERROR",
            status_code=422,
            details=details,
        )


class DatabaseException(CardioAIException):
    """Database error exception."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            status_code=500,
        )


class ExternalServiceException(CardioAIException):
    """External service error exception."""

    def __init__(self, message: str, service_name: str | None = None) -> None:
        details = {"service_name": service_name} if service_name else {}
        super().__init__(
            message=message,
            error_code="EXTERNAL_SERVICE_ERROR",
            status_code=502,
            details=details,
        )


class NonECGImageException(CardioAIException):
    """Non-ECG image detected exception with contextual response."""

    def __init__(
        self,
        message: str,
        category: str,
        contextual_response: dict[str, Any],
        confidence: float = 0.0,
    ) -> None:
        super().__init__(
            message=message,
            error_code="NON_ECG_IMAGE_DETECTED",
            status_code=422,
            details={
                "category": category,
                "confidence": confidence,
                "contextual_response": contextual_response,
            },
        )


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
