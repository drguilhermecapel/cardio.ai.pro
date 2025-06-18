"""
Core Exceptions
Custom exception classes for CardioAI Pro
"""

from typing import Any, Dict, Optional


class CardioAIException(Exception):
    """Base exception for CardioAI Pro"""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary"""
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details
        }


class DatabaseException(CardioAIException):
    """Database-related exceptions"""
    pass


class ResourceNotFoundException(CardioAIException):
    """Resource not found exception"""
    
    def __init__(self, message: str, resource_type: Optional[str] = None, resource_id: Optional[str] = None):
        details = {}
        if resource_type:
            details["resource_type"] = resource_type
        if resource_id:
            details["resource_id"] = resource_id
        super().__init__(message, "RESOURCE_NOT_FOUND", details)


class ValidationException(CardioAIException):
    """Validation exception"""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None):
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)
        super().__init__(message, "VALIDATION_ERROR", details)


class AuthenticationException(CardioAIException):
    """Authentication exception"""
    
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, "AUTHENTICATION_FAILED")


class AuthorizationException(CardioAIException):
    """Authorization exception"""
    
    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(message, "AUTHORIZATION_FAILED")


class ECGProcessingException(CardioAIException):
    """ECG processing exception"""
    
    def __init__(self, message: str, stage: Optional[str] = None, ecg_id: Optional[str] = None):
        details = {}
        if stage:
            details["processing_stage"] = stage
        if ecg_id:
            details["ecg_id"] = ecg_id
        super().__init__(message, "ECG_PROCESSING_ERROR", details)


class MLModelException(CardioAIException):
    """Machine learning model exception"""
    
    def __init__(self, message: str, model_name: Optional[str] = None, model_version: Optional[str] = None):
        details = {}
        if model_name:
            details["model_name"] = model_name
        if model_version:
            details["model_version"] = model_version
        super().__init__(message, "ML_MODEL_ERROR", details)


class FileOperationException(CardioAIException):
    """File operation exception"""
    
    def __init__(self, message: str, file_path: Optional[str] = None, operation: Optional[str] = None):
        details = {}
        if file_path:
            details["file_path"] = file_path
        if operation:
            details["operation"] = operation
        super().__init__(message, "FILE_OPERATION_ERROR", details)


class ConfigurationException(CardioAIException):
    """Configuration exception"""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        details = {}
        if config_key:
            details["config_key"] = config_key
        super().__init__(message, "CONFIGURATION_ERROR", details)


class ExternalServiceException(CardioAIException):
    """External service exception"""
    
    def __init__(self, message: str, service_name: Optional[str] = None, status_code: Optional[int] = None):
        details = {}
        if service_name:
            details["service_name"] = service_name
        if status_code:
            details["status_code"] = status_code
        super().__init__(message, "EXTERNAL_SERVICE_ERROR", details)


class RateLimitException(CardioAIException):
    """Rate limit exceeded exception"""
    
    def __init__(self, message: str = "Rate limit exceeded", retry_after: Optional[int] = None):
        details = {}
        if retry_after:
            details["retry_after_seconds"] = retry_after
        super().__init__(message, "RATE_LIMIT_EXCEEDED", details)


class DataIntegrityException(CardioAIException):
    """Data integrity exception"""
    
    def __init__(self, message: str, entity_type: Optional[str] = None, entity_id: Optional[str] = None):
        details = {}
        if entity_type:
            details["entity_type"] = entity_type
        if entity_id:
            details["entity_id"] = entity_id
        super().__init__(message, "DATA_INTEGRITY_ERROR", details)


class ConcurrencyException(CardioAIException):
    """Concurrency/locking exception"""
    
    def __init__(self, message: str = "Resource is locked by another process"):
        super().__init__(message, "CONCURRENCY_ERROR")


class MemoryException(CardioAIException):
    """Memory-related exception"""
    
    def __init__(self, message: str, available_memory: Optional[int] = None, required_memory: Optional[int] = None):
        details = {}
        if available_memory:
            details["available_memory_mb"] = available_memory / (1024 * 1024)
        if required_memory:
            details["required_memory_mb"] = required_memory / (1024 * 1024)
        super().__init__(message, "MEMORY_ERROR", details)


# Error code mappings for HTTP status codes
ERROR_CODE_TO_STATUS = {
    "RESOURCE_NOT_FOUND": 404,
    "VALIDATION_ERROR": 400,
    "AUTHENTICATION_FAILED": 401,
    "AUTHORIZATION_FAILED": 403,
    "RATE_LIMIT_EXCEEDED": 429,
    "CONCURRENCY_ERROR": 409,
    "MEMORY_ERROR": 507,
    "CONFIGURATION_ERROR": 500,
    "DATABASE_ERROR": 500,
    "ECG_PROCESSING_ERROR": 422,
    "ML_MODEL_ERROR": 500,
    "FILE_OPERATION_ERROR": 500,
    "EXTERNAL_SERVICE_ERROR": 502,
    "DATA_INTEGRITY_ERROR": 422,
}


def get_http_status_code(exception: CardioAIException) -> int:
    """Get HTTP status code for exception"""
    return ERROR_CODE_TO_STATUS.get(exception.error_code, 500)


__all__ = [
    "CardioAIException",
    "DatabaseException",
    "ResourceNotFoundException",
    "ValidationException",
    "AuthenticationException",
    "AuthorizationException",
    "ECGProcessingException",
    "MLModelException",
    "FileOperationException",
    "ConfigurationException",
    "ExternalServiceException",
    "RateLimitException",
    "DataIntegrityException",
    "ConcurrencyException",
    "MemoryException",
    "get_http_status_code",
]
