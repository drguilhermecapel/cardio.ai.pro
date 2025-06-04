"""
Custom exceptions for CardioAI Pro.
"""

from typing import Any
from app.services.i18n_service import i18n_service


class CardioAIException(Exception):
    """Base exception for CardioAI Pro."""

    def __init__(
        self,
        message_key: str,
        lang: str = "en",
        error_code: str = "CARDIOAI_ERROR",
        status_code: int = 500,
        details: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.message = i18n_service.translate(message_key, lang, **kwargs)
        self.message_key = message_key
        self.lang = lang
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ValidationException(CardioAIException):
    """Validation error exception."""

    def __init__(
        self,
        message_key: str = "errors.validation_error",
        lang: str = "en",
        errors: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        self.errors = errors or []
        super().__init__(
            message_key=message_key,
            lang=lang,
            error_code="VALIDATION_ERROR",
            status_code=422,
            details={"errors": self.errors},
            **kwargs,
        )


class AuthenticationException(CardioAIException):
    """Authentication error exception."""

    def __init__(
        self, 
        message_key: str = "errors.authentication_error",
        lang: str = "en",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            message_key=message_key,
            lang=lang,
            error_code="AUTHENTICATION_ERROR",
            status_code=401,
            **kwargs,
        )


class PermissionDeniedException(CardioAIException):
    """Permission denied exception."""

    def __init__(
        self, 
        message_key: str = "errors.permission_denied",
        lang: str = "en",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            message_key=message_key,
            lang=lang,
            error_code="PERMISSION_DENIED",
            status_code=403,
            **kwargs,
        )


class NotFoundException(CardioAIException):
    """Resource not found exception."""

    def __init__(
        self, 
        message_key: str = "errors.not_found",
        lang: str = "en",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            message_key=message_key,
            lang=lang,
            error_code="NOT_FOUND",
            status_code=404,
            **kwargs,
        )


class ConflictException(CardioAIException):
    """Resource conflict exception."""

    def __init__(
        self, 
        message_key: str = "errors.conflict",
        lang: str = "en",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            message_key=message_key,
            lang=lang,
            error_code="CONFLICT",
            status_code=409,
            **kwargs,
        )


class ECGProcessingException(CardioAIException):
    """ECG processing error exception."""

    def __init__(
        self, 
        message_key: str = "errors.ecg_processing_error",
        lang: str = "en",
        details: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            message_key=message_key,
            lang=lang,
            error_code="ECG_PROCESSING_ERROR",
            status_code=422,
            details=details,
            **kwargs,
        )


class MLModelException(CardioAIException):
    """ML model error exception."""

    def __init__(
        self, 
        message_key: str = "errors.ml_model_error",
        lang: str = "en",
        model_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        details = {"model_name": model_name} if model_name else {}
        super().__init__(
            message_key=message_key,
            lang=lang,
            error_code="ML_MODEL_ERROR",
            status_code=500,
            details=details,
            **kwargs,
        )


class ValidationNotFoundException(NotFoundException):
    """Validation not found exception."""

    def __init__(self, validation_id: str, lang: str = "en") -> None:
        super().__init__(
            message_key="errors.validation_not_found",
            lang=lang,
            validation_id=validation_id,
        )


class AnalysisNotFoundException(NotFoundException):
    """Analysis not found exception."""

    def __init__(self, analysis_id: str, lang: str = "en") -> None:
        super().__init__(
            message_key="errors.analysis_not_found",
            lang=lang,
            analysis_id=analysis_id,
        )


class ValidationAlreadyExistsException(ConflictException):
    """Validation already exists exception."""

    def __init__(self, analysis_id: str, lang: str = "en") -> None:
        super().__init__(
            message_key="errors.validation_already_exists",
            lang=lang,
            analysis_id=analysis_id,
        )


class InsufficientPermissionsException(PermissionDeniedException):
    """Insufficient permissions exception."""

    def __init__(self, required_permission: str, lang: str = "en") -> None:
        super().__init__(
            message_key="errors.insufficient_permissions",
            lang=lang,
            required_permission=required_permission,
        )


class RateLimitExceededException(CardioAIException):
    """Rate limit exceeded exception."""

    def __init__(
        self, 
        message_key: str = "errors.rate_limit_exceeded",
        lang: str = "en",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            message_key=message_key,
            lang=lang,
            error_code="RATE_LIMIT_EXCEEDED",
            status_code=429,
            **kwargs,
        )


class FileProcessingException(CardioAIException):
    """File processing error exception."""

    def __init__(
        self, 
        message_key: str = "errors.file_processing_error",
        lang: str = "en",
        filename: str | None = None,
        **kwargs: Any,
    ) -> None:
        details = {"filename": filename} if filename else {}
        super().__init__(
            message_key=message_key,
            lang=lang,
            error_code="FILE_PROCESSING_ERROR",
            status_code=422,
            details=details,
            **kwargs,
        )


class DatabaseException(CardioAIException):
    """Database error exception."""

    def __init__(
        self, 
        message_key: str = "errors.database_error",
        lang: str = "en",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            message_key=message_key,
            lang=lang,
            error_code="DATABASE_ERROR",
            status_code=500,
            **kwargs,
        )


class ExternalServiceException(CardioAIException):
    """External service error exception."""

    def __init__(
        self, 
        message_key: str = "errors.external_service_error",
        lang: str = "en",
        service_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        details = {"service_name": service_name} if service_name else {}
        super().__init__(
            message_key=message_key,
            lang=lang,
            error_code="EXTERNAL_SERVICE_ERROR",
            status_code=502,
            details=details,
            **kwargs,
        )
