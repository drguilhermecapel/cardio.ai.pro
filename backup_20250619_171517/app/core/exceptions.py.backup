# Substitua a classe ValidationException em backend/app/core/exceptions.py por:

class ValidationException(CardioAIException):
    """Validation exception."""

    def __init__(
        self,
        message: str = "Validation error",
        error_code: str = "VALIDATION_ERROR",
        status_code: int = 422,
        field: str | None = None,
        **kwargs
    ):
        super().__init__(message, error_code, status_code)
        self.field = field
        # Store any additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
