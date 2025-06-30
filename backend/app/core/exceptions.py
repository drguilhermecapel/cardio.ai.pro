"""
Sistema completo de exceções customizadas para CardioAI Pro
"""
from typing import Dict, Any, Optional, Union, List
from fastapi import HTTPException


class UnauthorizedException(Exception):
    """Exceção para usuário não autorizado"""
    
    def __init__(self, message: str = "Não autorizado"):
        super().__init__(message)
        self.message = message
        self.status_code = 401

class CardioAIException(Exception):
    """Exceção base para o sistema CardioAI"""
    
    def __init__(
        self,
        message: str,
        error_code: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Converte exceção para dicionário"""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "status_code": self.status_code
        }
    
    def to_http_exception(self) -> HTTPException:
        """Converte para HTTPException do FastAPI"""
        return HTTPException(
            status_code=self.status_code,
            detail=self.to_dict()
        )


class ECGNotFoundException(CardioAIException):
    """Exceção quando ECG não é encontrado"""
    
    def __init__(
        self,
        message: str = "Análise de ECG não encontrada",
        ecg_id: Optional[Union[str, int]] = None
    ):
        details = {}
        if ecg_id is not None:
            details["ecg_id"] = str(ecg_id)
            message = f"{message}: ID {ecg_id}"
        
        super().__init__(
            message=message,
            error_code="ECG_NOT_FOUND",
            status_code=404,
            details=details
        )


class ECGProcessingException(CardioAIException):
    """Exceção para erros no processamento de ECG com máxima flexibilidade"""
    
    def __init__(
        self,
        message: str,
        *args,
        ecg_id: Optional[Union[str, int]] = None,
        details: Optional[Dict[str, Any]] = None,
        detail: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None,
        status_code: int = 422,
        **kwargs
    ):
        # Máxima flexibilidade para aceitar diferentes formatos
        error_details = {}
        
        # Aceitar 'details' ou 'detail'
        if details:
            error_details.update(details)
        if detail:
            error_details.update(detail)
        
        # Adicionar ecg_id se fornecido
        if ecg_id is not None:
            error_details['ecg_id'] = str(ecg_id)
        
        # Adicionar kwargs extras
        for key, value in kwargs.items():
            if key not in ['details', 'detail', 'error_code', 'status_code']:
                error_details[key] = value
        
        # Se houver args adicionais, adicionar como 'additional_info'
        if args:
            error_details['additional_info'] = args
        
        super().__init__(
            message=message,
            error_code=error_code or "ECG_PROCESSING_ERROR",
            status_code=status_code,
            details=error_details
        )
        self.ecg_id = ecg_id


class ValidationException(CardioAIException):
    """Exceção para erros de validação"""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Any = None,
        validation_errors: Optional[Dict[str, Any]] = None
    ):
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)
        if validation_errors:
            details["validation_errors"] = validation_errors
        
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=400,
            details=details
        )


class AuthenticationException(CardioAIException):
    """Exceção para erros de autenticação"""
    
    def __init__(self, message: str = "Falha na autenticação"):
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            status_code=401
        )


class AuthorizationException(CardioAIException):
    """Exceção para erros de autorização"""
    
    def __init__(self, message: str = "Acesso negado"):
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_ERROR",
            status_code=403
        )


class NotFoundException(CardioAIException):
    """Exceção genérica para recursos não encontrados"""
    
    def __init__(self, resource: str, identifier: Optional[str] = None):
        message = f"{resource} não encontrado"
        if identifier:
            message = f"{message}: {identifier}"
        
        super().__init__(
            message=message,
            error_code="NOT_FOUND",
            status_code=404,
            details={"resource": resource, "identifier": identifier}
        )


class ConflictException(CardioAIException):
    """Exceção para conflitos de dados"""
    
    def __init__(self, message: str):
        super().__init__(
            message=message,
            error_code="CONFLICT",
            status_code=409
        )


class PermissionDeniedException(CardioAIException):
    """Exceção para negação de permissão"""
    
    def __init__(self, message: str = "Permissão negada"):
        super().__init__(
            message=message,
            error_code="PERMISSION_DENIED",
            status_code=403
        )


class ECGValidationException(ValidationException):
    """Exceção específica para validação de ECG"""
    
    def __init__(self, validation_id: str):
        super().__init__(f"Validation {validation_id} not found")


class AnalysisNotFoundException(NotFoundException):
    """Análise não encontrada"""
    
    def __init__(self, analysis_id: str):
        super().__init__("Analysis", analysis_id)


class ValidationAlreadyExistsException(ConflictException):
    """Validação já existe"""
    
    def __init__(self, analysis_id: str):
        super().__init__(f"Validation for analysis {analysis_id} already exists")


class InsufficientPermissionsException(PermissionDeniedException):
    """Permissões insuficientes"""
    
    def __init__(self, required_permission: str):
        super().__init__(f"Insufficient permissions. Required: {required_permission}")


class RateLimitExceededException(CardioAIException):
    """Rate limit excedido"""
    
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, "RATE_LIMIT_EXCEEDED", 429)


class FileProcessingException(CardioAIException):
    """Erro no processamento de arquivo"""
    
    def __init__(self, message: str, filename: Optional[str] = None):
        details = {"filename": filename} if filename else {}
        super().__init__(message, "FILE_PROCESSING_ERROR", 422, details)


class DatabaseException(CardioAIException):
    """Erro de banco de dados"""
    
    def __init__(self, message: str):
        super().__init__(message, "DATABASE_ERROR", 500)


class ExternalServiceException(CardioAIException):
    """Erro de serviço externo"""
    
    def __init__(self, message: str, service_name: Optional[str] = None):
        details = {"service_name": service_name} if service_name else {}
        super().__init__(message, "EXTERNAL_SERVICE_ERROR", 502, details)


class NonECGImageException(CardioAIException):
    """Imagem não é um ECG"""
    
    def __init__(self):
        super().__init__("Uploaded image is not an ECG", "NON_ECG_IMAGE", 422)


class MultiPathologyException(CardioAIException):
    """Exceção para erros de multi-patologia"""
    
    def __init__(self, message: str, pathologies: Optional[List[str]] = None):
        details = {"pathologies": pathologies} if pathologies else {}
        super().__init__(message, "MULTI_PATHOLOGY_ERROR", 500, details)


class ECGReaderException(CardioAIException):
    """Exceção para erros de leitura de ECG"""
    
    def __init__(self, message: str, file_format: Optional[str] = None):
        details = {"file_format": file_format} if file_format else {}
        super().__init__(message, "ECG_READER_ERROR", 422, details)
