from typing import Any, Optional

class BaseException(Exception):
    """Exceção base do sistema CardioAI"""
    def __init__(self, message: str, status_code: int = 400, details: Optional[Any] = None):
        self.message = message
        self.status_code = status_code
        self.details = details
        super().__init__(self.message)
    
    def __str__(self):
        return self.message
    
    def to_dict(self):
        """Converte a exceção para dicionário"""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "status_code": self.status_code,
            "details": self.details
        }

class ECGNotFoundException(BaseException):
    """Exceção quando ECG não é encontrado"""
    def __init__(self, ecg_id: int):
        super().__init__(
            message=f"ECG analysis with ID {ecg_id} not found",
            status_code=404,
            details={"ecg_id": ecg_id}
        )
        self.ecg_id = ecg_id

class ValidationException(BaseException):
    """Exceção de validação de dados"""
    def __init__(self, field: str, message: str):
        super().__init__(
            message=f"Validation error in field '{field}': {message}",
            status_code=422,
            details={"field": field, "validation_error": message}
        )
        self.field = field

class ProcessingException(BaseException):
    """Exceção durante processamento de ECG"""
    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(
            message=f"ECG processing error: {message}",
            status_code=500,
            details=details
        )

class PatientNotFoundException(BaseException):
    """Exceção quando paciente não é encontrado"""
    def __init__(self, patient_id: int):
        super().__init__(
            message=f"Patient with ID {patient_id} not found",
            status_code=404,
            details={"patient_id": patient_id}
        )
        self.patient_id = patient_id

class InvalidECGDataException(BaseException):
    """Exceção para dados de ECG inválidos"""
    def __init__(self, message: str, data_issues: Optional[list] = None):
        super().__init__(
            message=f"Invalid ECG data: {message}",
            status_code=400,
            details={"data_issues": data_issues or []}
        )

class AnalysisInProgressException(BaseException):
    """Exceção quando análise já está em progresso"""
    def __init__(self, ecg_id: int):
        super().__init__(
            message=f"Analysis for ECG {ecg_id} is already in progress",
            status_code=409,
            details={"ecg_id": ecg_id}
        )
        self.ecg_id = ecg_id

class UnauthorizedException(BaseException):
    """Exceção de acesso não autorizado"""
    def __init__(self, message: str = "Unauthorized access"):
        super().__init__(
            message=message,
            status_code=401
        )

class InsufficientPermissionsException(BaseException):
    """Exceção de permissões insuficientes"""
    def __init__(self, required_permission: str):
        super().__init__(
            message=f"Insufficient permissions. Required: {required_permission}",
            status_code=403,
            details={"required_permission": required_permission}
        )

