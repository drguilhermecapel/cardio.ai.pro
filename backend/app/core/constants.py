"""
Constantes do sistema CardioAI
"""
from enum import Enum


class FileType(str, Enum):
    """Tipos de arquivo suportados."""
    IMAGE = "image"
    PDF = "pdf"
    DICOM = "dicom"
    HL7 = "hl7"
    CSV = "csv"
    EDF = "edf"
    XML = "xml"
    JSON = "json"
    
    
class AnalysisStatus(str, Enum):
    """Status de análise."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    
    
class ClinicalUrgency(str, Enum):
    """Níveis de urgência clínica."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium" 
    NORMAL = "normal"
    LOW = "low"
    
    
class DiagnosisCategory(str, Enum):
    """Categorias de diagnóstico."""
    NORMAL = "normal"
    ARRHYTHMIA = "arrhythmia"
    CONDUCTION = "conduction"
    ISCHEMIA = "ischemia"
    HYPERTROPHY = "hypertrophy"
    OTHER = "other"
    
    
class ErrorCode(str, Enum):
    """Códigos de erro do sistema."""
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    ECG_PROCESSING_ERROR = "ECG_PROCESSING_ERROR"
    

class UserRoles(str, Enum):
    """Papéis de usuário."""
    ADMIN = "admin"
    PHYSICIAN = "physician"
    CARDIOLOGIST = "cardiologist"
    TECHNICIAN = "technician"
    RESEARCHER = "researcher"
    VIEWER = "viewer"
    

class ValidationStatus(str, Enum):
    """Status de validação."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    REQUIRES_REVIEW = "requires_review"
    

class NotificationType(str, Enum):
    """Tipos de notificação."""
    CRITICAL_FINDING = "critical_finding"
    ANALYSIS_COMPLETE = "analysis_complete"
    VALIDATION_REMINDER = "validation_reminder"
    SYSTEM_ALERT = "system_alert"
    

class NotificationPriority(str, Enum):
    """Prioridades de notificação."""
    LOW = "low"
    NORMAL = "normal"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    

class ECGLeads(str, Enum):
    """Derivações de ECG."""
    LEAD_I = "I"
    II = "II"
    III = "III"
    AVR = "aVR"
    AVL = "aVL"
    AVF = "aVF"
    V1 = "V1"
    V2 = "V2"
    V3 = "V3"
    V4 = "V4"
    V5 = "V5"
    V6 = "V6"
    

# Configurações padrão
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
ECG_FILE_EXTENSIONS = {".csv", ".txt", ".xml", ".dat", ".png", ".jpg", ".jpeg", ".edf", ".hea"}
