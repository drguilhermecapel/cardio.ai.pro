"""
Constants for CardioAI Pro.
"""

from enum import Enum


class UserRoles(str, Enum):
    """User roles."""

    ADMIN = "admin"
    PHYSICIAN = "physician"
    CARDIOLOGIST = "cardiologist"
    TECHNICIAN = "technician"
    RESEARCHER = "researcher"
    VIEWER = "viewer"


class AnalysisStatus(str, Enum):
    """Analysis status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ValidationStatus(str, Enum):
    """Validation status."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    REQUIRES_REVIEW = "requires_review"


class ECGLeads(str, Enum):
    """ECG leads."""

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


class DiagnosisCategory(str, Enum):
    """Diagnosis categories."""

    NORMAL = "normal"
    ARRHYTHMIA = "arrhythmia"
    CONDUCTION_DISORDER = "conduction_disorder"
    ISCHEMIA = "ischemia"
    HYPERTROPHY = "hypertrophy"
    OTHER = "other"


class ClinicalUrgency(str, Enum):
    """Clinical urgency levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FileType(str, Enum):
    """Supported file types."""

    PDF = "application/pdf"
    JPEG = "image/jpeg"
    PNG = "image/png"
    DICOM = "application/dicom"
    XML = "application/xml"
    TXT = "text/plain"


class NotificationChannel(str, Enum):
    """Notification channels."""

    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    IN_APP = "in_app"
    WEBHOOK = "webhook"
    PHONE_CALL = "phone_call"


class NotificationPriority(str, Enum):
    """Notification priority levels."""

    LOW = "low"
    NORMAL = "normal"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class NotificationType(str, Enum):
    """Notification types."""

    CRITICAL_FINDING = "critical_finding"
    ANALYSIS_COMPLETE = "analysis_complete"
    VALIDATION_REMINDER = "validation_reminder"
    QUALITY_ALERT = "quality_alert"
    SYSTEM_ALERT = "system_alert"
    APPOINTMENT_REMINDER = "appointment_reminder"
    REPORT_READY = "report_ready"


class AuditEventType(str, Enum):
    """Audit event types."""

    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    ANALYSIS_CREATED = "analysis_created"
    VALIDATION_SUBMITTED = "validation_submitted"
    REPORT_GENERATED = "report_generated"
    SYSTEM_ERROR = "system_error"


class ModelType(str, Enum):
    """ML model types."""

    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    DETECTION = "detection"
    REGRESSION = "regression"


class ModelStatus(str, Enum):
    """ML model status."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    TRAINING = "training"
    DEPRECATED = "deprecated"


ECG_SAMPLE_RATES = [250, 500, 1000]
ECG_STANDARD_DURATION = 10  # seconds
ECG_MINIMUM_DURATION = 5  # seconds
ECG_MAXIMUM_DURATION = 60  # seconds

HEART_RATE_NORMAL_RANGE = (60, 100)  # bpm
QT_INTERVAL_NORMAL_RANGE = (350, 450)  # ms
PR_INTERVAL_NORMAL_RANGE = (120, 200)  # ms
QRS_DURATION_NORMAL_RANGE = (80, 120)  # ms

CRITICAL_CONDITIONS = [
    "Ventricular Fibrillation",
    "Ventricular Tachycardia",
    "Complete Heart Block",
    "STEMI",
    "Asystole",
    "Torsades de Pointes",
]

ICD10_CODES = {
    "I47.2": "Ventricular Tachycardia",
    "I49.01": "Ventricular Fibrillation",
    "I44.2": "Complete Heart Block",
    "I21.9": "Acute Myocardial Infarction",
    "I49.9": "Cardiac Arrhythmia",
    "I25.2": "Old Myocardial Infarction",
    "I42.0": "Dilated Cardiomyopathy",
    "I42.1": "Obstructive Hypertrophic Cardiomyopathy",
}

ANVISA_RETENTION_YEARS = 7
FDA_CFR_PART_11_REQUIRED = True
LGPD_COMPLIANCE_REQUIRED = True
HIPAA_COMPLIANCE_REQUIRED = True

MAX_CONCURRENT_ANALYSES = 10
MAX_FILE_SIZE_MB = 100
MAX_BATCH_SIZE = 50
CACHE_TTL_SECONDS = 3600

RATE_LIMIT_PER_MINUTE = 100
RATE_LIMIT_PER_HOUR = 1000
RATE_LIMIT_PER_DAY = 10000
