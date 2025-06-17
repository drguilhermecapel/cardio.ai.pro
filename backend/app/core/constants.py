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
    PATIENT = "patient"
    NURSE = "nurse"


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
    AXIS_DEVIATION = "axis_deviation"
    REPOLARIZATION = "repolarization"
    PACEMAKER = "pacemaker"
    BUNDLE_BRANCH_BLOCK = "bundle_branch_block"
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


ECG_FILE_EXTENSIONS = {".csv", ".txt", ".xml", ".dat", ".png", ".jpg", ".jpeg", ".edf", ".hea"}


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
    ECG_ANALYSIS_COMPLETE = "ecg_analysis_complete"  # Adicionado
    VALIDATION_REMINDER = "validation_reminder"
    VALIDATION_REQUIRED = "validation_required"  # Adicionado
    VALIDATION_ASSIGNED = "validation_assigned"  # Adicionado
    VALIDATION_COMPLETE = "validation_complete"  # Adicionado
    QUALITY_ALERT = "quality_alert"
    SYSTEM_ALERT = "system_alert"
    APPOINTMENT_REMINDER = "appointment_reminder"
    REPORT_READY = "report_ready"
    URGENT_VALIDATION = "urgent_validation"  # Adicionado
    NO_VALIDATOR_AVAILABLE = "no_validator_available"  # Adicionado


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
    "Acute Myocardial Infarction",
    "Third Degree AV Block",
    "Ventricular Flutter",
    "Polymorphic Ventricular Tachycardia",
    "VF",  # Abbreviation
    "VT",  # Abbreviation
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
    "I48.0": "Atrial Fibrillation",
    "I48.3": "Atrial Flutter",
    "I44.0": "First Degree AV Block",
    "I44.1": "Second Degree AV Block",
    "I45.0": "Right Bundle Branch Block",
    "I45.2": "Left Bundle Branch Block",
    "I51.7": "Cardiomegaly",
    "I25.10": "Atherosclerotic Heart Disease",
}

SCP_ECG_CATEGORIES = {
    "NORMAL": ["NORM"],
    "ARRHYTHMIA": ["AFIB", "AFLT", "SVTAC", "VTAC", "BIGU", "TRIGU", "PVC", "PAC"],
    "CONDUCTION_DISORDER": ["AVB1", "AVB2", "AVB3", "RBBB", "LBBB", "LAFB", "LPFB", "WPW"],
    "ISCHEMIA": ["STEMI", "NSTEMI", "UAP", "ISCH", "QWAVE", "TWAVE"],
    "HYPERTROPHY": ["LVH", "RVH", "LAE", "RAE", "BIAE"],
    "AXIS_DEVIATION": ["LAD", "RAD", "EAXIS"],
    "REPOLARIZATION": ["LQTS", "SQTS", "EARLY", "TWAV", "UWAVE"],
    "PACEMAKER": ["PACE", "PACED", "PACEF"],
    "BUNDLE_BRANCH_BLOCK": ["RBBB", "LBBB", "IVCD", "BIFB"],
    "OTHER": ["LOWV", "ARTIF", "NOISE", "POOR"],
}

SCP_ECG_PERFORMANCE_TARGETS = {
    "NORMAL": {"sensitivity": 0.99, "specificity": 0.95, "npv": 0.99},
    "ARRHYTHMIA": {"sensitivity": 0.95, "specificity": 0.90, "ppv": 0.85},
    "CONDUCTION_DISORDER": {"sensitivity": 0.90, "specificity": 0.95, "ppv": 0.90},
    "ISCHEMIA": {"sensitivity": 0.98, "specificity": 0.85, "ppv": 0.80},
}
