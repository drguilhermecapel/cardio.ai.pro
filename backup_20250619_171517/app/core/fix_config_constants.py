"""
Fix configuration and constants issues
"""

from pathlib import Path

# Fix config.py to add missing attributes
CONFIG_ADDITIONS = """
    # CORS Settings
    BACKEND_CORS_ORIGINS: list = ["http://localhost:3000", "http://localhost:8000"]
    
    # Database Pool Settings
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20
    DATABASE_POOL_TIMEOUT: int = 30
    
    # Security Settings
    BCRYPT_ROUNDS: int = 12
    
    # Email Settings
    SMTP_HOST: Optional[str] = None
    SMTP_PORT: int = 587
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    
    # AWS Settings
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "us-east-1"
    S3_BUCKET_NAME: Optional[str] = None
    
    # Medical Compliance Settings
    ENABLE_AUDIT_TRAIL: bool = True
    ENABLE_FDA_COMPLIANCE: bool = True
    HIPAA_COMPLIANT: bool = True
    
    # ML Model Settings
    MODEL_CACHE_DIR: str = "/tmp/models"
    MAX_MODEL_CACHE_SIZE_GB: int = 10
    
    # Processing Settings
    MAX_SIGNAL_LENGTH_MINUTES: int = 60
    DEFAULT_SAMPLING_RATE: int = 500
    ENABLE_GPU: bool = False
"""

# Fix constants.py to add missing enums
CONSTANTS_ADDITIONS = '''
# Add missing enum values
class UserRoles(str, Enum):
    """User role types"""
    ADMIN = "admin"
    PHYSICIAN = "physician"
    CARDIOLOGIST = "cardiologist"
    TECHNICIAN = "technician"
    PATIENT = "patient"
    NURSE = "nurse"

class NotificationType(str, Enum):
    """Notification types"""
    ANALYSIS_READY = "analysis_ready"
    VALIDATION_REQUIRED = "validation_required"
    CRITICAL_FINDING = "critical_finding"
    QUALITY_ISSUE = "quality_issue"
    SYSTEM_ALERT = "system_alert"
    ECG_ANALYSIS_COMPLETE = "ecg_analysis_complete"
    VALIDATION_ASSIGNED = "validation_assigned"
    VALIDATION_COMPLETE = "validation_complete"
    URGENT_VALIDATION = "urgent_validation"
    NO_VALIDATOR_AVAILABLE = "no_validator_available"

# Ensure these are properly exported
__all__ = [
    "FileType", "AnalysisStatus", "ClinicalUrgency", "DiagnosisCode",
    "UserRoles", "NotificationType", "ValidationStatus", "DeviceType",
    "QualityLevel", "ECG_CONSTANTS", "MEDICAL_CONSTANTS", "SYSTEM_CONSTANTS"
]
'''

# Fix exceptions.py to add missing exception types
EXCEPTIONS_ADDITIONS = '''
class AuthorizationException(CardioAIException):
    """Raised when user lacks authorization"""
    def __init__(self, message: str = "Not authorized", resource: str = None):
        super().__init__(message)
        self.resource = resource

class RateLimitException(CardioAIException):
    """Raised when rate limit is exceeded"""
    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = None):
        super().__init__(message)
        self.retry_after = retry_after

class ConfigurationException(CardioAIException):
    """Raised when configuration is invalid"""
    def __init__(self, message: str = "Configuration error", config_key: str = None):
        super().__init__(message)
        self.config_key = config_key

# Update ValidationException to accept field parameter
class ValidationException(CardioAIException):
    """Raised when validation fails"""
    def __init__(self, message: str, field: str = None, details: dict = None):
        super().__init__(message)
        self.field = field
        self.details = details or {}
'''

# Fix logging.py to add missing audit logger methods
LOGGING_ADDITIONS = '''
class AuditLogger:
    """Enhanced audit logger for medical compliance"""
    
    def __init__(self, logger):
        self.logger = logger
    
    def log_data_access(self, user_id: int, resource_type: str, resource_id: str,
                       action: str, ip_address: str = None, **kwargs):
        """Log data access for HIPAA compliance"""
        self.logger.info(
            "data_access",
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            ip_address=ip_address,
            **kwargs
        )
    
    def log_medical_action(self, user_id: int, action: str, patient_id: str,
                          details: dict = None, **kwargs):
        """Log medical actions"""
        self.logger.info(
            "medical_action",
            user_id=user_id,
            action=action,
            patient_id=patient_id,
            details=details or {},
            **kwargs
        )
    
    def log_analysis_created(self, user_id: int, analysis_id: str, 
                           patient_id: str, **kwargs):
        """Log ECG analysis creation"""
        self.logger.info(
            "analysis_created",
            user_id=user_id,
            analysis_id=analysis_id,
            patient_id=patient_id,
            **kwargs
        )
    
    def log_clinical_decision(self, user_id: int, decision_type: str,
                            analysis_id: str, details: dict = None, **kwargs):
        """Log clinical decisions"""
        self.logger.info(
            "clinical_decision",
            user_id=user_id,
            decision_type=decision_type,
            analysis_id=analysis_id,
            details=details or {},
            **kwargs
        )

# Update the get_logger function
def get_logger(name: str) -> AuditLogger:
    """Get an audit logger instance"""
    base_logger = logging.getLogger(name)
    return AuditLogger(base_logger)

# Create global audit logger
audit_logger = get_logger("audit")
'''


def fix_config_file():
    """Add missing attributes to config.py"""
    config_path = Path(__file__).parent / "config.py"

    # Read existing content
    with open(config_path, "r") as f:
        content = f.read()

    # Check if additions are needed
    if "BACKEND_CORS_ORIGINS" not in content:
        # Find the class definition
        class_end = content.find("settings = Settings()")
        if class_end == -1:
            class_end = len(content) - 100  # Near the end

        # Insert additions before the end of the class
        insertion_point = content.rfind("\n", 0, class_end)
        new_content = (
            content[:insertion_point]
            + "\n"
            + CONFIG_ADDITIONS
            + content[insertion_point:]
        )

        # Write back
        with open(config_path, "w") as f:
            f.write(new_content)

        print("✓ config.py updated with missing attributes")
    else:
        print("✓ config.py already has required attributes")


def fix_constants_file():
    """Add missing enums to constants.py"""
    constants_path = Path(__file__).parent / "constants.py"

    # Read existing content
    with open(constants_path, "r") as f:
        content = f.read()

    # Check if additions are needed
    if "PATIENT = " not in content:
        # Append additions
        with open(constants_path, "a") as f:
            f.write("\n" + CONSTANTS_ADDITIONS)

        print("✓ constants.py updated with missing enums")
    else:
        print("✓ constants.py already has required enums")


def fix_exceptions_file():
    """Add missing exceptions to exceptions.py"""
    exceptions_path = Path(__file__).parent / "exceptions.py"

    # Read existing content
    with open(exceptions_path, "r") as f:
        content = f.read()

    # Check if additions are needed
    if "AuthorizationException" not in content:
        # Find a good insertion point (after other exception classes)
        insertion_point = content.rfind("class")
        if insertion_point != -1:
            # Find the end of that class
            next_class = content.find("\n\nclass", insertion_point + 5)
            if next_class == -1:
                next_class = len(content)

            # Insert after the last class
            new_content = (
                content[:next_class]
                + "\n\n"
                + EXCEPTIONS_ADDITIONS
                + content[next_class:]
            )
        else:
            new_content = content + "\n\n" + EXCEPTIONS_ADDITIONS

        # Write back
        with open(exceptions_path, "w") as f:
            f.write(new_content)

        print("✓ exceptions.py updated with missing exceptions")
    else:
        print("✓ exceptions.py already has required exceptions")


def fix_logging_file():
    """Add missing audit logger methods to logging.py"""
    logging_path = Path(__file__).parent / "logging.py"

    # Read existing content
    with open(logging_path, "r") as f:
        content = f.read()

    # Check if additions are needed
    if "AuditLogger" not in content or "log_medical_action" not in content:
        # Replace or add the audit logger class
        if "AuditLogger" in content:
            # Replace existing class
            start = content.find("class AuditLogger")
            end = content.find("\n\nclass", start)
            if end == -1:
                end = content.find("\n\ndef", start)
            if end == -1:
                end = len(content)

            new_content = content[:start] + LOGGING_ADDITIONS.strip() + content[end:]
        else:
            # Add new class
            new_content = content + "\n\n" + LOGGING_ADDITIONS

        # Write back
        with open(logging_path, "w") as f:
            f.write(new_content)

        print("✓ logging.py updated with audit logger methods")
    else:
        print("✓ logging.py already has required methods")


def main():
    """Run all fixes"""
    print("Fixing core module issues...")

    fix_config_file()
    fix_constants_file()
    fix_exceptions_file()
    fix_logging_file()

    print("\n✅ All core module fixes completed!")


if __name__ == "__main__":
    main()
