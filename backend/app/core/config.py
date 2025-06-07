"""
Application configuration settings.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

from pydantic import ValidationInfo, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    PROJECT_NAME: str = "CardioAI Pro"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    API_V1_STR: str = "/api/v1"

    SECRET_KEY: str = "dev-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    ALGORITHM: str = "HS256"

    DATABASE_URL: str = "sqlite+aiosqlite:///./cardioai.db"
    TEST_DATABASE_URL: str = "sqlite+aiosqlite:///./cardioai_test.db"



    ALLOWED_HOSTS: list[str] = ["*"]

    @field_validator("ALLOWED_HOSTS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: str | list[str]) -> list[str] | str:
        """Assemble CORS origins."""
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, list | str):
            return v
        raise ValueError(v)

    UPLOAD_DIR: str = "/app/uploads"
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_FILE_TYPES: list[str] = [
        "application/pdf",
        "image/jpeg",
        "image/png",
        "application/dicom",
        "text/plain",
        "application/xml",
    ]

    MODELS_DIR: str = "/app/models"
    MODEL_CACHE_SIZE: int = 3
    INFERENCE_TIMEOUT: int = 30

    ECG_SAMPLE_RATE: int = 500
    ECG_DURATION_SECONDS: int = 10
    ECG_LEADS: list[str] = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    MIN_VALIDATION_SCORE: float = 0.8
    REQUIRE_DOUBLE_VALIDATION_CRITICAL: bool = True
    VALIDATION_EXPIRY_HOURS: int = 72
    MIN_EXPERIENCE_YEARS_CRITICAL: int = 5

    ENABLE_NOTIFICATIONS: bool = True
    NOTIFICATION_RATE_LIMIT: int = 100
    WEBSOCKET_HEARTBEAT_INTERVAL: int = 30

    ENABLE_METRICS: bool = True
    SENTRY_DSN: str | None = None



    AUDIT_LOG_RETENTION_DAYS: int = 2555  # 7 years
    ENABLE_DIGITAL_SIGNATURES: bool = True
    
    MODELS_DIR: str = "models"
    REQUIRE_AUDIT_TRAIL: bool = True

    FIRST_SUPERUSER: str = "admin@cardioai.pro"
    FIRST_SUPERUSER_EMAIL: str = "admin@cardioai.pro"
    FIRST_SUPERUSER_PASSWORD: str = "changeme123"

    MAX_ECG_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    ECG_UPLOAD_DIR: str = "uploads/ecg"

    model_config = {
        "case_sensitive": True,
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }


settings = Settings()
