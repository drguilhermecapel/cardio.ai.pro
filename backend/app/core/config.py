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

    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "cardioai"
    POSTGRES_PASSWORD: str = "cardioai_dev_password"
    POSTGRES_DB: str = "cardioai_pro"
    POSTGRES_PORT: int = 5432
    DATABASE_URL: str | None = None
    TEST_DATABASE_URL: str | None = None
    STANDALONE_MODE: bool = True

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def assemble_db_connection(cls: "type[Settings]", v: str | None, info: ValidationInfo) -> Any:
        """Assemble database URL."""
        if isinstance(v, str):
            return v
        values = info.data
        
        if values.get("STANDALONE_MODE", True):
            return "sqlite+aiosqlite:///./cardioai.db"
            
        user = values.get("POSTGRES_USER")
        password = values.get("POSTGRES_PASSWORD")
        host = values.get("POSTGRES_SERVER")
        port = values.get("POSTGRES_PORT")
        db = values.get("POSTGRES_DB")
        return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db}"

    @field_validator("TEST_DATABASE_URL", mode="before")
    @classmethod
    def assemble_test_db_connection(cls: "type[Settings]", v: str | None, info: ValidationInfo) -> Any:
        """Assemble test database URL."""
        if isinstance(v, str):
            return v
        values = info.data
        user = values.get("POSTGRES_USER")
        password = values.get("POSTGRES_PASSWORD")
        host = values.get("POSTGRES_SERVER")
        port = values.get("POSTGRES_PORT")
        db = values.get("POSTGRES_DB")
        return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db}_test"

    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str | None = None
    REDIS_DB: int = 0
    REDIS_URL: str | None = None

    @field_validator("REDIS_URL", mode="before")
    @classmethod
    def assemble_redis_connection(cls: "type[Settings]", v: str | None, info: ValidationInfo) -> str:
        """Assemble Redis URL."""
        if isinstance(v, str):
            return v

        values = info.data
        
        if values.get("STANDALONE_MODE", True):
            return ""

        password = values.get("REDIS_PASSWORD")
        auth = f":{password}@" if password else ""

        return (
            f"redis://{auth}{values.get('REDIS_HOST')}:"
            f"{values.get('REDIS_PORT')}/{values.get('REDIS_DB')}"
        )

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

    CELERY_BROKER_URL: str | None = None
    CELERY_RESULT_BACKEND: str | None = None

    @field_validator("CELERY_BROKER_URL", mode="before")
    @classmethod
    def assemble_celery_broker(cls: "type[Settings]", v: str | None, info: ValidationInfo) -> str:
        """Assemble Celery broker URL."""
        if isinstance(v, str):
            return v
        values = info.data
        
        if values.get("STANDALONE_MODE", True):
            return ""
            
        redis_url = values.get("REDIS_URL")
        return str(redis_url) if redis_url else "redis://localhost:6379/0"

    @field_validator("CELERY_RESULT_BACKEND", mode="before")
    @classmethod
    def assemble_celery_backend(cls: "type[Settings]", v: str | None, info: ValidationInfo) -> str:
        """Assemble Celery result backend URL."""
        if isinstance(v, str):
            return v
        values = info.data
        
        if values.get("STANDALONE_MODE", True):
            return ""
            
        redis_url = values.get("REDIS_URL")
        return str(redis_url) if redis_url else "redis://localhost:6379/0"

    AUDIT_LOG_RETENTION_DAYS: int = 2555  # 7 years
    ENABLE_DIGITAL_SIGNATURES: bool = True
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
