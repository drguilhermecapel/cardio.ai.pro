"""
Application configuration settings.
"""

from typing import Any

from pydantic import validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    PROJECT_NAME: str = "CardioAI Pro"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    API_V1_STR: str = "/api/v1"

    SECRET_KEY: str
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

    @validator("DATABASE_URL", pre=True)
    def assemble_db_connection(cls, v: str | None, values: dict[str, Any]) -> Any:
        """Assemble database URL."""
        if isinstance(v, str):
            return v
        user = values.get("POSTGRES_USER")
        password = values.get("POSTGRES_PASSWORD")
        host = values.get("POSTGRES_SERVER")
        port = values.get("POSTGRES_PORT")
        db = values.get("POSTGRES_DB")
        return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db}"

    @validator("TEST_DATABASE_URL", pre=True)
    def assemble_test_db_connection(cls, v: str | None, values: dict[str, Any]) -> Any:
        """Assemble test database URL."""
        if isinstance(v, str):
            return v
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

    @validator("REDIS_URL", pre=True)
    def assemble_redis_connection(cls, v: str | None, values: dict[str, Any]) -> str:
        """Assemble Redis URL."""
        if isinstance(v, str):
            return v

        password = values.get("REDIS_PASSWORD")
        auth = f":{password}@" if password else ""

        return (
            f"redis://{auth}{values.get('REDIS_HOST')}:"
            f"{values.get('REDIS_PORT')}/{values.get('REDIS_DB')}"
        )

    ALLOWED_HOSTS: list[str] = ["*"]

    @validator("ALLOWED_HOSTS", pre=True)
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

    @validator("CELERY_BROKER_URL", pre=True)
    def assemble_celery_broker(cls, v: str | None, values: dict[str, Any]) -> str:
        """Assemble Celery broker URL."""
        if isinstance(v, str):
            return v
        return values.get("REDIS_URL", "redis://localhost:6379/0")

    @validator("CELERY_RESULT_BACKEND", pre=True)
    def assemble_celery_backend(cls, v: str | None, values: dict[str, Any]) -> str:
        """Assemble Celery result backend URL."""
        if isinstance(v, str):
            return v
        return values.get("REDIS_URL", "redis://localhost:6379/0")

    AUDIT_LOG_RETENTION_DAYS: int = 2555  # 7 years
    ENABLE_DIGITAL_SIGNATURES: bool = True
    REQUIRE_AUDIT_TRAIL: bool = True

    class Config:
        """Pydantic config."""
        case_sensitive = True
        env_file = ".env"


settings = Settings()
