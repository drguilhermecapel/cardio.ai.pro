"""
Core Configuration
Central configuration for CardioAI Pro system
"""

import os
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from pydantic import AnyHttpUrl, field_validator, PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings"""
    
    # App Info
    APP_NAME: str = "CardioAI Pro"
    APP_VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = True
    
    # Security
    SECRET_KEY: str = "your-super-secret-key-change-this-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # CORS
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []
    
    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    # Database
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "cardioai"
    POSTGRES_PASSWORD: str = "cardioai123"
    POSTGRES_DB: str = "cardioai_db"
    
    DATABASE_URL: Optional[str] = None
    DATABASE_SYNC_URL: Optional[str] = None
    
    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def assemble_db_connection(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        if isinstance(v, str):
            return v
        return PostgresDsn.build(
            scheme="postgresql+asyncpg",
            username=values.data.get("POSTGRES_USER"),
            password=values.data.get("POSTGRES_PASSWORD"),
            host=values.data.get("POSTGRES_SERVER"),
            port=values.data.get("POSTGRES_PORT"),
            path=values.data.get("POSTGRES_DB") or "",
        ).unicode_string()
    
    @field_validator("DATABASE_SYNC_URL", mode="before")  
    @classmethod
    def assemble_sync_db_connection(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        if isinstance(v, str):
            return v
        return PostgresDsn.build(
            scheme="postgresql",
            username=values.data.get("POSTGRES_USER"),
            password=values.data.get("POSTGRES_PASSWORD"),
            host=values.data.get("POSTGRES_SERVER"),
            port=values.data.get("POSTGRES_PORT"),
            path=values.data.get("POSTGRES_DB") or "",
        ).unicode_string()
    
    # Redis (optional)
    REDIS_URL: Optional[str] = None
    CACHE_EXPIRE_MINUTES: int = 60
    
    # File Upload
    UPLOAD_DIR: str = "uploads"
    MAX_UPLOAD_SIZE: int = 104857600  # 100MB
    ALLOWED_EXTENSIONS: List[str] = ["csv", "txt", "edf", "npy", "mat", "xml"]
    CHUNK_SIZE: int = 1024 * 1024  # 1MB chunks
    
    # ML Models
    MODEL_PATH: str = "app/ml/models"
    MODEL_CACHE_SIZE: int = 5
    MODEL_VERSION: str = "1.0.0"
    
    # ECG Processing
    DEFAULT_SAMPLING_RATE: int = 500  # Hz
    DEFAULT_LEADS: int = 12
    MIN_SIGNAL_DURATION: float = 10.0  # seconds
    MAX_SIGNAL_DURATION: float = 86400.0  # 24 hours
    
    # Analysis Settings
    R_PEAK_DETECTION_METHOD: str = "pan_tompkins"
    BASELINE_FILTER_CUTOFF: float = 0.5  # Hz
    POWERLINE_FREQUENCIES: List[int] = [50, 60]  # Hz
    BANDPASS_LOW: float = 0.5  # Hz
    BANDPASS_HIGH: float = 150.0  # Hz
    
    # Clinical Thresholds
    NORMAL_HR_MIN: int = 60
    NORMAL_HR_MAX: int = 100
    BRADYCARDIA_THRESHOLD: int = 60
    TACHYCARDIA_THRESHOLD: int = 100
    QTC_NORMAL_MAX: int = 450
    QTC_PROLONGED: int = 500
    PR_NORMAL_MIN: int = 120
    PR_NORMAL_MAX: int = 200
    QRS_NORMAL_MAX: int = 120
    
    # Monitoring
    ENABLE_MONITORING: bool = True
    MEMORY_CHECK_INTERVAL: int = 60  # seconds
    MEMORY_THRESHOLD: float = 80.0  # percentage
    PROCESS_MEMORY_THRESHOLD: float = 70.0  # percentage
    LOG_LEVEL: str = "INFO"
    
    # Email (optional)
    SMTP_TLS: bool = True
    SMTP_PORT: Optional[int] = 587
    SMTP_HOST: Optional[str] = None
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    EMAILS_FROM_EMAIL: Optional[str] = None
    EMAILS_FROM_NAME: Optional[str] = None
    
    # Celery (optional)
    CELERY_BROKER_URL: Optional[str] = None
    CELERY_RESULT_BACKEND: Optional[str] = None
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    STATIC_DIR: Path = BASE_DIR / "static"
    TEMPLATES_DIR: Path = BASE_DIR / "templates"
    LOGS_DIR: Path = BASE_DIR / "logs"
    REPORTS_DIR: Path = BASE_DIR / "reports"
    
    # First user
    FIRST_SUPERUSER_EMAIL: str = "admin@cardioai.com"
    FIRST_SUPERUSER_PASSWORD: str = "admin123"
    
    # Other
    SENTRY_DSN: Optional[str] = None
    ENVIRONMENT: str = "development"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True
    )
    
    def get_upload_path(self, filename: str) -> Path:
        """Get full upload path for a file"""
        upload_dir = self.BASE_DIR / self.UPLOAD_DIR
        upload_dir.mkdir(exist_ok=True)
        return upload_dir / filename
    
    def get_model_path(self, model_name: str) -> Path:
        """Get full path for a model file"""
        model_dir = self.BASE_DIR / self.MODEL_PATH
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir / model_name
    
    def get_report_path(self, report_name: str) -> Path:
        """Get full path for a report file"""
        self.REPORTS_DIR.mkdir(exist_ok=True)
        return self.REPORTS_DIR / report_name
    
    def get_log_path(self, log_name: str) -> Path:
        """Get full path for a log file"""
        self.LOGS_DIR.mkdir(exist_ok=True)
        return self.LOGS_DIR / log_name
    
    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.ENVIRONMENT.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.ENVIRONMENT.lower() == "development"
    
    def validate_file_extension(self, filename: str) -> bool:
        """Validate file extension"""
        ext = filename.split('.')[-1].lower()
        return ext in self.ALLOWED_EXTENSIONS
    
    def validate_file_size(self, size: int) -> bool:
        """Validate file size"""
        return 0 < size <= self.MAX_UPLOAD_SIZE


# Create settings instance
settings = Settings()

# Export commonly used values
PROJECT_NAME = settings.APP_NAME
API_V1_STR = settings.API_V1_STR
SECRET_KEY = settings.SECRET_KEY
ALGORITHM = settings.ALGORITHM

# Ensure required directories exist
for directory in [settings.LOGS_DIR, settings.REPORTS_DIR, settings.STATIC_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
