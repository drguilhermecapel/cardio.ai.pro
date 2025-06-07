"""
ECG Analysis models.
"""

from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.constants import AnalysisStatus, ClinicalUrgency, DiagnosisCategory
from app.models.base import Base

if TYPE_CHECKING:
    from app.models.patient import Patient
    from app.models.user import User
    from app.models.validation import Validation


class ECGAnalysis(Base):
    """ECG Analysis model."""

    __tablename__ = "ecg_analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    analysis_id: Mapped[str] = mapped_column(
        String(50), unique=True, index=True, nullable=False
    )

    patient_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("patients.id"), nullable=False, index=True
    )
    created_by: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False, index=True
    )

    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)

    acquisition_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    sample_rate: Mapped[int] = mapped_column(Integer, nullable=False)
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    leads_count: Mapped[int] = mapped_column(Integer, nullable=False)
    leads_names: Mapped[list[str]] = mapped_column(JSON, nullable=False)

    device_manufacturer: Mapped[str | None] = mapped_column(String(100))
    device_model: Mapped[str | None] = mapped_column(String(100))
    device_serial: Mapped[str | None] = mapped_column(String(50))

    status: Mapped[AnalysisStatus] = mapped_column(
        String(20), nullable=False, default=AnalysisStatus.PENDING
    )
    processing_started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    processing_completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    processing_duration_ms: Mapped[int | None] = mapped_column(Integer)

    ai_confidence: Mapped[float | None] = mapped_column(Float)
    ai_predictions: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    ai_interpretability: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    heart_rate_bpm: Mapped[int | None] = mapped_column(Integer)
    rhythm: Mapped[str | None] = mapped_column(String(100))
    pr_interval_ms: Mapped[int | None] = mapped_column(Integer)
    qrs_duration_ms: Mapped[int | None] = mapped_column(Integer)
    qt_interval_ms: Mapped[int | None] = mapped_column(Integer)
    qtc_interval_ms: Mapped[int | None] = mapped_column(Integer)

    primary_diagnosis: Mapped[str | None] = mapped_column(String(200))
    secondary_diagnoses: Mapped[list[str] | None] = mapped_column(JSON)
    diagnosis_category: Mapped[DiagnosisCategory | None] = mapped_column(String(50))
    icd10_codes: Mapped[list[str] | None] = mapped_column(JSON)

    clinical_urgency: Mapped[ClinicalUrgency] = mapped_column(
        String(20), nullable=False, default=ClinicalUrgency.LOW
    )
    requires_immediate_attention: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    clinical_notes: Mapped[str | None] = mapped_column(Text)
    recommendations: Mapped[list[str] | None] = mapped_column(JSON)

    signal_quality_score: Mapped[float | None] = mapped_column(Float)
    noise_level: Mapped[float | None] = mapped_column(Float)
    baseline_wander: Mapped[float | None] = mapped_column(Float)

    is_validated: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    validation_required: Mapped[bool] = mapped_column(
        Boolean, default=True, nullable=False
    )

    error_message: Mapped[str | None] = mapped_column(Text)
    retry_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    patient: Mapped["Patient"] = relationship("Patient", back_populates="analyses")
    created_by_user: Mapped["User"] = relationship(
        "User", back_populates="analyses", foreign_keys=[created_by]
    )
    validations: Mapped[list["Validation"]] = relationship(
        "Validation", back_populates="analysis", cascade="all, delete-orphan"
    )
    measurements: Mapped[list["ECGMeasurement"]] = relationship(
        "ECGMeasurement", back_populates="analysis", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<ECGAnalysis(id={self.id}, analysis_id='{self.analysis_id}', status='{self.status}')>"

    @property
    def is_critical(self) -> bool:
        """Check if analysis indicates critical condition."""
        return (
            self.clinical_urgency == ClinicalUrgency.CRITICAL
            or self.requires_immediate_attention
        )

    @property
    def processing_time_seconds(self) -> float | None:
        """Get processing time in seconds."""
        if self.processing_duration_ms:
            return self.processing_duration_ms / 1000
        return None


class ECGMeasurement(Base):
    """ECG measurement model for storing detailed measurements."""

    __tablename__ = "ecg_measurements"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    analysis_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("ecg_analyses.id"), nullable=False, index=True
    )

    measurement_type: Mapped[str] = mapped_column(String(50), nullable=False)
    lead_name: Mapped[str] = mapped_column(String(10), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    unit: Mapped[str] = mapped_column(String(20), nullable=False)

    start_time_ms: Mapped[float | None] = mapped_column(Float)
    end_time_ms: Mapped[float | None] = mapped_column(Float)

    confidence: Mapped[float | None] = mapped_column(Float)
    is_normal: Mapped[bool | None] = mapped_column(Boolean)

    normal_min: Mapped[float | None] = mapped_column(Float)
    normal_max: Mapped[float | None] = mapped_column(Float)

    analysis: Mapped["ECGAnalysis"] = relationship(
        "ECGAnalysis", back_populates="measurements"
    )

    def __repr__(self) -> str:
        return f"<ECGMeasurement(id={self.id}, type='{self.measurement_type}', value={self.value})>"


class ECGAnnotation(Base):
    """ECG annotation model for storing beat annotations and events."""

    __tablename__ = "ecg_annotations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    analysis_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("ecg_analyses.id"), nullable=False, index=True
    )

    annotation_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # beat, event, artifact
    label: Mapped[str] = mapped_column(String(100), nullable=False)

    time_ms: Mapped[float] = mapped_column(Float, nullable=False)
    lead_name: Mapped[str | None] = mapped_column(String(10))

    confidence: Mapped[float | None] = mapped_column(Float)
    properties: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    source: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # ai, manual, algorithm
    created_by: Mapped[int | None] = mapped_column(Integer, ForeignKey("users.id"))

    def __repr__(self) -> str:
        return f"<ECGAnnotation(id={self.id}, type='{self.annotation_type}', label='{self.label}')>"
