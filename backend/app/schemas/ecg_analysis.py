"""
ECG Analysis schemas.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.core.constants import (
    AnalysisStatus,
    ClinicalUrgency,
    DiagnosisCategory,
    ECGLeads,
)


class ECGAnalysisBase(BaseModel):
    """Base ECG analysis schema."""
    patient_id: int
    original_filename: str = Field(..., min_length=1, max_length=255)
    acquisition_date: datetime
    sample_rate: int = Field(..., ge=100, le=2000)
    duration_seconds: float = Field(..., ge=1.0, le=300.0)
    leads_count: int = Field(..., ge=1, le=15)
    leads_names: list[str] = Field(..., min_length=1, max_length=15)
    device_manufacturer: str | None = Field(None, max_length=100)
    device_model: str | None = Field(None, max_length=100)
    device_serial: str | None = Field(None, max_length=50)

    @field_validator('leads_names')
    @classmethod
    def validate_leads(cls, v: list[str]) -> list[str]:
        """Validate ECG leads."""
        valid_leads = [lead.value for lead in ECGLeads]
        for lead in v:
            if lead not in valid_leads:
                raise ValueError(f'Invalid ECG lead: {lead}')
        return v


class ECGAnalysisCreate(ECGAnalysisBase):
    """ECG analysis creation schema."""
    pass


class ECGAnalysisUpdate(BaseModel):
    """ECG analysis update schema."""
    clinical_notes: str | None = None
    recommendations: list[str] | None = None
    clinical_urgency: ClinicalUrgency | None = None


class ECGAnalysisInDB(ECGAnalysisBase):
    """ECG analysis in database schema."""
    id: int
    analysis_id: str
    created_by: int
    file_path: str
    file_hash: str
    file_size: int
    status: AnalysisStatus
    processing_started_at: datetime | None
    processing_completed_at: datetime | None
    processing_duration_ms: int | None
    ai_confidence: float | None
    ai_predictions: dict[str, Any] | None
    ai_interpretability: dict[str, Any] | None
    heart_rate_bpm: int | None
    rhythm: str | None
    pr_interval_ms: int | None
    qrs_duration_ms: int | None
    qt_interval_ms: int | None
    qtc_interval_ms: int | None
    primary_diagnosis: str | None
    secondary_diagnoses: list[str] | None
    diagnosis_category: DiagnosisCategory | None
    icd10_codes: list[str] | None
    clinical_urgency: ClinicalUrgency
    requires_immediate_attention: bool
    clinical_notes: str | None
    recommendations: list[str] | None
    signal_quality_score: float | None
    noise_level: float | None
    baseline_wander: float | None
    is_validated: bool
    validation_required: bool
    error_message: str | None
    retry_count: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ECGAnalysis(ECGAnalysisInDB):
    """ECG analysis response schema."""
    is_critical: bool
    processing_time_seconds: float | None

    class Config:
        from_attributes = True


class ECGAnalysisList(BaseModel):
    """ECG analysis list response schema."""
    analyses: list[ECGAnalysis]
    total: int
    page: int
    size: int


class ECGAnalysisSearch(BaseModel):
    """ECG analysis search schema."""
    patient_id: int | None = None
    status: AnalysisStatus | None = None
    clinical_urgency: ClinicalUrgency | None = None
    diagnosis_category: DiagnosisCategory | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    is_validated: bool | None = None
    requires_validation: bool | None = None


class ECGMeasurementBase(BaseModel):
    """Base ECG measurement schema."""
    measurement_type: str = Field(..., min_length=1, max_length=50)
    lead_name: str = Field(..., min_length=1, max_length=10)
    value: float
    unit: str = Field(..., min_length=1, max_length=20)
    start_time_ms: float | None = None
    end_time_ms: float | None = None
    confidence: float | None = Field(None, ge=0.0, le=1.0)
    is_normal: bool | None = None
    normal_min: float | None = None
    normal_max: float | None = None


class ECGMeasurementCreate(ECGMeasurementBase):
    """ECG measurement creation schema."""
    pass


class ECGMeasurement(ECGMeasurementBase):
    """ECG measurement response schema."""
    id: int
    analysis_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ECGAnnotationBase(BaseModel):
    """Base ECG annotation schema."""
    annotation_type: str = Field(..., min_length=1, max_length=50)
    label: str = Field(..., min_length=1, max_length=100)
    time_ms: float = Field(..., ge=0.0)
    lead_name: str | None = Field(None, max_length=10)
    confidence: float | None = Field(None, ge=0.0, le=1.0)
    properties: dict[str, Any] | None = None
    source: str = Field(..., pattern="^(ai|manual|algorithm)$")


class ECGAnnotationCreate(ECGAnnotationBase):
    """ECG annotation creation schema."""
    pass


class ECGAnnotation(ECGAnnotationBase):
    """ECG annotation response schema."""
    id: int
    analysis_id: int
    created_by: int | None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ECGProcessingResult(BaseModel):
    """ECG processing result schema."""
    analysis_id: str
    status: AnalysisStatus
    ai_confidence: float | None
    predictions: dict[str, Any] | None
    interpretability: dict[str, Any] | None
    measurements: list[ECGMeasurement]
    annotations: list[ECGAnnotation]
    clinical_findings: dict[str, Any]
    quality_metrics: dict[str, float]
    processing_time_ms: int
    error_message: str | None


class ECGUploadResponse(BaseModel):
    """ECG upload response schema."""
    analysis_id: str
    message: str
    status: AnalysisStatus
    estimated_processing_time_seconds: int
    file_type: str | None = None
    document_scanning_metadata: dict[str, Any] | None = None
