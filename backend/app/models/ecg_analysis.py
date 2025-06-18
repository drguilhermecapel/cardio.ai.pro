"""
ECG Analysis Database Model
SQLAlchemy model for ECG analysis records
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from uuid import UUID, uuid4

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, 
    Text, JSON, ForeignKey, Enum as SQLEnum, Index
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.ext.hybrid import hybrid_property

from app.db.base import Base
from app.schemas.ecg_analysis import ProcessingStatus, ClinicalUrgency


class ECGAnalysis(Base):
    """ECG Analysis database model"""
    
    __tablename__ = "ecg_analyses"
    
    # Primary key
    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), 
        primary_key=True, 
        default=uuid4,
        index=True
    )
    
    # Foreign keys
    patient_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("patients.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    user_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )
    
    # Status and timestamps
    status: Mapped[ProcessingStatus] = mapped_column(
        SQLEnum(ProcessingStatus),
        default=ProcessingStatus.PENDING,
        nullable=False,
        index=True
    )
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
        index=True
    )
    
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime,
        onupdate=datetime.utcnow,
        nullable=True
    )
    
    processed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime,
        nullable=True,
        index=True
    )
    
    # File information
    file_info: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        default=dict,
        nullable=False
    )
    
    # Recording information
    recording_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime,
        nullable=True,
        index=True
    )
    
    device_info: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True
    )
    
    # Analysis results
    measurements: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True
    )
    
    annotations: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(
        JSON,
        nullable=True
    )
    
    pathologies: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(
        JSON,
        nullable=True
    )
    
    clinical_urgency: Mapped[Optional[ClinicalUrgency]] = mapped_column(
        SQLEnum(ClinicalUrgency),
        nullable=True,
        index=True
    )
    
    # Clinical information
    recommendations: Mapped[Optional[List[str]]] = mapped_column(
        JSON,
        nullable=True
    )
    
    clinical_notes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True
    )
    
    clinical_context: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True
    )
    
    # Quality metrics
    quality_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        index=True
    )
    
    confidence_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        index=True
    )
    
    # Validation
    validated: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True
    )
    
    validated_by: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True
    )
    
    validation_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime,
        nullable=True
    )
    
    # Additional metadata
    metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True
    )
    
    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True
    )
    
    # Relationships
    patient: Mapped["Patient"] = relationship(
        "Patient",
        back_populates="ecg_analyses",
        lazy="joined"
    )
    
    user: Mapped[Optional["User"]] = relationship(
        "User",
        foreign_keys=[user_id],
        back_populates="ecg_analyses"
    )
    
    validator: Mapped[Optional["User"]] = relationship(
        "User",
        foreign_keys=[validated_by]
    )
    
    validations: Mapped[List["Validation"]] = relationship(
        "Validation",
        back_populates="ecg_analysis",
        cascade="all, delete-orphan"
    )
    
    # Hybrid properties
    @hybrid_property
    def is_urgent(self) -> bool:
        """Check if analysis requires urgent attention"""
        return self.clinical_urgency in [
            ClinicalUrgency.HIGH,
            ClinicalUrgency.CRITICAL
        ]
    
    @hybrid_property
    def is_completed(self) -> bool:
        """Check if analysis is completed"""
        return self.status == ProcessingStatus.COMPLETED
    
    @hybrid_property
    def has_pathologies(self) -> bool:
        """Check if pathologies were detected"""
        return bool(self.pathologies) and len(self.pathologies) > 0
    
    @hybrid_property
    def processing_duration(self) -> Optional[float]:
        """Calculate processing duration in seconds"""
        if self.processed_at and self.created_at:
            return (self.processed_at - self.created_at).total_seconds()
        return None
    
    @hybrid_property
    def heart_rate(self) -> Optional[float]:
        """Get heart rate from measurements"""
        if self.measurements:
            return self.measurements.get("heart_rate")
        return None
    
    @hybrid_property
    def primary_pathology(self) -> Optional[str]:
        """Get primary detected pathology"""
        if self.pathologies and len(self.pathologies) > 0:
            # Sort by probability and return highest
            sorted_pathologies = sorted(
                self.pathologies,
                key=lambda x: x.get("probability", 0),
                reverse=True
            )
            return sorted_pathologies[0].get("condition")
        return None
    
    # Methods
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": str(self.id),
            "patient_id": str(self.patient_id),
            "user_id": str(self.user_id) if self.user_id else None,
            "status": self.status.value,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "file_info": self.file_info,
            "recording_date": self.recording_date.isoformat() if self.recording_date else None,
            "device_info": self.device_info,
            "measurements": self.measurements,
            "annotations": self.annotations,
            "pathologies": self.pathologies,
            "clinical_urgency": self.clinical_urgency.value if self.clinical_urgency else None,
            "recommendations": self.recommendations,
            "clinical_notes": self.clinical_notes,
            "clinical_context": self.clinical_context,
            "quality_score": self.quality_score,
            "confidence_score": self.confidence_score,
            "validated": self.validated,
            "validated_by": str(self.validated_by) if self.validated_by else None,
            "validation_date": self.validation_date.isoformat() if self.validation_date else None,
            "metadata": self.metadata,
            "error_message": self.error_message,
            "is_urgent": self.is_urgent,
            "is_completed": self.is_completed,
            "has_pathologies": self.has_pathologies,
            "processing_duration": self.processing_duration,
            "heart_rate": self.heart_rate,
            "primary_pathology": self.primary_pathology
        }
    
    def update_measurements(self, measurements: Dict[str, Any]) -> None:
        """Update measurements"""
        self.measurements = measurements
        self.updated_at = datetime.utcnow()
    
    def update_pathologies(self, pathologies: List[Dict[str, Any]]) -> None:
        """Update detected pathologies"""
        self.pathologies = pathologies
        self.updated_at = datetime.utcnow()
        
        # Update clinical urgency based on pathologies
        if pathologies:
            max_severity = max(
                p.get("severity", "low") for p in pathologies
            )
            if max_severity == "critical":
                self.clinical_urgency = ClinicalUrgency.CRITICAL
            elif max_severity == "high":
                self.clinical_urgency = ClinicalUrgency.HIGH
            elif max_severity == "moderate":
                self.clinical_urgency = ClinicalUrgency.MODERATE
    
    def mark_as_validated(self, validator_id: UUID) -> None:
        """Mark analysis as validated"""
        self.validated = True
        self.validated_by = validator_id
        self.validation_date = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def add_clinical_note(self, note: str, append: bool = True) -> None:
        """Add clinical note"""
        if append and self.clinical_notes:
            self.clinical_notes += f"\n\n{note}"
        else:
            self.clinical_notes = note
        self.updated_at = datetime.utcnow()
    
    def __repr__(self) -> str:
        """String representation"""
        return (
            f"<ECGAnalysis(id={self.id}, "
            f"patient_id={self.patient_id}, "
            f"status={self.status.value}, "
            f"urgency={self.clinical_urgency.value if self.clinical_urgency else 'N/A'})>"
        )


# Indexes for performance
Index("idx_ecg_patient_created", ECGAnalysis.patient_id, ECGAnalysis.created_at.desc())
Index("idx_ecg_user_created", ECGAnalysis.user_id, ECGAnalysis.created_at.desc())
Index("idx_ecg_status_urgency", ECGAnalysis.status, ECGAnalysis.clinical_urgency)
Index("idx_ecg_validated_date", ECGAnalysis.validated, ECGAnalysis.validation_date)
